import asyncio
import json
import struct
import zlib

import pytest

from modmex_ai import Agent, tool
from modmex_ai.errors import ModelRefusalError, ProviderError, RateLimitError
from modmex_ai.http import HttpResponse
from modmex_ai.messages import FileInput, ImageInput, Message, TextInput
from modmex_ai.models import BedrockSettings, ModelRequest, ModelSettings, ModelStreamEventType
from modmex_ai.providers.bedrock import BedrockConverseModel
from modmex_ai.providers.bedrock.mapper import to_converse_payload
from modmex_ai.sessions import SessionItem
from tests.aws.sigv4_reference import verify

SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# Signing and the AWS event-stream decoder use botocore (the `aws` extra; part of
# the Lambda runtime). Without it these tests are skipped, not failed.
needs_botocore = pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("botocore") is None,
    reason="botocore not installed (pip install 'modmex-ai[aws]')",
)


@pytest.fixture(autouse=True)
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
    monkeypatch.setenv("AWS_SESSION_TOKEN", "session-token")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


class FakeHttp:
    def __init__(self, *bodies, stream=b""):
        self.bodies = list(bodies)
        self.stream_bytes = stream
        self.calls = []

    def post_json(self, url, *, headers=None, data=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "data": data})
        return HttpResponse(status_code=200, headers={"x-amzn-RequestId": "rid-1"}, body=self.bodies.pop(0))

    def post_json_stream(self, url, *, headers=None, data=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "data": data})
        return iter([self.stream_bytes[i:i + 7] for i in range(0, len(self.stream_bytes), 7)])  # odd chunking


def _reply(content, stop="end_turn", usage=None):
    return {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": stop,
            "usage": usage or {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}


def _payload(messages, model="us.anthropic.claude-opus-5-5", load_s3=None, **kwargs):
    return to_converse_payload(ModelRequest(messages=messages, **kwargs), model, load_s3=load_s3)


# --- signing -----------------------------------------------------------------


@needs_botocore
def test_requests_are_sigv4_signed_over_the_exact_bytes_sent():
    http = FakeHttp(_reply([{"text": "hola"}]))
    model = BedrockConverseModel("us.anthropic.claude-opus-5-5:0", http_client=http)
    response = model.complete(ModelRequest(messages=[Message(role="user", content="hi")]))

    call = http.calls[0]
    assert response.output_text == "hola" and response.request_id == "rid-1"
    assert call["url"] == "https://bedrock-runtime.us-west-2.amazonaws.com/model/us.anthropic.claude-opus-5-5%3A0/converse"
    assert isinstance(call["data"], bytes)
    assert call["headers"]["X-Amz-Security-Token"] == "session-token"
    assert "Credential=AKIDEXAMPLE/" in call["headers"]["Authorization"]
    assert "/us-west-2/bedrock/aws4_request" in call["headers"]["Authorization"]
    assert verify("POST", call["url"], call["headers"], call["data"], secret_key=SECRET)
    tampered = call["data"].replace(b"hi", b"ho")
    assert not verify("POST", call["url"], call["headers"], tampered, secret_key=SECRET)


def test_bedrock_api_key_uses_bearer_instead_of_signing():
    http = FakeHttp(_reply([{"text": "ok"}]))
    BedrockConverseModel("m", api_key="bedrock-key", http_client=http).complete(
        ModelRequest(messages=[Message(role="user", content="hi")])
    )
    assert http.calls[0]["headers"]["authorization"] == "Bearer bedrock-key"
    assert isinstance(http.calls[0]["data"], dict)
    with pytest.raises(ValueError, match="either"):
        BedrockConverseModel("m", api_key="k", auth=object())


@needs_botocore
def test_openai_and_anthropic_adapters_sign_when_given_auth():
    from modmex_ai.aws import AwsSigV4
    from modmex_ai.providers.anthropic import AnthropicMessagesModel
    from modmex_ai.providers.openai import OpenAIResponsesModel

    openai_http = FakeHttp({"id": "r", "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]})
    OpenAIResponsesModel(
        "openai.gpt-x", base_url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
        http_client=openai_http, auth=AwsSigV4(service="bedrock"),
    ).complete(ModelRequest(messages=[Message(role="user", content="hi")]))
    call = openai_http.calls[0]
    assert call["url"].endswith("/openai/v1/responses")
    assert "authorization" not in {k for k in call["headers"] if call["headers"][k].startswith("Bearer")}
    assert verify("POST", call["url"], call["headers"], call["data"], secret_key=SECRET)

    anthropic_http = FakeHttp({"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {}})
    AnthropicMessagesModel(
        "anthropic.claude-opus-5-5", base_url="https://bedrock-mantle.us-west-2.api.aws/anthropic/v1",
        http_client=anthropic_http, auth=AwsSigV4(service="bedrock-mantle"),
    ).complete(ModelRequest(messages=[Message(role="user", content="hi")]))
    call = anthropic_http.calls[0]
    assert "x-api-key" not in {k.lower() for k in call["headers"]}  # never leak ANTHROPIC_API_KEY to AWS
    assert "/bedrock-mantle/aws4_request" in call["headers"]["Authorization"]
    assert verify("POST", call["url"], call["headers"], call["data"], secret_key=SECRET)


# --- request mapping ---------------------------------------------------------


def test_payload_maps_system_tools_output_schema_and_settings():
    schema = {"type": "object", "properties": {"city": {"type": "string", "minLength": 1}}, "required": ["city"]}
    payload = _payload(
        [Message(role="system", content="Be brief."), Message(role="user", content="weather?")],
        tools=[{"name": "get_weather", "description": "Weather", "parameters": schema}],
        output_schema={"title": "Out", "type": "object", "properties": {"x": {"type": "string"}}},
        settings=ModelSettings(max_tokens=800, temperature=0.2, bedrock=BedrockSettings(
            cache_ttl="1h", additional_model_request_fields={"top_k": 5},
            guardrail_identifier="gr-1", guardrail_version="2")),
    )
    assert payload["system"] == [{"text": "Be brief."}, {"cachePoint": {"type": "default", "ttl": "1h"}}]
    assert payload["messages"] == [{"role": "user", "content": [
        {"text": "weather?"}, {"cachePoint": {"type": "default", "ttl": "1h"}}]}]
    spec = payload["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["strict"] is True and spec["inputSchema"]["json"]["additionalProperties"] is False
    assert "minLength" not in spec["inputSchema"]["json"]["properties"]["city"]
    fmt = payload["outputConfig"]["textFormat"]
    assert fmt["type"] == "json_schema" and fmt["structure"]["jsonSchema"]["name"] == "Out"
    assert json.loads(fmt["structure"]["jsonSchema"]["schema"])["additionalProperties"] is False
    assert payload["inferenceConfig"] == {"maxTokens": 800, "temperature": 0.2}
    assert payload["additionalModelRequestFields"] == {"top_k": 5}
    assert payload["guardrailConfig"] == {"guardrailIdentifier": "gr-1", "guardrailVersion": "2"}


def test_reasoning_effort_and_foreign_settings_are_rejected_with_guidance():
    from modmex_ai.models import OpenAISettings

    with pytest.raises(ValueError, match="additional_model_request_fields"):
        _payload([Message(role="user", content="x")], settings=ModelSettings(reasoning_effort="high"))
    # Another provider's settings are ignored (one request may go to several providers).
    payload = _payload([Message(role="user", content="x")], settings=ModelSettings(openai=OpenAISettings(prompt_cache_key="k")))
    assert "prompt_cache_key" not in str(payload)


def test_mid_conversation_instructions_become_user_text():
    payload = _payload([
        Message(role="user", content="q"),
        Message(role="assistant", content="a"),
        Message(role="user", content="q2"),
        Message(role="developer", content="Cliente: Acme"),  # e.g. turn_context
    ])
    assert payload["messages"][-1] == {"role": "user", "content": [
        {"text": "q2"}, {"text": "<system>\nCliente: Acme\n</system>"}]}


# --- files -------------------------------------------------------------------


def test_images_and_documents_travel_as_bytes_with_formats_and_safe_names():
    payload = _payload([Message(role="user", content=[
        TextInput(text="Revisa la prueba de entrega"),
        ImageInput(data=b"\x89PNG", media_type="image/png"),
        FileInput(data=b"%PDF-1.7", filename="POD #123 (final).pdf"),
        FileInput(data=b"%PDF-1.7", filename="POD #123 (final).pdf"),
    ])])
    blocks = payload["messages"][0]["content"]
    assert blocks[1] == {"image": {"format": "png", "source": {"bytes": "iVBORw=="}}}
    assert blocks[2]["document"]["format"] == "pdf"
    assert blocks[2]["document"]["name"] == "POD -123 (final)"
    assert blocks[3]["document"]["name"] == "POD -123 (final)-2"  # names must be unique


def test_s3_uris_go_direct_to_nova_and_are_resolved_for_other_models():
    loads = []
    loader = lambda uri: loads.append(uri) or b"%PDF-1.7"  # noqa: E731
    message = Message(role="user", content=[TextInput(text="lee"), FileInput(url="s3://media/pod.pdf", media_type="application/pdf")])

    nova = _payload([message], model="us.amazon.nova-pro-v1:0", load_s3=loader)
    assert nova["messages"][0]["content"][1]["document"]["source"] == {"s3Location": {"uri": "s3://media/pod.pdf"}}
    assert loads == []

    claude = _payload([message], load_s3=loader)
    assert claude["messages"][0]["content"][1]["document"]["source"] == {"bytes": "JVBERi0xLjc="}
    assert loads == ["s3://media/pod.pdf"]


@pytest.mark.parametrize("part, model, match", [
    (ImageInput(url="https://cdn.example.com/a.png", media_type="image/png"), "us.anthropic.claude-opus-5-5", "does not fetch URLs"),
    (FileInput(file_id="file_1", media_type="application/pdf"), "us.anthropic.claude-opus-5-5", "no Files API"),
    (FileInput(data=b"%PDF", media_type="application/pdf"), "us.openai.gpt-6-sol", "OpenAIResponsesModel"),
    (FileInput(data=b"x", media_type="application/zip"), "us.anthropic.claude-opus-5-5", "document format"),
    (ImageInput(data=b"x", media_type="image/bmp"), "us.anthropic.claude-opus-5-5", "image/png"),
])
def test_unsupported_file_inputs_fail_with_actionable_errors(part, model, match):
    with pytest.raises(ValueError, match=match):
        _payload([Message(role="user", content=[TextInput(text="x"), part])], model=model)


def test_document_without_text_is_rejected():
    with pytest.raises(ValueError, match="text block"):
        _payload([Message(role="user", content=[FileInput(data=b"%PDF", media_type="application/pdf")])])


# --- tool loop and reasoning replay -------------------------------------------


REASONING = {"reasoningContent": {"reasoningText": {"text": "", "signature": "sig-1"}}}


def test_agent_tool_loop_replays_signed_reasoning_and_reports_tool_errors():
    @tool
    def lookup_load(load_id: str) -> dict:
        """Look up a load."""
        if load_id == "BAD":
            raise RuntimeError("backend down")
        return {"status": "in transit"}

    turn = [REASONING, {"text": "Déjame revisar."},
            {"toolUse": {"toolUseId": "t1", "name": "lookup_load", "input": {"load_id": "BAD"}}}]
    http = FakeHttp(_reply(turn, stop="tool_use"), _reply([{"text": "No pude consultarla."}]))
    agent = Agent(name="a", instructions="Help.", tools=[lookup_load], tool_error_policy="return_error",
                  model=BedrockConverseModel("us.anthropic.claude-opus-5-5", api_key="k", http_client=http))

    assert agent.run("¿Y la carga BAD?").output == "No pude consultarla."
    second = http.calls[1]["data"]
    assert second["messages"][1] == {"role": "assistant", "content": turn}  # verbatim, signature included
    result = second["messages"][2]["content"][0]["toolResult"]
    assert result["toolUseId"] == "t1" and result["status"] == "error"


def test_reasoning_bound_to_an_old_tool_set_is_dropped():
    from modmex_ai.providers.bedrock.mapper import binding_digest

    stale = binding_digest({"system": [{"text": "otro agente"}]})
    item = SessionItem(type="function_call", tool_call_id="t1", name="a", arguments={},
                       provider_data={"bedrock": {"binding": stale, "content": [
                           REASONING, {"toolUse": {"toolUseId": "t1", "name": "a", "input": {}}}]}})
    payload = _payload([
        Message(role="system", content="agente actual"), Message(role="user", content="go"), item,
        SessionItem(type="function_call_output", tool_call_id="t1", output="ok"),
    ])
    assert payload["messages"][1]["content"] == [{"toolUse": {"toolUseId": "t1", "name": "a", "input": {}}}]


# --- responses ---------------------------------------------------------------


def test_usage_is_normalized_and_stop_reasons_map():
    http = FakeHttp(_reply([{"text": "ok"}], usage={
        "inputTokens": 20, "outputTokens": 5, "cacheReadInputTokens": 100, "cacheWriteInputTokens": 0, "totalTokens": 125}))
    response = BedrockConverseModel("m", api_key="k", http_client=http).complete(
        ModelRequest(messages=[Message(role="user", content="hi")]))
    assert (response.usage.input_tokens, response.usage.cached_input_tokens, response.usage.total_tokens) == (120, 100, 125)
    assert response.stop_reason == "end_turn" and response.provider == "bedrock"


@pytest.mark.parametrize("stop, content, error", [
    ("content_filtered", [], ModelRefusalError),
    ("max_tokens", [{"toolUse": {"toolUseId": "t", "name": "a", "input": {}}}], ProviderError),
    ("model_context_window_exceeded", [], ProviderError),
])
def test_failure_stop_reasons_raise(stop, content, error):
    http = FakeHttp(_reply(content, stop=stop))
    with pytest.raises(error):
        BedrockConverseModel("m", api_key="k", http_client=http).complete(
            ModelRequest(messages=[Message(role="user", content="hi")]))


# --- streaming (AWS binary event stream) --------------------------------------


def _frame(event_type, payload, message_type="event"):
    def header(name, value):
        n, v = name.encode(), value.encode()
        return bytes([len(n)]) + n + bytes([7]) + struct.pack(">H", len(v)) + v

    type_key = ":event-type" if message_type == "event" else ":exception-type"
    headers = header(":message-type", message_type) + header(type_key, event_type) + header(":content-type", "application/json")
    body = json.dumps(payload).encode()
    total = 12 + len(headers) + len(body) + 4
    prelude = struct.pack(">II", total, len(headers))
    prelude += struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
    message = prelude + headers + body
    return message + struct.pack(">I", zlib.crc32(message) & 0xFFFFFFFF)


STREAM = b"".join([
    _frame("messageStart", {"role": "assistant"}),
    _frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "pienso"}}}),
    _frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig-9"}}}),
    _frame("contentBlockStop", {"contentBlockIndex": 0}),
    _frame("contentBlockDelta", {"contentBlockIndex": 1, "delta": {"text": "Déjame "}}),
    _frame("contentBlockDelta", {"contentBlockIndex": 1, "delta": {"text": "revisar."}}),
    _frame("contentBlockStop", {"contentBlockIndex": 1}),
    _frame("contentBlockStart", {"contentBlockIndex": 2, "start": {"toolUse": {"toolUseId": "t9", "name": "lookup"}}}),
    _frame("contentBlockDelta", {"contentBlockIndex": 2, "delta": {"toolUse": {"input": '{"id": '}}}),
    _frame("contentBlockDelta", {"contentBlockIndex": 2, "delta": {"toolUse": {"input": '"L1"}'}}}),
    _frame("contentBlockStop", {"contentBlockIndex": 2}),
    _frame("messageStop", {"stopReason": "tool_use"}),
    _frame("metadata", {"usage": {"inputTokens": 30, "outputTokens": 9, "totalTokens": 39}}),
])


@needs_botocore
def test_converse_stream_decodes_binary_events_and_rebuilds_the_turn():
    model = BedrockConverseModel("us.anthropic.claude-opus-5-5", api_key="k", http_client=FakeHttp(stream=STREAM))
    events = list(model.stream(ModelRequest(messages=[Message(role="user", content="hi")])))

    text = [e.text_delta for e in events if e.type == ModelStreamEventType.TEXT_DELTA]
    tool = [e.tool_call.arguments for e in events if e.type == ModelStreamEventType.TOOL_CALL_DELTA]
    final = events[-1].response
    assert text == ["Déjame ", "revisar."]
    assert tool == ["", '{"id": ', '"L1"}']
    assert final.tool_calls[0].arguments == {"id": "L1"} and final.stop_reason == "tool_use"
    assert final.usage.input_tokens == 30
    replay = final.tool_calls[0].provider_data["bedrock"]["content"]
    assert replay[0] == {"reasoningContent": {"reasoningText": {"text": "pienso", "signature": "sig-9"}}}
    assert model.http_client.calls[0]["url"].endswith("/converse-stream")


@needs_botocore
def test_stream_exception_frames_raise_typed_errors():
    stream = _frame("messageStart", {"role": "assistant"}) + _frame(
        "throttlingException", {"message": "Too many requests"}, message_type="exception")
    model = BedrockConverseModel("m", api_key="k", http_client=FakeHttp(stream=stream))
    with pytest.raises(RateLimitError, match="Too many requests"):
        list(model.stream(ModelRequest(messages=[Message(role="user", content="hi")])))


@needs_botocore
def test_async_stream_uses_the_same_decoder():
    class AsyncFake:
        async def post_json_stream_gen(self):
            for i in range(0, len(STREAM), 11):
                yield STREAM[i:i + 11]

        def post_json_stream(self, url, *, headers=None, data=None, timeout=None):
            return self.post_json_stream_gen()

        async def close(self):
            pass

    model = BedrockConverseModel("m", api_key="k", async_http_client=AsyncFake())

    async def run():
        return [e async for e in model.astream(ModelRequest(messages=[Message(role="user", content="hi")]))]

    events = asyncio.run(run())
    assert events[-1].response.tool_calls[0].arguments == {"id": "L1"}


# --- S3 loader -----------------------------------------------------------------


def test_s3_loader_parses_uris_and_caches_objects():
    from modmex_ai.aws import S3ObjectLoader, parse_s3_uri

    class Client:
        def __init__(self):
            self.calls = 0

        def get_object(self, Bucket, Key):
            self.calls += 1

            class Body:
                def read(self_inner):
                    return f"{Bucket}/{Key}".encode()

            return {"Body": Body()}

    client = Client()
    loader = S3ObjectLoader(client=client)
    assert parse_s3_uri("s3://media/whatsapp/a%20b.jpg") == ("media", "whatsapp/a b.jpg")
    assert loader.load("s3://media/x.pdf") == b"media/x.pdf"
    assert loader.load("s3://media/x.pdf") == b"media/x.pdf"
    assert client.calls == 1
    with pytest.raises(ValueError):
        parse_s3_uri("https://media/x.pdf")


def test_anthropic_adapter_on_bedrock_rejects_unsupported_features():
    from modmex_ai.aws import AwsSigV4
    from modmex_ai.models import AnthropicSettings
    from modmex_ai.providers.anthropic import AnthropicMessagesModel

    model = AnthropicMessagesModel(
        "anthropic.claude-opus-5-5", base_url="https://bedrock-mantle.us-west-2.api.aws/anthropic/v1",
        http_client=FakeHttp(), auth=AwsSigV4(service="bedrock-mantle"),
    )
    with pytest.raises(ValueError, match="FallbackModel"):
        model.complete(ModelRequest(messages=[Message(role="user", content="hi")],
                                    settings=ModelSettings(anthropic=AnthropicSettings(fallbacks="default"))))
    with pytest.raises(ValueError, match="bytes"):
        model.complete(ModelRequest(messages=[Message(role="user", content=[
            TextInput(text="x"), ImageInput(url="https://cdn.example.com/a.png")])]))


def test_openai_structured_output_adds_the_strict_text_format_field():
    schema = {"title": "Out", "type": "object", "properties": {"x": {"type": "string"}}}
    gpt = _payload([Message(role="user", content="x")], model="us.openai.gpt-6.1-terra", output_schema=schema,
                   settings=ModelSettings(bedrock=BedrockSettings(additional_model_request_fields={"reasoning": {"effort": "low"}})))
    assert gpt["additionalModelRequestFields"] == {"reasoning": {"effort": "low"}, "text": {"format": {"strict": True}}}
    assert "outputConfig" in gpt
    claude = _payload([Message(role="user", content="x")], output_schema=schema)
    assert "additionalModelRequestFields" not in claude
