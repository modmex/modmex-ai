import asyncio

import pytest

from modmex_ai.http import HttpResponse
from modmex_ai.messages import Message
from modmex_ai.models import ModelRequest
from modmex_ai.providers.openai import OpenAIChatModel, OpenAIResponsesModel
from modmex_ai.models import ModelStreamEventType
from modmex_ai.providers.openai_compatible import OpenAICompatibleChatModel


class FakeHttpClient:
    def __init__(self, body):
        self.body = body
        self.calls = []

    def post_json(self, url, *, headers=None, data=None, timeout=None):
        self.calls.append({
            "url": url,
            "headers": headers,
            "data": data,
            "timeout": timeout,
        })
        return HttpResponse(status_code=200, headers={"x-request-id": "req-1"}, body=self.body)


class FakeAsyncHttpClient:
    def __init__(self, body):
        self.body = body
        self.calls = []

    async def post_json(self, url, *, headers=None, data=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "data": data, "timeout": timeout})
        return HttpResponse(status_code=200, headers={"x-request-id": "req-async"}, body=self.body)

    async def close(self):
        self.closed = True


def test_openai_responses_model_posts_to_responses_api():
    http = FakeHttpClient({
        "id": "resp-1",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
    })
    model = OpenAIResponsesModel(
        "gpt-test",
        api_key="key",
        organization="org",
        project="proj",
        http_client=http,
    )

    response = model.complete(ModelRequest(messages=[Message(role="user", content="hello")]))

    assert response.output_text == "ok"
    assert http.calls[0]["url"] == "https://api.openai.com/v1/responses"
    assert http.calls[0]["headers"]["authorization"] == "Bearer key"
    assert http.calls[0]["headers"]["OpenAI-Organization"] == "org"
    assert http.calls[0]["headers"]["OpenAI-Project"] == "proj"


def test_openai_chat_model_posts_to_chat_completions_api():
    http = FakeHttpClient({
        "id": "chat-1",
        "choices": [{"message": {"content": "ok"}}],
    })
    model = OpenAIChatModel("gpt-test", api_key="key", http_client=http)

    response = model.complete(ModelRequest(messages=[Message(role="user", content="hello")]))

    assert response.output_text == "ok"
    assert http.calls[0]["url"] == "https://api.openai.com/v1/chat/completions"
    assert http.calls[0]["headers"] == {"authorization": "Bearer key"}


def test_openai_compatible_chat_model_is_chat_model_alias():
    assert OpenAICompatibleChatModel is OpenAIChatModel


def test_openai_models_close_owned_async_clients_only():
    async def run():
        chat_http = FakeAsyncHttpClient({})
        responses_http = FakeAsyncHttpClient({})
        chat = OpenAIChatModel("gpt", async_http_client=chat_http)
        responses = OpenAIResponsesModel("gpt", async_http_client=responses_http)
        chat._owns_async_http_client = True
        responses._owns_async_http_client = True
        await chat.aclose()
        await responses.aclose()
        assert chat_http.closed and responses_http.closed

    asyncio.run(run())


def test_openai_models_expose_stream_methods():
    responses_model = OpenAIResponsesModel("gpt-test", http_client=FakeHttpClient({}))
    chat_model = OpenAIChatModel("gpt-test", http_client=FakeHttpClient({}))

    assert callable(responses_model.stream)
    assert callable(chat_model.stream)


def test_openai_models_expose_async_completion():
    async def run():
        responses = OpenAIResponsesModel(
            "gpt-test",
            http_client=FakeHttpClient({
                "id": "resp-1",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
            }),
        )
        chat = OpenAIChatModel(
            "gpt-test",
            http_client=FakeHttpClient({"id": "chat-1", "choices": [{"message": {"content": "ok"}}]}),
        )

        assert (await responses.acomplete(ModelRequest(messages=[]))).output_text == "ok"
        assert (await chat.acomplete(ModelRequest(messages=[]))).output_text == "ok"

    asyncio.run(run())


def test_openai_models_use_injected_native_async_transport():
    async def run():
        responses_http = FakeAsyncHttpClient({
            "id": "resp-1",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
        })
        chat_http = FakeAsyncHttpClient({
            "id": "chat-1",
            "choices": [{"message": {"content": "ok"}}],
        })
        responses = OpenAIResponsesModel("gpt-test", async_http_client=responses_http)
        chat = OpenAIChatModel("gpt-test", async_http_client=chat_http)

        assert (await responses.acomplete(ModelRequest(messages=[]))).output_text == "ok"
        assert (await chat.acomplete(ModelRequest(messages=[]))).output_text == "ok"
        assert responses_http.calls[0]["url"].endswith("/responses")
        assert chat_http.calls[0]["url"].endswith("/chat/completions")

    asyncio.run(run())


def test_openai_models_translate_sse_streams():
    class StreamHttp:
        def __init__(self, chunks):
            self.chunks = chunks
            self.calls = []

        def post_json_stream(self, url, *, headers=None, data=None, timeout=None):
            self.calls.append((url, headers, data, timeout))
            yield from self.chunks

    chat_http = StreamHttp([
        b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
        b'data: [DONE]\n\n',
    ])
    responses_http = StreamHttp([
        b'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n',
        b'data: {"type":"response.completed","response":{"id":"resp-1","output":[{"type":"message","content":[{"type":"output_text","text":"Hi"}]}]}}\n\n',
    ])

    chat_events = list(OpenAIChatModel("gpt-test", http_client=chat_http).stream(ModelRequest(messages=[])))
    responses_events = list(OpenAIResponsesModel("gpt-test", http_client=responses_http).stream(ModelRequest(messages=[])))

    assert chat_http.calls[0][2]["stream"] is True
    assert responses_http.calls[0][2]["stream"] is True
    assert chat_events[0].type == ModelStreamEventType.TEXT_DELTA
    assert chat_events[-1].response.output_text == "Hi"
    assert responses_events[0].type == ModelStreamEventType.TEXT_DELTA
    assert responses_events[-1].response.output_text == "Hi"


def test_openai_models_translate_sse_streams_asynchronously():
    class AsyncStreamHttp:
        def __init__(self, chunks):
            self.chunks = chunks
            self.calls = []

        async def post_json_stream(self, url, *, headers=None, data=None, timeout=None):
            self.calls.append((url, headers, data, timeout))
            for chunk in self.chunks:
                yield chunk

    async def collect():
        chat_http = AsyncStreamHttp([
            b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
            b'data: [DONE]\n\n',
        ])
        responses_http = AsyncStreamHttp([
            b'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n',
            b'data: {"type":"response.completed","response":{"id":"resp-1","output":[{"type":"message","content":[{"type":"output_text","text":"Hi"}]}]}}\n\n',
        ])
        chat_events = [
            event
            async for event in OpenAIChatModel(
                "gpt-test",
                async_http_client=chat_http,
            ).astream(ModelRequest(messages=[]))
        ]
        responses_events = [
            event
            async for event in OpenAIResponsesModel(
                "gpt-test",
                async_http_client=responses_http,
            ).astream(ModelRequest(messages=[]))
        ]
        return chat_http, responses_http, chat_events, responses_events

    chat_http, responses_http, chat_events, responses_events = asyncio.run(collect())

    assert chat_http.calls[0][2]["stream"] is True
    assert responses_http.calls[0][2]["stream"] is True
    assert chat_events[0].type == ModelStreamEventType.TEXT_DELTA
    assert chat_events[-1].response.output_text == "Hi"
    assert responses_events[0].type == ModelStreamEventType.TEXT_DELTA
    assert responses_events[-1].response.output_text == "Hi"
