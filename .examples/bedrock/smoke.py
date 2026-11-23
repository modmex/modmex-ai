"""Live Bedrock check: one agent on several catalog models, plus a PDF size probe.

Calls the real Amazon Bedrock APIs with your AWS account (cost: cents for the
matrix; the PDF probe sends large requests but very few tokens).

Run (credentials from the standard chain: AWS_PROFILE / SSO / env vars):

    MODMEX_AI_RUN_LIVE=1 AWS_PROFILE=... AWS_REGION=us-west-2 \
    BEDROCK_MODELS="<model or inference profile id>,<another id>,..." \
    poetry run python .examples/bedrock/smoke.py

Copy the ids from each model card in the Bedrock console ("Programmatic access";
prefer the us./global. inference profile ids). Optional:

    BEDROCK_API_KEY=...                Bedrock API key instead of SigV4
    BEDROCK_OPENAI_MODEL=<GPT id>      GPT model for the Responses API + PDF probe
    BEDROCK_S3_URI=s3://bucket/key.pdf a PDF in S3 to read through an s3:// input
    BEDROCK_PDF_SIZES_MB="1,5,10,20"   PDF sizes for the probe (default 1,5,10,20)

Results are a matrix: ok / FAIL (with the error) per model and capability, so one
run tells you what each model supports with your account and region.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
import zlib

from modmex import BaseModel

from modmex_ai import Agent, ModelSettings, tool
from modmex_ai.aws import AwsSigV4
from modmex_ai.messages import FileInput, ImageInput, Message, TextInput
from modmex_ai.models import ModelRequest
from modmex_ai.providers.bedrock import BedrockConverseModel
from modmex_ai.providers.openai import OpenAIResponsesModel

REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2"
API_KEY = os.getenv("BEDROCK_API_KEY")
SETTINGS = ModelSettings(max_tokens=2000)


# --- test inputs built with the standard library ------------------------------


def make_png(width: int = 64, height: int = 64, rgb=(220, 20, 20)) -> bytes:
    row = b"\x00" + bytes(rgb) * width
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(row * height)) + chunk(b"IEND", b"")


def make_pdf(text: str, *, padding_bytes: int = 0) -> bytes:
    """One-page PDF with ``text``; ``padding_bytes`` of incompressible data in an
    unreferenced stream make it as large as a scanned proof of delivery."""
    content = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    if padding_bytes:
        objects.append(b"<< /Length %d >>\nstream\n" % padding_bytes + os.urandom(padding_bytes) + b"\nendstream")
    out = bytearray(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


# --- checks --------------------------------------------------------------------


class Capital(BaseModel):
    country: str
    capital: str


calls: list[str] = []


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    calls.append(city)
    return {"city": city, "temp_c": 21, "conditions": "sunny"}


def converse(model_id: str) -> BedrockConverseModel:
    if API_KEY:
        return BedrockConverseModel(model_id, region=REGION, api_key=API_KEY)
    return BedrockConverseModel(model_id, region=REGION)


def check_text(model):
    r = model.complete(ModelRequest(messages=[Message(role="user", content="Reply with exactly: pong")], settings=SETTINGS))
    assert r.output_text and "pong" in r.output_text.lower(), r.output_text
    return f"{r.usage.input_tokens} in / {r.usage.output_tokens} out"


def check_tools(model):
    calls.clear()
    agent = Agent(name="w", instructions="Use get_weather to answer weather questions. One sentence.",
                  tools=[get_weather], model=model, settings=SETTINGS)
    out = agent.run("What's the weather in Paris?").output
    assert calls == ["Paris"], f"tool calls: {calls}"
    return repr(out)[:60]


def check_structured(model):
    agent = Agent(name="c", instructions="Return the capital of the given country.", output_type=Capital,
                  model=model, settings=SETTINGS)
    out = agent.run("France").output
    assert out.capital.lower() == "paris", out
    return str(out)


def check_stream(model):
    chunks = [e.text_delta for e in model.stream(ModelRequest(
        messages=[Message(role="user", content="Count from 1 to 5, comma separated.")], settings=SETTINGS)) if e.text_delta]
    text = "".join(chunks)
    assert "1" in text and "5" in text, text
    return f"{len(chunks)} chunks"


def check_image(model):
    r = model.complete(ModelRequest(messages=[Message(role="user", content=[
        TextInput(text="What color is this square? One word."), ImageInput(data=make_png(), media_type="image/png")])],
        settings=SETTINGS))
    assert any(word in (r.output_text or "").lower() for word in ("red", "rojo")), r.output_text
    return repr(r.output_text)


def check_pdf(model):
    r = model.complete(ModelRequest(messages=[Message(role="user", content=[
        TextInput(text="What is the delivery reference in this document? Reply with the reference only."),
        FileInput(data=make_pdf("Proof of delivery ABC-123"), media_type="application/pdf", filename="proof of delivery.pdf")])], settings=SETTINGS))
    assert "ABC-123" in (r.output_text or ""), r.output_text
    return repr(r.output_text)


def check_s3_pdf(model, uri):
    r = model.complete(ModelRequest(messages=[Message(role="user", content=[
        TextInput(text="Summarize this document in one sentence."),
        FileInput(url=uri, media_type="application/pdf")])], settings=SETTINGS))
    assert r.output_text, "empty answer"
    return repr(r.output_text)[:60]


def run_check(name, fn, *args):
    started = time.perf_counter()
    try:
        detail = fn(*args)
        status = "ok  "
    except ValueError as exc:
        if isinstance(exc, json.JSONDecodeError):
            detail, status = f"{type(exc).__name__}: {str(exc)[:300]}", "FAIL"
        else:  # modmex-ai rejected an input this model/endpoint does not support
            detail, status = str(exc)[:300], "n/a "
    except Exception as exc:  # noqa: BLE001 - report and continue
        detail = f"{type(exc).__name__}: {str(exc)[:300]}"
        status = "FAIL"
    print(f"    {status} {name:11s} ({time.perf_counter() - started:5.1f}s) {detail}")
    return status == "ok  "


def pdf_probe(label, send):
    sizes = [int(x) for x in os.getenv("BEDROCK_PDF_SIZES_MB", "1,5,10,20").split(",") if x.strip()]
    print(f"\n  PDF size probe: {label}")
    for mb in sizes:
        pdf = make_pdf("Proof of delivery ABC-123", padding_bytes=mb * 1024 * 1024)
        started = time.perf_counter()
        try:
            text = send(pdf)
            ok = "ABC-123" in (text or "")
            print(f"    {'ok  ' if ok else 'WRONG'} {len(pdf) / 1e6:6.1f} MB ({time.perf_counter() - started:5.1f}s) {text!r:.60}")
        except Exception as exc:  # noqa: BLE001
            print(f"    FAIL {len(pdf) / 1e6:6.1f} MB ({time.perf_counter() - started:5.1f}s) {type(exc).__name__}: {str(exc)[:160]}")


def main() -> None:
    if os.getenv("MODMEX_AI_RUN_LIVE") != "1":
        raise SystemExit("Set MODMEX_AI_RUN_LIVE=1 to run this live API example.")
    models = [m.strip() for m in os.getenv("BEDROCK_MODELS", "").split(",") if m.strip()]
    openai_model = os.getenv("BEDROCK_OPENAI_MODEL")
    if not models and not openai_model:
        raise SystemExit("Set BEDROCK_MODELS (comma-separated ids from the model cards) and/or BEDROCK_OPENAI_MODEL.")
    print(f"region: {REGION} | auth: {'Bedrock API key' if API_KEY else 'SigV4 (AWS credential chain)'}")
    if not API_KEY:
        try:  # who is signing: the first thing to check on a 403
            signer = AwsSigV4(region=REGION)
            identity = signer.session().create_client("sts", region_name=REGION).get_caller_identity()
            print(f"identity: {identity['Arn']} (account {identity['Account']})")
        except Exception as exc:  # noqa: BLE001
            print(f"identity: could not resolve AWS credentials ({type(exc).__name__}: {exc})")

    for model_id in models:
        print(f"\n  Converse: {model_id}")
        model = converse(model_id)
        run_check("text", check_text, model)
        run_check("tools", check_tools, model)
        run_check("structured", check_structured, model)
        run_check("stream", check_stream, model)
        run_check("image", check_image, model)
        run_check("pdf", check_pdf, model)
        if os.getenv("BEDROCK_S3_URI"):
            run_check("s3 pdf", check_s3_pdf, model, os.environ["BEDROCK_S3_URI"])

    if openai_model:
        responses = OpenAIResponsesModel(
            openai_model,
            base_url=f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1",
            api_key=API_KEY,
            auth=None if API_KEY else AwsSigV4(region=REGION, service="bedrock"),
        )
        print(f"\n  OpenAI Responses API on Bedrock: {openai_model}")
        run_check("text", check_text, responses)
        run_check("tools", check_tools, responses)
        run_check("structured", check_structured, responses)
        run_check("pdf", check_pdf, responses)

        def send_gpt(pdf):
            return responses.complete(ModelRequest(messages=[Message(role="user", content=[
                TextInput(text="What is the delivery reference in this document? Reply with the reference only."),
                FileInput(data=pdf, media_type="application/pdf", filename="pod.pdf")])], settings=SETTINGS)).output_text

        pdf_probe(f"{openai_model} (Responses input_file)", send_gpt)

    claude = next((m for m in models if "anthropic." in m), None)
    if claude:
        model = converse(claude)

        def send_claude(pdf):
            return model.complete(ModelRequest(messages=[Message(role="user", content=[
                TextInput(text="What is the delivery reference in this document? Reply with the reference only."),
                FileInput(data=pdf, filename="proof of delivery.pdf")])], settings=SETTINGS)).output_text

        pdf_probe(f"{claude} (Converse document)", send_claude)
    print("\ndone")


if __name__ == "__main__":
    sys.exit(main())
