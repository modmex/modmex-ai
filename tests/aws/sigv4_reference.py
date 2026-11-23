"""Independent SigV4 verifier for tests (AWS Signature Version 4 specification).

Recomputes the signature from the request that was actually sent, so a test can
prove the signature is valid for exactly the bytes on the wire.
"""

import hashlib
import hmac
from urllib.parse import quote, urlparse


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def verify(method: str, url: str, headers: dict, body: bytes, *, secret_key: str) -> bool:
    lower = {k.lower(): v for k, v in headers.items()}
    auth = lower["authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 ")
    fields = dict(part.strip().split("=", 1) for part in auth[len("AWS4-HMAC-SHA256 "):].split(","))
    _, date, region, service, _ = fields["Credential"].split("/")
    signed_headers = fields["SignedHeaders"].split(";")
    parsed = urlparse(url)
    host = parsed.netloc
    values = {**lower, "host": host}
    canonical_headers = "".join(f"{name}:{' '.join(str(values[name]).split())}\n" for name in signed_headers)
    canonical_path = quote(parsed.path or "/", safe="/~")  # SigV4: URI-encode each path segment once more
    canonical_request = "\n".join([
        method,
        canonical_path,
        parsed.query,
        canonical_headers,
        ";".join(signed_headers),
        hashlib.sha256(body).hexdigest(),
    ])
    amz_date = lower["x-amz-date"]
    scope = f"{date}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical_request.encode()).hexdigest()
    ])
    key = _hmac(_hmac(_hmac(_hmac(f"AWS4{secret_key}".encode(), date), region), service), "aws4_request")
    expected = hmac.new(key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, fields["Signature"])
