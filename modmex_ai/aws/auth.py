"""AWS request signing for model adapters, delegated to botocore.

botocore is part of the AWS Lambda Python runtime (and of the ``aws`` extra), so it
adds nothing to deployment packages. It is imported only when an ``AwsSigV4``
instance is used, so adapters that do not talk to AWS never pay for it.
"""

from __future__ import annotations

import os
from typing import Any


class AwsSigV4:
    """Sign requests with AWS Signature Version 4.

    Credentials come from botocore's standard chain: on Lambda the execution
    role's temporary credentials (environment variables), locally a profile,
    SSO or instance metadata. Temporary credentials are refreshed by botocore
    before they expire.

    ``service`` is the signing name of the endpoint: ``"bedrock"`` for
    ``bedrock-runtime`` (Converse and the OpenAI-compatible API) and
    ``"bedrock-mantle"`` for the ``bedrock-mantle`` endpoint.
    """

    def __init__(
        self,
        *,
        region: str | None = None,
        service: str = "bedrock",
        profile: str | None = None,
        credentials: Any = None,
    ) -> None:
        self.region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if not self.region:
            raise ValueError("AwsSigV4 needs a region (argument, AWS_REGION or AWS_DEFAULT_REGION)")
        self.service = service
        self.profile = profile
        self._credentials = credentials
        self._session: Any = None

    def session(self) -> Any:
        """The botocore session used for credentials (and for S3 reads)."""
        if self._session is None:
            from botocore.session import Session

            self._session = Session(profile=self.profile)
        return self._session

    def credentials(self) -> Any:
        if self._credentials is None:
            self._credentials = self.session().get_credentials()
            if self._credentials is None:
                raise RuntimeError(
                    "No AWS credentials found (execution role, environment, profile or SSO)"
                )
        # Frozen per request: refreshable credentials renew themselves here.
        return self._credentials.get_frozen_credentials()

    def sign(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
    ) -> dict[str, str]:
        """Return ``headers`` plus the SigV4 headers for exactly this ``body``."""
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        request = AWSRequest(method=method, url=url, data=body, headers=dict(headers))
        SigV4Auth(self.credentials(), self.service, self.region).add_auth(request)
        return dict(request.headers.items())
