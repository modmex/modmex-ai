"""AWS helpers for model adapters: SigV4 signing, S3 inputs, event streams.

botocore is imported only when one of these is used.
"""

from modmex_ai.aws.auth import AwsSigV4
from modmex_ai.aws.s3 import S3ObjectLoader, parse_s3_uri

__all__ = ["AwsSigV4", "S3ObjectLoader", "parse_s3_uri"]
