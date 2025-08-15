from urllib.parse import urlparse
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from .settings import S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY, S3_REGION


def _make_client():
    """
    Build a boto3 S3 client for MinIO or AWS.
    For MinIO/non-AWS endpoints: use path-style addressing + endpoint_url
    For AWS: use virtual-hosted style (default), omit endpoint_url
    """
    parsed = urlparse(S3_ENDPOINT)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError("S3 endpoint must start with http:// or https://")

    is_aws = "amazonaws.com" in parsed.netloc

    # Always use SigV4. For MinIO/path-style, force addressing_style='path'
    s3_config = {"addressing_style": "virtual" if is_aws else "path"}
    cfg = Config(signature_version="s3v4", s3=s3_config)

    session = boto3.session.Session(
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
    )

    if is_aws:
        # Let boto3 decide the endpoint for AWS
        return session.client("s3", config=cfg)
    else:
        # Non-AWS (MinIO/LocalStack): pass endpoint_url and use path-style
        return session.client("s3", endpoint_url=S3_ENDPOINT, config=cfg)


def ensure_bucket_if_missing():
    s3 = _make_client()
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
        return
    except ClientError as e:
        # If not found, create it
        error_code = e.response.get("Error", {}).get("Code")
        if error_code not in ("404", "NoSuchBucket"):
            raise
    # Create bucket (LocationConstraint required for non-us-east-1 in AWS)
    params = {"Bucket": S3_BUCKET}
    if S3_REGION and S3_REGION != "us-east-1":
        params["CreateBucketConfiguration"] = {"LocationConstraint": S3_REGION}
    s3.create_bucket(**params)


def upload_fileobj(fileobj, object_name: str, content_type: str | None = None):
    s3 = _make_client()
    extra = {"ContentType": content_type} if content_type else None
    s3.upload_fileobj(fileobj, S3_BUCKET, object_name, ExtraArgs=extra or {})


def presigned_get_url(object_name: str, expires_seconds: int = 1800) -> str:
    s3 = _make_client()
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": object_name},
        ExpiresIn=expires_seconds,
    )


def head_object(object_name: str) -> dict:
    s3 = _make_client()
    resp = s3.head_object(Bucket=S3_BUCKET, Key=object_name)
    return resp
