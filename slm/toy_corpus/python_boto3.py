"""AWS data-engineering patterns: S3, Glue, Athena, Secrets Manager via boto3."""
import json
import boto3
from botocore.exceptions import ClientError


def get_secret(secret_arn: str, region: str = "us-east-1") -> dict:
    """Fetch a JSON secret from AWS Secrets Manager and return it as a dict."""
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_arn)
    return json.loads(response["SecretString"])


def upload_file_to_s3(local_path: str, bucket: str, key: str) -> None:
    """Upload a local file to s3://bucket/key with server-side encryption."""
    s3 = boto3.client("s3")
    s3.upload_file(
        Filename=local_path,
        Bucket=bucket,
        Key=key,
        ExtraArgs={"ServerSideEncryption": "AES256"},
    )


def download_s3_object(bucket: str, key: str, local_path: str) -> None:
    """Download s3://bucket/key to a local path; create parent dirs as needed."""
    import os
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)


def list_s3_prefix(bucket: str, prefix: str) -> list[str]:
    """List all object keys under a prefix, handling pagination."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def stream_s3_object(bucket: str, key: str):
    """Yield decoded lines from a (possibly large) S3 text object without buffering."""
    s3 = boto3.client("s3")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    for raw in body.iter_lines():
        yield raw.decode("utf-8")


def trigger_glue_job(job_name: str, arguments: dict[str, str] | None = None) -> str:
    """Start a Glue job run and return the run id."""
    glue = boto3.client("glue")
    kwargs = {"JobName": job_name}
    if arguments:
        kwargs["Arguments"] = arguments
    return glue.start_job_run(**kwargs)["JobRunId"]


def wait_for_glue_job(job_name: str, run_id: str, poll_seconds: int = 30) -> str:
    """Poll Glue until the run finishes; return the final state (SUCCEEDED/FAILED/...)."""
    import time
    glue = boto3.client("glue")
    while True:
        run = glue.get_job_run(JobName=job_name, RunId=run_id)["JobRun"]
        state = run["JobRunState"]
        if state in {"SUCCEEDED", "FAILED", "TIMEOUT", "STOPPED"}:
            return state
        time.sleep(poll_seconds)


def run_athena_query(sql: str, database: str, output_s3: str) -> str:
    """Submit an Athena query and return the QueryExecutionId."""
    athena = boto3.client("athena")
    resp = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_s3},
    )
    return resp["QueryExecutionId"]


def copy_csv_to_redshift(table: str, s3_path: str, iam_role: str, redshift_conn) -> None:
    """Issue a Redshift COPY from S3 with IAM auth and CSV format."""
    sql = f"""
        COPY {table}
        FROM '{s3_path}'
        IAM_ROLE '{iam_role}'
        FORMAT AS CSV
        IGNOREHEADER 1
        TIMEFORMAT 'auto';
    """
    with redshift_conn.cursor() as cur:
        cur.execute(sql)
    redshift_conn.commit()


def head_object_safely(bucket: str, key: str) -> dict | None:
    """Return object metadata, or None if it doesn't exist (no exception)."""
    s3 = boto3.client("s3")
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return None
        raise