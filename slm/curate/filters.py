"""Keyword filters for routing streamed docs into curation buckets."""

DE_KEYWORDS = (
    # Storage / warehouses
    "postgres", "postgresql", "bigquery", "snowflake", "redshift", "athena",
    "iceberg", "parquet", "delta lake", "data lake", "data warehouse",
    # Cloud data services
    "s3 ", "aws glue", "boto3", "google.cloud", "google-cloud", "pubsub", "pub/sub",
    "cloud storage", "gcs", "secrets manager", "kms",
    # Compute / dataframes
    "pandas", "pyspark", "polars", "duckdb", "spark.sql",
    # Orchestration / pipelines
    "airflow", "dbt ", "etl", "elt ", "data pipeline", "data engineering",
    # SQL concepts
    "window function", "partition by", "jsonb", "group by",
    "create table", "select count", "row_number",
    # Ingestion
    "streaming ingestion", "kafka", "kinesis", "pubsub",
)

CLOUD_IMPORTS = (
    "import boto3",
    "from boto3",
    "from google.cloud",
    "import google.cloud",
)


def has_de_keyword(text: str, head_chars: int = 4000) -> bool:
    head = text[:head_chars].lower()
    return any(kw in head for kw in DE_KEYWORDS)


def has_cloud_import(code: str, head_chars: int = 3000) -> bool:
    head = code[:head_chars]
    return any(imp in head for imp in CLOUD_IMPORTS)