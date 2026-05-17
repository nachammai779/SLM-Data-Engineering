"""GCP data-engineering patterns: BigQuery, GCS, Pub/Sub via google-cloud client libraries."""
import json
from google.cloud import bigquery, storage, pubsub_v1


def run_bq_query(sql: str, project: str) -> list[dict]:
    """Run a BigQuery SQL query and return rows as dicts."""
    client = bigquery.Client(project=project)
    job = client.query(sql)
    return [dict(row.items()) for row in job.result()]


def load_csv_to_bq(
    uri: str, dataset: str, table: str, project: str, write_disposition: str = "WRITE_APPEND"
) -> None:
    """Load a GCS CSV into a BigQuery table with autodetected schema."""
    client = bigquery.Client(project=project)
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition=write_disposition,
    )
    table_ref = f"{project}.{dataset}.{table}"
    load_job = client.load_table_from_uri(uri, table_ref, job_config=job_config)
    load_job.result()


def stream_rows_to_bq(rows: list[dict], dataset: str, table: str, project: str) -> list[dict]:
    """Insert rows into BigQuery via the streaming API; return any per-row errors."""
    client = bigquery.Client(project=project)
    table_ref = client.dataset(dataset).table(table)
    errors = client.insert_rows_json(table_ref, rows)
    return errors


def create_partitioned_table(dataset: str, table: str, project: str) -> None:
    """Create a date-partitioned, clustered BigQuery table."""
    client = bigquery.Client(project=project)
    schema = [
        bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("event_time", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("amount", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("country", "STRING", mode="NULLABLE"),
    ]
    table_ref = bigquery.Table(f"{project}.{dataset}.{table}", schema=schema)
    table_ref.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="event_time",
    )
    table_ref.clustering_fields = ["country", "user_id"]
    client.create_table(table_ref, exists_ok=True)


def upload_blob(bucket_name: str, source_path: str, dest_blob: str) -> None:
    """Upload a local file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(source_path)


def download_blob_to_memory(bucket_name: str, blob_name: str) -> bytes:
    """Download a GCS object directly into memory."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    return bucket.blob(blob_name).download_as_bytes()


def list_gcs_prefix(bucket_name: str, prefix: str) -> list[str]:
    """List all blob names under a prefix."""
    client = storage.Client()
    return [b.name for b in client.list_blobs(bucket_name, prefix=prefix)]


def publish_event(project: str, topic: str, payload: dict, **attrs: str) -> str:
    """Publish a JSON-serializable payload to Pub/Sub; return the published message id."""
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project, topic)
    data = json.dumps(payload).encode("utf-8")
    future = publisher.publish(topic_path, data, **attrs)
    return future.result()


def pull_messages(project: str, subscription: str, max_messages: int = 10) -> list[dict]:
    """Pull a batch of messages from a Pub/Sub subscription, ack them, return payloads."""
    subscriber = pubsub_v1.SubscriberClient()
    sub_path = subscriber.subscription_path(project, subscription)
    response = subscriber.pull(
        request={"subscription": sub_path, "max_messages": max_messages},
        timeout=10,
    )
    payloads = []
    ack_ids = []
    for received in response.received_messages:
        payloads.append(json.loads(received.message.data.decode("utf-8")))
        ack_ids.append(received.ack_id)
    if ack_ids:
        subscriber.acknowledge(request={"subscription": sub_path, "ack_ids": ack_ids})
    return payloads