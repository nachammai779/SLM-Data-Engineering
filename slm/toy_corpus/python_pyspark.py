"""PySpark patterns: sessions, joins, window functions, partitioned writes."""
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F


def build_session(app_name: str = "etl") -> SparkSession:
    """Create a Spark session with sensible defaults for batch ETL on S3."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


def read_parquet_glob(spark: SparkSession, path: str) -> DataFrame:
    """Read all Parquet files under a glob pattern."""
    return spark.read.parquet(path)


def sessionize(events: DataFrame, gap_minutes: int = 30) -> DataFrame:
    """Assign session_id per user where the gap between consecutive events exceeds gap_minutes."""
    w = Window.partitionBy("user_id").orderBy("event_time")
    with_lag = events.withColumn("prev_time", F.lag("event_time").over(w))
    with_gap = with_lag.withColumn(
        "new_session",
        F.when(
            F.col("prev_time").isNull()
            | (F.col("event_time").cast("long") - F.col("prev_time").cast("long") > gap_minutes * 60),
            1,
        ).otherwise(0),
    )
    return with_gap.withColumn("session_id", F.sum("new_session").over(w))


def top_n_per_group(df: DataFrame, partition_col: str, order_col: str, n: int = 10) -> DataFrame:
    """Top-N rows per partition using ROW_NUMBER()."""
    w = Window.partitionBy(partition_col).orderBy(F.col(order_col).desc())
    return df.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") <= n).drop("rn")


def daily_revenue(orders: DataFrame) -> DataFrame:
    """Compute revenue per country per day."""
    return (
        orders.withColumn("order_date", F.to_date("order_time"))
        .groupBy("order_date", "country")
        .agg(
            F.sum("amount").alias("revenue"),
            F.count("order_id").alias("n_orders"),
            F.approx_count_distinct("user_id").alias("n_users"),
        )
        .orderBy("order_date", "country")
    )


def join_with_broadcast(facts: DataFrame, dims: DataFrame, key: str) -> DataFrame:
    """Broadcast the smaller dim table to avoid a shuffle on the larger facts."""
    return facts.join(F.broadcast(dims), on=key, how="left")


def deduplicate_latest(df: DataFrame, key_cols: list[str], order_col: str) -> DataFrame:
    """Keep only the latest row per key based on order_col (e.g. updated_at)."""
    w = Window.partitionBy(*key_cols).orderBy(F.col(order_col).desc())
    return df.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn")


def write_iceberg_partitioned(df: DataFrame, table: str, partition_col: str) -> None:
    """Append to an Iceberg table partitioned by day."""
    (
        df.writeTo(table)
        .partitionedBy(F.days(partition_col))
        .using("iceberg")
        .append()
    )


def quarantine_invalid_rows(df: DataFrame, rule: str, quarantine_path: str) -> DataFrame:
    """Split a DataFrame into valid (returned) and invalid (written to quarantine) rows."""
    invalid = df.filter(f"NOT ({rule})")
    invalid.write.mode("append").parquet(quarantine_path)
    return df.filter(rule)