"""Pandas patterns for data-engineering pipelines: read, transform, aggregate, write."""
import pandas as pd
import numpy as np


def read_csv_chunked(path: str, chunksize: int = 100_000):
    """Stream a large CSV in chunks to avoid loading the whole file into memory."""
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield chunk


def load_events(path: str) -> pd.DataFrame:
    """Load event CSV, parse timestamps, drop rows missing the user_id key."""
    df = pd.read_csv(path, parse_dates=["event_time"])
    df = df.dropna(subset=["user_id"])
    df["event_date"] = df["event_time"].dt.date
    return df


def daily_active_users(events: pd.DataFrame) -> pd.DataFrame:
    """Count distinct users per day. Returns a frame with event_date, dau."""
    return (
        events.groupby("event_date")["user_id"]
        .nunique()
        .reset_index(name="dau")
        .sort_values("event_date")
    )


def session_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-session: first/last event time, count, total revenue."""
    return events.groupby("session_id").agg(
        started_at=("event_time", "min"),
        ended_at=("event_time", "max"),
        n_events=("event_time", "count"),
        revenue=("amount", "sum"),
    )


def pivot_revenue_by_country(orders: pd.DataFrame) -> pd.DataFrame:
    """Pivot total revenue: rows = month, columns = country, values = sum(amount)."""
    orders["month"] = orders["order_date"].dt.to_period("M")
    return orders.pivot_table(
        index="month",
        columns="country",
        values="amount",
        aggfunc="sum",
        fill_value=0,
    )


def join_orders_with_users(orders: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Inner-join orders to users on user_id; keep only enriched columns."""
    merged = orders.merge(users, on="user_id", how="inner", suffixes=("", "_user"))
    return merged[["order_id", "user_id", "country", "amount", "order_date", "signup_date"]]


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip a Series to its [lower, upper] quantile bounds."""
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)


def write_parquet(df: pd.DataFrame, path: str, partition_cols: list[str] | None = None) -> None:
    """Write to Parquet with snappy compression, optionally Hive-partitioned."""
    df.to_parquet(path, engine="pyarrow", compression="snappy", partition_cols=partition_cols)


def polars_equivalent_groupby():
    """Polars version of a groupby+agg: same idea, lazy, ~5x faster on large data."""
    import polars as pl
    lf = pl.scan_csv("events.csv")
    return (
        lf.group_by("user_id")
        .agg(pl.col("amount").sum().alias("total_revenue"))
        .sort("total_revenue", descending=True)
        .collect()
    )


def duckdb_query_parquet():
    """Query a Parquet directory with DuckDB — no Spark cluster required."""
    import duckdb
    return duckdb.query(
        "SELECT country, SUM(amount) AS rev "
        "FROM read_parquet('s3://bucket/orders/*.parquet') "
        "GROUP BY country ORDER BY rev DESC"
    ).to_df()