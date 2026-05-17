-- PostgreSQL patterns for analytical and operational workloads.

-- Create a partitioned events table with a primary key and supporting indexes.
CREATE TABLE IF NOT EXISTS events (
    event_id    BIGSERIAL    PRIMARY KEY,
    user_id     UUID         NOT NULL,
    event_type  TEXT         NOT NULL,
    event_time  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    amount      NUMERIC(12, 2),
    payload     JSONB
) PARTITION BY RANGE (event_time);

CREATE TABLE IF NOT EXISTS events_2026_05
    PARTITION OF events
    FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');

CREATE INDEX IF NOT EXISTS events_user_time_idx
    ON events (user_id, event_time DESC);

-- GIN index for fast JSONB containment queries.
CREATE INDEX IF NOT EXISTS events_payload_gin
    ON events USING gin (payload jsonb_path_ops);

-- Bulk load from a server-side CSV file.
COPY events (user_id, event_type, event_time, amount, payload)
    FROM '/var/lib/postgresql/import/events.csv'
    WITH (FORMAT csv, HEADER true);

-- Daily active users.
SELECT
    DATE_TRUNC('day', event_time)::date AS day,
    COUNT(DISTINCT user_id) AS dau
FROM events
WHERE event_time >= now() - INTERVAL '30 days'
GROUP BY 1
ORDER BY 1;

-- Top 10 spenders per country using a window function.
SELECT user_id, country, total_spend
FROM (
    SELECT
        e.user_id,
        u.country,
        SUM(e.amount) AS total_spend,
        ROW_NUMBER() OVER (PARTITION BY u.country ORDER BY SUM(e.amount) DESC) AS rk
    FROM events e
    JOIN users u USING (user_id)
    WHERE e.event_type = 'purchase'
      AND e.event_time >= now() - INTERVAL '90 days'
    GROUP BY e.user_id, u.country
) ranked
WHERE rk <= 10;

-- Sessionize: assign a session id when the gap between events exceeds 30 minutes.
WITH ordered AS (
    SELECT
        user_id,
        event_time,
        event_time - LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS gap
    FROM events
), flagged AS (
    SELECT
        *,
        SUM(CASE WHEN gap IS NULL OR gap > INTERVAL '30 minutes' THEN 1 ELSE 0 END)
            OVER (PARTITION BY user_id ORDER BY event_time) AS session_id
    FROM ordered
)
SELECT user_id, session_id, MIN(event_time) AS started, MAX(event_time) AS ended
FROM flagged
GROUP BY user_id, session_id;

-- JSONB containment + path queries.
SELECT event_id, payload -> 'item' ->> 'name' AS item_name
FROM events
WHERE payload @> '{"channel": "web"}'
  AND payload -> 'amount' IS NOT NULL;

-- Upsert (insert on conflict) for slowly-changing dim tables.
INSERT INTO users (user_id, country, signup_date)
VALUES ($1, $2, $3)
ON CONFLICT (user_id)
DO UPDATE SET
    country = EXCLUDED.country,
    updated_at = now();

-- Safe online migration: add a column with a default in two steps.
ALTER TABLE events ADD COLUMN source TEXT;
UPDATE events SET source = 'unknown' WHERE source IS NULL;
ALTER TABLE events ALTER COLUMN source SET NOT NULL;

-- Materialized view for a frequently-queried aggregate, refreshed concurrently.
CREATE MATERIALIZED VIEW IF NOT EXISTS revenue_daily AS
SELECT
    DATE_TRUNC('day', event_time)::date AS day,
    SUM(amount) FILTER (WHERE event_type = 'purchase') AS revenue,
    COUNT(*) FILTER (WHERE event_type = 'purchase')    AS n_purchases
FROM events
GROUP BY 1;

CREATE UNIQUE INDEX IF NOT EXISTS revenue_daily_day_uq ON revenue_daily (day);
REFRESH MATERIALIZED VIEW CONCURRENTLY revenue_daily;