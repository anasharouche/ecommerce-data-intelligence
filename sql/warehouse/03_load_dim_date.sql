
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 3 - Load Date Dimension
-- =====================================================

BEGIN;

WITH all_dates AS (

    SELECT order_purchase_timestamp::date AS event_date
    FROM staging.orders_clean

    UNION ALL

    SELECT order_approved_at::date
    FROM staging.orders_clean

    UNION ALL

    SELECT order_delivered_carrier_date::date
    FROM staging.orders_clean

    UNION ALL

    SELECT order_delivered_customer_date::date
    FROM staging.orders_clean

    UNION ALL

    SELECT order_estimated_delivery_date::date
    FROM staging.orders_clean

    UNION ALL

    SELECT shipping_limit_date::date
    FROM staging.order_items_clean

    UNION ALL

    SELECT review_creation_date::date
    FROM staging.order_reviews_clean

    UNION ALL

    SELECT review_answer_timestamp::date
    FROM staging.order_reviews_clean
),

date_bounds AS (
    SELECT
        MIN(event_date) AS min_date,
        MAX(event_date) AS max_date
    FROM all_dates
    WHERE event_date IS NOT NULL
),

calendar AS (
    SELECT gs.calendar_date::date AS full_date
    FROM date_bounds
    CROSS JOIN LATERAL generate_series(
        min_date,
        max_date,
        INTERVAL '1 day'
    ) AS gs(calendar_date)
)

INSERT INTO dw.dim_date (
    date_key,
    full_date,
    year,
    quarter,
    month,
    month_name,
    day,
    day_of_week,
    day_name,
    is_weekend
)
SELECT
    TO_CHAR(full_date, 'YYYYMMDD')::integer,
    full_date,
    EXTRACT(YEAR FROM full_date)::integer,
    EXTRACT(QUARTER FROM full_date)::integer,
    EXTRACT(MONTH FROM full_date)::integer,
    TO_CHAR(full_date, 'TMMonth'),
    EXTRACT(DAY FROM full_date)::integer,
    EXTRACT(ISODOW FROM full_date)::integer,
    TO_CHAR(full_date, 'TMDay'),
    EXTRACT(ISODOW FROM full_date) IN (6, 7)
FROM calendar

ON CONFLICT (date_key) DO NOTHING;

COMMIT;

-- =====================================================
-- VALIDATION
-- =====================================================

SELECT
    COUNT(*) AS total_days,
    MIN(full_date) AS first_date,
    MAX(full_date) AS last_date,
    COUNT(DISTINCT year) AS total_years
FROM dw.dim_date;

-- Sample records
SELECT
    date_key,
    full_date,
    year,
    quarter,
    month,
    month_name,
    day_name,
    is_weekend
FROM dw.dim_date
ORDER BY full_date
LIMIT 10;
