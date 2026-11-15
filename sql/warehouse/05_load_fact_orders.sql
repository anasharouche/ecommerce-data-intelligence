
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 3 - Load Fact Orders
-- =====================================================

BEGIN;

-- =====================================================
-- 1. AGGREGATE ORDER ITEMS
-- =====================================================

WITH item_aggregates AS (
    SELECT
        order_id,
        COUNT(*) AS item_count,
        SUM(price) AS total_item_value,
        SUM(freight_value) AS total_freight_value,
        SUM(item_total_value) AS total_order_value
    FROM staging.order_items_clean
    GROUP BY order_id
),

-- =====================================================
-- 2. PREPARE ORDER DATASET
-- =====================================================

order_dataset AS (
    SELECT
        o.*,
        c.customer_key,

        ia.item_count,
        ia.total_item_value,
        ia.total_freight_value,
        ia.total_order_value,

        p.total_payment_value

    FROM staging.orders_clean o

    INNER JOIN dw.dim_customer c
        ON o.customer_id = c.customer_id

    LEFT JOIN item_aggregates ia
        ON o.order_id = ia.order_id

    LEFT JOIN staging.order_payments_aggregated p
        ON o.order_id = p.order_id
)

-- =====================================================
-- 3. LOAD FACT ORDERS
-- =====================================================

INSERT INTO dw.fact_orders (
    order_id,
    customer_key,

    purchase_date_key,
    approved_date_key,
    carrier_date_key,
    delivered_date_key,
    estimated_date_key,

    order_status,
    purchase_timestamp,
    delivered_timestamp,
    estimated_delivery_timestamp,

    delivery_days,
    delivery_delay_days,
    is_delayed,

    item_count,
    total_item_value,
    total_freight_value,
    total_order_value,
    total_payment_value,

    flag_delivered_without_date,
    flag_canceled_with_delivery
)

SELECT
    order_id,
    customer_key,

    -- Date keys
    TO_CHAR(
        order_purchase_timestamp,
        'YYYYMMDD'
    )::integer,

    CASE
        WHEN order_approved_at IS NOT NULL
        THEN TO_CHAR(
            order_approved_at,
            'YYYYMMDD'
        )::integer
    END,

    CASE
        WHEN order_delivered_carrier_date IS NOT NULL
        THEN TO_CHAR(
            order_delivered_carrier_date,
            'YYYYMMDD'
        )::integer
    END,

    CASE
        WHEN order_delivered_customer_date IS NOT NULL
        THEN TO_CHAR(
            order_delivered_customer_date,
            'YYYYMMDD'
        )::integer
    END,

    TO_CHAR(
        order_estimated_delivery_date,
        'YYYYMMDD'
    )::integer,

    -- Order information
    order_status,
    order_purchase_timestamp,
    order_delivered_customer_date,
    order_estimated_delivery_date,

    -- =================================================
    -- DELIVERY DAYS
    -- Actual delivery duration in decimal days
    -- =================================================

    CASE
        WHEN order_status = 'delivered'
             AND order_delivered_customer_date IS NOT NULL
        THEN EXTRACT(
            EPOCH FROM (
                order_delivered_customer_date
                - order_purchase_timestamp
            )
        ) / 86400.0
        ELSE NULL
    END,

    -- =================================================
    -- DELIVERY DELAY DAYS
    -- Calendar-day difference
    -- Positive = late
    -- Zero = on estimated date
    -- Negative = early
    -- =================================================

    CASE
        WHEN order_status = 'delivered'
             AND order_delivered_customer_date IS NOT NULL
        THEN (
            order_delivered_customer_date::date
            - order_estimated_delivery_date::date
        )::double precision
        ELSE NULL
    END,

    -- =================================================
    -- IS DELAYED
    -- Compare calendar dates, not timestamps
    -- =================================================

    CASE
        WHEN order_status = 'delivered'
             AND order_delivered_customer_date IS NOT NULL
        THEN (
            order_delivered_customer_date::date
            > order_estimated_delivery_date::date
        )
        ELSE NULL
    END,

    -- Financial measures
    item_count,
    total_item_value,
    total_freight_value,
    total_order_value,
    total_payment_value,

    -- Data quality flags
    flag_delivered_without_date,
    flag_canceled_with_delivery

FROM order_dataset

-- Prevent duplicate business keys
ON CONFLICT (order_id) DO NOTHING;

COMMIT;

-- =====================================================
-- 4. VALIDATION - ORDER COUNTS
-- =====================================================

SELECT
    COUNT(*) AS total_orders,

    COUNT(*) FILTER (
        WHERE order_status = 'delivered'
    ) AS delivered_orders,

    COUNT(*) FILTER (
        WHERE is_delayed = TRUE
    ) AS delayed_orders,

    COUNT(*) FILTER (
        WHERE is_delayed = FALSE
    ) AS on_time_orders,

    COUNT(*) FILTER (
        WHERE is_delayed IS NULL
    ) AS unknown_delivery_status

FROM dw.fact_orders;

-- =====================================================
-- 5. VALIDATION - DELIVERY KPIs
-- =====================================================

SELECT
    ROUND(
        AVG(delivery_days)::numeric,
        2
    ) AS avg_delivery_days,

    ROUND(
        AVG(delivery_delay_days)::numeric,
        2
    ) AS avg_delivery_delay_days,

    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE is_delayed = TRUE
        )
        / NULLIF(
            COUNT(*) FILTER (
                WHERE is_delayed IS NOT NULL
            ),
            0
        ),
        2
    ) AS delayed_percentage

FROM dw.fact_orders;

-- =====================================================
-- 6. VALIDATION - DATA CONSISTENCY
-- =====================================================

SELECT
    COUNT(*) AS total_orders,

    COUNT(*) FILTER (
        WHERE delivery_delay_days > 0
          AND is_delayed IS DISTINCT FROM TRUE
    ) AS inconsistent_positive_delays,

    COUNT(*) FILTER (
        WHERE delivery_delay_days <= 0
          AND is_delayed IS DISTINCT FROM FALSE
    ) AS inconsistent_non_positive_delays,

    COUNT(*) FILTER (
        WHERE delivery_delay_days IS NULL
          AND is_delayed IS NOT NULL
    ) AS inconsistent_null_delays

FROM dw.fact_orders;
