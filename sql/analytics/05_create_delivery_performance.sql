
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 4 - Delivery Performance Analytics
-- Grain: One row per purchase month
-- =====================================================

CREATE OR REPLACE VIEW analytics.vw_delivery_performance AS

WITH monthly_delivery AS (
    SELECT
        DATE_TRUNC(
            'month',
            purchase_timestamp
        )::date AS purchase_month,

        COUNT(*) AS total_orders,

        COUNT(*) FILTER (
            WHERE order_status = 'delivered'
        ) AS delivered_orders,

        COUNT(*) FILTER (
            WHERE is_delayed IS NOT NULL
        ) AS evaluated_orders,

        COUNT(*) FILTER (
            WHERE is_delayed = TRUE
        ) AS delayed_orders,

        COUNT(*) FILTER (
            WHERE is_delayed = FALSE
        ) AS on_time_orders,

        COUNT(*) FILTER (
            WHERE is_delayed IS NULL
        ) AS unknown_delivery_status,

        ROUND(
            AVG(delivery_days)::numeric,
            2
        ) AS avg_delivery_days,

        ROUND(
            AVG(delivery_delay_days)::numeric,
            2
        ) AS avg_delivery_delay_days

    FROM dw.fact_orders

    GROUP BY 1
)

SELECT
    purchase_month,
    total_orders,
    delivered_orders,
    evaluated_orders,
    delayed_orders,
    on_time_orders,
    unknown_delivery_status,

    ROUND(
        100.0 * delayed_orders /
        NULLIF(evaluated_orders, 0),
        2
    ) AS delay_rate,

    ROUND(
        100.0 * on_time_orders /
        NULLIF(evaluated_orders, 0),
        2
    ) AS on_time_rate,

    avg_delivery_days,
    avg_delivery_delay_days

FROM monthly_delivery;

-- =====================================================
-- VALIDATION 1 - MONTHLY DELIVERY PERFORMANCE
-- =====================================================

SELECT *
FROM analytics.vw_delivery_performance
ORDER BY purchase_month;

-- =====================================================
-- VALIDATION 2 - GLOBAL COUNTS
-- =====================================================

SELECT
    SUM(total_orders) AS total_orders,
    SUM(delivered_orders) AS delivered_orders,
    SUM(evaluated_orders) AS evaluated_orders,
    SUM(delayed_orders) AS delayed_orders,
    SUM(on_time_orders) AS on_time_orders,
    SUM(unknown_delivery_status) AS unknown_orders,

    ROUND(
        100.0 * SUM(delayed_orders) /
        NULLIF(SUM(evaluated_orders), 0),
        2
    ) AS global_delay_rate

FROM analytics.vw_delivery_performance;

-- =====================================================
-- VALIDATION 3 - CONSISTENCY
-- =====================================================

SELECT
    COUNT(*) AS inconsistent_months
FROM analytics.vw_delivery_performance
WHERE total_orders <>
    evaluated_orders + unknown_delivery_status
   OR evaluated_orders <>
    delayed_orders + on_time_orders;
