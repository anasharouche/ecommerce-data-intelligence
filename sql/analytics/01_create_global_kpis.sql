
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 4 - Global Business KPIs
-- =====================================================

CREATE SCHEMA IF NOT EXISTS analytics;

CREATE OR REPLACE VIEW analytics.vw_global_kpis AS

WITH order_metrics AS (
    SELECT
        COUNT(*) AS total_orders,

        COUNT(*) FILTER (
            WHERE order_status = 'delivered'
        ) AS delivered_orders,

        COUNT(*) FILTER (
            WHERE is_delayed = TRUE
        ) AS delayed_orders,

        COUNT(*) FILTER (
            WHERE is_delayed IS NOT NULL
        ) AS evaluated_orders,

        ROUND(
            AVG(delivery_days)::numeric, 2
        ) AS avg_delivery_days

    FROM dw.fact_orders
),

sales_metrics AS (
    SELECT
        COUNT(DISTINCT order_id) AS orders_with_items,

        COALESCE(
            SUM(item_total_value), 0
        ) AS total_sales,

        COALESCE(
            SUM(price), 0
        ) AS total_product_revenue,

        COALESCE(
            SUM(freight_value), 0
        ) AS total_freight

    FROM dw.fact_order_items
),

payment_metrics AS (
    SELECT
        COALESCE(
            SUM(payment_value), 0
        ) AS total_payments

    FROM dw.fact_payments
),

review_metrics AS (
    SELECT
        COUNT(*) AS total_reviews,

        ROUND(
            AVG(review_score)::numeric, 2
        ) AS avg_review_score,

        ROUND(
            100.0 * COUNT(*) FILTER (
                WHERE review_score >= 4
            ) / NULLIF(COUNT(*), 0),
            2
        ) AS satisfaction_rate

    FROM dw.fact_reviews
)

SELECT
    o.total_orders,
    o.delivered_orders,

    s.orders_with_items,
    s.total_sales,
    s.total_product_revenue,
    s.total_freight,

    ROUND(
        s.total_sales /
        NULLIF(s.orders_with_items, 0),
        2
    ) AS avg_order_value,

    p.total_payments,

    o.delayed_orders,

    ROUND(
        100.0 * o.delayed_orders /
        NULLIF(o.evaluated_orders, 0),
        2
    ) AS delay_rate,

    o.avg_delivery_days,

    r.total_reviews,
    r.avg_review_score,
    r.satisfaction_rate

FROM order_metrics o
CROSS JOIN sales_metrics s
CROSS JOIN payment_metrics p
CROSS JOIN review_metrics r;

-- =====================================================
-- VALIDATION
-- =====================================================

SELECT *
FROM analytics.vw_global_kpis;
