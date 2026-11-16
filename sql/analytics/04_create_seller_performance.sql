
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 4 - Seller Performance Analytics
-- Grain: One row per seller
-- =====================================================

CREATE OR REPLACE VIEW analytics.vw_seller_performance AS

WITH seller_order_metrics AS (
    SELECT
        oi.seller_key,
        oi.order_key,

        COUNT(*) AS items_sold,

        SUM(oi.price) AS product_revenue,
        SUM(oi.freight_value) AS freight_revenue,
        SUM(oi.item_total_value) AS total_sales

    FROM dw.fact_order_items oi

    GROUP BY
        oi.seller_key,
        oi.order_key
),

seller_metrics AS (
    SELECT
        s.seller_id,
        s.seller_city,
        s.seller_state,

        COUNT(*) AS total_orders,

        SUM(som.items_sold) AS items_sold,

        SUM(som.product_revenue) AS product_revenue,
        SUM(som.freight_revenue) AS freight_revenue,
        SUM(som.total_sales) AS total_sales,

        ROUND(
            SUM(som.total_sales) /
            NULLIF(COUNT(*), 0),
            2
        ) AS avg_order_value,

        COUNT(*) FILTER (
            WHERE o.order_status = 'delivered'
        ) AS delivered_orders,

        COUNT(*) FILTER (
            WHERE o.is_delayed IS NOT NULL
        ) AS evaluated_orders,

        COUNT(*) FILTER (
            WHERE o.is_delayed = TRUE
        ) AS delayed_orders

    FROM seller_order_metrics som

    JOIN dw.dim_seller s
        ON som.seller_key = s.seller_key

    JOIN dw.fact_orders o
        ON som.order_key = o.order_key

    GROUP BY
        s.seller_id,
        s.seller_city,
        s.seller_state
)

SELECT
    seller_id,
    seller_city,
    seller_state,

    total_orders,
    items_sold,

    product_revenue,
    freight_revenue,
    total_sales,
    avg_order_value,

    delivered_orders,
    evaluated_orders,
    delayed_orders,

    ROUND(
        100.0 * delayed_orders /
        NULLIF(evaluated_orders, 0),
        2
    ) AS delay_rate,

    ROUND(
        100.0 * total_sales /
        NULLIF(SUM(total_sales) OVER (), 0),
        6
    ) AS sales_share_pct,

    DENSE_RANK() OVER (
        ORDER BY total_sales DESC
    ) AS sales_rank

FROM seller_metrics;

-- =====================================================
-- VALIDATION 1 - TOP 15 SELLERS
-- =====================================================

SELECT *
FROM analytics.vw_seller_performance
ORDER BY total_sales DESC
LIMIT 15;

-- =====================================================
-- VALIDATION 2 - FINANCIAL RECONCILIATION
-- =====================================================

SELECT
    COUNT(*) AS active_sellers,
    SUM(items_sold) AS total_items,
    SUM(total_sales) AS total_sales,
    SUM(product_revenue) AS product_revenue,
    SUM(freight_revenue) AS freight_revenue,
    ROUND(SUM(sales_share_pct), 2)
        AS total_sales_share_pct

FROM analytics.vw_seller_performance;

-- =====================================================
-- VALIDATION 3 - DELIVERY PERFORMANCE
-- =====================================================

SELECT
    SUM(delivered_orders) AS seller_delivered_orders,
    SUM(evaluated_orders) AS seller_evaluated_orders,
    SUM(delayed_orders) AS seller_delayed_orders,

    ROUND(
        100.0 * SUM(delayed_orders) /
        NULLIF(SUM(evaluated_orders), 0),
        2
    ) AS seller_delay_rate

FROM analytics.vw_seller_performance;
