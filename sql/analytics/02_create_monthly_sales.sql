
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 4 - Monthly Sales Analytics (Improved)
-- =====================================================

CREATE OR REPLACE VIEW analytics.vw_monthly_sales AS

WITH month_range AS (
    SELECT
        GENERATE_SERIES(
            DATE_TRUNC(
                'month',
                MIN(purchase_timestamp)
            ),
            DATE_TRUNC(
                'month',
                MAX(purchase_timestamp)
            ),
            INTERVAL '1 month'
        )::date AS sales_month
    FROM dw.fact_orders
),

monthly_orders AS (
    SELECT
        DATE_TRUNC(
            'month', purchase_timestamp
        )::date AS sales_month,

        COUNT(*) AS total_orders,

        COUNT(*) FILTER (
            WHERE order_status = 'delivered'
        ) AS delivered_orders

    FROM dw.fact_orders
    GROUP BY 1
),

monthly_sales AS (
    SELECT
        DATE_TRUNC(
            'month', o.purchase_timestamp
        )::date AS sales_month,

        COUNT(DISTINCT oi.order_id)
            AS orders_with_items,

        SUM(oi.price) AS product_revenue,
        SUM(oi.freight_value) AS freight_revenue,
        SUM(oi.item_total_value) AS total_sales

    FROM dw.fact_order_items oi

    JOIN dw.fact_orders o
        ON oi.order_key = o.order_key

    GROUP BY 1
),

monthly_metrics AS (
    SELECT
        mr.sales_month,

        COALESCE(mo.total_orders, 0)
            AS total_orders,

        COALESCE(mo.delivered_orders, 0)
            AS delivered_orders,

        COALESCE(ms.orders_with_items, 0)
            AS orders_with_items,

        COALESCE(ms.product_revenue, 0)
            AS product_revenue,

        COALESCE(ms.freight_revenue, 0)
            AS freight_revenue,

        COALESCE(ms.total_sales, 0)
            AS total_sales,

        ROUND(
            ms.total_sales /
            NULLIF(ms.orders_with_items, 0),
            2
        ) AS avg_order_value

    FROM month_range mr

    LEFT JOIN monthly_orders mo
        ON mr.sales_month = mo.sales_month

    LEFT JOIN monthly_sales ms
        ON mr.sales_month = ms.sales_month
),

monthly_growth AS (
    SELECT
        *,

        LAG(total_sales) OVER (
            ORDER BY sales_month
        ) AS previous_month_sales

    FROM monthly_metrics
)

SELECT
    sales_month,
    total_orders,
    delivered_orders,
    orders_with_items,
    product_revenue,
    freight_revenue,
    total_sales,
    avg_order_value,

    CASE
        WHEN previous_month_sales IS NULL
          OR previous_month_sales = 0
        THEN NULL

        ELSE ROUND(
            100.0 * (
                total_sales - previous_month_sales
            ) / previous_month_sales,
            2
        )
    END AS sales_growth_pct

FROM monthly_growth;

-- =====================================================
-- VALIDATION 1 - MONTHLY RESULTS
-- =====================================================

SELECT *
FROM analytics.vw_monthly_sales
ORDER BY sales_month;

-- =====================================================
-- VALIDATION 2 - RECONCILIATION
-- =====================================================

SELECT
    COUNT(*) AS total_months,
    SUM(total_orders) AS total_orders,
    SUM(orders_with_items) AS orders_with_items,
    SUM(total_sales) AS total_sales,
    SUM(product_revenue) AS product_revenue,
    SUM(freight_revenue) AS freight_revenue
FROM analytics.vw_monthly_sales;

-- =====================================================
-- VALIDATION 3 - MONTHS WITHOUT ORDERS
-- =====================================================

SELECT
    sales_month,
    total_orders,
    total_sales,
    sales_growth_pct
FROM analytics.vw_monthly_sales
WHERE total_orders = 0
ORDER BY sales_month;
