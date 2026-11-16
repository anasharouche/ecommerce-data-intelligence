
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 4 - Product Category Sales Analytics
-- =====================================================

CREATE OR REPLACE VIEW analytics.vw_category_sales AS

WITH category_metrics AS (
    SELECT
        COALESCE(
            p.product_category_name_english,
            'unknown'
        ) AS category_name,

        COUNT(*) AS items_sold,

        COUNT(DISTINCT oi.order_id)
            AS total_orders,

        COUNT(DISTINCT oi.product_key)
            AS distinct_products,

        SUM(oi.price)
            AS product_revenue,

        SUM(oi.freight_value)
            AS freight_revenue,

        SUM(oi.item_total_value)
            AS total_sales,

        ROUND(
            AVG(oi.price),
            2
        ) AS avg_item_price

    FROM dw.fact_order_items oi

    INNER JOIN dw.dim_product p
        ON oi.product_key = p.product_key

    GROUP BY 1
)

SELECT
    category_name,
    items_sold,
    total_orders,
    distinct_products,
    product_revenue,
    freight_revenue,
    total_sales,
    avg_item_price,

    ROUND(
        100.0 * total_sales /
        NULLIF(
            SUM(total_sales) OVER (), 0
        ),
        2
    ) AS sales_share_pct,

    DENSE_RANK() OVER (
        ORDER BY total_sales DESC
    ) AS sales_rank

FROM category_metrics;

-- =====================================================
-- VALIDATION 1 - TOP 15 CATEGORIES
-- =====================================================

SELECT *
FROM analytics.vw_category_sales
ORDER BY total_sales DESC
LIMIT 15;

-- =====================================================
-- VALIDATION 2 - RECONCILIATION
-- =====================================================

SELECT
    COUNT(*) AS total_categories,

    SUM(items_sold) AS total_items,

    ROUND(SUM(product_revenue), 2)
        AS product_revenue,

    ROUND(SUM(freight_revenue), 2)
        AS freight_revenue,

    ROUND(SUM(total_sales), 2)
        AS total_sales,

    ROUND(SUM(sales_share_pct), 2)
        AS total_sales_share_pct

FROM analytics.vw_category_sales;

-- =====================================================
-- VALIDATION 3 - UNKNOWN CATEGORY
-- =====================================================

SELECT *
FROM analytics.vw_category_sales
WHERE category_name = 'unknown';
