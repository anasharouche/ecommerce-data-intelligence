
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 3 - Data Warehouse Validation
-- =====================================================

-- =====================================================
-- 1. ROW COUNT VALIDATION
-- =====================================================

WITH counts AS (
    SELECT
        'dim_customer' AS table_name,
        (SELECT COUNT(*) FROM staging.customers_clean) AS staging_rows,
        (SELECT COUNT(*) FROM dw.dim_customer) AS dw_rows

    UNION ALL SELECT
        'dim_product',
        (SELECT COUNT(*) FROM staging.products_clean),
        (SELECT COUNT(*) FROM dw.dim_product)

    UNION ALL SELECT
        'dim_seller',
        (SELECT COUNT(*) FROM staging.sellers_clean),
        (SELECT COUNT(*) FROM dw.dim_seller)

    UNION ALL SELECT
        'fact_orders',
        (SELECT COUNT(*) FROM staging.orders_clean),
        (SELECT COUNT(*) FROM dw.fact_orders)

    UNION ALL SELECT
        'fact_order_items',
        (SELECT COUNT(*) FROM staging.order_items_clean),
        (SELECT COUNT(*) FROM dw.fact_order_items)

    UNION ALL SELECT
        'fact_payments',
        (SELECT COUNT(*) FROM staging.order_payments_clean),
        (SELECT COUNT(*) FROM dw.fact_payments)

    UNION ALL SELECT
        'fact_reviews',
        (SELECT COUNT(*) FROM staging.order_reviews_clean),
        (SELECT COUNT(*) FROM dw.fact_reviews)
)

SELECT
    table_name,
    staging_rows,
    dw_rows,
    dw_rows - staging_rows AS difference,
    CASE
        WHEN staging_rows = dw_rows THEN 'PASS'
        ELSE 'FAIL'
    END AS status
FROM counts
ORDER BY table_name;

-- =====================================================
-- 2. BUSINESS KEY UNIQUENESS
-- =====================================================

SELECT
    'fact_orders' AS table_name,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT order_id) AS distinct_keys
FROM dw.fact_orders

UNION ALL SELECT
    'fact_order_items',
    COUNT(*),
    COUNT(DISTINCT (order_id, order_item_id))
FROM dw.fact_order_items

UNION ALL SELECT
    'fact_payments',
    COUNT(*),
    COUNT(DISTINCT (order_id, payment_sequential))
FROM dw.fact_payments

UNION ALL SELECT
    'fact_reviews',
    COUNT(*),
    COUNT(DISTINCT (review_id, order_id))
FROM dw.fact_reviews;

-- =====================================================
-- 3. FINANCIAL RECONCILIATION
-- =====================================================

SELECT
    ROUND(
        (SELECT SUM(item_total_value)
         FROM staging.order_items_clean),
        2
    ) AS staging_items_total,

    ROUND(
        (SELECT SUM(item_total_value)
         FROM dw.fact_order_items),
        2
    ) AS dw_items_total,

    ROUND(
        (SELECT SUM(payment_value)
         FROM staging.order_payments_clean),
        2
    ) AS staging_payments_total,

    ROUND(
        (SELECT SUM(payment_value)
         FROM dw.fact_payments),
        2
    ) AS dw_payments_total;

-- =====================================================
-- 4. DELIVERY CONSISTENCY
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

-- =====================================================
-- 5. FOREIGN KEY INTEGRITY
-- =====================================================

SELECT
    COUNT(*) AS orphan_order_items
FROM dw.fact_order_items oi
LEFT JOIN dw.fact_orders o
    ON oi.order_key = o.order_key
WHERE o.order_key IS NULL;

SELECT
    COUNT(*) AS orphan_payments
FROM dw.fact_payments p
LEFT JOIN dw.fact_orders o
    ON p.order_key = o.order_key
WHERE o.order_key IS NULL;

SELECT
    COUNT(*) AS orphan_reviews
FROM dw.fact_reviews r
LEFT JOIN dw.fact_orders o
    ON r.order_key = o.order_key
WHERE o.order_key IS NULL;

-- =====================================================
-- 6. WAREHOUSE TABLE SUMMARY
-- =====================================================

SELECT
    'dim_customer' AS table_name,
    COUNT(*) AS row_count
FROM dw.dim_customer

UNION ALL SELECT 'dim_product', COUNT(*)
FROM dw.dim_product

UNION ALL SELECT 'dim_seller', COUNT(*)
FROM dw.dim_seller

UNION ALL SELECT 'dim_date', COUNT(*)
FROM dw.dim_date

UNION ALL SELECT 'fact_orders', COUNT(*)
FROM dw.fact_orders

UNION ALL SELECT 'fact_order_items', COUNT(*)
FROM dw.fact_order_items

UNION ALL SELECT 'fact_payments', COUNT(*)
FROM dw.fact_payments

UNION ALL SELECT 'fact_reviews', COUNT(*)
FROM dw.fact_reviews

ORDER BY table_name;
