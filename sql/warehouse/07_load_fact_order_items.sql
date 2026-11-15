
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 3 - Load Fact Order Items
-- =====================================================

BEGIN;

-- =====================================================
-- 1. LOAD ORDER ITEMS
-- =====================================================

INSERT INTO dw.fact_order_items (
    order_id,
    order_item_id,
    order_key,
    product_key,
    seller_key,
    shipping_limit_date_key,
    price,
    freight_value,
    item_total_value
)

SELECT
    oi.order_id,
    oi.order_item_id,

    fo.order_key,
    dp.product_key,
    ds.seller_key,

    TO_CHAR(
        oi.shipping_limit_date,
        'YYYYMMDD'
    )::integer,

    oi.price,
    oi.freight_value,
    oi.item_total_value

FROM staging.order_items_clean oi

INNER JOIN dw.fact_orders fo
    ON oi.order_id = fo.order_id

INNER JOIN dw.dim_product dp
    ON oi.product_id = dp.product_id

INNER JOIN dw.dim_seller ds
    ON oi.seller_id = ds.seller_id

ON CONFLICT (order_id, order_item_id) DO NOTHING;

COMMIT;

-- =====================================================
-- 2. VALIDATION - ROW COUNTS
-- =====================================================

SELECT
    (SELECT COUNT(*)
     FROM staging.order_items_clean)
        AS staging_rows,

    (SELECT COUNT(*)
     FROM dw.fact_order_items)
        AS fact_rows;

-- =====================================================
-- 3. VALIDATION - FINANCIAL MEASURES
-- =====================================================

SELECT
    COUNT(*) AS total_items,

    ROUND(SUM(price), 2)
        AS total_product_revenue,

    ROUND(SUM(freight_value), 2)
        AS total_freight_revenue,

    ROUND(SUM(item_total_value), 2)
        AS total_order_items_value

FROM dw.fact_order_items;

-- =====================================================
-- 4. VALIDATION - BUSINESS KEYS
-- =====================================================

SELECT
    COUNT(*) AS total_rows,

    COUNT(DISTINCT (order_id, order_item_id))
        AS distinct_order_items

FROM dw.fact_order_items;
