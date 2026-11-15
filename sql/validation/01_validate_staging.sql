
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 3 - Staging validation
-- =====================================================

-- 1. Vérification des volumes
SELECT 'customers_clean' AS table_name, COUNT(*) AS rows
FROM staging.customers_clean
UNION ALL
SELECT 'sellers_clean', COUNT(*)
FROM staging.sellers_clean
UNION ALL
SELECT 'products_clean', COUNT(*)
FROM staging.products_clean
UNION ALL
SELECT 'geolocation_zip', COUNT(*)
FROM staging.geolocation_zip
UNION ALL
SELECT 'orders_clean', COUNT(*)
FROM staging.orders_clean
UNION ALL
SELECT 'order_items_clean', COUNT(*)
FROM staging.order_items_clean
UNION ALL
SELECT 'order_payments_clean', COUNT(*)
FROM staging.order_payments_clean
UNION ALL
SELECT 'order_payments_aggregated', COUNT(*)
FROM staging.order_payments_aggregated
UNION ALL
SELECT 'order_reviews_clean', COUNT(*)
FROM staging.order_reviews_clean
UNION ALL
SELECT 'order_reviews_aggregated', COUNT(*)
FROM staging.order_reviews_aggregated
ORDER BY table_name;

-- 2. Intégrité référentielle
SELECT
    COUNT(*) AS orphan_order_items
FROM staging.order_items_clean oi
LEFT JOIN staging.orders_clean o
    ON oi.order_id = o.order_id
WHERE o.order_id IS NULL;

-- 3. Vérification des montants
SELECT
    SUM(price) AS total_product_value,
    SUM(freight_value) AS total_freight_value,
    SUM(item_total_value) AS total_items_value
FROM staging.order_items_clean;

-- 4. Vérification des dates de livraison
SELECT
    COUNT(*) FILTER (
        WHERE order_status = 'delivered'
    ) AS delivered_orders,
    COUNT(*) FILTER (
        WHERE flag_delivered_without_date
    ) AS flagged_missing_delivery,
    COUNT(*) FILTER (
        WHERE flag_canceled_with_delivery
    ) AS flagged_canceled_with_delivery
FROM staging.orders_clean;

-- 5. Vérification des avis
SELECT
    COUNT(*) AS total_reviews,
    COUNT(DISTINCT order_id) AS reviewed_orders,
    ROUND(AVG(review_score), 3) AS avg_review_score
FROM staging.order_reviews_clean;
