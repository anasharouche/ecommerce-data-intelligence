
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 3 - Load Business Dimensions
-- =====================================================

BEGIN;

-- =====================================================
-- 1. CUSTOMER DIMENSION
-- =====================================================

INSERT INTO dw.dim_customer (
    customer_id,
    customer_unique_id,
    customer_zip_code_prefix,
    customer_city,
    customer_state,
    latitude,
    longitude,
    gps_quality_ratio,
    flag_has_gps
)
SELECT
    customer_id,
    customer_unique_id,
    customer_zip_code_prefix,
    customer_city,
    customer_state,
    latitude,
    longitude,
    gps_quality_ratio,
    flag_has_gps
FROM staging.customers_clean
ON CONFLICT (customer_id) DO NOTHING;

-- =====================================================
-- 2. PRODUCT DIMENSION
-- =====================================================

INSERT INTO dw.dim_product (
    product_id,
    product_category_name,
    product_category_name_english,
    product_name_length,
    product_description_length,
    product_photos_qty,
    product_weight_g,
    product_length_cm,
    product_height_cm,
    product_width_cm,
    flag_missing_category,
    flag_zero_weight,
    flag_missing_dimensions,
    flag_missing_translation
)
SELECT
    product_id,
    product_category_name,
    product_category_name_english,
    product_name_lenght,
    product_description_lenght,
    product_photos_qty,
    product_weight_g,
    product_length_cm,
    product_height_cm,
    product_width_cm,
    flag_missing_category,
    flag_zero_weight,
    flag_missing_dimensions,
    flag_missing_translation
FROM staging.products_clean
ON CONFLICT (product_id) DO NOTHING;

-- =====================================================
-- 3. SELLER DIMENSION
-- =====================================================

INSERT INTO dw.dim_seller (
    seller_id,
    seller_zip_code_prefix,
    seller_city,
    seller_state,
    latitude,
    longitude,
    gps_quality_ratio,
    flag_has_gps
)
SELECT
    seller_id,
    seller_zip_code_prefix,
    seller_city,
    seller_state,
    latitude,
    longitude,
    gps_quality_ratio,
    flag_has_gps
FROM staging.sellers_clean
ON CONFLICT (seller_id) DO NOTHING;

COMMIT;

-- =====================================================
-- VALIDATION
-- =====================================================

SELECT 'dim_customer' AS dimension, COUNT(*) AS rows
FROM dw.dim_customer

UNION ALL

SELECT 'dim_product', COUNT(*)
FROM dw.dim_product

UNION ALL

SELECT 'dim_seller', COUNT(*)
FROM dw.dim_seller

ORDER BY dimension;
