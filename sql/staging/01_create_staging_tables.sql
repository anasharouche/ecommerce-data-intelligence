
-- ==========================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 3 - Staging tables
-- ==========================================================

-- 1. CUSTOMERS

CREATE TABLE IF NOT EXISTS staging.customers_clean (
    customer_id TEXT PRIMARY KEY,
    customer_unique_id TEXT NOT NULL,
    customer_zip_code_prefix INTEGER NOT NULL,
    customer_city TEXT NOT NULL,
    customer_state TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    gps_quality_ratio DOUBLE PRECISION,
    flag_has_gps BOOLEAN NOT NULL
);

-- 2. SELLERS

CREATE TABLE IF NOT EXISTS staging.sellers_clean (
    seller_id TEXT PRIMARY KEY,
    seller_zip_code_prefix INTEGER NOT NULL,
    seller_city TEXT NOT NULL,
    seller_state TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    gps_quality_ratio DOUBLE PRECISION,
    flag_has_gps BOOLEAN NOT NULL
);

-- 3. PRODUCTS

CREATE TABLE IF NOT EXISTS staging.products_clean (
    product_id TEXT PRIMARY KEY,
    product_category_name TEXT NOT NULL,
    product_name_lenght INTEGER,
    product_description_lenght INTEGER,
    product_photos_qty INTEGER,
    product_weight_g DOUBLE PRECISION,
    product_length_cm DOUBLE PRECISION,
    product_height_cm DOUBLE PRECISION,
    product_width_cm DOUBLE PRECISION,
    flag_missing_category BOOLEAN NOT NULL,
    flag_zero_weight BOOLEAN NOT NULL,
    flag_missing_dimensions BOOLEAN NOT NULL,
    product_category_name_english TEXT NOT NULL,
    flag_missing_translation BOOLEAN NOT NULL
);

-- 4. ORDERS

CREATE TABLE IF NOT EXISTS staging.orders_clean (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    order_status TEXT NOT NULL,
    order_purchase_timestamp TIMESTAMP NOT NULL,
    order_approved_at TIMESTAMP,
    order_delivered_carrier_date TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP NOT NULL,
    flag_delivery_before_purchase BOOLEAN NOT NULL,
    flag_delivered_without_date BOOLEAN NOT NULL,
    flag_canceled_with_delivery BOOLEAN NOT NULL,

    CONSTRAINT fk_staging_order_customer
        FOREIGN KEY (customer_id)
        REFERENCES staging.customers_clean(customer_id)
);
