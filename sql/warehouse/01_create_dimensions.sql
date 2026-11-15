
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 3 - Data Warehouse Dimensions
-- =====================================================

-- 1. CUSTOMER DIMENSION

CREATE TABLE IF NOT EXISTS dw.dim_customer (
    customer_key BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id TEXT NOT NULL UNIQUE,
    customer_unique_id TEXT NOT NULL,
    customer_zip_code_prefix INTEGER NOT NULL,
    customer_city TEXT NOT NULL,
    customer_state TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    gps_quality_ratio DOUBLE PRECISION,
    flag_has_gps BOOLEAN NOT NULL
);

-- 2. PRODUCT DIMENSION

CREATE TABLE IF NOT EXISTS dw.dim_product (
    product_key BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    product_id TEXT NOT NULL UNIQUE,
    product_category_name TEXT NOT NULL,
    product_category_name_english TEXT NOT NULL,
    product_name_length INTEGER,
    product_description_length INTEGER,
    product_photos_qty INTEGER,
    product_weight_g DOUBLE PRECISION,
    product_length_cm DOUBLE PRECISION,
    product_height_cm DOUBLE PRECISION,
    product_width_cm DOUBLE PRECISION,
    flag_missing_category BOOLEAN NOT NULL,
    flag_zero_weight BOOLEAN NOT NULL,
    flag_missing_dimensions BOOLEAN NOT NULL,
    flag_missing_translation BOOLEAN NOT NULL
);

-- 3. SELLER DIMENSION

CREATE TABLE IF NOT EXISTS dw.dim_seller (
    seller_key BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    seller_id TEXT NOT NULL UNIQUE,
    seller_zip_code_prefix INTEGER NOT NULL,
    seller_city TEXT NOT NULL,
    seller_state TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    gps_quality_ratio DOUBLE PRECISION,
    flag_has_gps BOOLEAN NOT NULL
);

-- 4. DATE DIMENSION

CREATE TABLE IF NOT EXISTS dw.dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date DATE NOT NULL UNIQUE,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    month INTEGER NOT NULL,
    month_name TEXT NOT NULL,
    day INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,
    day_name TEXT NOT NULL,
    is_weekend BOOLEAN NOT NULL,

    CONSTRAINT chk_dim_date_month
        CHECK (month BETWEEN 1 AND 12),

    CONSTRAINT chk_dim_date_quarter
        CHECK (quarter BETWEEN 1 AND 4),

    CONSTRAINT chk_dim_date_day_of_week
        CHECK (day_of_week BETWEEN 1 AND 7)
);
