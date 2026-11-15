
-- ==========================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 3 - Remaining staging tables
-- ==========================================================

-- 1. GEOLOCATION

CREATE TABLE IF NOT EXISTS staging.geolocation_zip (
    geolocation_zip_code_prefix INTEGER PRIMARY KEY,
    observation_count BIGINT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    valid_observation_count BIGINT NOT NULL,
    invalid_observation_count BIGINT NOT NULL,
    gps_quality_ratio DOUBLE PRECISION NOT NULL,

    CONSTRAINT chk_geo_quality
        CHECK (gps_quality_ratio BETWEEN 0 AND 1),

    CONSTRAINT chk_geo_counts
        CHECK (
            observation_count >= 0
            AND valid_observation_count >= 0
            AND invalid_observation_count >= 0
            AND observation_count =
                valid_observation_count + invalid_observation_count
        )
);

-- 2. ORDER ITEMS

CREATE TABLE IF NOT EXISTS staging.order_items_clean (
    order_id TEXT NOT NULL,
    order_item_id BIGINT NOT NULL,
    product_id TEXT NOT NULL,
    seller_id TEXT NOT NULL,
    shipping_limit_date TIMESTAMP NOT NULL,
    price NUMERIC(15,2) NOT NULL,
    freight_value NUMERIC(15,2) NOT NULL,
    item_total_value NUMERIC(15,2) NOT NULL,

    CONSTRAINT pk_staging_order_items
        PRIMARY KEY (order_id, order_item_id),

    CONSTRAINT fk_items_order
        FOREIGN KEY (order_id)
        REFERENCES staging.orders_clean(order_id),

    CONSTRAINT fk_items_product
        FOREIGN KEY (product_id)
        REFERENCES staging.products_clean(product_id),

    CONSTRAINT fk_items_seller
        FOREIGN KEY (seller_id)
        REFERENCES staging.sellers_clean(seller_id),

    CONSTRAINT chk_items_price
        CHECK (price > 0 AND freight_value >= 0),

    CONSTRAINT chk_items_total
        CHECK (item_total_value = price + freight_value)
);

-- 3. ORDER PAYMENTS

CREATE TABLE IF NOT EXISTS staging.order_payments_clean (
    order_id TEXT NOT NULL,
    payment_sequential BIGINT NOT NULL,
    payment_type TEXT NOT NULL,
    payment_installments BIGINT NOT NULL,
    payment_value NUMERIC(15,2) NOT NULL,
    flag_zero_payment BOOLEAN NOT NULL,
    flag_zero_installments BOOLEAN NOT NULL,
    flag_undefined_payment_type BOOLEAN NOT NULL,

    CONSTRAINT pk_staging_payments
        PRIMARY KEY (order_id, payment_sequential),

    CONSTRAINT fk_payments_order
        FOREIGN KEY (order_id)
        REFERENCES staging.orders_clean(order_id),

    CONSTRAINT chk_payment_value
        CHECK (payment_value >= 0)
);

-- 4. AGGREGATED PAYMENTS

CREATE TABLE IF NOT EXISTS staging.order_payments_aggregated (
    order_id TEXT PRIMARY KEY,
    total_payment_value NUMERIC(15,2) NOT NULL,
    payment_count BIGINT NOT NULL,
    payment_method_count BIGINT NOT NULL,
    max_installments BIGINT NOT NULL,
    has_zero_payment BOOLEAN NOT NULL,
    has_undefined_payment_type BOOLEAN NOT NULL,

    CONSTRAINT fk_agg_payments_order
        FOREIGN KEY (order_id)
        REFERENCES staging.orders_clean(order_id),

    CONSTRAINT chk_agg_payment_total
        CHECK (total_payment_value >= 0),

    CONSTRAINT chk_payment_count
        CHECK (payment_count > 0)
);

-- 5. ORDER REVIEWS

CREATE TABLE IF NOT EXISTS staging.order_reviews_clean (
    review_id TEXT NOT NULL,
    order_id TEXT NOT NULL,
    review_score BIGINT NOT NULL,
    review_comment_title TEXT,
    review_comment_message TEXT,
    review_creation_date TIMESTAMP NOT NULL,
    review_answer_timestamp TIMESTAMP NOT NULL,
    flag_has_title BOOLEAN NOT NULL,
    flag_has_message BOOLEAN NOT NULL,
    review_response_days DOUBLE PRECISION NOT NULL,
    flag_negative_response_time BOOLEAN NOT NULL,
    flag_long_response_time BOOLEAN NOT NULL,

    CONSTRAINT pk_staging_reviews
        PRIMARY KEY (review_id, order_id),

    CONSTRAINT fk_reviews_order
        FOREIGN KEY (order_id)
        REFERENCES staging.orders_clean(order_id),

    CONSTRAINT chk_review_score
        CHECK (review_score BETWEEN 1 AND 5)
);

-- 6. AGGREGATED REVIEWS

CREATE TABLE IF NOT EXISTS staging.order_reviews_aggregated (
    order_id TEXT PRIMARY KEY,
    review_count BIGINT NOT NULL,
    avg_review_score DOUBLE PRECISION NOT NULL,
    min_review_score BIGINT NOT NULL,
    max_review_score BIGINT NOT NULL,
    has_review_message BOOLEAN NOT NULL,
    has_review_title BOOLEAN NOT NULL,
    avg_response_days DOUBLE PRECISION NOT NULL,

    CONSTRAINT fk_agg_reviews_order
        FOREIGN KEY (order_id)
        REFERENCES staging.orders_clean(order_id),

    CONSTRAINT chk_review_count
        CHECK (review_count > 0),

    CONSTRAINT chk_avg_review_score
        CHECK (avg_review_score BETWEEN 1 AND 5)
);
