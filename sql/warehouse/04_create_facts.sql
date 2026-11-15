
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 3 - Fact Tables
-- =====================================================

BEGIN;

-- 1. ORDERS FACT

CREATE TABLE IF NOT EXISTS dw.fact_orders (
    order_key BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id TEXT NOT NULL UNIQUE,
    customer_key BIGINT NOT NULL
        REFERENCES dw.dim_customer(customer_key),

    purchase_date_key INTEGER NOT NULL
        REFERENCES dw.dim_date(date_key),
    approved_date_key INTEGER
        REFERENCES dw.dim_date(date_key),
    carrier_date_key INTEGER
        REFERENCES dw.dim_date(date_key),
    delivered_date_key INTEGER
        REFERENCES dw.dim_date(date_key),
    estimated_date_key INTEGER NOT NULL
        REFERENCES dw.dim_date(date_key),

    order_status TEXT NOT NULL,
    purchase_timestamp TIMESTAMP NOT NULL,
    delivered_timestamp TIMESTAMP,
    estimated_delivery_timestamp TIMESTAMP NOT NULL,

    delivery_days DOUBLE PRECISION,
    delivery_delay_days DOUBLE PRECISION,
    is_delayed BOOLEAN,

    item_count BIGINT,
    total_item_value NUMERIC(15,2),
    total_freight_value NUMERIC(15,2),
    total_order_value NUMERIC(15,2),
    total_payment_value NUMERIC(15,2),

    flag_delivered_without_date BOOLEAN NOT NULL,
    flag_canceled_with_delivery BOOLEAN NOT NULL
);

-- 2. ORDER ITEMS FACT

CREATE TABLE IF NOT EXISTS dw.fact_order_items (
    order_item_key BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id TEXT NOT NULL,
    order_item_id BIGINT NOT NULL,

    order_key BIGINT NOT NULL
        REFERENCES dw.fact_orders(order_key),
    product_key BIGINT NOT NULL
        REFERENCES dw.dim_product(product_key),
    seller_key BIGINT NOT NULL
        REFERENCES dw.dim_seller(seller_key),

    shipping_limit_date_key INTEGER NOT NULL
        REFERENCES dw.dim_date(date_key),

    price NUMERIC(15,2) NOT NULL,
    freight_value NUMERIC(15,2) NOT NULL,
    item_total_value NUMERIC(15,2) NOT NULL,

    CONSTRAINT uq_fact_order_items
        UNIQUE (order_id, order_item_id)
);

-- 3. PAYMENTS FACT

CREATE TABLE IF NOT EXISTS dw.fact_payments (
    payment_key BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id TEXT NOT NULL,
    payment_sequential BIGINT NOT NULL,

    order_key BIGINT NOT NULL
        REFERENCES dw.fact_orders(order_key),

    payment_type TEXT NOT NULL,
    payment_installments BIGINT NOT NULL,
    payment_value NUMERIC(15,2) NOT NULL,

    flag_zero_payment BOOLEAN NOT NULL,
    flag_zero_installments BOOLEAN NOT NULL,
    flag_undefined_payment_type BOOLEAN NOT NULL,

    CONSTRAINT uq_fact_payments
        UNIQUE (order_id, payment_sequential)
);

-- 4. REVIEWS FACT

CREATE TABLE IF NOT EXISTS dw.fact_reviews (
    review_key BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    review_id TEXT NOT NULL,
    order_id TEXT NOT NULL,

    order_key BIGINT NOT NULL
        REFERENCES dw.fact_orders(order_key),

    review_creation_date_key INTEGER NOT NULL
        REFERENCES dw.dim_date(date_key),
    review_answer_date_key INTEGER NOT NULL
        REFERENCES dw.dim_date(date_key),

    review_score BIGINT NOT NULL,
    review_response_days DOUBLE PRECISION NOT NULL,

    flag_has_title BOOLEAN NOT NULL,
    flag_has_message BOOLEAN NOT NULL,
    flag_negative_response_time BOOLEAN NOT NULL,
    flag_long_response_time BOOLEAN NOT NULL,

    CONSTRAINT uq_fact_reviews
        UNIQUE (review_id, order_id)
);

COMMIT;
