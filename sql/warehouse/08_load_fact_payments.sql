
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 3 - Load Fact Payments
-- =====================================================

BEGIN;

-- =====================================================
-- 1. LOAD PAYMENTS
-- =====================================================

INSERT INTO dw.fact_payments (
    order_id,
    payment_sequential,
    order_key,
    payment_type,
    payment_installments,
    payment_value,
    flag_zero_payment,
    flag_zero_installments,
    flag_undefined_payment_type
)

SELECT
    p.order_id,
    p.payment_sequential,
    fo.order_key,
    p.payment_type,
    p.payment_installments,
    p.payment_value,
    p.flag_zero_payment,
    p.flag_zero_installments,
    p.flag_undefined_payment_type

FROM staging.order_payments_clean p

INNER JOIN dw.fact_orders fo
    ON p.order_id = fo.order_id

ON CONFLICT (order_id, payment_sequential) DO NOTHING;

COMMIT;

-- =====================================================
-- 2. VALIDATION - ROW COUNTS
-- =====================================================

SELECT
    (SELECT COUNT(*)
     FROM staging.order_payments_clean)
        AS staging_rows,

    (SELECT COUNT(*)
     FROM dw.fact_payments)
        AS fact_rows;

-- =====================================================
-- 3. VALIDATION - FINANCIAL MEASURES
-- =====================================================

SELECT
    COUNT(*) AS total_payments,

    ROUND(SUM(payment_value), 2)
        AS total_payment_value,

    ROUND(AVG(payment_value), 2)
        AS avg_payment_value,

    MAX(payment_installments)
        AS max_installments

FROM dw.fact_payments;

-- =====================================================
-- 4. VALIDATION - PAYMENT METHODS
-- =====================================================

SELECT
    payment_type,
    COUNT(*) AS payment_count,
    ROUND(SUM(payment_value), 2)
        AS total_payment_value

FROM dw.fact_payments

GROUP BY payment_type
ORDER BY payment_count DESC;

-- =====================================================
-- 5. VALIDATION - BUSINESS KEYS
-- =====================================================

SELECT
    COUNT(*) AS total_rows,

    COUNT(DISTINCT (order_id, payment_sequential))
        AS distinct_payments

FROM dw.fact_payments;
