
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 3 - Load Fact Reviews
-- =====================================================

BEGIN;

-- =====================================================
-- 1. LOAD REVIEWS
-- =====================================================

INSERT INTO dw.fact_reviews (
    review_id,
    order_id,
    order_key,
    review_creation_date_key,
    review_answer_date_key,
    review_score,
    review_response_days,
    flag_has_title,
    flag_has_message,
    flag_negative_response_time,
    flag_long_response_time
)

SELECT
    r.review_id,
    r.order_id,
    fo.order_key,

    TO_CHAR(
        r.review_creation_date,
        'YYYYMMDD'
    )::integer,

    TO_CHAR(
        r.review_answer_timestamp,
        'YYYYMMDD'
    )::integer,

    r.review_score,
    r.review_response_days,

    r.flag_has_title,
    r.flag_has_message,
    r.flag_negative_response_time,
    r.flag_long_response_time

FROM staging.order_reviews_clean r

INNER JOIN dw.fact_orders fo
    ON r.order_id = fo.order_id

ON CONFLICT (review_id, order_id) DO NOTHING;

COMMIT;

-- =====================================================
-- 2. VALIDATION - ROW COUNTS
-- =====================================================

SELECT
    (SELECT COUNT(*)
     FROM staging.order_reviews_clean)
        AS staging_rows,

    (SELECT COUNT(*)
     FROM dw.fact_reviews)
        AS fact_rows;

-- =====================================================
-- 3. VALIDATION - REVIEW SCORES
-- =====================================================

SELECT
    review_score,
    COUNT(*) AS review_count
FROM dw.fact_reviews
GROUP BY review_score
ORDER BY review_score;

-- =====================================================
-- 4. VALIDATION - SATISFACTION KPIs
-- =====================================================

SELECT
    COUNT(*) AS total_reviews,

    ROUND(AVG(review_score)::numeric, 2)
        AS avg_review_score,

    ROUND(AVG(review_response_days)::numeric, 2)
        AS avg_response_days,

    COUNT(*) FILTER (
        WHERE flag_has_message = TRUE
    ) AS reviews_with_message,

    COUNT(*) FILTER (
        WHERE flag_negative_response_time = TRUE
    ) AS negative_response_times

FROM dw.fact_reviews;

-- =====================================================
-- 5. VALIDATION - BUSINESS KEYS
-- =====================================================

SELECT
    COUNT(*) AS total_rows,

    COUNT(DISTINCT (review_id, order_id))
        AS distinct_reviews

FROM dw.fact_reviews;
