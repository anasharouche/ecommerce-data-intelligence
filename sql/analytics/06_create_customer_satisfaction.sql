
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 4 - Customer Satisfaction Analytics
-- Grain: One row per review creation month
-- =====================================================

CREATE OR REPLACE VIEW analytics.vw_customer_satisfaction AS

WITH monthly_reviews AS (
    SELECT
        DATE_TRUNC(
            'month',
            TO_DATE(
                r.review_creation_date_key::text,
                'YYYYMMDD'
            )
        )::date AS review_month,

        COUNT(*) AS total_reviews,

        COUNT(DISTINCT r.order_key)
            AS reviewed_orders,

        COUNT(*) FILTER (
            WHERE r.review_score >= 4
        ) AS positive_reviews,

        COUNT(*) FILTER (
            WHERE r.review_score <= 2
        ) AS negative_reviews,

        COUNT(*) FILTER (
            WHERE r.review_score = 3
        ) AS neutral_reviews,

        ROUND(
            AVG(r.review_score)::numeric,
            2
        ) AS avg_review_score,

        ROUND(
            AVG(r.review_response_days)::numeric,
            2
        ) AS avg_response_days,

        COUNT(*) FILTER (
            WHERE r.flag_has_message = TRUE
        ) AS reviews_with_message,

        COUNT(*) FILTER (
            WHERE o.is_delayed = TRUE
        ) AS reviews_for_delayed_orders,

        COUNT(*) FILTER (
            WHERE o.is_delayed = FALSE
        ) AS reviews_for_on_time_orders

    FROM dw.fact_reviews r

    JOIN dw.fact_orders o
        ON r.order_key = o.order_key

    GROUP BY 1
)

SELECT
    review_month,
    total_reviews,
    reviewed_orders,

    positive_reviews,
    negative_reviews,
    neutral_reviews,

    avg_review_score,
    avg_response_days,

    reviews_with_message,

    reviews_for_delayed_orders,
    reviews_for_on_time_orders,

    ROUND(
        100.0 * positive_reviews /
        NULLIF(total_reviews, 0),
        2
    ) AS satisfaction_rate,

    ROUND(
        100.0 * negative_reviews /
        NULLIF(total_reviews, 0),
        2
    ) AS dissatisfaction_rate,

    ROUND(
        100.0 * reviews_with_message /
        NULLIF(total_reviews, 0),
        2
    ) AS message_rate

FROM monthly_reviews;

-- =====================================================
-- VALIDATION 1 - MONTHLY SATISFACTION
-- =====================================================

SELECT *
FROM analytics.vw_customer_satisfaction
ORDER BY review_month;

-- =====================================================
-- VALIDATION 2 - GLOBAL COUNTS
-- =====================================================

SELECT
    SUM(total_reviews) AS total_reviews,
    SUM(positive_reviews) AS positive_reviews,
    SUM(negative_reviews) AS negative_reviews,
    SUM(neutral_reviews) AS neutral_reviews,
    SUM(reviews_with_message) AS reviews_with_message,

    ROUND(
        100.0 * SUM(positive_reviews) /
        NULLIF(SUM(total_reviews), 0),
        2
    ) AS global_satisfaction_rate

FROM analytics.vw_customer_satisfaction;

-- =====================================================
-- VALIDATION 3 - CONSISTENCY
-- =====================================================

SELECT
    COUNT(*) AS inconsistent_months
FROM analytics.vw_customer_satisfaction
WHERE total_reviews <>
    positive_reviews + negative_reviews + neutral_reviews;
