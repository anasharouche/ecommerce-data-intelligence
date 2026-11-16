
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 4 - Delivery vs Customer Satisfaction
-- Grain: One row per delivery status
-- =====================================================

CREATE OR REPLACE VIEW
analytics.vw_delivery_satisfaction AS

WITH review_delivery AS (
    SELECT
        r.review_key,
        r.order_key,
        r.review_score,

        CASE
            WHEN o.is_delayed = TRUE
                THEN 'delayed'
            WHEN o.is_delayed = FALSE
                THEN 'on_time'
            ELSE 'unknown'
        END AS delivery_status

    FROM dw.fact_reviews r

    INNER JOIN dw.fact_orders o
        ON r.order_key = o.order_key
)

SELECT
    delivery_status,

    COUNT(*) AS total_reviews,

    COUNT(DISTINCT order_key)
        AS reviewed_orders,

    ROUND(
        AVG(review_score)::numeric,
        2
    ) AS avg_review_score,

    COUNT(*) FILTER (
        WHERE review_score >= 4
    ) AS positive_reviews,

    COUNT(*) FILTER (
        WHERE review_score <= 2
    ) AS negative_reviews,

    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE review_score >= 4
        ) / NULLIF(COUNT(*), 0),
        2
    ) AS satisfaction_rate,

    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE review_score <= 2
        ) / NULLIF(COUNT(*), 0),
        2
    ) AS dissatisfaction_rate

FROM review_delivery
GROUP BY delivery_status;

-- =====================================================
-- VALIDATION 1 - DELIVERY VS SATISFACTION
-- =====================================================

SELECT *
FROM analytics.vw_delivery_satisfaction
ORDER BY delivery_status;

-- =====================================================
-- VALIDATION 2 - REVIEW RECONCILIATION
-- =====================================================

SELECT
    SUM(total_reviews) AS total_reviews,
    SUM(positive_reviews) AS positive_reviews,
    SUM(negative_reviews) AS negative_reviews
FROM analytics.vw_delivery_satisfaction;

-- =====================================================
-- VALIDATION 3 - DISTINCT REVIEWED ORDERS
-- =====================================================

SELECT
    COUNT(DISTINCT order_key)
        AS distinct_reviewed_orders
FROM dw.fact_reviews;
