
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 3 - Correct delivery delay business rule
-- =====================================================

BEGIN;

UPDATE dw.fact_orders
SET
    is_delayed = CASE
        WHEN order_status = 'delivered'
             AND delivered_timestamp IS NOT NULL
        THEN (
            delivered_timestamp::date
            > estimated_delivery_timestamp::date
        )
        ELSE NULL
    END,

    delivery_delay_days = CASE
        WHEN order_status = 'delivered'
             AND delivered_timestamp IS NOT NULL
        THEN (
            delivered_timestamp::date
            - estimated_delivery_timestamp::date
        )::double precision
        ELSE NULL
    END;

COMMIT;

-- =====================================================
-- VALIDATION
-- =====================================================

SELECT
    COUNT(*) AS total_orders,
    COUNT(*) FILTER (
        WHERE is_delayed = TRUE
    ) AS delayed_orders,
    COUNT(*) FILTER (
        WHERE is_delayed = FALSE
    ) AS on_time_orders,
    COUNT(*) FILTER (
        WHERE is_delayed IS NULL
    ) AS unknown_delivery_status,
    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE is_delayed = TRUE
        ) / NULLIF(
            COUNT(*) FILTER (
                WHERE is_delayed IS NOT NULL
            ), 0
        ),
        2
    ) AS delayed_percentage
FROM dw.fact_orders;

SELECT
    ROUND(AVG(delivery_days)::numeric, 2)
        AS avg_delivery_days,
    ROUND(AVG(delivery_delay_days)::numeric, 2)
        AS avg_delivery_delay_days
FROM dw.fact_orders;
