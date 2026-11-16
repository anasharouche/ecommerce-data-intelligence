
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 4 - Analytics Layer Validation
-- =====================================================

WITH checks AS (

    -- 1. Global orders
    SELECT
        '01_global_orders' AS test_name,
        (SELECT total_orders
         FROM analytics.vw_global_kpis)::numeric
            AS actual_value,
        (SELECT COUNT(*)
         FROM dw.fact_orders)::numeric
            AS expected_value

    UNION ALL

    -- 2. Monthly orders
    SELECT
        '02_monthly_orders',
        (SELECT SUM(total_orders)
         FROM analytics.vw_monthly_sales),
        (SELECT COUNT(*)
         FROM dw.fact_orders)

    UNION ALL

    -- 3. Monthly sales
    SELECT
        '03_monthly_sales',
        (SELECT SUM(total_sales)
         FROM analytics.vw_monthly_sales),
        (SELECT SUM(item_total_value)
         FROM dw.fact_order_items)

    UNION ALL

    -- 4. Category sales
    SELECT
        '04_category_sales',
        (SELECT SUM(total_sales)
         FROM analytics.vw_category_sales),
        (SELECT SUM(item_total_value)
         FROM dw.fact_order_items)

    UNION ALL

    -- 5. Seller sales
    SELECT
        '05_seller_sales',
        (SELECT SUM(total_sales)
         FROM analytics.vw_seller_performance),
        (SELECT SUM(item_total_value)
         FROM dw.fact_order_items)

    UNION ALL

    -- 6. Seller items
    SELECT
        '06_seller_items',
        (SELECT SUM(items_sold)
         FROM analytics.vw_seller_performance),
        (SELECT COUNT(*)
         FROM dw.fact_order_items)

    UNION ALL

    -- 7. Delivery delayed orders
    SELECT
        '07_delayed_orders',
        (SELECT SUM(delayed_orders)
         FROM analytics.vw_delivery_performance),
        (SELECT COUNT(*)
         FROM dw.fact_orders
         WHERE is_delayed = TRUE)

    UNION ALL

    -- 8. Monthly reviews
    SELECT
        '08_monthly_reviews',
        (SELECT SUM(total_reviews)
         FROM analytics.vw_customer_satisfaction),
        (SELECT COUNT(*)
         FROM dw.fact_reviews)

    UNION ALL

    -- 9. Delivery satisfaction reviews
    SELECT
        '09_delivery_satisfaction',
        (SELECT SUM(total_reviews)
         FROM analytics.vw_delivery_satisfaction),
        (SELECT COUNT(*)
         FROM dw.fact_reviews)

    UNION ALL

    -- 10. Calendar continuity
    SELECT
        '10_calendar_months',
        (SELECT COUNT(*)
         FROM analytics.vw_monthly_sales),
        (
            SELECT
                (
                    (
                        EXTRACT(YEAR FROM MAX(sales_month))
                        - EXTRACT(YEAR FROM MIN(sales_month))
                    ) * 12
                    + EXTRACT(MONTH FROM MAX(sales_month))
                    - EXTRACT(MONTH FROM MIN(sales_month))
                    + 1
                )
            FROM analytics.vw_monthly_sales
        )

    UNION ALL

    -- 11. Category items
    SELECT
        '11_category_items',
        (SELECT SUM(items_sold)
         FROM analytics.vw_category_sales),
        (SELECT COUNT(*)
         FROM dw.fact_order_items)

    UNION ALL

    -- 12. Review score distribution
    SELECT
        '12_review_distribution',
        (
            SELECT SUM(
                positive_reviews
                + negative_reviews
                + neutral_reviews
            )
            FROM analytics.vw_customer_satisfaction
        ),
        (SELECT COUNT(*)
         FROM dw.fact_reviews)
)

SELECT
    test_name,
    actual_value,
    expected_value,

    CASE
        WHEN actual_value = expected_value
            THEN 'PASS'
        ELSE 'FAIL'
    END AS status

FROM checks
ORDER BY test_name;
