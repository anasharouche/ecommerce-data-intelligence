
# E-Commerce Data Intelligence Platform — Codex Instructions

## Project context

This repository contains an end-to-end e-commerce data intelligence platform built with the Brazilian Olist dataset.

The project combines:
- Python ETL / data preparation
- PostgreSQL Data Warehouse
- SQL analytics
- Power BI dashboards
- Machine Learning for delivery-delay prediction
- FastAPI and an interactive web interface
- Testing, orchestration and deployment

## Current status

Phase 1: Data Exploration — completed.
Phase 2: ETL and Data Preparation — completed.
Phase 3: PostgreSQL Data Warehouse — completed.
Phase 4: SQL Analytics and Power BI — in progress.

Do not rebuild completed phases without explicit authorization.

## Database

PostgreSQL database: ecommerce_dw
Host: localhost
Port: 5432

Schemas:
- staging
- dw
- analytics

The analytics schema contains seven validated views:

1. analytics.vw_global_kpis
2. analytics.vw_monthly_sales
3. analytics.vw_category_sales
4. analytics.vw_seller_performance
5. analytics.vw_delivery_performance
6. analytics.vw_customer_satisfaction
7. analytics.vw_delivery_satisfaction

The SQL analytics validation suite has 12 passing checks.

Use the SQL files in sql/analytics/ and sql/validation/
as the authoritative sources for definitions and expected results.

## Power BI

The powerbi/ directory contains a PBIP project.

Prefer editing version-controlled PBIP, PBIR and TMDL
definitions instead of binary PBIX files.

Inspect the actual file structure and existing semantic model
before making any changes.

Preserve all existing data sources, tables and measures.

Do not invent column names or relationships.

The seven analytics views have different granularities.
Do not create relationships between aggregated views
without proving that they are semantically correct.

Do not assume slicers will cross-filter disconnected tables.

## Business KPI definitions

Total Sales:
15,843,553.24 BRL

Total Orders:
99,441

Average Order Value:
160.58 BRL

Delay Rate:
6.77%

On-Time Delivery Rate:
93.23%

Customer Satisfaction Rate:
77.07%

Total Sales represents item prices plus freight,
not necessarily recognized accounting revenue.

Average Order Value uses orders containing items.

Delivery rates must use evaluated orders as denominator.

Satisfaction rate is the percentage of individual reviews
with scores of 4 or 5.

For DAX percentages, return fractions between 0 and 1
and apply Power BI percentage formatting.

Never average precomputed monthly percentage values
to obtain global rates.

## Development rules

1. Inspect before modifying.
2. Present a concise implementation plan.
3. Work in small, reviewable increments.
4. Preserve existing project functionality.
5. Do not change the PostgreSQL schema unless requested.
6. Do not modify existing validated SQL logic without approval.
7. Never expose credentials or secrets.
8. Never commit local credentials or cache files.
9. Prefer maintainable, documented solutions.
10. Validate generated Power BI definitions when tooling permits.
11. Explain any validation that requires Power BI Desktop.
12. Do not push to GitHub without explicit approval.

## Communication

Respond in French.

Explain technical choices clearly.

At the end of each task, report:
- Files created or modified
- Changes implemented
- Tests performed
- Remaining manual validation steps

Wait for user confirmation before starting the next major task.
