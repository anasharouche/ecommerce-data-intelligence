
-- =====================================================
-- E-COMMERCE DATA INTELLIGENCE PLATFORM
-- Phase 3 - Data Warehouse PostgreSQL
-- Step 1: Create database schemas
-- =====================================================

-- Create staging schema
CREATE SCHEMA IF NOT EXISTS staging
AUTHORIZATION ecommerce_user;

-- Create analytical warehouse schema
CREATE SCHEMA IF NOT EXISTS dw
AUTHORIZATION ecommerce_user;

-- Verify schemas
SELECT
    schema_name,
    schema_owner
FROM information_schema.schemata
WHERE schema_name IN ('staging', 'dw')
ORDER BY schema_name;
