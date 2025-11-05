-- GALILEO V2.0 Database Initialization Script
-- This script is automatically run when PostgreSQL container starts

-- Ensure database and user exist (already created by env vars, but double-check)
DO
$$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'gravity_ops') THEN
        CREATE DATABASE gravity_ops;
    END IF;
END
$$;

-- Grant all privileges
GRANT ALL PRIVILEGES ON DATABASE gravity_ops TO gravity;

-- Connect to the database
\c gravity_ops

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schemas (if needed for multi-tenancy later)
CREATE SCHEMA IF NOT EXISTS public;
GRANT ALL ON SCHEMA public TO gravity;

-- Set default privileges
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO gravity;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO gravity;

-- Create initial tables (these will be managed by SQLAlchemy, but we can add indexes)
-- Note: SQLAlchemy will create the actual tables via Base.metadata.create_all()

COMMENT ON DATABASE gravity_ops IS 'GALILEO V2.0 Gravity Operations Database';
