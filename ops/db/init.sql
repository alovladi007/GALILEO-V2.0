-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Processing jobs table
CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(50) NOT NULL, -- 'plan', 'ingest', 'process', 'catalog'
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    config JSONB,
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Satellite observations time-series table
CREATE TABLE IF NOT EXISTS satellite_observations (
    time TIMESTAMPTZ NOT NULL,
    satellite_id VARCHAR(50) NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    altitude DOUBLE PRECISION NOT NULL,
    gravity_value DOUBLE PRECISION,
    gravity_uncertainty DOUBLE PRECISION,
    metadata JSONB,
    job_id UUID REFERENCES processing_jobs(id) ON DELETE CASCADE,
    PRIMARY KEY (time, satellite_id)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('satellite_observations', 'time', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Gravity field products table
CREATE TABLE IF NOT EXISTS gravity_products (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_type VARCHAR(50) NOT NULL, -- 'spherical_harmonic', 'grid', 'mascon'
    version VARCHAR(20) NOT NULL,
    time_start TIMESTAMPTZ NOT NULL,
    time_end TIMESTAMPTZ NOT NULL,
    degree_max INTEGER,
    spatial_resolution DOUBLE PRECISION,
    s3_path TEXT NOT NULL,
    metadata JSONB,
    job_id UUID REFERENCES processing_jobs(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Baseline vectors table
CREATE TABLE IF NOT EXISTS baseline_vectors (
    time TIMESTAMPTZ NOT NULL,
    satellite_1 VARCHAR(50) NOT NULL,
    satellite_2 VARCHAR(50) NOT NULL,
    range_rate DOUBLE PRECISION,
    range_acceleration DOUBLE PRECISION,
    baseline_length DOUBLE PRECISION,
    metadata JSONB,
    job_id UUID REFERENCES processing_jobs(id) ON DELETE CASCADE,
    PRIMARY KEY (time, satellite_1, satellite_2)
);

-- Convert to hypertable
SELECT create_hypertable('baseline_vectors', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Audit log table for provenance tracking
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT
);

-- Convert to hypertable for efficient time-based queries
SELECT create_hypertable('audit_logs', 'timestamp',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE);

-- Create indexes for better query performance
CREATE INDEX idx_processing_jobs_status ON processing_jobs(status);
CREATE INDEX idx_processing_jobs_user_id ON processing_jobs(user_id);
CREATE INDEX idx_processing_jobs_created_at ON processing_jobs(created_at DESC);

CREATE INDEX idx_satellite_obs_satellite_id ON satellite_observations(satellite_id, time DESC);
CREATE INDEX idx_satellite_obs_location ON satellite_observations USING GIST (
    point(longitude, latitude)
);

CREATE INDEX idx_gravity_products_type ON gravity_products(product_type);
CREATE INDEX idx_gravity_products_version ON gravity_products(version);
CREATE INDEX idx_gravity_products_time_range ON gravity_products(time_start, time_end);

CREATE INDEX idx_baseline_vectors_satellites ON baseline_vectors(satellite_1, satellite_2, time DESC);

CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);

-- Create default admin user (password: admin123 - change in production!)
INSERT INTO users (username, email, hashed_password, is_superuser)
VALUES ('admin', 'admin@gravity.ops', 
        '$2b$12$iWPwB3LAVguD5uFjRWPGxOxH7Lm3JHXI3VR2TXGOm.yLXIzBAOqQe', TRUE)
ON CONFLICT DO NOTHING;
