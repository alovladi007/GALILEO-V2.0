# Backend Operations Documentation

## Overview

The Gravity Processing Backend provides a comprehensive REST API for satellite gravity data processing, featuring asynchronous job execution, time-series data management, and object storage integration.

## Architecture

### Components

1. **FastAPI Application** - REST API server with automatic OpenAPI documentation
2. **PostgreSQL + TimescaleDB** - Time-series optimized database
3. **Celery Workers** - Asynchronous task processing
4. **Redis** - Message broker and result backend
5. **MinIO** - S3-compatible object storage
6. **Flower** - Celery monitoring dashboard

### Technology Stack

- **Language**: Python 3.11
- **Framework**: FastAPI
- **ORM**: SQLAlchemy
- **Database**: PostgreSQL 15 with TimescaleDB
- **Queue**: Celery with Redis
- **Storage**: MinIO (S3-compatible)
- **Authentication**: JWT (OAuth2)

## API Endpoints

### Authentication

#### Register User
```http
POST /auth/register
Content-Type: application/json

{
  "username": "user123",
  "email": "user@example.com",
  "password": "securepassword"
}
```

#### Login
```http
POST /auth/token
Content-Type: application/x-www-form-urlencoded

username=user123&password=securepassword
```

#### Get Current User
```http
GET /auth/me
Authorization: Bearer <token>
```

### Processing Operations

#### Create Processing Plan
```http
POST /ops/plan
Authorization: Bearer <token>
Content-Type: application/json

{
  "start_time": "2025-01-01T00:00:00Z",
  "end_time": "2025-01-31T23:59:59Z",
  "satellites": ["GRACE-A", "GRACE-B"],
  "time_step": 60,
  "processing_mode": "standard",
  "output_format": "spherical_harmonic",
  "degree_max": 60
}
```

#### Ingest Data
```http
POST /ops/ingest
Authorization: Bearer <token>
Content-Type: application/json

{
  "data_source": "s3://gravity-data/raw/2025-01/",
  "data_type": "l1b",
  "satellites": ["GRACE-A", "GRACE-B"],
  "start_time": "2025-01-01T00:00:00Z",
  "end_time": "2025-01-31T23:59:59Z",
  "validate_data": true
}
```

#### Process Gravity Field
```http
POST /ops/process
Authorization: Bearer <token>
Content-Type: application/json

{
  "algorithm": "variational",
  "degree_max": 60,
  "regularization": {
    "kaula": 1e-5,
    "tikhonov": 0.0
  },
  "output_products": ["coefficients", "grids", "uncertainties"]
}
```

#### Catalog Products
```http
POST /ops/catalog
Authorization: Bearer <token>
Content-Type: application/json

{
  "product_ids": ["550e8400-e29b-41d4-a716-446655440000"],
  "metadata": {
    "mission": "GRACE-FO",
    "processing_center": "CSR",
    "version": "RL06.1"
  },
  "publish": false,
  "archive": true
}
```

### Job Management

#### List Jobs
```http
GET /ops/jobs?status=running&skip=0&limit=100
Authorization: Bearer <token>
```

#### Get Job Details
```http
GET /ops/jobs/{job_id}
Authorization: Bearer <token>
```

#### Cancel Job
```http
DELETE /ops/jobs/{job_id}
Authorization: Bearer <token>
```

## Database Schema

### Core Tables

1. **users** - User authentication and profiles
2. **processing_jobs** - Job tracking and status
3. **satellite_observations** - Time-series satellite data (hypertable)
4. **gravity_products** - Processed gravity field products
5. **baseline_vectors** - Inter-satellite measurements (hypertable)
6. **audit_logs** - Comprehensive audit trail (hypertable)

### TimescaleDB Optimizations

- Automatic time-based partitioning for observations
- Compression policies for historical data
- Continuous aggregates for performance
- Retention policies for data lifecycle

## Asynchronous Processing

### Celery Tasks

1. **execute_plan_task** - Generate processing plans
2. **execute_ingest_task** - Ingest and validate raw data
3. **execute_process_task** - Compute gravity fields
4. **execute_catalog_task** - Catalog and archive products

### Task Monitoring

Access Flower dashboard at `http://localhost:5555` for:
- Real-time task monitoring
- Worker status
- Task history and results
- Performance metrics

## Object Storage

### MinIO Buckets

- **raw-data/** - Original satellite observations
- **processed-data/** - Intermediate processing results
- **products/** - Final gravity field products
- **temp/** - Temporary processing files

### Storage Patterns

```python
# Upload example
from minio_client import get_minio_client, upload_to_minio

client = get_minio_client()
upload_to_minio(
    client,
    bucket="products",
    key="gravity_field_202501.nc",
    data=field_data
)

# Generate presigned URL
url = client.presigned_get_object(
    "products",
    "gravity_field_202501.nc",
    expires=3600
)
```

## Security & Audit

### Authentication Flow

1. User registers with username/email/password
2. Password hashed with bcrypt
3. Login returns JWT token (30 min expiry)
4. Token included in Authorization header
5. Token validated on each request

### Audit Logging

Every API request is logged with:
- Timestamp
- User ID
- Action performed
- Resource accessed
- IP address
- Response status
- Request duration

### Data Provenance

Complete tracking of:
- Data lineage from raw to products
- Processing parameters
- Algorithm versions
- Quality metrics
- User actions

## Performance Optimization

### Database
- Connection pooling
- Prepared statements
- Index optimization
- Query result caching

### API
- Request/response compression
- Pagination for large datasets
- Async request handling
- Rate limiting

### Storage
- Multipart uploads for large files
- CDN integration for static assets
- Lifecycle policies for data retention

## Testing

### Unit Tests
```bash
pytest tests/unit -v
```

### Integration Tests
```bash
pytest test_integration.py -v
```

### Load Testing
```bash
locust -f tests/load_test.py --host=http://localhost:8000
```

### Postman Collection
Import `postman_collection.json` for interactive API testing

## Deployment

### Docker Compose
```bash
docker-compose up -d
```

### Environment Variables
```env
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://redis:6379
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
JWT_SECRET_KEY=your-secret-key
CORS_ORIGINS=["http://localhost:3000"]
```

### Health Checks
```bash
curl http://localhost:8000/health
```

### Scaling
- Horizontal scaling with multiple API instances
- Celery worker autoscaling
- Database read replicas
- MinIO distributed mode

## Monitoring

### Metrics
- Prometheus metrics at `/metrics`
- Application logs in JSON format
- Database slow query log
- Storage usage alerts

### Dashboards
- Grafana for metrics visualization
- Flower for Celery monitoring
- pgAdmin for database management
- MinIO Console for storage

## API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI Schema: `http://localhost:8000/openapi.json`

## Error Handling

### HTTP Status Codes
- 200 - Success
- 201 - Created
- 400 - Bad Request
- 401 - Unauthorized
- 403 - Forbidden
- 404 - Not Found
- 422 - Validation Error
- 500 - Internal Server Error

### Error Response Format
```json
{
  "detail": "Error message",
  "status_code": 400,
  "errors": [
    {
      "field": "satellites",
      "message": "At least one satellite required"
    }
  ]
}
```

## Best Practices

1. **Use async/await** for I/O operations
2. **Implement retry logic** for external services
3. **Validate all inputs** with Pydantic
4. **Log everything** for debugging
5. **Monitor performance** continuously
6. **Document API changes** in changelog
7. **Version your API** for compatibility
8. **Secure sensitive data** with encryption
