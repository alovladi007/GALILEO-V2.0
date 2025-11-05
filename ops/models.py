from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Text, JSON, Float, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
from datetime import datetime
import uuid
import os

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://gravity:gravity_secret@localhost:5432/gravity_ops")

# Create engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    jobs = relationship("ProcessingJob", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type = Column(String(50), nullable=False)
    status = Column(String(50), default="pending")
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    config = Column(JSONB)
    result = Column(JSONB)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="jobs")
    observations = relationship("SatelliteObservation", back_populates="job")
    products = relationship("GravityProduct", back_populates="job")
    baseline_vectors = relationship("BaselineVector", back_populates="job")

class SatelliteObservation(Base):
    __tablename__ = "satellite_observations"
    
    time = Column(DateTime(timezone=True), primary_key=True)
    satellite_id = Column(String(50), primary_key=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    altitude = Column(Float, nullable=False)
    gravity_value = Column(Float)
    gravity_uncertainty = Column(Float)
    metadata = Column(JSONB)
    job_id = Column(UUID(as_uuid=True), ForeignKey("processing_jobs.id", ondelete="CASCADE"))
    
    # Relationships
    job = relationship("ProcessingJob", back_populates="observations")

class GravityProduct(Base):
    __tablename__ = "gravity_products"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_type = Column(String(50), nullable=False)
    version = Column(String(20), nullable=False)
    time_start = Column(DateTime(timezone=True), nullable=False)
    time_end = Column(DateTime(timezone=True), nullable=False)
    degree_max = Column(Integer)
    spatial_resolution = Column(Float)
    s3_path = Column(Text, nullable=False)
    metadata = Column(JSONB)
    job_id = Column(UUID(as_uuid=True), ForeignKey("processing_jobs.id", ondelete="CASCADE"))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    job = relationship("ProcessingJob", back_populates="products")

class BaselineVector(Base):
    __tablename__ = "baseline_vectors"
    
    time = Column(DateTime(timezone=True), primary_key=True)
    satellite_1 = Column(String(50), primary_key=True)
    satellite_2 = Column(String(50), primary_key=True)
    range_rate = Column(Float)
    range_acceleration = Column(Float)
    baseline_length = Column(Float)
    metadata = Column(JSONB)
    job_id = Column(UUID(as_uuid=True), ForeignKey("processing_jobs.id", ondelete="CASCADE"))
    
    # Relationships
    job = relationship("ProcessingJob", back_populates="baseline_vectors")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(UUID(as_uuid=True))
    details = Column(JSONB)
    ip_address = Column(INET)
    user_agent = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
