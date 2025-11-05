#!/usr/bin/env python3
"""
Initialize GALILEO V2.0 Database Schema
Creates all tables defined in ops/models.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops.models import Base, engine, SessionLocal
from sqlalchemy import text

def init_database():
    """Initialize database schema and create tables."""

    print("="*70)
    print("GALILEO V2.0 - Database Initialization")
    print("="*70)

    # Test connection
    print("\n1. Testing database connection...")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"   ✓ Connected to PostgreSQL")
            print(f"   Version: {version.split(',')[0]}")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return False

    # Create all tables
    print("\n2. Creating database schema...")
    try:
        Base.metadata.create_all(engine)
        print("   ✓ Schema created successfully")
    except Exception as e:
        print(f"   ✗ Schema creation failed: {e}")
        return False

    # Verify tables
    print("\n3. Verifying tables...")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result.fetchall()]

            expected_tables = [
                'users',
                'processing_jobs',
                'satellite_observations',
                'gravity_products',
                'baseline_vectors',
                'audit_logs'
            ]

            for table in expected_tables:
                if table in tables:
                    print(f"   ✓ {table}")
                else:
                    print(f"   ✗ {table} (missing)")

            if all(t in tables for t in expected_tables):
                print(f"\n   All {len(expected_tables)} tables created successfully")
            else:
                print(f"\n   Warning: Some tables are missing")

    except Exception as e:
        print(f"   ✗ Verification failed: {e}")
        return False

    # Create initial admin user (optional)
    print("\n4. Creating initial admin user...")
    try:
        from ops.models import User
        import uuid
        from datetime import datetime

        db = SessionLocal()

        # Check if admin exists
        admin = db.query(User).filter(User.username == 'admin').first()

        if not admin:
            admin = User(
                id=uuid.uuid4(),
                username='admin',
                email='admin@galileo.space',
                hashed_password='$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5iJxYZN3YQqOe',  # 'admin'
                is_active=True,
                is_superuser=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(admin)
            db.commit()
            print(f"   ✓ Admin user created (username: admin, password: admin)")
            print(f"   ⚠️  CHANGE DEFAULT PASSWORD IN PRODUCTION!")
        else:
            print(f"   ℹ️  Admin user already exists")

        db.close()

    except Exception as e:
        print(f"   ⚠️  Could not create admin user: {e}")
        print(f"   (This is optional and can be done later)")

    print("\n" + "="*70)
    print("✓ Database initialization complete")
    print("="*70)

    return True

if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)
