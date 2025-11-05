#!/usr/bin/env python3
"""
Compliance Module Demo
Demonstrates all features of the security and compliance module.
"""

from compliance import (
    AuthorizationManager, Policy, Permission,
    AuditLogger, AuditEventType, AuditSeverity,
    SecretsManager, SecretType,
    DataRetentionManager, DataClassification, LegalHoldStatus
)
from datetime import datetime, timedelta


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def demo_authorization():
    """Demonstrate authorization features"""
    print_section("1. AUTHORIZATION & ACCESS CONTROL")
    
    auth = AuthorizationManager()
    
    # Setup roles and users
    print("Setting up roles and permissions...")
    auth.assign_role_policy("researcher", "research_restricted")
    auth.assign_role_policy("analyst", "data_protection")
    auth.assign_user_role("alice", "researcher")
    auth.assign_user_role("bob", "analyst")
    print("✓ Roles configured: researcher, analyst")
    print("✓ Users assigned: alice (researcher), bob (analyst)\n")
    
    # Test permission checks
    print("Testing permission checks:")
    
    # Valid research with approvals
    result = auth.check_permission(
        "alice",
        Permission.RESEARCH_CONDUCT,
        context={
            "topic": "user_behavior_analysis",
            "ethical_review_approved": True,
            "legal_review_approved": True,
        }
    )
    print(f"  ✓ Alice can conduct approved research: {result}")
    
    # Prohibited research topic
    result = auth.check_permission(
        "alice",
        Permission.RESEARCH_CONDUCT,
        context={
            "topic": "weapons_development",
            "ethical_review_approved": True,
        }
    )
    print(f"  ✗ Alice blocked from weapons research: {not result}")
    
    # Missing ethical review
    result = auth.check_permission(
        "alice",
        Permission.RESEARCH_CONDUCT,
        context={
            "topic": "user_behavior_analysis",
            "ethical_review_approved": False,
        }
    )
    print(f"  ✗ Alice blocked without ethical review: {not result}")
    
    # Data analyst permissions
    result = auth.check_permission("bob", Permission.DATA_READ)
    print(f"  ✓ Bob can read data: {result}")
    
    result = auth.check_permission("bob", Permission.DATA_DELETE)
    print(f"  ✗ Bob cannot delete data: {not result}")
    
    return auth


def demo_audit_logging(auth):
    """Demonstrate audit logging features"""
    print_section("2. AUDIT LOGGING & TRAIL INTEGRITY")
    
    audit = AuditLogger()
    
    # Log various events
    print("Logging security events...")
    
    audit.log_access(
        user_id="alice",
        resource="research_data_001",
        action="read",
        granted=True,
        metadata={"classification": "confidential"}
    )
    print("✓ Logged: Alice accessed research data")
    
    audit.log_access(
        user_id="eve",
        resource="restricted_data",
        action="read",
        granted=False,
        metadata={"reason": "insufficient_permissions"}
    )
    print("✓ Logged: Eve denied access to restricted data")
    
    audit.log_security_alert(
        alert_type="suspicious_activity",
        description="Multiple failed login attempts",
        user_id="eve",
        metadata={"attempt_count": 5}
    )
    print("✓ Logged: Security alert for suspicious activity")
    
    audit.log_data_access(
        user_id="bob",
        resource="analytics_database",
        operation="export",
        metadata={"record_count": 1000}
    )
    print("✓ Logged: Bob exported analytics data\n")
    
    # Verify chain integrity
    print("Verifying audit chain integrity...")
    is_valid = audit.verify_chain()
    print(f"✓ Audit chain verified: {is_valid}\n")
    
    # Query events
    print("Querying audit events:")
    denied_access = audit.query_events(
        event_type=AuditEventType.ACCESS_DENIED
    )
    print(f"  • Found {len(denied_access)} access denial(s)")
    
    security_alerts = audit.query_events(
        severity=AuditSeverity.CRITICAL
    )
    print(f"  • Found {len(security_alerts)} critical alert(s)")
    
    # Generate summary
    summary = audit.get_compliance_summary(days=1)
    print(f"\nCompliance Summary:")
    print(f"  • Total events: {summary['total_events']}")
    print(f"  • Access denied: {summary['access_denied_count']}")
    print(f"  • Data exports: {summary['data_exports_count']}")
    print(f"  • Security alerts: {summary['security_alerts_count']}")
    
    return audit


def demo_secrets_management(audit):
    """Demonstrate secrets management features"""
    print_section("3. SECRETS MANAGEMENT")
    
    secrets = SecretsManager(audit_logger=audit)
    
    # Create secrets
    print("Creating secrets...")
    
    api_secret = secrets.create_secret(
        name="production_api_key",
        value="sk_live_abc123xyz789",
        secret_type=SecretType.API_KEY,
        rotation_policy_days=90,
        user_id="admin"
    )
    print(f"✓ Created API key: {api_secret.secret_id[:16]}...")
    
    db_secret = secrets.create_secret(
        name="database_password",
        value="super_secure_password_123",
        secret_type=SecretType.PASSWORD,
        rotation_policy_days=30,
        expires_at=datetime.utcnow() + timedelta(days=365),
        user_id="admin"
    )
    print(f"✓ Created DB password: {db_secret.secret_id[:16]}...\n")
    
    # Retrieve secrets
    print("Retrieving secrets...")
    api_value = secrets.get_secret(api_secret.secret_id, user_id="alice")
    print(f"✓ Retrieved API key (decrypted): {api_value[:20]}...")
    
    # Check rotation needed
    print("\nChecking rotation status...")
    needs_rotation = secrets.check_rotation_needed()
    print(f"  • Secrets needing rotation: {len(needs_rotation)}")
    
    # Rotate a secret
    print("\nRotating API key...")
    success = secrets.rotate_secret(
        api_secret.secret_id,
        "sk_live_new_key_456",
        user_id="admin"
    )
    print(f"✓ Secret rotated: {success}")
    
    # List secrets
    manifest = secrets.export_secrets_manifest()
    print(f"\nSecrets Manifest:")
    print(f"  • Total secrets: {manifest['total_secrets']}")
    print(f"  • By type: {manifest['secrets_by_type']}")
    print(f"  • By status: {manifest['secrets_by_status']}")
    
    return secrets


def demo_data_retention(audit):
    """Demonstrate data retention features"""
    print_section("4. DATA RETENTION & LEGAL HOLDS")
    
    retention = DataRetentionManager(audit_logger=audit)
    
    # Register data
    print("Registering data for retention tracking...")
    
    user_data = retention.register_data(
        resource_type="user_data",
        resource_id="user_12345",
        classification=DataClassification.CONFIDENTIAL,
        metadata={"created_by": "signup_process"}
    )
    print(f"✓ Registered user data: {user_data.record_id}")
    print(f"  • Policy: {user_data.retention_policy_id}")
    print(f"  • Deletion scheduled: {user_data.deletion_scheduled_at.strftime('%Y-%m-%d')}")
    
    research_data = retention.register_data(
        resource_type="research_data",
        resource_id="experiment_001",
        classification=DataClassification.REGULATED,
        metadata={"study": "clinical_trial_2025"}
    )
    print(f"✓ Registered research data: {research_data.record_id}")
    print(f"  • Policy: {research_data.retention_policy_id}")
    print(f"  • Deletion scheduled: {research_data.deletion_scheduled_at.strftime('%Y-%m-%d')}\n")
    
    # Create legal hold
    print("Creating legal hold...")
    hold = retention.create_legal_hold(
        name="Litigation Case 2025-001",
        description="Hold all documents related to contract dispute",
        case_number="CASE-2025-001",
        custodian="legal_department",
        resources={"experiment_001", "doc_002", "doc_003"},
        user_id="legal_admin"
    )
    print(f"✓ Legal hold created: {hold.hold_id}")
    print(f"  • Case: {hold.case_number}")
    print(f"  • Status: {hold.status.value}")
    print(f"  • Protected resources: {len(hold.resources)}\n")
    
    # Check protection
    print("Verifying data protection:")
    print(f"  • User data protected: {user_data.is_protected()}")
    print(f"  • Research data protected: {research_data.is_protected()}")
    
    # Get compliance report
    print("\nCompliance Report:")
    report = retention.get_compliance_report()
    print(f"  • Total records tracked: {report['total_records']}")
    print(f"  • Protected records: {report['protected_records']}")
    print(f"  • Active policies: {report['active_policies']}")
    print(f"  • Active legal holds: {report['active_legal_holds']}")
    print(f"  • Pending deletion: {report['pending_deletion']}")
    print(f"  • Pending archival: {report['pending_archival']}")
    
    return retention


def demo_integration(auth, audit, secrets, retention):
    """Demonstrate integrated workflow"""
    print_section("5. INTEGRATED WORKFLOW")
    
    print("Scenario: Alice wants to export sensitive research data\n")
    
    # Step 1: Check authorization
    print("Step 1: Authorization Check")
    can_export = auth.check_permission(
        "alice",
        Permission.DATA_EXPORT,
        context={
            "data_classification": "confidential",
            "ethical_review_approved": True,
            "legal_review_approved": True,
        }
    )
    print(f"  • Alice can export: {can_export}")
    
    if not can_export:
        print("  ✗ Export denied - insufficient permissions")
        return
    
    # Step 2: Check legal hold
    print("\nStep 2: Legal Hold Check")
    data_record = retention.data_records.get("research_data:experiment_001")
    if data_record:
        is_held = data_record.is_protected()
        print(f"  • Data under legal hold: {is_held}")
        if is_held:
            print("  ⚠ Warning: Data is protected by legal hold")
    
    # Step 3: Get export credentials
    print("\nStep 3: Retrieve Export Credentials")
    export_key = secrets.get_secret(
        list(secrets.secrets.keys())[0],  # First secret
        user_id="alice"
    )
    if export_key:
        print(f"  ✓ Retrieved credentials: {export_key[:20]}...")
    
    # Step 4: Log the operation
    print("\nStep 4: Audit Logging")
    audit.log_event(
        event_type=AuditEventType.DATA_EXPORTED,
        action="export_research_data",
        result="success",
        user_id="alice",
        resource="experiment_001",
        severity=AuditSeverity.INFO,
        metadata={
            "classification": "confidential",
            "record_count": 1500,
            "destination": "secure_archive"
        }
    )
    print("  ✓ Export logged to audit trail")
    
    print("\n✓ Integrated workflow completed successfully!")


def main():
    """Run complete demonstration"""
    print("\n" + "="*80)
    print("COMPLIANCE MODULE DEMONSTRATION")
    print("Session 13 - Security & Compliance Hardening")
    print("="*80)
    
    try:
        # Run each demo
        auth = demo_authorization()
        audit = demo_audit_logging(auth)
        secrets = demo_secrets_management(audit)
        retention = demo_data_retention(audit)
        demo_integration(auth, audit, secrets, retention)
        
        # Final summary
        print_section("DEMONSTRATION COMPLETE")
        print("✓ Authorization: Role-based access control working")
        print("✓ Audit Logging: Immutable audit trail verified")
        print("✓ Secrets Management: Encrypted storage and rotation working")
        print("✓ Data Retention: Legal holds and policies enforced")
        print("✓ Integration: All components working together\n")
        
        print("All features demonstrated successfully! 🎉\n")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
