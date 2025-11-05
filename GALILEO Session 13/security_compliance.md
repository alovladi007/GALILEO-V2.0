# Security & Compliance Documentation

## Overview

This document provides comprehensive documentation for the Security and Compliance Module implemented in Session 13. The module provides enterprise-grade security controls, compliance enforcement, and audit capabilities.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Authorization System](#authorization-system)
3. [Audit Logging](#audit-logging)
4. [Secrets Management](#secrets-management)
5. [Data Retention](#data-retention)
6. [Security Testing](#security-testing)
7. [Compliance Requirements](#compliance-requirements)
8. [API Reference](#api-reference)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### Components

The compliance module consists of four main components:

```
compliance/
├── __init__.py           # Module initialization
├── authorization.py      # RBAC and policy enforcement
├── audit.py             # Immutable audit logging
├── secrets.py           # Encrypted secrets management
└── retention.py         # Data lifecycle management
```

### Design Principles

1. **Security by Default**: All operations require explicit authorization
2. **Immutable Audit Trail**: Cryptographically chained audit logs
3. **Defense in Depth**: Multiple layers of security controls
4. **Compliance First**: Built-in support for GDPR, CCPA, HIPAA
5. **Zero Trust**: Verify every access attempt

---

## Authorization System

### Overview

Role-Based Access Control (RBAC) with fine-grained permissions and policy conditions.

### Key Features

- **Role-based permissions** with flexible policy assignment
- **Conditional access** based on context (ethical review, data classification, etc.)
- **Prohibited topics** enforcement for research restrictions
- **Policy versioning** with cryptographic hashing
- **Audit integration** for all authorization decisions

### Permission Types

```python
from compliance import Permission

# Data permissions
Permission.DATA_READ
Permission.DATA_WRITE
Permission.DATA_DELETE
Permission.DATA_EXPORT

# Research permissions
Permission.RESEARCH_CONDUCT
Permission.RESEARCH_REVIEW
Permission.RESEARCH_APPROVE

# Compliance permissions
Permission.COMPLIANCE_VIEW
Permission.COMPLIANCE_CONFIGURE
Permission.COMPLIANCE_AUDIT

# Admin permissions
Permission.ADMIN_USERS
Permission.ADMIN_SYSTEM
Permission.ADMIN_POLICIES

# Secrets permissions
Permission.SECRETS_READ
Permission.SECRETS_WRITE
Permission.SECRETS_ROTATE
```

### Usage Examples

#### Basic Setup

```python
from compliance import AuthorizationManager, Policy, Permission

# Initialize manager
auth_manager = AuthorizationManager()

# Create custom policy
policy = Policy(
    name="data_analyst",
    description="Data analyst access policy",
    permissions={
        Permission.DATA_READ,
        Permission.DATA_EXPORT,
    },
    conditions={
        "encryption_required": True,
    }
)

# Add policy
auth_manager.add_policy(policy)

# Assign to role
auth_manager.assign_role_policy("analyst", "data_analyst")

# Assign user to role
auth_manager.assign_user_role("user123", "analyst")
```

#### Permission Checking

```python
# Simple permission check
has_access = auth_manager.check_permission(
    user_id="user123",
    permission=Permission.DATA_READ,
    resource="document_001"
)

# Context-aware check
has_access = auth_manager.check_permission(
    user_id="researcher_456",
    permission=Permission.RESEARCH_CONDUCT,
    context={
        "topic": "user_behavior_analysis",
        "ethical_review_approved": True,
        "legal_review_approved": True,
    }
)
```

### Default Policies

#### Research Restricted Policy
- Permissions: RESEARCH_CONDUCT, DATA_READ
- Requires: Ethical review, Legal review
- Blocks: Weapons, surveillance, manipulation, privacy invasion

#### Data Protection Policy
- Permissions: DATA_READ, DATA_WRITE
- Requires: Encryption, PII handling, audit logging

#### Compliance Officer Policy
- Full compliance management access
- Read-only data access
- Audit trail access

---

## Audit Logging

### Overview

Immutable, cryptographically-chained audit log system providing tamper detection and comprehensive event tracking.

### Key Features

- **Cryptographic chaining** prevents log tampering
- **Comprehensive event types** covering all security-relevant operations
- **Severity levels** for event classification
- **Query capabilities** with multiple filter options
- **Compliance reports** for regulatory requirements
- **Export functionality** for external systems

### Event Types

- **Authentication**: login, logout, failed attempts
- **Authorization**: access granted/denied, permission changes
- **Data Operations**: access, modification, deletion, export
- **Research**: started, completed, blocked
- **Compliance**: policy changes, retention, legal holds
- **Secrets**: access, creation, rotation, deletion
- **System**: configuration changes, errors, security alerts

### Usage Examples

#### Basic Logging

```python
from compliance import AuditLogger, AuditEventType, AuditSeverity

# Initialize logger
audit_logger = AuditLogger()

# Log an event
event = audit_logger.log_event(
    event_type=AuditEventType.DATA_ACCESSED,
    action="read_document",
    result="success",
    user_id="user123",
    resource="document_001",
    severity=AuditSeverity.INFO,
    metadata={"classification": "confidential"},
    ip_address="192.168.1.100",
)
```

#### Specialized Logging

```python
# Log access attempt
audit_logger.log_access(
    user_id="user123",
    resource="sensitive_data",
    action="read",
    granted=True,
    metadata={"method": "API"}
)

# Log data operation
audit_logger.log_data_access(
    user_id="user123",
    resource="database_001",
    operation="export",
    metadata={"record_count": 1000}
)

# Log security alert
audit_logger.log_security_alert(
    alert_type="unauthorized_access",
    description="Multiple failed login attempts detected",
    user_id="attacker_ip",
    metadata={"attempt_count": 5}
)
```

#### Querying and Reporting

```python
# Query events
events = audit_logger.query_events(
    event_type=AuditEventType.DATA_ACCESSED,
    user_id="user123",
    start_time=datetime(2025, 1, 1),
    end_time=datetime(2025, 12, 31)
)

# Generate audit report
report = audit_logger.generate_audit_report(
    start_time=datetime(2025, 1, 1),
    end_time=datetime(2025, 3, 31)
)

# Get compliance summary
summary = audit_logger.get_compliance_summary(days=30)

# Verify chain integrity
is_valid = audit_logger.verify_chain()
```

---

## Secrets Management

### Overview

Encrypted storage and management of sensitive credentials with automatic rotation and access tracking.

### Key Features

- **Encryption at rest** using Fernet (symmetric encryption)
- **Automatic rotation** based on configurable policies
- **Access tracking** with usage statistics
- **Expiration support** for time-limited secrets
- **Version history** for rotation tracking
- **Audit integration** for all operations

### Secret Types

```python
from compliance import SecretType

SecretType.API_KEY
SecretType.PASSWORD
SecretType.TOKEN
SecretType.CERTIFICATE
SecretType.ENCRYPTION_KEY
SecretType.DATABASE_CREDENTIAL
SecretType.SSH_KEY
```

### Usage Examples

#### Creating Secrets

```python
from compliance import SecretsManager, SecretType
from datetime import datetime, timedelta

# Initialize manager
secrets_manager = SecretsManager()

# Create a secret
secret = secrets_manager.create_secret(
    name="api_key_production",
    value="sk_live_abc123xyz789",
    secret_type=SecretType.API_KEY,
    rotation_policy_days=90,
    expires_at=datetime.utcnow() + timedelta(days=365),
    metadata={"environment": "production"},
    user_id="admin123"
)
```

#### Retrieving Secrets

```python
# Get and decrypt secret
value = secrets_manager.get_secret(
    secret_id=secret.secret_id,
    user_id="app_server"
)

# Get metadata only (no decryption)
metadata = secrets_manager.get_secret_metadata(secret.secret_id)
```

#### Rotating Secrets

```python
# Manual rotation
success = secrets_manager.rotate_secret(
    secret_id=secret.secret_id,
    new_value="sk_live_new_key_456",
    user_id="admin123"
)

# Check which secrets need rotation
needs_rotation = secrets_manager.check_rotation_needed()

# Auto-rotate with callback
def generate_new_api_key(secret):
    return f"sk_live_{secrets.token_urlsafe(32)}"

rotated_count = secrets_manager.auto_rotate_expired(
    rotation_callback=generate_new_api_key
)
```

---

## Data Retention

### Overview

Automated data lifecycle management with retention policies and legal hold support for compliance.

### Key Features

- **Retention policies** by data classification
- **Legal holds** preventing data deletion
- **Scheduled deletion** and archival
- **Compliance reporting** for audits
- **GDPR/CCPA support** for data subject rights
- **Audit integration** for all operations

### Data Classifications

```python
from compliance import DataClassification

DataClassification.PUBLIC
DataClassification.INTERNAL
DataClassification.CONFIDENTIAL
DataClassification.RESTRICTED
DataClassification.REGULATED
```

### Usage Examples

#### Registering Data

```python
from compliance import DataRetentionManager, DataClassification

# Initialize manager
retention_manager = DataRetentionManager()

# Register data for tracking
record = retention_manager.register_data(
    resource_type="user_data",
    resource_id="user_12345",
    classification=DataClassification.CONFIDENTIAL,
    metadata={"source": "web_app"}
)
```

#### Legal Holds

```python
# Create legal hold
hold = retention_manager.create_legal_hold(
    name="Litigation Case 2025-001",
    description="Hold all documents related to contract dispute",
    case_number="CASE-2025-001",
    custodian="legal_department",
    resources={"doc_001", "doc_002", "doc_003"},
    expires_at=datetime(2026, 12, 31),
    user_id="legal_admin"
)

# Check if data is protected
is_protected = record.is_protected()

# Release legal hold
retention_manager.release_legal_hold(
    hold_id=hold.hold_id,
    user_id="legal_admin"
)
```

#### Retention Policy Management

```python
from compliance import RetentionPolicy, RetentionAction

# Create custom policy
policy = RetentionPolicy(
    policy_id="financial_records_7y",
    name="Financial Records 7 Year Retention",
    description="Retain financial records for 7 years per regulation",
    retention_days=2555,  # ~7 years
    action=RetentionAction.ARCHIVE,
    data_classification=DataClassification.REGULATED,
    applicable_types=["invoice", "receipt", "statement"]
)

retention_manager.add_policy(policy)
```

#### Executing Retention Actions

```python
# Get records ready for action
to_delete = retention_manager.get_records_for_deletion()
to_archive = retention_manager.get_records_for_archival()

# Execute retention actions
results = retention_manager.execute_retention_actions()
# Returns: {'deleted': 10, 'archived': 25, 'skipped_protected': 3}

# Generate compliance report
report = retention_manager.get_compliance_report()
```

---

## Security Testing

### Test Coverage

The module includes comprehensive unit and integration tests:

- **Authorization Tests** (9 tests): Policy creation, role assignment, permission checking
- **Audit Logging Tests** (8 tests): Event logging, chain verification, querying
- **Secrets Management Tests** (7 tests): Encryption, rotation, expiration
- **Data Retention Tests** (8 tests): Policies, legal holds, compliance
- **Integration Tests** (3 tests): Cross-module workflows

### Running Tests

```bash
# Run all tests
cd /home/claude/project
python tests/security/test_compliance.py

# Run specific test class
python -m unittest tests.security.test_compliance.TestAuthorizationPolicies

# Run with verbose output
python tests/security/test_compliance.py -v
```

### Test Results

```
Ran 35 tests in 0.007s
OK
```

All tests pass successfully, demonstrating:
- Proper authorization enforcement
- Audit trail integrity
- Secret encryption/decryption
- Retention policy application
- Legal hold protection
- Integration between modules

---

## Compliance Requirements

### GDPR Compliance

The module supports GDPR requirements:

1. **Lawful Basis**: Context-based authorization
2. **Data Subject Rights**: Retention and deletion controls
3. **Privacy by Design**: Encryption and access controls
4. **Audit Requirements**: Comprehensive logging
5. **Data Retention**: Automated lifecycle management
6. **Legal Basis Documentation**: Policy metadata

### CCPA Compliance

Support for California Consumer Privacy Act:

1. **Consumer Rights**: Data access and deletion
2. **Privacy Notice**: Audit trail for transparency
3. **Opt-Out Mechanisms**: Configurable policies
4. **Security Requirements**: Encryption and access controls
5. **Breach Notification**: Security alert logging

### HIPAA Compliance

Healthcare data protection features:

1. **Access Controls**: Role-based permissions
2. **Audit Trails**: Comprehensive logging required by HIPAA
3. **Encryption**: At-rest and in-transit protection
4. **Minimum Necessary**: Context-aware access
5. **Retention Requirements**: Configurable policies

---

## API Reference

### Authorization Manager

```python
class AuthorizationManager:
    def add_policy(policy: Policy) -> bool
    def remove_policy(policy_name: str) -> bool
    def assign_role_policy(role: str, policy_name: str) -> bool
    def assign_user_role(user_id: str, role: str) -> bool
    def check_permission(user_id: str, permission: Permission, 
                        resource: str = None, context: dict = None) -> bool
    def get_user_permissions(user_id: str) -> Set[Permission]
    def export_policies() -> dict
```

### Audit Logger

```python
class AuditLogger:
    def log_event(event_type, action, result, user_id=None, 
                 resource=None, severity=INFO, metadata=None) -> AuditEvent
    def log_access(user_id, resource, action, granted, metadata=None) -> AuditEvent
    def log_data_access(user_id, resource, operation, metadata=None) -> AuditEvent
    def log_security_alert(alert_type, description, user_id=None, 
                          metadata=None) -> AuditEvent
    def verify_chain() -> bool
    def query_events(**filters) -> List[AuditEvent]
    def generate_audit_report(start_time=None, end_time=None) -> dict
    def get_compliance_summary(days=30) -> dict
```

### Secrets Manager

```python
class SecretsManager:
    def create_secret(name, value, secret_type, rotation_policy_days=90,
                     expires_at=None, metadata=None, user_id=None) -> Secret
    def get_secret(secret_id, user_id=None) -> str
    def rotate_secret(secret_id, new_value, user_id=None) -> bool
    def delete_secret(secret_id, user_id=None) -> bool
    def check_rotation_needed() -> List[Secret]
    def auto_rotate_expired(rotation_callback=None) -> int
    def list_secrets(secret_type=None, status=None) -> List[dict]
    def export_secrets_manifest() -> dict
```

### Data Retention Manager

```python
class DataRetentionManager:
    def add_policy(policy: RetentionPolicy) -> bool
    def create_legal_hold(name, description, case_number, custodian,
                         resources=None, expires_at=None, user_id=None) -> LegalHold
    def release_legal_hold(hold_id, user_id=None) -> bool
    def register_data(resource_type, resource_id, classification,
                     created_at=None, metadata=None) -> DataRecord
    def get_records_for_deletion() -> List[DataRecord]
    def get_records_for_archival() -> List[DataRecord]
    def execute_retention_actions() -> dict
    def get_compliance_report() -> dict
```

---

## Best Practices

### Authorization

1. **Principle of Least Privilege**: Grant minimum permissions required
2. **Regular Policy Reviews**: Audit and update policies quarterly
3. **Context-Aware Access**: Use conditions for sensitive operations
4. **Emergency Access**: Have break-glass procedures documented

### Audit Logging

1. **Log Everything**: All security-relevant events should be logged
2. **Verify Chain Regularly**: Check audit integrity periodically
3. **Retention Policy**: Keep logs for required compliance periods
4. **Monitor Alerts**: Set up monitoring for security alerts
5. **Regular Reviews**: Conduct periodic log analysis

### Secrets Management

1. **Rotate Regularly**: Enforce rotation policies
2. **Minimize Access**: Limit who can access secrets
3. **Use Short Expiration**: Set reasonable expiration dates
4. **Audit Access**: Review secret access patterns
5. **Secure Generation**: Use cryptographically secure random generation

### Data Retention

1. **Document Policies**: Clearly document all retention policies
2. **Legal Hold Process**: Have clear procedures for legal holds
3. **Regular Cleanup**: Execute retention actions regularly
4. **Classification**: Properly classify all data
5. **Compliance Reports**: Generate regular compliance reports

---

## Troubleshooting

### Common Issues

#### Permission Denied Errors

**Problem**: User cannot access resource despite having role

**Solution**:
```python
# Check user's roles
roles = auth_manager.user_roles.get("user_id")

# Check role's policies
for role in roles:
    policies = auth_manager.role_policies.get(role)
    
# Verify policy conditions
context = {"ethical_review_approved": True}
has_perm = auth_manager.check_permission(user_id, permission, context=context)
```

#### Audit Chain Verification Failure

**Problem**: `verify_chain()` returns False

**Solution**:
- This indicates potential tampering
- Investigate immediately
- Check for concurrent modifications
- Review recent audit events
- Restore from backup if necessary

#### Secret Retrieval Returns None

**Problem**: `get_secret()` returns None

**Possible Causes**:
1. Secret expired: Check `expires_at`
2. Secret deleted: Check secret history
3. Wrong secret_id: Verify ID is correct

#### Legal Hold Not Preventing Deletion

**Problem**: Data deleted despite active legal hold

**Solution**:
```python
# Verify hold is active
hold = retention_manager.legal_holds.get(hold_id)
is_active = hold.is_active()

# Check if resource in hold
if resource_id not in hold.resources:
    hold.resources.add(resource_id)

# Verify record has hold
record = retention_manager.data_records.get(record_id)
if hold_id not in record.legal_holds:
    retention_manager._apply_legal_hold(resource_id, hold_id)
```

---

## Security Considerations

### Threat Model

The module protects against:

1. **Unauthorized Access**: RBAC prevents unauthorized operations
2. **Privilege Escalation**: Conditional policies prevent escalation
3. **Log Tampering**: Cryptographic chaining detects modifications
4. **Secret Exposure**: Encryption protects secrets at rest
5. **Data Loss**: Retention policies prevent premature deletion
6. **Compliance Violations**: Automated enforcement of policies

### Security Assumptions

1. **Master Key Protection**: Secrets encryption key must be secured
2. **Audit Storage**: Audit logs must be stored securely
3. **Access Control**: Platform-level access controls in place
4. **Network Security**: TLS/HTTPS for data in transit
5. **Physical Security**: Servers physically secured

### Security Recommendations

1. Use hardware security modules (HSM) for master key storage
2. Implement backup and disaster recovery for audit logs
3. Regular security audits and penetration testing
4. Incident response procedures documented
5. Security awareness training for all users

---

## Performance Considerations

### Scalability

- **Authorization checks**: O(n) where n = number of user roles
- **Audit logging**: O(1) for writes, O(n) for queries
- **Secret operations**: O(1) for encryption/decryption
- **Retention checks**: O(n) where n = number of tracked records

### Optimization Tips

1. **Cache authorization decisions** for frequently accessed resources
2. **Batch audit log exports** for large queries
3. **Index audit events** by user_id and event_type
4. **Asynchronous secret rotation** for bulk operations
5. **Scheduled retention execution** during off-peak hours

---

## Changelog

### Version 1.0.0 (2025-11-04)

**Initial Release - Session 13**

Features:
- Role-based access control with conditional policies
- Cryptographically-chained audit logging
- Encrypted secrets management with rotation
- Data retention and legal hold management
- Comprehensive security testing
- GDPR/CCPA/HIPAA compliance support
- ETHICS.md and LEGAL.md documentation

---

## Support and Contact

For security issues or questions:
- **Security Team**: security@example.com
- **Compliance Officer**: compliance@example.com
- **Documentation**: https://docs.example.com/security

**Report Security Vulnerabilities**: security-reports@example.com (PGP key available)

---

*Last Updated: 2025-11-04*
*Document Version: 1.0.0*
*Classification: Internal Use Only*
