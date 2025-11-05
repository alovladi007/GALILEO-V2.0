"""
Compliance Module - Security and Compliance Hardening
Provides authorization, audit logging, secrets management, and data retention controls.
"""

from .authorization import AuthorizationManager, Policy, Permission
from .audit import AuditLogger, AuditEvent, AuditEventType, AuditSeverity
from .secrets import SecretsManager, SecretType, SecretStatus
from .retention import DataRetentionManager, LegalHold, LegalHoldStatus, DataClassification

__all__ = [
    'AuthorizationManager',
    'Policy',
    'Permission',
    'AuditLogger',
    'AuditEvent',
    'AuditEventType',
    'AuditSeverity',
    'SecretsManager',
    'SecretType',
    'SecretStatus',
    'DataRetentionManager',
    'LegalHold',
    'LegalHoldStatus',
    'DataClassification',
]

__version__ = '1.0.0'
