"""
Audit Logging - Comprehensive audit trail for compliance
Provides immutable audit logs with cryptographic verification and tamper detection.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import uuid


class AuditEventType(Enum):
    """Types of auditable events"""
    # Authentication events
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"
    
    # Authorization events
    ACCESS_GRANTED = "authz.access_granted"
    ACCESS_DENIED = "authz.access_denied"
    PERMISSION_CHANGED = "authz.permission_changed"
    
    # Data events
    DATA_ACCESSED = "data.accessed"
    DATA_MODIFIED = "data.modified"
    DATA_DELETED = "data.deleted"
    DATA_EXPORTED = "data.exported"
    
    # Research events
    RESEARCH_STARTED = "research.started"
    RESEARCH_COMPLETED = "research.completed"
    RESEARCH_BLOCKED = "research.blocked"
    
    # Compliance events
    POLICY_CREATED = "compliance.policy_created"
    POLICY_MODIFIED = "compliance.policy_modified"
    POLICY_DELETED = "compliance.policy_deleted"
    RETENTION_APPLIED = "compliance.retention_applied"
    LEGAL_HOLD_APPLIED = "compliance.legal_hold_applied"
    
    # Secrets events
    SECRET_ACCESSED = "secrets.accessed"
    SECRET_CREATED = "secrets.created"
    SECRET_ROTATED = "secrets.rotated"
    SECRET_DELETED = "secrets.deleted"
    
    # System events
    SYSTEM_CONFIG_CHANGED = "system.config_changed"
    SYSTEM_ERROR = "system.error"
    SECURITY_ALERT = "security.alert"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Immutable audit event record"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: AuditEventType = AuditEventType.SYSTEM_ERROR
    severity: AuditSeverity = AuditSeverity.INFO
    user_id: Optional[str] = None
    resource: Optional[str] = None
    action: str = ""
    result: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    previous_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'user_id': self.user_id,
            'resource': self.resource,
            'action': self.action,
            'result': self.result,
            'metadata': self.metadata,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'previous_hash': self.previous_hash,
        }
    
    def hash(self) -> str:
        """Generate cryptographic hash of event"""
        event_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()


class AuditLogger:
    """Manages audit logging with tamper detection"""
    
    def __init__(self, storage_backend: Optional[Any] = None):
        self.events: List[AuditEvent] = []
        self.storage_backend = storage_backend
        self.chain_verified = True
        self.last_hash: Optional[str] = None
    
    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        result: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditEvent:
        """Log an audit event with cryptographic chaining"""
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            metadata=metadata or {},
            ip_address=ip_address,
            user_agent=user_agent,
            previous_hash=self.last_hash,
        )
        
        # Calculate hash for this event
        event_hash = event.hash()
        self.last_hash = event_hash
        
        # Store event
        self.events.append(event)
        
        # Persist to backend if available
        if self.storage_backend:
            self._persist_event(event)
        
        return event
    
    def log_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        granted: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log an access attempt"""
        event_type = AuditEventType.ACCESS_GRANTED if granted else AuditEventType.ACCESS_DENIED
        severity = AuditSeverity.INFO if granted else AuditSeverity.WARNING
        result = "success" if granted else "denied"
        
        return self.log_event(
            event_type=event_type,
            action=action,
            result=result,
            user_id=user_id,
            resource=resource,
            severity=severity,
            metadata=metadata,
        )
    
    def log_data_access(
        self,
        user_id: str,
        resource: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log data access event"""
        event_type_map = {
            'read': AuditEventType.DATA_ACCESSED,
            'write': AuditEventType.DATA_MODIFIED,
            'delete': AuditEventType.DATA_DELETED,
            'export': AuditEventType.DATA_EXPORTED,
        }
        
        event_type = event_type_map.get(operation, AuditEventType.DATA_ACCESSED)
        
        return self.log_event(
            event_type=event_type,
            action=operation,
            result="completed",
            user_id=user_id,
            resource=resource,
            severity=AuditSeverity.INFO,
            metadata=metadata,
        )
    
    def log_security_alert(
        self,
        alert_type: str,
        description: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log a security alert"""
        return self.log_event(
            event_type=AuditEventType.SECURITY_ALERT,
            action=alert_type,
            result=description,
            user_id=user_id,
            severity=AuditSeverity.CRITICAL,
            metadata=metadata,
        )
    
    def verify_chain(self) -> bool:
        """Verify integrity of audit log chain"""
        if not self.events:
            return True
        
        previous_hash = None
        for event in self.events:
            # Verify previous hash matches
            if event.previous_hash != previous_hash:
                self.chain_verified = False
                return False
            
            # Calculate expected hash
            expected_hash = event.hash()
            previous_hash = expected_hash
        
        self.chain_verified = True
        return True
    
    def query_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[AuditSeverity] = None,
    ) -> List[AuditEvent]:
        """Query audit events with filters"""
        filtered_events = self.events
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if resource:
            filtered_events = [e for e in filtered_events if e.resource == resource]
        
        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        return filtered_events
    
    def generate_audit_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        events = self.query_events(start_time=start_time, end_time=end_time)
        
        # Calculate statistics
        event_types = {}
        severities = {}
        users = {}
        
        for event in events:
            event_types[event.event_type.value] = event_types.get(event.event_type.value, 0) + 1
            severities[event.severity.value] = severities.get(event.severity.value, 0) + 1
            if event.user_id:
                users[event.user_id] = users.get(event.user_id, 0) + 1
        
        return {
            'report_generated': datetime.utcnow().isoformat(),
            'period_start': start_time.isoformat() if start_time else None,
            'period_end': end_time.isoformat() if end_time else None,
            'total_events': len(events),
            'event_types': event_types,
            'severities': severities,
            'top_users': sorted(users.items(), key=lambda x: x[1], reverse=True)[:10],
            'chain_verified': self.chain_verified,
        }
    
    def get_compliance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get compliance-focused summary"""
        start_time = datetime.utcnow() - timedelta(days=days)
        events = self.query_events(start_time=start_time)
        
        # Focus on compliance-relevant events
        access_denied = len([e for e in events if e.event_type == AuditEventType.ACCESS_DENIED])
        data_exports = len([e for e in events if e.event_type == AuditEventType.DATA_EXPORTED])
        policy_changes = len([e for e in events if 'policy' in e.event_type.value])
        security_alerts = len([e for e in events if e.event_type == AuditEventType.SECURITY_ALERT])
        
        return {
            'summary_period_days': days,
            'total_events': len(events),
            'access_denied_count': access_denied,
            'data_exports_count': data_exports,
            'policy_changes_count': policy_changes,
            'security_alerts_count': security_alerts,
            'chain_integrity': self.chain_verified,
            'generated_at': datetime.utcnow().isoformat(),
        }
    
    def _persist_event(self, event: AuditEvent):
        """Persist event to storage backend"""
        # Placeholder for actual storage implementation
        # Could be database, file system, or external audit service
        pass
    
    def export_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = 'json'
    ) -> str:
        """Export audit logs in specified format"""
        events = self.query_events(start_time=start_time, end_time=end_time)
        
        if format == 'json':
            return json.dumps(
                [event.to_dict() for event in events],
                indent=2,
                default=str
            )
        
        # Add other formats as needed (CSV, XML, etc.)
        return json.dumps([event.to_dict() for event in events], default=str)
