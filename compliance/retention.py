"""
Data Retention and Legal Hold - Compliance-driven data lifecycle management
Implements retention policies, legal holds, and data disposal with audit trails.
"""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json


class RetentionAction(Enum):
    """Actions to take when retention period expires"""
    DELETE = "delete"
    ARCHIVE = "archive"
    REVIEW = "review"
    NOTIFY = "notify"


class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    REGULATED = "regulated"


class LegalHoldStatus(Enum):
    """Status of legal hold"""
    ACTIVE = "active"
    RELEASED = "released"
    EXPIRED = "expired"


@dataclass
class RetentionPolicy:
    """Data retention policy definition"""
    policy_id: str
    name: str
    description: str
    retention_days: int
    action: RetentionAction
    data_classification: DataClassification
    applicable_types: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary"""
        return {
            'policy_id': self.policy_id,
            'name': self.name,
            'description': self.description,
            'retention_days': self.retention_days,
            'action': self.action.value,
            'data_classification': self.data_classification.value,
            'applicable_types': self.applicable_types,
            'exceptions': self.exceptions,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'active': self.active,
        }


@dataclass
class LegalHold:
    """Legal hold preventing data deletion"""
    hold_id: str
    name: str
    description: str
    case_number: str
    custodian: str
    resources: Set[str] = field(default_factory=set)
    status: LegalHoldStatus = LegalHoldStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    released_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if hold is currently active"""
        if self.status != LegalHoldStatus.ACTIVE:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hold to dictionary"""
        return {
            'hold_id': self.hold_id,
            'name': self.name,
            'description': self.description,
            'case_number': self.case_number,
            'custodian': self.custodian,
            'resources': list(self.resources),
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'released_at': self.released_at.isoformat() if self.released_at else None,
            'metadata': self.metadata,
        }


@dataclass
class DataRecord:
    """Tracked data record for retention"""
    record_id: str
    resource_type: str
    resource_id: str
    classification: DataClassification
    created_at: datetime
    retention_policy_id: Optional[str] = None
    deletion_scheduled_at: Optional[datetime] = None
    legal_holds: Set[str] = field(default_factory=set)
    archived: bool = False
    archived_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_protected(self) -> bool:
        """Check if record is protected from deletion"""
        return len(self.legal_holds) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary"""
        return {
            'record_id': self.record_id,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'classification': self.classification.value,
            'created_at': self.created_at.isoformat(),
            'retention_policy_id': self.retention_policy_id,
            'deletion_scheduled_at': self.deletion_scheduled_at.isoformat() if self.deletion_scheduled_at else None,
            'legal_holds': list(self.legal_holds),
            'archived': self.archived,
            'archived_at': self.archived_at.isoformat() if self.archived_at else None,
            'metadata': self.metadata,
        }


class DataRetentionManager:
    """Manages data retention policies and legal holds"""
    
    def __init__(self, audit_logger: Optional[Any] = None):
        self.policies: Dict[str, RetentionPolicy] = {}
        self.legal_holds: Dict[str, LegalHold] = {}
        self.data_records: Dict[str, DataRecord] = {}
        self.audit_logger = audit_logger
        self._init_default_policies()
    
    def _init_default_policies(self):
        """Initialize default retention policies"""
        # Research data - 7 years
        self.add_policy(RetentionPolicy(
            policy_id="research_data_7y",
            name="Research Data 7 Year Retention",
            description="Retain research data for 7 years per regulatory requirements",
            retention_days=2555,  # ~7 years
            action=RetentionAction.ARCHIVE,
            data_classification=DataClassification.REGULATED,
            applicable_types=["research_data", "experiment_results", "clinical_data"],
        ))
        
        # Audit logs - 10 years
        self.add_policy(RetentionPolicy(
            policy_id="audit_logs_10y",
            name="Audit Logs 10 Year Retention",
            description="Retain audit logs for 10 years for compliance",
            retention_days=3650,  # 10 years
            action=RetentionAction.ARCHIVE,
            data_classification=DataClassification.REGULATED,
            applicable_types=["audit_log", "security_log"],
        ))
        
        # User data - 90 days after deletion request
        self.add_policy(RetentionPolicy(
            policy_id="user_data_90d",
            name="User Data 90 Day Grace Period",
            description="Retain user data for 90 days after deletion request",
            retention_days=90,
            action=RetentionAction.DELETE,
            data_classification=DataClassification.CONFIDENTIAL,
            applicable_types=["user_data", "personal_info"],
        ))
        
        # Public data - indefinite (reviewed annually)
        self.add_policy(RetentionPolicy(
            policy_id="public_data_review",
            name="Public Data Annual Review",
            description="Review public data annually",
            retention_days=365,
            action=RetentionAction.REVIEW,
            data_classification=DataClassification.PUBLIC,
            applicable_types=["public_content"],
        ))
        
        # Temporary data - 30 days
        self.add_policy(RetentionPolicy(
            policy_id="temp_data_30d",
            name="Temporary Data 30 Day Retention",
            description="Delete temporary data after 30 days",
            retention_days=30,
            action=RetentionAction.DELETE,
            data_classification=DataClassification.INTERNAL,
            applicable_types=["temp_file", "cache", "session_data"],
        ))
    
    def add_policy(self, policy: RetentionPolicy) -> bool:
        """Add or update a retention policy"""
        self.policies[policy.policy_id] = policy
        
        if self.audit_logger:
            from .audit import AuditEventType
            self.audit_logger.log_event(
                event_type=AuditEventType.POLICY_CREATED,
                action="add_retention_policy",
                result="success",
                resource=policy.policy_id,
                metadata={'policy_name': policy.name},
            )
        
        return True
    
    def create_legal_hold(
        self,
        name: str,
        description: str,
        case_number: str,
        custodian: str,
        resources: Optional[Set[str]] = None,
        expires_at: Optional[datetime] = None,
        user_id: Optional[str] = None,
    ) -> LegalHold:
        """Create a new legal hold"""
        hold_id = f"hold_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        hold = LegalHold(
            hold_id=hold_id,
            name=name,
            description=description,
            case_number=case_number,
            custodian=custodian,
            resources=resources or set(),
            expires_at=expires_at,
        )
        
        self.legal_holds[hold_id] = hold
        
        # Apply hold to resources
        for resource_id in hold.resources:
            self._apply_legal_hold(resource_id, hold_id)
        
        # Audit log
        if self.audit_logger:
            from .audit import AuditEventType, AuditSeverity
            self.audit_logger.log_event(
                event_type=AuditEventType.LEGAL_HOLD_APPLIED,
                action="create_legal_hold",
                result="success",
                user_id=user_id,
                resource=hold_id,
                severity=AuditSeverity.WARNING,
                metadata={
                    'case_number': case_number,
                    'resource_count': len(hold.resources),
                },
            )
        
        return hold
    
    def release_legal_hold(
        self,
        hold_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """Release a legal hold"""
        hold = self.legal_holds.get(hold_id)
        if not hold:
            return False
        
        # Remove hold from all resources
        for resource_id in hold.resources:
            self._remove_legal_hold(resource_id, hold_id)
        
        # Update hold status
        hold.status = LegalHoldStatus.RELEASED
        hold.released_at = datetime.utcnow()
        
        # Audit log
        if self.audit_logger:
            from .audit import AuditEventType, AuditSeverity
            self.audit_logger.log_event(
                event_type=AuditEventType.LEGAL_HOLD_APPLIED,
                action="release_legal_hold",
                result="success",
                user_id=user_id,
                resource=hold_id,
                severity=AuditSeverity.WARNING,
            )
        
        return True
    
    def register_data(
        self,
        resource_type: str,
        resource_id: str,
        classification: DataClassification,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataRecord:
        """Register data for retention tracking"""
        record_id = f"{resource_type}:{resource_id}"
        
        # Find applicable policy
        policy = self._find_applicable_policy(resource_type, classification)
        
        # Calculate deletion schedule
        deletion_scheduled_at = None
        if policy:
            creation_time = created_at or datetime.utcnow()
            deletion_scheduled_at = creation_time + timedelta(days=policy.retention_days)
        
        record = DataRecord(
            record_id=record_id,
            resource_type=resource_type,
            resource_id=resource_id,
            classification=classification,
            created_at=created_at or datetime.utcnow(),
            retention_policy_id=policy.policy_id if policy else None,
            deletion_scheduled_at=deletion_scheduled_at,
            metadata=metadata or {},
        )
        
        self.data_records[record_id] = record
        
        # Audit log
        if self.audit_logger:
            from .audit import AuditEventType
            self.audit_logger.log_event(
                event_type=AuditEventType.RETENTION_APPLIED,
                action="register_data",
                result="success",
                resource=record_id,
                metadata={
                    'policy_id': policy.policy_id if policy else None,
                    'deletion_scheduled': deletion_scheduled_at.isoformat() if deletion_scheduled_at else None,
                },
            )
        
        return record
    
    def _find_applicable_policy(
        self,
        resource_type: str,
        classification: DataClassification
    ) -> Optional[RetentionPolicy]:
        """Find applicable retention policy for resource"""
        for policy in self.policies.values():
            if not policy.active:
                continue
            
            if policy.data_classification != classification:
                continue
            
            if policy.applicable_types and resource_type not in policy.applicable_types:
                continue
            
            return policy
        
        return None
    
    def _apply_legal_hold(self, resource_id: str, hold_id: str):
        """Apply legal hold to a resource"""
        for record in self.data_records.values():
            if record.resource_id == resource_id:
                record.legal_holds.add(hold_id)
    
    def _remove_legal_hold(self, resource_id: str, hold_id: str):
        """Remove legal hold from a resource"""
        for record in self.data_records.values():
            if record.resource_id == resource_id:
                record.legal_holds.discard(hold_id)
    
    def get_records_for_deletion(self) -> List[DataRecord]:
        """Get records scheduled for deletion"""
        now = datetime.utcnow()
        return [
            record for record in self.data_records.values()
            if (record.deletion_scheduled_at and 
                record.deletion_scheduled_at <= now and
                not record.is_protected() and
                not record.archived)
        ]
    
    def get_records_for_archival(self) -> List[DataRecord]:
        """Get records scheduled for archival"""
        now = datetime.utcnow()
        eligible_records = []
        
        for record in self.data_records.values():
            if record.archived:
                continue
            
            policy = self.policies.get(record.retention_policy_id)
            if policy and policy.action == RetentionAction.ARCHIVE:
                if record.deletion_scheduled_at and record.deletion_scheduled_at <= now:
                    eligible_records.append(record)
        
        return eligible_records
    
    def execute_retention_actions(self) -> Dict[str, int]:
        """Execute scheduled retention actions"""
        results = {
            'deleted': 0,
            'archived': 0,
            'skipped_protected': 0,
        }
        
        # Process deletions
        for record in self.get_records_for_deletion():
            if record.is_protected():
                results['skipped_protected'] += 1
                continue
            
            policy = self.policies.get(record.retention_policy_id)
            if policy and policy.action == RetentionAction.DELETE:
                # Actual deletion would happen here
                results['deleted'] += 1
        
        # Process archival
        for record in self.get_records_for_archival():
            if not record.is_protected():
                record.archived = True
                record.archived_at = datetime.utcnow()
                results['archived'] += 1
        
        return results
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report for data retention"""
        now = datetime.utcnow()
        
        # Calculate statistics
        total_records = len(self.data_records)
        protected_records = sum(1 for r in self.data_records.values() if r.is_protected())
        archived_records = sum(1 for r in self.data_records.values() if r.archived)
        active_holds = sum(1 for h in self.legal_holds.values() if h.is_active())
        
        # Records by classification
        by_classification = {}
        for record in self.data_records.values():
            cls = record.classification.value
            by_classification[cls] = by_classification.get(cls, 0) + 1
        
        return {
            'report_generated': now.isoformat(),
            'total_records': total_records,
            'protected_records': protected_records,
            'archived_records': archived_records,
            'active_policies': len([p for p in self.policies.values() if p.active]),
            'active_legal_holds': active_holds,
            'records_by_classification': by_classification,
            'pending_deletion': len(self.get_records_for_deletion()),
            'pending_archival': len(self.get_records_for_archival()),
        }
    
    def export_policies(self) -> Dict[str, Any]:
        """Export all retention policies"""
        return {
            'policies': [p.to_dict() for p in self.policies.values()],
            'legal_holds': [h.to_dict() for h in self.legal_holds.values()],
            'exported_at': datetime.utcnow().isoformat(),
        }
