"""
Authorization Policies - Role-Based Access Control (RBAC)
Implements fine-grained authorization policies with support for roles, permissions, and resources.
"""

from typing import Dict, List, Set, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib


class Permission(Enum):
    """Enumeration of system permissions"""
    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"
    
    # Research permissions
    RESEARCH_CONDUCT = "research:conduct"
    RESEARCH_REVIEW = "research:review"
    RESEARCH_APPROVE = "research:approve"
    
    # Compliance permissions
    COMPLIANCE_VIEW = "compliance:view"
    COMPLIANCE_CONFIGURE = "compliance:configure"
    COMPLIANCE_AUDIT = "compliance:audit"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_POLICIES = "admin:policies"
    
    # Secrets permissions
    SECRETS_READ = "secrets:read"
    SECRETS_WRITE = "secrets:write"
    SECRETS_ROTATE = "secrets:rotate"


@dataclass
class Policy:
    """Authorization policy definition"""
    name: str
    description: str
    permissions: Set[Permission]
    resources: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'permissions': [p.value for p in self.permissions],
            'resources': self.resources,
            'conditions': self.conditions,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }
    
    def hash(self) -> str:
        """Generate policy hash for integrity verification"""
        policy_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(policy_str.encode()).hexdigest()


class AuthorizationManager:
    """Manages authorization policies and access control"""
    
    def __init__(self):
        self.policies: Dict[str, Policy] = {}
        self.role_policies: Dict[str, Set[str]] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self._init_default_policies()
    
    def _init_default_policies(self):
        """Initialize default security policies"""
        # Research policy - restricted operations
        self.add_policy(Policy(
            name="research_restricted",
            description="Restricted research operations policy",
            permissions={
                Permission.RESEARCH_CONDUCT,
                Permission.DATA_READ,
            },
            conditions={
                "require_ethical_review": True,
                "require_legal_review": True,
                "prohibited_topics": [
                    "weapons_development",
                    "surveillance_systems",
                    "manipulation_techniques",
                    "privacy_invasion",
                ],
            }
        ))
        
        # Data protection policy
        self.add_policy(Policy(
            name="data_protection",
            description="Data protection and privacy policy",
            permissions={
                Permission.DATA_READ,
                Permission.DATA_WRITE,
            },
            conditions={
                "encryption_required": True,
                "pii_handling": "strict",
                "audit_all_access": True,
            }
        ))
        
        # Compliance officer policy
        self.add_policy(Policy(
            name="compliance_officer",
            description="Full compliance management access",
            permissions={
                Permission.COMPLIANCE_VIEW,
                Permission.COMPLIANCE_CONFIGURE,
                Permission.COMPLIANCE_AUDIT,
                Permission.DATA_READ,
            },
        ))
        
        # Admin policy
        self.add_policy(Policy(
            name="system_admin",
            description="System administration access",
            permissions={
                Permission.ADMIN_USERS,
                Permission.ADMIN_SYSTEM,
                Permission.ADMIN_POLICIES,
                Permission.COMPLIANCE_VIEW,
            },
        ))
        
        # Read-only policy
        self.add_policy(Policy(
            name="read_only",
            description="Read-only access to non-sensitive data",
            permissions={
                Permission.DATA_READ,
                Permission.COMPLIANCE_VIEW,
            },
            conditions={
                "exclude_sensitive": True,
            }
        ))
    
    def add_policy(self, policy: Policy) -> bool:
        """Add or update a policy"""
        self.policies[policy.name] = policy
        return True
    
    def remove_policy(self, policy_name: str) -> bool:
        """Remove a policy"""
        if policy_name in self.policies:
            del self.policies[policy_name]
            return True
        return False
    
    def assign_role_policy(self, role: str, policy_name: str) -> bool:
        """Assign a policy to a role"""
        if policy_name not in self.policies:
            raise ValueError(f"Policy '{policy_name}' does not exist")
        
        if role not in self.role_policies:
            self.role_policies[role] = set()
        
        self.role_policies[role].add(policy_name)
        return True
    
    def assign_user_role(self, user_id: str, role: str) -> bool:
        """Assign a role to a user"""
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        self.user_roles[user_id].add(role)
        return True
    
    def check_permission(
        self,
        user_id: str,
        permission: Permission,
        resource: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if user has permission for a resource"""
        user_roles = self.user_roles.get(user_id, set())
        
        for role in user_roles:
            policies = self.role_policies.get(role, set())
            
            for policy_name in policies:
                policy = self.policies.get(policy_name)
                if not policy:
                    continue
                
                # Check if permission is granted
                if permission not in policy.permissions:
                    continue
                
                # Check resource restrictions
                if resource and policy.resources:
                    if resource not in policy.resources:
                        continue
                
                # Check conditions
                if context and not self._check_conditions(policy.conditions, context):
                    continue
                
                return True
        
        return False
    
    def _check_conditions(
        self,
        conditions: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate policy conditions against context"""
        for key, value in conditions.items():
            if key == "require_ethical_review" and value:
                if not context.get("ethical_review_approved", False):
                    return False
            
            if key == "require_legal_review" and value:
                if not context.get("legal_review_approved", False):
                    return False
            
            if key == "prohibited_topics":
                topic = context.get("topic", "")
                if any(prohibited in topic.lower() for prohibited in value):
                    return False
            
            if key == "exclude_sensitive":
                if context.get("data_classification") == "sensitive":
                    return False
        
        return True
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user"""
        permissions = set()
        user_roles = self.user_roles.get(user_id, set())
        
        for role in user_roles:
            policies = self.role_policies.get(role, set())
            for policy_name in policies:
                policy = self.policies.get(policy_name)
                if policy:
                    permissions.update(policy.permissions)
        
        return permissions
    
    def audit_authorization(self, user_id: str, action: str, result: bool) -> Dict[str, Any]:
        """Create audit record for authorization decision"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'action': action,
            'result': 'allowed' if result else 'denied',
            'roles': list(self.user_roles.get(user_id, set())),
        }
    
    def export_policies(self) -> Dict[str, Any]:
        """Export all policies for backup/audit"""
        return {
            'policies': {name: policy.to_dict() for name, policy in self.policies.items()},
            'role_policies': {role: list(policies) for role, policies in self.role_policies.items()},
            'policy_hashes': {name: policy.hash() for name, policy in self.policies.items()},
            'exported_at': datetime.utcnow().isoformat(),
        }
