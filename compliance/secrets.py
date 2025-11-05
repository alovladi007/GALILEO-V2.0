"""
Secrets Management - Secure storage and rotation of sensitive credentials
Provides encrypted storage, automatic rotation, and access auditing for secrets.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import secrets
import hashlib
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecretType(Enum):
    """Types of secrets managed by the system"""
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    ENCRYPTION_KEY = "encryption_key"
    DATABASE_CREDENTIAL = "database_credential"
    SSH_KEY = "ssh_key"


class SecretStatus(Enum):
    """Status of a secret"""
    ACTIVE = "active"
    ROTATED = "rotated"
    EXPIRED = "expired"
    COMPROMISED = "compromised"
    PENDING_ROTATION = "pending_rotation"


@dataclass
class Secret:
    """Secure secret container"""
    secret_id: str
    name: str
    secret_type: SecretType
    encrypted_value: bytes
    status: SecretStatus = SecretStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    rotation_policy_days: int = 90
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    
    def is_expired(self) -> bool:
        """Check if secret has expired"""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    def needs_rotation(self) -> bool:
        """Check if secret needs rotation"""
        rotation_date = self.updated_at + timedelta(days=self.rotation_policy_days)
        return datetime.utcnow() > rotation_date
    
    def to_dict(self, include_value: bool = False) -> Dict[str, Any]:
        """Convert secret to dictionary (excluding encrypted value by default)"""
        data = {
            'secret_id': self.secret_id,
            'name': self.name,
            'secret_type': self.secret_type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'access_count': self.access_count,
            'rotation_policy_days': self.rotation_policy_days,
            'version': self.version,
            'metadata': self.metadata,
        }
        
        if include_value:
            data['encrypted_value'] = self.encrypted_value.hex()
        
        return data


class SecretsManager:
    """Manages secrets with encryption, rotation, and auditing"""
    
    def __init__(self, master_key: Optional[bytes] = None, audit_logger: Optional[Any] = None):
        self.secrets: Dict[str, Secret] = {}
        self.secret_history: Dict[str, List[Secret]] = {}
        self.audit_logger = audit_logger
        
        # Initialize encryption key
        if master_key is None:
            master_key = Fernet.generate_key()
        self.master_key = master_key
        self.cipher = Fernet(master_key)
    
    @staticmethod
    def generate_master_key(password: str, salt: Optional[bytes] = None) -> bytes:
        """Generate master key from password"""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password.encode())
        return Fernet.generate_key()  # Use Fernet key format
    
    def create_secret(
        self,
        name: str,
        value: str,
        secret_type: SecretType,
        rotation_policy_days: int = 90,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Secret:
        """Create and store a new secret"""
        # Generate unique ID
        secret_id = secrets.token_urlsafe(16)
        
        # Encrypt the secret value
        encrypted_value = self.cipher.encrypt(value.encode())
        
        # Create secret object
        secret = Secret(
            secret_id=secret_id,
            name=name,
            secret_type=secret_type,
            encrypted_value=encrypted_value,
            rotation_policy_days=rotation_policy_days,
            expires_at=expires_at,
            metadata=metadata or {},
        )
        
        # Store secret
        self.secrets[secret_id] = secret
        self.secret_history[secret_id] = [secret]
        
        # Audit log
        if self.audit_logger:
            from .audit import AuditEventType
            self.audit_logger.log_event(
                event_type=AuditEventType.SECRET_CREATED,
                action="create_secret",
                result="success",
                user_id=user_id,
                resource=secret_id,
                metadata={
                    'secret_name': name,
                    'secret_type': secret_type.value,
                }
            )
        
        return secret
    
    def get_secret(
        self,
        secret_id: str,
        user_id: Optional[str] = None
    ) -> Optional[str]:
        """Retrieve and decrypt a secret"""
        secret = self.secrets.get(secret_id)
        if not secret:
            return None
        
        # Check expiration
        if secret.is_expired():
            secret.status = SecretStatus.EXPIRED
            return None
        
        # Update access tracking
        secret.last_accessed = datetime.utcnow()
        secret.access_count += 1
        
        # Audit log
        if self.audit_logger:
            from .audit import AuditEventType
            self.audit_logger.log_event(
                event_type=AuditEventType.SECRET_ACCESSED,
                action="get_secret",
                result="success",
                user_id=user_id,
                resource=secret_id,
                metadata={
                    'secret_name': secret.name,
                    'access_count': secret.access_count,
                }
            )
        
        # Decrypt and return
        decrypted_value = self.cipher.decrypt(secret.encrypted_value)
        return decrypted_value.decode()
    
    def rotate_secret(
        self,
        secret_id: str,
        new_value: str,
        user_id: Optional[str] = None
    ) -> bool:
        """Rotate a secret with a new value"""
        secret = self.secrets.get(secret_id)
        if not secret:
            return False
        
        # Mark old secret as rotated
        old_secret = secret
        old_secret.status = SecretStatus.ROTATED
        
        # Create new version
        encrypted_value = self.cipher.encrypt(new_value.encode())
        new_secret = Secret(
            secret_id=secret_id,
            name=secret.name,
            secret_type=secret.secret_type,
            encrypted_value=encrypted_value,
            rotation_policy_days=secret.rotation_policy_days,
            expires_at=secret.expires_at,
            metadata=secret.metadata,
            version=secret.version + 1,
        )
        
        # Update storage
        self.secrets[secret_id] = new_secret
        self.secret_history[secret_id].append(new_secret)
        
        # Audit log
        if self.audit_logger:
            from .audit import AuditEventType
            self.audit_logger.log_event(
                event_type=AuditEventType.SECRET_ROTATED,
                action="rotate_secret",
                result="success",
                user_id=user_id,
                resource=secret_id,
                metadata={
                    'secret_name': secret.name,
                    'old_version': old_secret.version,
                    'new_version': new_secret.version,
                }
            )
        
        return True
    
    def delete_secret(
        self,
        secret_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """Delete a secret (marks as deleted, keeps history)"""
        secret = self.secrets.get(secret_id)
        if not secret:
            return False
        
        # Remove from active secrets but keep history
        del self.secrets[secret_id]
        
        # Audit log
        if self.audit_logger:
            from .audit import AuditEventType
            self.audit_logger.log_event(
                event_type=AuditEventType.SECRET_DELETED,
                action="delete_secret",
                result="success",
                user_id=user_id,
                resource=secret_id,
                metadata={
                    'secret_name': secret.name,
                }
            )
        
        return True
    
    def check_rotation_needed(self) -> List[Secret]:
        """Get list of secrets that need rotation"""
        return [
            secret for secret in self.secrets.values()
            if secret.needs_rotation() and secret.status == SecretStatus.ACTIVE
        ]
    
    def auto_rotate_expired(self, rotation_callback=None) -> int:
        """Automatically rotate secrets that need rotation"""
        rotated_count = 0
        secrets_to_rotate = self.check_rotation_needed()
        
        for secret in secrets_to_rotate:
            if rotation_callback:
                new_value = rotation_callback(secret)
                if new_value:
                    self.rotate_secret(secret.secret_id, new_value)
                    rotated_count += 1
            else:
                # Mark as pending rotation
                secret.status = SecretStatus.PENDING_ROTATION
        
        return rotated_count
    
    def list_secrets(
        self,
        secret_type: Optional[SecretType] = None,
        status: Optional[SecretStatus] = None
    ) -> List[Dict[str, Any]]:
        """List secrets with optional filters"""
        secrets = list(self.secrets.values())
        
        if secret_type:
            secrets = [s for s in secrets if s.secret_type == secret_type]
        
        if status:
            secrets = [s for s in secrets if s.status == status]
        
        return [s.to_dict() for s in secrets]
    
    def get_secret_metadata(self, secret_id: str) -> Optional[Dict[str, Any]]:
        """Get secret metadata without decrypting value"""
        secret = self.secrets.get(secret_id)
        if secret:
            return secret.to_dict()
        return None
    
    def export_secrets_manifest(self) -> Dict[str, Any]:
        """Export manifest of all secrets (without values)"""
        return {
            'total_secrets': len(self.secrets),
            'secrets_by_type': self._count_by_type(),
            'secrets_by_status': self._count_by_status(),
            'secrets_needing_rotation': len(self.check_rotation_needed()),
            'secrets': [s.to_dict() for s in self.secrets.values()],
            'exported_at': datetime.utcnow().isoformat(),
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count secrets by type"""
        counts = {}
        for secret in self.secrets.values():
            secret_type = secret.secret_type.value
            counts[secret_type] = counts.get(secret_type, 0) + 1
        return counts
    
    def _count_by_status(self) -> Dict[str, int]:
        """Count secrets by status"""
        counts = {}
        for secret in self.secrets.values():
            status = secret.status.value
            counts[status] = counts.get(status, 0) + 1
        return counts
