"""
FalconOne Multi-Tenant Support Module (v3.0)
Enterprise multi-tenancy with data isolation and resource management

Implements:
- Tenant isolation and management
- Per-tenant configuration and customization
- Resource quotas and rate limiting
- Data encryption per tenant
- Tenant-aware database operations

Version: 3.0.0
"""

import hashlib
import secrets
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging
from enum import Enum

try:
    from ..utils.logger import ModuleLogger
except ImportError:
    class ModuleLogger:
        def __init__(self, name, parent):
            self.logger = logging.getLogger(name) if parent else logging.getLogger(__name__)
        def info(self, msg, **kw): self.logger.info(f"{msg} {kw if kw else ''}")
        def warning(self, msg, **kw): self.logger.warning(f"{msg} {kw if kw else ''}")
        def error(self, msg, **kw): self.logger.error(f"{msg} {kw if kw else ''}")


class TenantStatus(Enum):
    """Tenant status"""
    ACTIVE = 'active'
    SUSPENDED = 'suspended'
    TRIAL = 'trial'
    INACTIVE = 'inactive'


class SubscriptionTier(Enum):
    """Subscription tier"""
    FREE = 'free'
    BASIC = 'basic'
    PROFESSIONAL = 'professional'
    ENTERPRISE = 'enterprise'


@dataclass
class ResourceQuota:
    """Resource quota limits per tenant"""
    max_targets: int = 100
    max_scans_per_hour: int = 10
    max_storage_mb: int = 1000
    max_api_calls_per_minute: int = 100
    max_users: int = 5
    max_concurrent_operations: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class TenantConfig:
    """Tenant-specific configuration"""
    tenant_id: str
    tenant_name: str
    status: str = TenantStatus.ACTIVE.value
    subscription_tier: str = SubscriptionTier.BASIC.value
    created_at: float = None
    updated_at: float = None
    
    # Resource quotas
    quota: ResourceQuota = field(default_factory=ResourceQuota)
    
    # Customization
    custom_logo_url: Optional[str] = None
    custom_domain: Optional[str] = None
    branding_color: Optional[str] = None
    
    # Security
    encryption_key: Optional[str] = None
    ip_whitelist: List[str] = field(default_factory=list)
    require_2fa: bool = False
    
    # Features
    enabled_features: List[str] = field(default_factory=list)
    disabled_features: List[str] = field(default_factory=list)
    
    # Contact
    admin_email: Optional[str] = None
    admin_phone: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().timestamp()
        if self.updated_at is None:
            self.updated_at = datetime.now().timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['quota'] = self.quota.to_dict()
        return data
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class TenantUsage:
    """Current resource usage for a tenant"""
    tenant_id: str
    targets_count: int = 0
    scans_last_hour: int = 0
    storage_used_mb: float = 0
    api_calls_last_minute: int = 0
    users_count: int = 0
    concurrent_operations: int = 0
    last_activity: float = None
    
    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = datetime.now().timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class MultiTenantManager:
    """
    Multi-tenant management system
    
    Provides:
    - Tenant creation and management
    - Resource quota enforcement
    - Data isolation
    - Per-tenant configuration
    - Usage tracking and rate limiting
    """
    
    def __init__(self, logger=None):
        """
        Initialize multi-tenant manager
        
        Args:
            logger: Optional logger instance
        """
        self.logger = ModuleLogger('MultiTenant', logger)
        
        # Tenant registry
        self.tenants: Dict[str, TenantConfig] = {}
        self.tenant_usage: Dict[str, TenantUsage] = {}
        
        # Subscription tier quotas
        self.tier_quotas = {
            SubscriptionTier.FREE.value: ResourceQuota(
                max_targets=10,
                max_scans_per_hour=5,
                max_storage_mb=100,
                max_api_calls_per_minute=10,
                max_users=1,
                max_concurrent_operations=1
            ),
            SubscriptionTier.BASIC.value: ResourceQuota(
                max_targets=100,
                max_scans_per_hour=10,
                max_storage_mb=1000,
                max_api_calls_per_minute=100,
                max_users=5,
                max_concurrent_operations=3
            ),
            SubscriptionTier.PROFESSIONAL.value: ResourceQuota(
                max_targets=500,
                max_scans_per_hour=50,
                max_storage_mb=10000,
                max_api_calls_per_minute=500,
                max_users=20,
                max_concurrent_operations=10
            ),
            SubscriptionTier.ENTERPRISE.value: ResourceQuota(
                max_targets=99999,
                max_scans_per_hour=999,
                max_storage_mb=999999,
                max_api_calls_per_minute=9999,
                max_users=999,
                max_concurrent_operations=100
            ),
        }
        
        self.statistics = {
            'total_tenants': 0,
            'active_tenants': 0,
            'suspended_tenants': 0,
            'quota_exceeded_events': 0,
            'api_calls_blocked': 0,
        }
    
    def create_tenant(self, tenant_name: str, admin_email: str,
                     subscription_tier: str = SubscriptionTier.BASIC.value,
                     tenant_id: Optional[str] = None) -> TenantConfig:
        """
        Create a new tenant
        
        Args:
            tenant_name: Tenant organization name
            admin_email: Admin email address
            subscription_tier: Subscription tier (free, basic, professional, enterprise)
            tenant_id: Optional tenant ID (generated if not provided)
        
        Returns:
            Created TenantConfig
        """
        if tenant_id is None:
            tenant_id = self._generate_tenant_id(tenant_name)
        
        if tenant_id in self.tenants:
            raise ValueError(f"Tenant {tenant_id} already exists")
        
        # Get quota for tier
        quota = self.tier_quotas.get(subscription_tier, ResourceQuota())
        
        # Generate encryption key
        encryption_key = secrets.token_hex(32)
        
        # Create tenant config
        tenant = TenantConfig(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            status=TenantStatus.ACTIVE.value,
            subscription_tier=subscription_tier,
            quota=quota,
            encryption_key=encryption_key,
            admin_email=admin_email,
            enabled_features=['targets', 'scanning', 'exploits', 'monitoring']
        )
        
        self.tenants[tenant_id] = tenant
        self.tenant_usage[tenant_id] = TenantUsage(tenant_id=tenant_id)
        
        self.statistics['total_tenants'] += 1
        self.statistics['active_tenants'] += 1
        
        self.logger.info(f"Tenant created: {tenant_name}",
                        tenant_id=tenant_id,
                        tier=subscription_tier)
        
        return tenant
    
    def _generate_tenant_id(self, tenant_name: str) -> str:
        """Generate unique tenant ID"""
        # Create ID from tenant name + random component
        name_hash = hashlib.sha256(tenant_name.encode()).hexdigest()[:8]
        random_suffix = secrets.token_hex(4)
        return f"tenant_{name_hash}_{random_suffix}"
    
    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration"""
        return self.tenants.get(tenant_id)
    
    def update_tenant(self, tenant_id: str, updates: Dict[str, Any]) -> TenantConfig:
        """
        Update tenant configuration
        
        Args:
            tenant_id: Tenant ID
            updates: Dictionary of fields to update
        
        Returns:
            Updated TenantConfig
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        # Update allowed fields
        allowed_fields = ['tenant_name', 'status', 'subscription_tier', 'admin_email',
                         'admin_phone', 'custom_logo_url', 'custom_domain',
                         'branding_color', 'require_2fa', 'ip_whitelist',
                         'enabled_features', 'disabled_features', 'metadata']
        
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(tenant, field, value)
        
        # Update subscription tier quota if tier changed
        if 'subscription_tier' in updates:
            tenant.quota = self.tier_quotas.get(updates['subscription_tier'], ResourceQuota())
        
        tenant.updated_at = datetime.now().timestamp()
        
        self.logger.info(f"Tenant updated: {tenant_id}", updates=list(updates.keys()))
        
        return tenant
    
    def delete_tenant(self, tenant_id: str):
        """Delete a tenant (soft delete - mark as inactive)"""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        tenant.status = TenantStatus.INACTIVE.value
        tenant.updated_at = datetime.now().timestamp()
        
        self.statistics['active_tenants'] -= 1
        
        self.logger.info(f"Tenant deleted (marked inactive): {tenant_id}")
    
    def suspend_tenant(self, tenant_id: str, reason: str = None):
        """Suspend a tenant"""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        tenant.status = TenantStatus.SUSPENDED.value
        tenant.metadata['suspension_reason'] = reason
        tenant.metadata['suspended_at'] = datetime.now().timestamp()
        tenant.updated_at = datetime.now().timestamp()
        
        self.statistics['active_tenants'] -= 1
        self.statistics['suspended_tenants'] += 1
        
        self.logger.warning(f"Tenant suspended: {tenant_id}", reason=reason)
    
    def reactivate_tenant(self, tenant_id: str):
        """Reactivate a suspended tenant"""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        if tenant.status == TenantStatus.SUSPENDED.value:
            tenant.status = TenantStatus.ACTIVE.value
            tenant.metadata.pop('suspension_reason', None)
            tenant.metadata.pop('suspended_at', None)
            tenant.updated_at = datetime.now().timestamp()
            
            self.statistics['suspended_tenants'] -= 1
            self.statistics['active_tenants'] += 1
            
            self.logger.info(f"Tenant reactivated: {tenant_id}")
    
    def check_quota(self, tenant_id: str, resource: str, requested: int = 1) -> bool:
        """
        Check if tenant has quota available for resource
        
        Args:
            tenant_id: Tenant ID
            resource: Resource name (targets, scans, storage, api_calls, users, operations)
            requested: Amount requested (default 1)
        
        Returns:
            True if quota available, False if exceeded
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        if tenant.status != TenantStatus.ACTIVE.value:
            self.logger.warning(f"Quota check failed: tenant {tenant_id} not active")
            return False
        
        usage = self.tenant_usage.get(tenant_id)
        if not usage:
            return True
        
        quota = tenant.quota
        
        # Check quota based on resource type
        if resource == 'targets':
            available = quota.max_targets - usage.targets_count
        elif resource == 'scans':
            available = quota.max_scans_per_hour - usage.scans_last_hour
        elif resource == 'storage':
            available = quota.max_storage_mb - usage.storage_used_mb
        elif resource == 'api_calls':
            available = quota.max_api_calls_per_minute - usage.api_calls_last_minute
        elif resource == 'users':
            available = quota.max_users - usage.users_count
        elif resource == 'operations':
            available = quota.max_concurrent_operations - usage.concurrent_operations
        else:
            self.logger.warning(f"Unknown resource type: {resource}")
            return True
        
        if available < requested:
            self.statistics['quota_exceeded_events'] += 1
            self.logger.warning(f"Quota exceeded for {tenant_id}",
                              resource=resource,
                              requested=requested,
                              available=available)
            return False
        
        return True
    
    def increment_usage(self, tenant_id: str, resource: str, amount: int = 1):
        """Increment resource usage counter"""
        usage = self.tenant_usage.get(tenant_id)
        if not usage:
            usage = TenantUsage(tenant_id=tenant_id)
            self.tenant_usage[tenant_id] = usage
        
        if resource == 'targets':
            usage.targets_count += amount
        elif resource == 'scans':
            usage.scans_last_hour += amount
        elif resource == 'storage':
            usage.storage_used_mb += amount
        elif resource == 'api_calls':
            usage.api_calls_last_minute += amount
        elif resource == 'users':
            usage.users_count += amount
        elif resource == 'operations':
            usage.concurrent_operations += amount
        
        usage.last_activity = datetime.now().timestamp()
    
    def decrement_usage(self, tenant_id: str, resource: str, amount: int = 1):
        """Decrement resource usage counter"""
        usage = self.tenant_usage.get(tenant_id)
        if not usage:
            return
        
        if resource == 'targets':
            usage.targets_count = max(0, usage.targets_count - amount)
        elif resource == 'scans':
            usage.scans_last_hour = max(0, usage.scans_last_hour - amount)
        elif resource == 'storage':
            usage.storage_used_mb = max(0, usage.storage_used_mb - amount)
        elif resource == 'operations':
            usage.concurrent_operations = max(0, usage.concurrent_operations - amount)
    
    def reset_time_based_quotas(self, tenant_id: str):
        """Reset time-based quota counters (hourly, minute)"""
        usage = self.tenant_usage.get(tenant_id)
        if usage:
            usage.scans_last_hour = 0
            usage.api_calls_last_minute = 0
            
            self.logger.info(f"Time-based quotas reset for {tenant_id}")
    
    def get_usage(self, tenant_id: str) -> Optional[TenantUsage]:
        """Get current usage for tenant"""
        return self.tenant_usage.get(tenant_id)
    
    def list_tenants(self, status: Optional[str] = None,
                     tier: Optional[str] = None) -> List[TenantConfig]:
        """
        List tenants with optional filters
        
        Args:
            status: Filter by status (active, suspended, trial, inactive)
            tier: Filter by subscription tier
        
        Returns:
            List of TenantConfig
        """
        tenants = list(self.tenants.values())
        
        if status:
            tenants = [t for t in tenants if t.status == status]
        
        if tier:
            tenants = [t for t in tenants if t.subscription_tier == tier]
        
        return tenants
    
    def encrypt_tenant_data(self, tenant_id: str, data: bytes) -> bytes:
        """
        Encrypt data with tenant-specific key
        
        Args:
            tenant_id: Tenant ID
            data: Data to encrypt
        
        Returns:
            Encrypted data
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant or not tenant.encryption_key:
            raise ValueError(f"Encryption key not found for tenant {tenant_id}")
        
        from Crypto.Cipher import AES
        from Crypto.Random import get_random_bytes
        
        # Derive AES key from tenant encryption key
        key = hashlib.sha256(tenant.encryption_key.encode()).digest()
        
        # Encrypt with AES-GCM
        cipher = AES.new(key, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(data)
        
        # Return nonce + tag + ciphertext
        return cipher.nonce + tag + ciphertext
    
    def decrypt_tenant_data(self, tenant_id: str, encrypted_data: bytes) -> bytes:
        """
        Decrypt data with tenant-specific key
        
        Args:
            tenant_id: Tenant ID
            encrypted_data: Encrypted data (nonce + tag + ciphertext)
        
        Returns:
            Decrypted data
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant or not tenant.encryption_key:
            raise ValueError(f"Encryption key not found for tenant {tenant_id}")
        
        from Crypto.Cipher import AES
        
        # Derive AES key from tenant encryption key
        key = hashlib.sha256(tenant.encryption_key.encode()).digest()
        
        # Extract components
        nonce = encrypted_data[:16]
        tag = encrypted_data[16:32]
        ciphertext = encrypted_data[32:]
        
        # Decrypt with AES-GCM
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        
        return plaintext
    
    def is_feature_enabled(self, tenant_id: str, feature: str) -> bool:
        """Check if feature is enabled for tenant"""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        # Check if explicitly disabled
        if feature in tenant.disabled_features:
            return False
        
        # Check if in enabled list
        return feature in tenant.enabled_features
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get multi-tenant statistics"""
        return {
            **self.statistics,
            'tier_distribution': {
                tier: len([t for t in self.tenants.values() 
                          if t.subscription_tier == tier])
                for tier in [t.value for t in SubscriptionTier]
            }
        }
