"""
O-RAN Near-RT RIC (RAN Intelligent Controller) Integration

Provides xApp hosting, RIC services, and E2 node management

Features:
- xApp lifecycle management
- E2 node discovery and management
- Shared data layer (SDL)
- Conflict mitigation
- A1 policy interface
- RIC message routing

Author: FalconOne Team
Version: 3.0.0
"""

import logging
import threading
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import redis

from .e2_interface import E2Interface, E2Node, RICSubscription, E2Indication


class XAppState(Enum):
    """xApp lifecycle states"""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class XAppDescriptor:
    """xApp descriptor"""
    name: str
    version: str
    description: str
    service_models: List[str]
    config: Dict[str, Any]
    state: XAppState = XAppState.CREATED
    instance: Optional[Any] = None
    start_time: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class NearRTRIC:
    """
    Near-RT RIC (RAN Intelligent Controller)
    
    Manages xApps, E2 nodes, and provides RIC platform services
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Near-RT RIC
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # E2 interface
        self.e2_interface = E2Interface(
            ric_address=config.get('ric_address', 'localhost'),
            ric_port=config.get('ric_port', 36421),
            config=config,
            logger=logger
        )
        
        # xApps registry
        self.xapps: Dict[str, XAppDescriptor] = {}
        self.xapp_lock = threading.Lock()
        
        # E2 nodes
        self.e2_nodes: Dict[str, E2Node] = {}
        
        # Shared Data Layer (SDL) - using Redis
        self.sdl_enabled = config.get('sdl_enabled', True)
        self.sdl_client = None
        if self.sdl_enabled:
            try:
                redis_host = config.get('sdl_redis_host', 'localhost')
                redis_port = config.get('sdl_redis_port', 6379)
                self.sdl_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=config.get('sdl_redis_db', 0),
                    decode_responses=True
                )
                self.logger.info(f"SDL connected to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                self.logger.error(f"Failed to connect to SDL: {e}")
                self.sdl_enabled = False
        
        # A1 policy interface (simplified)
        self.policies: Dict[str, Dict[str, Any]] = {}
        
        # Message routing
        self.indication_routers: Dict[int, List[str]] = {}  # ran_function_id -> [xapp_names]
        
        # Statistics
        self.stats = {
            'xapps_deployed': 0,
            'e2_nodes_connected': 0,
            'subscriptions_active': 0,
            'indications_routed': 0,
            'policies_installed': 0
        }
        
        self.logger.info("Near-RT RIC initialized")
    
    def start(self) -> bool:
        """
        Start the RIC platform
        
        Returns:
            Success status
        """
        try:
            # Connect E2 interface
            if not self.e2_interface.connect():
                self.logger.error("Failed to start E2 interface")
                return False
            
            self.logger.info("Near-RT RIC started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start RIC: {e}")
            return False
    
    def stop(self):
        """Stop the RIC platform"""
        # Stop all xApps
        with self.xapp_lock:
            for xapp_name in list(self.xapps.keys()):
                self.stop_xapp(xapp_name)
        
        # Disconnect E2 interface
        self.e2_interface.disconnect()
        
        self.logger.info("Near-RT RIC stopped")
    
    def deploy_xapp(
        self,
        xapp_class: type,
        name: str,
        config: Dict[str, Any],
        service_models: List[str]
    ) -> bool:
        """
        Deploy and start an xApp
        
        Args:
            xapp_class: xApp class to instantiate
            name: xApp name
            config: xApp configuration
            service_models: Supported E2 service models
        
        Returns:
            Success status
        """
        with self.xapp_lock:
            if name in self.xapps:
                self.logger.error(f"xApp {name} already deployed")
                return False
            
            try:
                # Create descriptor
                descriptor = XAppDescriptor(
                    name=name,
                    version=config.get('version', '1.0.0'),
                    description=config.get('description', ''),
                    service_models=service_models,
                    config=config
                )
                
                # Instantiate xApp
                instance = xapp_class(
                    ric=self,
                    config=config,
                    logger=self.logger
                )
                
                descriptor.instance = instance
                descriptor.state = XAppState.STARTING
                
                # Start xApp
                instance.start()
                
                descriptor.state = XAppState.RUNNING
                descriptor.start_time = time.time()
                
                self.xapps[name] = descriptor
                self.stats['xapps_deployed'] += 1
                
                self.logger.info(f"Deployed xApp: {name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to deploy xApp {name}: {e}")
                if name in self.xapps:
                    self.xapps[name].state = XAppState.FAILED
                return False
    
    def stop_xapp(self, name: str) -> bool:
        """
        Stop an xApp
        
        Args:
            name: xApp name
        
        Returns:
            Success status
        """
        with self.xapp_lock:
            if name not in self.xapps:
                self.logger.warning(f"xApp {name} not found")
                return False
            
            descriptor = self.xapps[name]
            
            try:
                descriptor.state = XAppState.STOPPING
                
                if descriptor.instance and hasattr(descriptor.instance, 'stop'):
                    descriptor.instance.stop()
                
                descriptor.state = XAppState.STOPPED
                
                self.logger.info(f"Stopped xApp: {name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to stop xApp {name}: {e}")
                descriptor.state = XAppState.FAILED
                return False
    
    def undeploy_xapp(self, name: str) -> bool:
        """
        Undeploy an xApp
        
        Args:
            name: xApp name
        
        Returns:
            Success status
        """
        self.stop_xapp(name)
        
        with self.xapp_lock:
            if name in self.xapps:
                del self.xapps[name]
                self.stats['xapps_deployed'] -= 1
                self.logger.info(f"Undeployed xApp: {name}")
                return True
            return False
    
    def register_indication_router(
        self,
        ran_function_id: int,
        xapp_name: str
    ):
        """
        Register xApp to receive indications for a RAN function
        
        Args:
            ran_function_id: RAN Function ID
            xapp_name: xApp name
        """
        if ran_function_id not in self.indication_routers:
            self.indication_routers[ran_function_id] = []
        
        if xapp_name not in self.indication_routers[ran_function_id]:
            self.indication_routers[ran_function_id].append(xapp_name)
            self.logger.debug(
                f"Registered {xapp_name} for indications from RAN function {ran_function_id}"
            )
    
    def route_indication(self, indication: E2Indication):
        """
        Route indication to registered xApps
        
        Args:
            indication: E2 indication message
        """
        ran_function_id = indication.ran_function_id
        
        if ran_function_id not in self.indication_routers:
            self.logger.warning(
                f"No xApp registered for RAN function {ran_function_id}"
            )
            return
        
        xapp_names = self.indication_routers[ran_function_id]
        
        with self.xapp_lock:
            for xapp_name in xapp_names:
                if xapp_name in self.xapps:
                    descriptor = self.xapps[xapp_name]
                    if descriptor.instance and hasattr(descriptor.instance, 'handle_indication'):
                        try:
                            descriptor.instance.handle_indication(indication)
                            self.stats['indications_routed'] += 1
                        except Exception as e:
                            self.logger.error(
                                f"Error routing indication to {xapp_name}: {e}"
                            )
    
    # Shared Data Layer (SDL) methods
    
    def sdl_set(self, namespace: str, key: str, value: Any) -> bool:
        """
        Store data in SDL
        
        Args:
            namespace: Data namespace
            key: Key
            value: Value (will be JSON serialized)
        
        Returns:
            Success status
        """
        if not self.sdl_enabled or not self.sdl_client:
            return False
        
        try:
            full_key = f"{namespace}:{key}"
            value_json = json.dumps(value)
            self.sdl_client.set(full_key, value_json)
            return True
        except Exception as e:
            self.logger.error(f"SDL set failed: {e}")
            return False
    
    def sdl_get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Retrieve data from SDL
        
        Args:
            namespace: Data namespace
            key: Key
        
        Returns:
            Value or None if not found
        """
        if not self.sdl_enabled or not self.sdl_client:
            return None
        
        try:
            full_key = f"{namespace}:{key}"
            value_json = self.sdl_client.get(full_key)
            
            if value_json:
                return json.loads(value_json)
            return None
        except Exception as e:
            self.logger.error(f"SDL get failed: {e}")
            return None
    
    def sdl_delete(self, namespace: str, key: str) -> bool:
        """
        Delete data from SDL
        
        Args:
            namespace: Data namespace
            key: Key
        
        Returns:
            Success status
        """
        if not self.sdl_enabled or not self.sdl_client:
            return False
        
        try:
            full_key = f"{namespace}:{key}"
            self.sdl_client.delete(full_key)
            return True
        except Exception as e:
            self.logger.error(f"SDL delete failed: {e}")
            return False
    
    # A1 Policy Interface methods
    
    def install_policy(
        self,
        policy_id: str,
        policy_type: str,
        policy_payload: Dict[str, Any]
    ) -> bool:
        """
        Install A1 policy
        
        Args:
            policy_id: Policy identifier
            policy_type: Policy type
            policy_payload: Policy configuration
        
        Returns:
            Success status
        """
        try:
            policy = {
                'policy_id': policy_id,
                'policy_type': policy_type,
                'payload': policy_payload,
                'installed_at': time.time()
            }
            
            self.policies[policy_id] = policy
            self.stats['policies_installed'] += 1
            
            # Notify relevant xApps
            with self.xapp_lock:
                for xapp_name, descriptor in self.xapps.items():
                    if descriptor.instance and hasattr(descriptor.instance, 'on_policy_update'):
                        try:
                            descriptor.instance.on_policy_update(policy)
                        except Exception as e:
                            self.logger.error(
                                f"Error notifying {xapp_name} of policy update: {e}"
                            )
            
            self.logger.info(f"Installed policy: {policy_id} (type: {policy_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to install policy: {e}")
            return False
    
    def get_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Get policy by ID"""
        return self.policies.get(policy_id)
    
    def delete_policy(self, policy_id: str) -> bool:
        """Delete policy by ID"""
        if policy_id in self.policies:
            del self.policies[policy_id]
            self.logger.info(f"Deleted policy: {policy_id}")
            return True
        return False
    
    def get_xapp_status(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get xApp status
        
        Args:
            name: xApp name
        
        Returns:
            Status dictionary or None
        """
        if name not in self.xapps:
            return None
        
        descriptor = self.xapps[name]
        
        uptime = None
        if descriptor.start_time:
            uptime = time.time() - descriptor.start_time
        
        return {
            'name': descriptor.name,
            'version': descriptor.version,
            'state': descriptor.state.value,
            'service_models': descriptor.service_models,
            'uptime_seconds': uptime,
            'metrics': descriptor.metrics
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RIC platform statistics"""
        return {
            'xapps_deployed': self.stats['xapps_deployed'],
            'xapps_running': sum(
                1 for d in self.xapps.values() if d.state == XAppState.RUNNING
            ),
            'e2_nodes_connected': len(self.e2_nodes),
            'subscriptions_active': len(self.e2_interface.subscriptions),
            'indications_routed': self.stats['indications_routed'],
            'policies_installed': len(self.policies),
            'sdl_enabled': self.sdl_enabled,
            'e2_interface_stats': self.e2_interface.get_statistics()
        }
