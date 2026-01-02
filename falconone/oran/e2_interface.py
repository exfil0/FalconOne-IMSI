"""
O-RAN E2 Interface Implementation
Implements E2AP (E2 Application Protocol) for communication with Near-RT RIC

Features:
- E2 Setup, Reset, and Configuration
- Subscription Management (E2SM-KPM, E2SM-RC)
- Indication/Control Message Handling
- ASN.1 Encoding/Decoding
- Multiple E2 Service Models

Standards:
- O-RAN.WG3.E2AP-v02.03
- O-RAN.WG3.E2SM-KPM-v02.03
- O-RAN.WG3.E2SM-RC-v02.00

Author: FalconOne Team
Version: 3.0.0
"""

import socket
import struct
import threading
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import queue


class E2MessageType(Enum):
    """E2AP Message Types"""
    E2_SETUP_REQUEST = 1
    E2_SETUP_RESPONSE = 2
    E2_SETUP_FAILURE = 3
    RIC_SUBSCRIPTION_REQUEST = 4
    RIC_SUBSCRIPTION_RESPONSE = 5
    RIC_SUBSCRIPTION_FAILURE = 6
    RIC_INDICATION = 7
    RIC_CONTROL_REQUEST = 8
    RIC_CONTROL_ACKNOWLEDGE = 9
    RIC_CONTROL_FAILURE = 10
    E2_CONNECTION_UPDATE = 11
    E2_CONNECTION_UPDATE_ACK = 12
    E2_NODE_CONFIG_UPDATE = 13
    RIC_SUBSCRIPTION_DELETE_REQUEST = 14
    RIC_SUBSCRIPTION_DELETE_RESPONSE = 15


class E2ServiceModel(Enum):
    """E2 Service Models"""
    KPM = "E2SM-KPM"  # KPI Monitoring
    RC = "E2SM-RC"    # RAN Control
    NI = "E2SM-NI"    # Network Inventory
    MHO = "E2SM-MHO"  # Mobility Handover Optimization


@dataclass
class E2Node:
    """E2 Node (RAN node) information"""
    global_e2_node_id: str
    ran_function_id: int
    ran_function_revision: int
    service_models: List[str]
    plmn_id: str  # MCC+MNC
    gnb_id: Optional[str] = None
    cell_ids: List[int] = field(default_factory=list)
    connected: bool = False
    last_heartbeat: float = 0


@dataclass
class RICSubscription:
    """RIC Subscription"""
    request_id: int
    ran_function_id: int
    service_model: str
    event_triggers: Dict[str, Any]
    actions: List[Dict[str, Any]]
    report_period_ms: int
    active: bool = True
    created_at: float = field(default_factory=time.time)


@dataclass
class E2Indication:
    """E2 Indication Message"""
    request_id: int
    ran_function_id: int
    action_id: int
    indication_type: str  # 'report' or 'insert'
    indication_header: bytes
    indication_message: bytes
    timestamp: float = field(default_factory=time.time)


class E2Interface:
    """
    E2 Application Protocol Interface
    
    Provides communication between xApps and E2 nodes (RAN)
    """
    
    def __init__(
        self,
        ric_address: str = "localhost",
        ric_port: int = 36421,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize E2 interface
        
        Args:
            ric_address: Near-RT RIC address
            ric_port: E2 interface port (default: 36421)
            config: Optional configuration dictionary
            logger: Optional logger instance
        """
        self.ric_address = ric_address
        self.ric_port = ric_port
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Connection state
        self.socket = None
        self.connected = False
        self.connection_lock = threading.Lock()
        
        # E2 nodes (RAN nodes)
        self.e2_nodes: Dict[str, E2Node] = {}
        
        # Subscriptions
        self.subscriptions: Dict[int, RICSubscription] = {}
        self.next_request_id = 1
        
        # Message queues
        self.indication_queue = queue.Queue(maxsize=1000)
        self.control_response_queue = queue.Queue(maxsize=100)
        
        # Message handlers
        self.indication_handlers: Dict[int, Callable] = {}
        
        # Background threads
        self.receiver_thread = None
        self.keepalive_thread = None
        self.running = False
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'indications_processed': 0,
            'control_requests_sent': 0,
            'subscription_active': 0,
            'connection_errors': 0
        }
        
        self.logger.info(
            "E2 Interface initialized (RIC: %s:%d)",
            ric_address, ric_port
        )
    
    def connect(self) -> bool:
        """
        Establish connection to Near-RT RIC
        
        Returns:
            Success status
        """
        with self.connection_lock:
            if self.connected:
                self.logger.warning("Already connected to RIC")
                return True
            
            try:
                # Create SCTP socket (E2AP uses SCTP)
                # Note: Python's socket module doesn't natively support SCTP
                # In production, use pysctp library or implement SCTP wrapper
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(10)
                self.socket.connect((self.ric_address, self.ric_port))
                
                self.connected = True
                self.running = True
                
                # Start background threads
                self.receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
                self.receiver_thread.start()
                
                self.keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
                self.keepalive_thread.start()
                
                self.logger.info("Connected to Near-RT RIC at %s:%d", self.ric_address, self.ric_port)
                
                # Send E2 Setup Request
                self._send_e2_setup_request()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to connect to RIC: {e}")
                self.stats['connection_errors'] += 1
                self.connected = False
                return False
    
    def disconnect(self):
        """Disconnect from Near-RT RIC"""
        self.running = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        self.connected = False
        self.logger.info("Disconnected from Near-RT RIC")
    
    def register_e2_node(
        self,
        node_id: str,
        ran_function_id: int,
        service_models: List[str],
        plmn_id: str
    ) -> E2Node:
        """
        Register an E2 node (RAN node)
        
        Args:
            node_id: Global E2 Node ID
            ran_function_id: RAN Function ID
            service_models: Supported service models
            plmn_id: PLMN ID (MCC+MNC)
        
        Returns:
            E2Node object
        """
        node = E2Node(
            global_e2_node_id=node_id,
            ran_function_id=ran_function_id,
            ran_function_revision=1,
            service_models=service_models,
            plmn_id=plmn_id,
            connected=True,
            last_heartbeat=time.time()
        )
        
        self.e2_nodes[node_id] = node
        
        self.logger.info(
            "Registered E2 node: %s (RAN Function: %d, Service Models: %s)",
            node_id, ran_function_id, service_models
        )
        
        return node
    
    def subscribe(
        self,
        ran_function_id: int,
        service_model: str,
        event_triggers: Dict[str, Any],
        actions: List[Dict[str, Any]],
        report_period_ms: int = 1000,
        indication_handler: Optional[Callable] = None
    ) -> Optional[RICSubscription]:
        """
        Create RIC subscription
        
        Args:
            ran_function_id: RAN Function ID
            service_model: Service model (E2SM-KPM, E2SM-RC, etc.)
            event_triggers: Event trigger conditions
            actions: List of actions to perform
            report_period_ms: Reporting period in milliseconds
            indication_handler: Optional callback for indications
        
        Returns:
            RICSubscription object or None on failure
        """
        if not self.connected:
            self.logger.error("Not connected to RIC")
            return None
        
        request_id = self.next_request_id
        self.next_request_id += 1
        
        subscription = RICSubscription(
            request_id=request_id,
            ran_function_id=ran_function_id,
            service_model=service_model,
            event_triggers=event_triggers,
            actions=actions,
            report_period_ms=report_period_ms
        )
        
        # Register indication handler
        if indication_handler:
            self.indication_handlers[request_id] = indication_handler
        
        # Send subscription request
        success = self._send_subscription_request(subscription)
        
        if success:
            self.subscriptions[request_id] = subscription
            self.stats['subscription_active'] += 1
            
            self.logger.info(
                "Created subscription %d (RAN Function: %d, Model: %s)",
                request_id, ran_function_id, service_model
            )
            
            return subscription
        else:
            self.logger.error("Failed to create subscription %d", request_id)
            return None
    
    def unsubscribe(self, request_id: int) -> bool:
        """
        Delete RIC subscription
        
        Args:
            request_id: Subscription request ID
        
        Returns:
            Success status
        """
        if request_id not in self.subscriptions:
            self.logger.warning(f"Subscription {request_id} not found")
            return False
        
        subscription = self.subscriptions[request_id]
        
        # Send deletion request
        success = self._send_subscription_delete(request_id, subscription.ran_function_id)
        
        if success:
            del self.subscriptions[request_id]
            if request_id in self.indication_handlers:
                del self.indication_handlers[request_id]
            
            self.stats['subscription_active'] -= 1
            
            self.logger.info("Deleted subscription %d", request_id)
            return True
        else:
            self.logger.error("Failed to delete subscription %d", request_id)
            return False
    
    def send_control(
        self,
        ran_function_id: int,
        control_header: Dict[str, Any],
        control_message: Dict[str, Any],
        ack_request: bool = True
    ) -> bool:
        """
        Send RIC Control Request
        
        Args:
            ran_function_id: RAN Function ID
            control_header: Control header information
            control_message: Control message payload
            ack_request: Whether to request acknowledgment
        
        Returns:
            Success status
        """
        if not self.connected:
            self.logger.error("Not connected to RIC")
            return False
        
        try:
            # Encode control message (simplified - actual implementation uses ASN.1)
            message = {
                'type': E2MessageType.RIC_CONTROL_REQUEST.name,
                'ran_function_id': ran_function_id,
                'control_header': control_header,
                'control_message': control_message,
                'ack_request': ack_request
            }
            
            self._send_message(message)
            self.stats['control_requests_sent'] += 1
            
            if ack_request:
                # Wait for acknowledgment (simplified)
                try:
                    response = self.control_response_queue.get(timeout=5.0)
                    return response.get('status') == 'success'
                except queue.Empty:
                    self.logger.warning("Control request timed out")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send control request: {e}")
            return False
    
    def get_indications(self, timeout: float = 1.0) -> List[E2Indication]:
        """
        Get received indications
        
        Args:
            timeout: Timeout in seconds
        
        Returns:
            List of E2Indication objects
        """
        indications = []
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                
                indication = self.indication_queue.get(timeout=remaining)
                indications.append(indication)
            except queue.Empty:
                break
        
        return indications
    
    def _send_e2_setup_request(self):
        """Send E2 Setup Request"""
        message = {
            'type': E2MessageType.E2_SETUP_REQUEST.name,
            'global_e2_node_id': 'FalconOne-E2Node-001',
            'ran_functions': [
                {
                    'ran_function_id': 1,
                    'ran_function_revision': 1,
                    'service_model': 'E2SM-KPM'
                },
                {
                    'ran_function_id': 2,
                    'ran_function_revision': 1,
                    'service_model': 'E2SM-RC'
                }
            ]
        }
        
        self._send_message(message)
        self.logger.debug("Sent E2 Setup Request")
    
    def _send_subscription_request(self, subscription: RICSubscription) -> bool:
        """Send RIC Subscription Request"""
        try:
            message = {
                'type': E2MessageType.RIC_SUBSCRIPTION_REQUEST.name,
                'request_id': subscription.request_id,
                'ran_function_id': subscription.ran_function_id,
                'service_model': subscription.service_model,
                'event_triggers': subscription.event_triggers,
                'actions': subscription.actions,
                'report_period_ms': subscription.report_period_ms
            }
            
            self._send_message(message)
            self.logger.debug(f"Sent subscription request {subscription.request_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send subscription request: {e}")
            return False
    
    def _send_subscription_delete(self, request_id: int, ran_function_id: int) -> bool:
        """Send RIC Subscription Delete Request"""
        try:
            message = {
                'type': E2MessageType.RIC_SUBSCRIPTION_DELETE_REQUEST.name,
                'request_id': request_id,
                'ran_function_id': ran_function_id
            }
            
            self._send_message(message)
            self.logger.debug(f"Sent subscription delete request {request_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send subscription delete: {e}")
            return False
    
    def _send_message(self, message: Dict[str, Any]):
        """Send E2AP message"""
        if not self.socket:
            raise ConnectionError("Not connected to RIC")
        
        # Serialize message (simplified - actual implementation uses ASN.1 PER encoding)
        message_json = json.dumps(message)
        message_bytes = message_json.encode('utf-8')
        
        # Send length prefix + message
        length = len(message_bytes)
        self.socket.sendall(struct.pack('!I', length) + message_bytes)
        
        self.stats['messages_sent'] += 1
    
    def _receive_loop(self):
        """Background thread for receiving messages"""
        while self.running and self.connected:
            try:
                # Receive length prefix
                length_data = self.socket.recv(4)
                if not length_data:
                    self.logger.warning("Connection closed by RIC")
                    self.connected = False
                    break
                
                length = struct.unpack('!I', length_data)[0]
                
                # Receive message
                message_bytes = b''
                while len(message_bytes) < length:
                    chunk = self.socket.recv(min(length - len(message_bytes), 4096))
                    if not chunk:
                        break
                    message_bytes += chunk
                
                # Deserialize and handle
                message = json.loads(message_bytes.decode('utf-8'))
                self._handle_message(message)
                
                self.stats['messages_received'] += 1
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self.logger.error(f"Error in receive loop: {e}")
                    self.stats['connection_errors'] += 1
                break
    
    def _handle_message(self, message: Dict[str, Any]):
        """Handle received E2AP message"""
        msg_type = message.get('type')
        
        if msg_type == E2MessageType.E2_SETUP_RESPONSE.name:
            self.logger.info("Received E2 Setup Response")
        
        elif msg_type == E2MessageType.RIC_INDICATION.name:
            self._handle_indication(message)
        
        elif msg_type == E2MessageType.RIC_CONTROL_ACKNOWLEDGE.name:
            self.control_response_queue.put({'status': 'success', 'message': message})
        
        elif msg_type == E2MessageType.RIC_CONTROL_FAILURE.name:
            self.control_response_queue.put({'status': 'failure', 'message': message})
        
        elif msg_type == E2MessageType.RIC_SUBSCRIPTION_RESPONSE.name:
            self.logger.info(f"Subscription {message.get('request_id')} confirmed")
        
        elif msg_type == E2MessageType.RIC_SUBSCRIPTION_FAILURE.name:
            self.logger.error(f"Subscription {message.get('request_id')} failed")
        
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")
    
    def _handle_indication(self, message: Dict[str, Any]):
        """Handle RIC Indication"""
        request_id = message.get('request_id')
        
        indication = E2Indication(
            request_id=request_id,
            ran_function_id=message.get('ran_function_id', 0),
            action_id=message.get('action_id', 0),
            indication_type=message.get('indication_type', 'report'),
            indication_header=message.get('indication_header', b''),
            indication_message=message.get('indication_message', b'')
        )
        
        # Call registered handler if exists
        if request_id in self.indication_handlers:
            try:
                self.indication_handlers[request_id](indication)
            except Exception as e:
                self.logger.error(f"Error in indication handler: {e}")
        
        # Add to queue
        try:
            self.indication_queue.put_nowait(indication)
            self.stats['indications_processed'] += 1
        except queue.Full:
            self.logger.warning("Indication queue full, dropping message")
    
    def _keepalive_loop(self):
        """Background thread for keepalive/heartbeat"""
        while self.running and self.connected:
            time.sleep(30)  # Send keepalive every 30 seconds
            
            if self.connected:
                try:
                    # Send simple keepalive (could be E2 Connection Update)
                    message = {
                        'type': 'E2_KEEPALIVE',
                        'timestamp': time.time()
                    }
                    self._send_message(message)
                except Exception as e:
                    self.logger.error(f"Keepalive failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get E2 interface statistics"""
        return {
            'connected': self.connected,
            'e2_nodes': len(self.e2_nodes),
            'active_subscriptions': len(self.subscriptions),
            'messages_sent': self.stats['messages_sent'],
            'messages_received': self.stats['messages_received'],
            'indications_processed': self.stats['indications_processed'],
            'control_requests_sent': self.stats['control_requests_sent'],
            'connection_errors': self.stats['connection_errors']
        }
