"""
FalconOne O-RAN RIC xApp Framework (v3.0)
Comprehensive xApp framework for intelligent RAN control, monitoring, and security analysis

Features:
- E2 interface integration (E2AP v3.0)
- E2SM-KPM subscription for KPI monitoring
- E2SM-RC control for RAN control
- Anomaly detection
- Traffic steering
- Resource optimization
- Security monitoring

References:
- O-RAN E2AP v3.0 specification
- O-RAN WG3 E2SM-KPM v3.0
- O-RAN WG2 Near-RT RIC architecture
- srsRAN/OSC RIC integration

Version: 3.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import threading
import time
import hashlib
from abc import ABC, abstractmethod

try:
    from ..utils.logger import ModuleLogger
except ImportError:
    class ModuleLogger:
        def __init__(self, name, parent):
            self.logger = logging.getLogger(name) if parent else logging.getLogger(__name__)
        def info(self, msg, **kw): self.logger.info(f"{msg} {kw if kw else ''}")
        def warning(self, msg, **kw): self.logger.warning(f"{msg} {kw if kw else ''}")
        def error(self, msg, **kw): self.logger.error(f"{msg} {kw if kw else ''}")
        def debug(self, msg, **kw): self.logger.debug(f"{msg} {kw if kw else ''}")


@dataclass
class E2NodeInfo:
    """E2 Node (gNB) information"""
    node_id: str
    plmn_id: str
    cell_ids: List[int]
    supported_functions: List[str] = field(default_factory=list)
    connected: bool = False
    last_heartbeat: float = field(default_factory=time.time)


@dataclass
class KPIReport:
    """KPI report from E2SM-KPM"""
    node_id: str
    cell_id: int
    timestamp: datetime
    prb_utilization_dl: float
    prb_utilization_ul: float
    active_ues: int
    throughput_dl_mbps: float
    throughput_ul_mbps: float
    avg_cqi: float
    handover_success_rate: float
    latency_ms: float = 0.0
    packet_loss_rate: float = 0.0


@dataclass
class HandoverDecision:
    """RAN control handover decision"""
    source_cell: int
    target_cell: int
    ue_rnti: int
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: Optional[bool] = None


@dataclass
class AnomalyAlert:
    """Security/performance anomaly alert"""
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    cell_id: int
    description: str
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


class XAppBase(ABC):
    """
    Base class for O-RAN xApps
    
    All xApps should inherit from this class and implement:
    - on_start(): Initialization logic
    - on_kpi_indication(): Handle KPI reports
    - on_policy_update(): Handle A1 policy updates
    - on_stop(): Cleanup logic
    """
    
    def __init__(self, ric: Any, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize xApp
        
        Args:
            ric: Near-RT RIC instance
            config: xApp configuration
            logger: Optional logger
        """
        self.ric = ric
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        self.running = False
        self.thread = None
    
    def start(self):
        """Start xApp"""
        self.running = True
        self.on_start()
        
        # Start main loop in background thread
        self.thread = threading.Thread(target=self._main_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop xApp"""
        self.running = False
        self.on_stop()
        
        if self.thread:
            self.thread.join(timeout=5.0)
    
    def _main_loop(self):
        """Main xApp loop (runs in background thread)"""
        while self.running:
            try:
                self.on_iteration()
                time.sleep(self.config.get('iteration_interval', 1.0))
            except Exception as e:
                self.logger.error(f"Error in xApp main loop: {e}")
    
    @abstractmethod
    def on_start(self):
        """Called when xApp starts"""
        pass
    
    @abstractmethod
    def on_stop(self):
        """Called when xApp stops"""
        pass
    
    @abstractmethod
    def on_iteration(self):
        """Called periodically in main loop"""
        pass
    
    @abstractmethod
    def on_kpi_indication(self, kpi: KPIReport):
        """
        Handle KPI indication from E2SM-KPM
        
        Args:
            kpi: KPI report
        """
        pass
    
    @abstractmethod
    def on_policy_update(self, policy: Dict[str, Any]):
        """
        Handle A1 policy update
        
        Args:
            policy: Policy configuration
        """
        pass
    
    def handle_indication(self, indication: Any):
        """
        Handle E2 indication (called by RIC)
        
        Args:
            indication: E2 indication message
        """
        # Parse indication and route to appropriate handler
        # This is a simplified implementation
        pass


class TrafficSteeringXApp(XAppBase):
    """
    xApp for intelligent traffic steering and load balancing
    """
    
    def __init__(self, ric: Any, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(ric, config, logger)
        
        self.cell_loads: Dict[int, float] = {}
        self.steering_threshold = config.get('steering_threshold', 0.7)
    
    def on_start(self):
        self.logger.info("Traffic Steering xApp started")
        
        # Subscribe to KPM for all cells
        # self.ric.e2_interface.subscribe(...)
    
    def on_stop(self):
        self.logger.info("Traffic Steering xApp stopped")
    
    def on_iteration(self):
        # Check cell loads and make steering decisions
        overloaded_cells = [
            cell_id for cell_id, load in self.cell_loads.items()
            if load > self.steering_threshold
        ]
        
        if overloaded_cells:
            self.logger.info(f"Overloaded cells detected: {overloaded_cells}")
            # Trigger steering actions
    
    def on_kpi_indication(self, kpi: KPIReport):
        # Update cell load
        self.cell_loads[kpi.cell_id] = kpi.prb_utilization_dl
        
        # Check if steering is needed
        if kpi.prb_utilization_dl > self.steering_threshold:
            self._perform_traffic_steering(kpi.cell_id)
    
    def on_policy_update(self, policy: Dict[str, Any]):
        if 'steering_threshold' in policy:
            self.steering_threshold = policy['steering_threshold']
            self.logger.info(f"Updated steering threshold to {self.steering_threshold}")
    
    def _perform_traffic_steering(self, cell_id: int):
        """Steer traffic away from overloaded cell"""
        self.logger.info(f"Performing traffic steering for cell {cell_id}")
        
        # Find neighbor cell with lowest load
        neighbor_cells = [cid for cid in self.cell_loads.keys() if cid != cell_id]
        
        if not neighbor_cells:
            return
        
        target_cell = min(neighbor_cells, key=lambda c: self.cell_loads.get(c, 1.0))
        
        # Send RIC control to steer UEs
        # self.ric.e2_interface.send_control(...)


class AnomalyDetectionXApp(XAppBase):
    """
    xApp for detecting network anomalies and security threats
    """
    
    def __init__(self, ric: Any, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(ric, config, logger)
        
        self.kpi_history: Dict[int, List[KPIReport]] = {}
        self.anomaly_threshold = config.get('anomaly_threshold', 2.0)  # Standard deviations
        self.alerts: List[AnomalyAlert] = []
    
    def on_start(self):
        self.logger.info("Anomaly Detection xApp started")
    
    def on_stop(self):
        self.logger.info("Anomaly Detection xApp stopped")
    
    def on_iteration(self):
        # Periodic anomaly analysis
        for cell_id, history in self.kpi_history.items():
            if len(history) >= 10:
                self._detect_anomalies(cell_id, history)
    
    def on_kpi_indication(self, kpi: KPIReport):
        # Store KPI in history
        if kpi.cell_id not in self.kpi_history:
            self.kpi_history[kpi.cell_id] = []
        
        self.kpi_history[kpi.cell_id].append(kpi)
        
        # Keep last 100 reports
        if len(self.kpi_history[kpi.cell_id]) > 100:
            self.kpi_history[kpi.cell_id].pop(0)
        
        # Immediate anomaly check
        if len(self.kpi_history[kpi.cell_id]) >= 10:
            self._detect_anomalies(kpi.cell_id, self.kpi_history[kpi.cell_id])
    
    def on_policy_update(self, policy: Dict[str, Any]):
        if 'anomaly_threshold' in policy:
            self.anomaly_threshold = policy['anomaly_threshold']
    
    def _detect_anomalies(self, cell_id: int, history: List[KPIReport]):
        """Detect statistical anomalies in KPIs"""
        recent = history[-10:]
        
        # Check throughput anomalies
        throughputs = [kpi.throughput_dl_mbps for kpi in recent]
        mean_throughput = np.mean(throughputs)
        std_throughput = np.std(throughputs)
        
        latest = recent[-1]
        
        if std_throughput > 0:
            z_score = abs(latest.throughput_dl_mbps - mean_throughput) / std_throughput
            
            if z_score > self.anomaly_threshold:
                alert = AnomalyAlert(
                    alert_type='throughput_anomaly',
                    severity='high' if z_score > 3.0 else 'medium',
                    cell_id=cell_id,
                    description=f"Abnormal throughput: {latest.throughput_dl_mbps:.2f} Mbps (z-score: {z_score:.2f})",
                    metrics={'z_score': z_score, 'throughput': latest.throughput_dl_mbps}
                )
                
                self.alerts.append(alert)
                self.logger.warning(f"Anomaly detected: {alert.description}")
        
        # Check PRB utilization spikes
        prb_utils = [kpi.prb_utilization_dl for kpi in recent]
        if max(prb_utils) - min(prb_utils) > 0.5:  # >50% variation
            alert = AnomalyAlert(
                alert_type='prb_spike',
                severity='medium',
                cell_id=cell_id,
                description=f"PRB utilization spike detected (range: {min(prb_utils):.2f}-{max(prb_utils):.2f})",
                metrics={'min_prb': min(prb_utils), 'max_prb': max(prb_utils)}
            )
            
            self.alerts.append(alert)
            self.logger.warning(f"Anomaly detected: {alert.description}")


class ResourceOptimizationXApp(XAppBase):
    """
    xApp for RAN resource optimization using ML
    """
    
    def __init__(self, ric: Any, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(ric, config, logger)
        
        self.dqn_agent = None
        self._init_dqn()
    
    def _init_dqn(self):
        """Initialize DQN agent"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(10,)),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(8, activation='linear')
            ])
            
            model.compile(optimizer='adam', loss='mse')
            self.dqn_agent = model
            
            self.logger.info("DQN agent initialized")
            
        except ImportError:
            self.logger.warning("TensorFlow not available, DQN disabled")
    
    def on_start(self):
        self.logger.info("Resource Optimization xApp started")
    
    def on_stop(self):
        self.logger.info("Resource Optimization xApp stopped")
    
    def on_iteration(self):
        # Periodic optimization
        pass
    
    def on_kpi_indication(self, kpi: KPIReport):
        if self.dqn_agent:
            # Extract features
            features = np.array([
                kpi.prb_utilization_dl,
                kpi.prb_utilization_ul,
                kpi.active_ues / 100.0,  # Normalize
                kpi.throughput_dl_mbps / 1000.0,
                kpi.avg_cqi / 15.0,
                0, 0, 0, 0, 0  # Placeholder for neighbor metrics
            ])
            
            # Get action from DQN
            action = self.dqn_agent.predict(features.reshape(1, -1), verbose=0)[0]
            best_action = np.argmax(action)
            
            # Apply action (simplified)
            if best_action > 0:
                self.logger.debug(f"DQN suggests action {best_action} for cell {kpi.cell_id}")
    
    def on_policy_update(self, policy: Dict[str, Any]):
        pass


class RICxAppController:
    """
    O-RAN Near-RT RIC xApp Controller (Legacy/Compatibility)
    Use XAppBase for new xApp development
    """
    
    def __init__(self, ric_endpoint: str, config, logger=None):
        self.ric_endpoint = ric_endpoint
        self.config = config
        self.logger = ModuleLogger('O-RAN-xApp', logger)
        
        # E2 nodes (gNBs)
        self.e2_nodes: Dict[str, E2NodeInfo] = {}
        
        # KPI database
        self.kpi_history: List[KPIReport] = []
        
        # Handover decisions
        self.handover_decisions: List[HandoverDecision] = []
        
        # DQN agent for intelligent control
        self.dqn_agent = self._init_dqn_agent()
        
        self.logger.info(f"RIC xApp initialized, endpoint: {ric_endpoint}")
    
    def _init_dqn_agent(self):
        """
        Initialize DQN agent for handover optimization
        State: [PRB util, throughput, CQI, UE count, neighbor RSRP]
        Action: [No handover, handover to neighbor 1, 2, 3, ...]
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # Simple DQN model
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(10,)),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(8, activation='linear'),  # 8 actions: no HO + 7 neighbors
            ])
            
            model.compile(optimizer='adam', loss='mse')
            return model
            
        except ImportError:
            self.logger.warning("TensorFlow not available, DQN disabled")
            return None
    
    def connect_e2_node(self, node_id: str, plmn_id: str, cell_ids: List[int]) -> bool:
        """
        Establish E2 connection with gNB
        In production: uses SCTP + E2AP
        """
        try:
            node = E2NodeInfo(
                node_id=node_id,
                plmn_id=plmn_id,
                cell_ids=cell_ids,
                supported_functions=['E2SM-KPM', 'E2SM-RC'],
                connected=True
            )
            
            self.e2_nodes[node_id] = node
            
            # Subscribe to E2SM-KPM for KPI monitoring
            self._subscribe_kpm(node_id)
            
            self.logger.info(f"E2 node connected: {node_id}, cells: {cell_ids}")
            return True
            
        except Exception as e:
            self.logger.error(f"E2 connection failed: {e}")
            return False
    
    def _subscribe_kpm(self, node_id: str):
        """
        Subscribe to E2SM-KPM for periodic KPI reports
        Real implementation: E2AP Subscription Request
        """
        subscription = {
            'node_id': node_id,
            'ran_function_id': 2,  # E2SM-KPM
            'reporting_period_ms': 1000,
            'kpis': [
                'DRB.UEThpDl',
                'DRB.UEThpUl',
                'RRU.PrbUsedDl',
                'RRU.PrbUsedUl',
                'L3.ActiveUEs',
                'QosFlow.PdcpPduVolumeDl',
            ]
        }
        
        # In production: send via E2AP
        self.logger.info(f"Subscribed to E2SM-KPM for node {node_id}")
    
    def process_kpi_indication(self, node_id: str, cell_id: int, kpi_data: Dict[str, float]):
        """
        Process KPI indication message from E2SM-KPM
        
        Args:
            node_id: Source E2 node
            cell_id: Cell ID
            kpi_data: Dictionary of KPI values
        """
        try:
            report = KPIReport(
                node_id=node_id,
                cell_id=cell_id,
                timestamp=datetime.now(),
                prb_utilization_dl=kpi_data.get('prb_util_dl', 0.0),
                prb_utilization_ul=kpi_data.get('prb_util_ul', 0.0),
                active_ues=int(kpi_data.get('active_ues', 0)),
                throughput_dl_mbps=kpi_data.get('throughput_dl', 0.0),
                throughput_ul_mbps=kpi_data.get('throughput_ul', 0.0),
                avg_cqi=kpi_data.get('avg_cqi', 0.0),
                handover_success_rate=kpi_data.get('ho_success_rate', 1.0),
            )
            
            self.kpi_history.append(report)
            
            # Trigger intelligent control if needed
            if self.dqn_agent:
                self._evaluate_control_action(report)
            
        except Exception as e:
            self.logger.error(f"KPI processing failed: {e}")
    
    def _evaluate_control_action(self, report: KPIReport):
        """
        Use DQN to evaluate if control action needed
        Examples:
        - Handover if PRB utilization > 90%
        - Load balancing across cells
        - Attack: Force unnecessary handovers for DoS
        """
        # Build state vector
        state = np.array([
            report.prb_utilization_dl,
            report.prb_utilization_ul,
            report.throughput_dl_mbps / 1000.0,  # Normalize
            report.throughput_ul_mbps / 1000.0,
            report.active_ues / 100.0,  # Normalize
            report.avg_cqi / 15.0,  # CQI 0-15
            report.handover_success_rate,
            0.0, 0.0, 0.0  # Placeholder for neighbor RSRP
        ])
        
        # Get DQN action
        q_values = self.dqn_agent.predict(state.reshape(1, -1), verbose=0)[0]
        action = np.argmax(q_values)
        
        # Action 0: No handover
        # Action 1-7: Handover to neighbor cells
        if action > 0:
            target_cell = report.cell_id + action  # Simplified neighbor mapping
            self._trigger_handover(report.node_id, report.cell_id, target_cell, "DQN optimization")
    
    def _trigger_handover(self, node_id: str, source_cell: int, target_cell: int, reason: str):
        """
        Trigger handover via E2SM-RC
        Real implementation: E2AP RAN Control Request
        """
        # In production: send E2SM-RC Control Request with handover command
        decision = HandoverDecision(
            source_cell=source_cell,
            target_cell=target_cell,
            ue_rnti=0,  # Placeholder - real UE RNTI from tracking
            reason=reason
        )
        
        self.handover_decisions.append(decision)
        self.logger.info(f"Handover triggered: Cell {source_cell} -> {target_cell} ({reason})")
    
    def manipulate_handover_parameters(self, node_id: str, cell_id: int, 
                                       new_hysteresis_db: float, new_time_to_trigger_ms: int) -> bool:
        """
        Manipulate handover parameters for attack scenarios
        - Increase hysteresis: Prevent legitimate handovers (DoS)
        - Decrease hysteresis: Cause handover storms
        - Modify TTT: Trigger premature/delayed handovers
        
        Args:
            node_id: Target E2 node
            cell_id: Target cell
            new_hysteresis_db: New handover hysteresis (dB)
            new_time_to_trigger_ms: New time-to-trigger (ms)
        """
        try:
            # Build E2SM-RC Control message
            control_message = {
                'node_id': node_id,
                'cell_id': cell_id,
                'ran_function_id': 3,  # E2SM-RC
                'control_action': 'ModifyHandoverConfig',
                'parameters': {
                    'hysteresis_db': new_hysteresis_db,
                    'time_to_trigger_ms': new_time_to_trigger_ms,
                }
            }
            
            # In production: send via E2AP Control Request
            self.logger.warning(f"Handover params modified for cell {cell_id}: "
                              f"Hysteresis={new_hysteresis_db}dB, TTT={new_time_to_trigger_ms}ms")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Handover manipulation failed: {e}")
            return False
    
    def inject_false_kpis(self, node_id: str, cell_id: int, fake_kpis: Dict[str, float]) -> bool:
        """
        Inject false KPIs into RIC data plane
        Attack: Trigger wrong control decisions by poisoning KPI stream
        
        Args:
            node_id: Target node
            cell_id: Target cell
            fake_kpis: Fake KPI values (e.g., high PRB util to trigger handovers)
        """
        try:
            # Craft fake KPI indication message
            self.process_kpi_indication(node_id, cell_id, fake_kpis)
            
            self.logger.warning(f"False KPIs injected for cell {cell_id}: {fake_kpis}")
            return True
            
        except Exception as e:
            self.logger.error(f"KPI injection failed: {e}")
            return False
    
    def force_handover_storm(self, target_cell: int, duration_sec: int = 60) -> bool:
        """
        Create handover storm for DoS attack
        Attack: Continuously trigger handovers between cells
        
        Args:
            target_cell: Initial target cell
            duration_sec: Attack duration
        """
        try:
            import time
            
            start_time = time.time()
            neighbor_cells = [target_cell + i for i in range(1, 4)]
            current_idx = 0
            
            while time.time() - start_time < duration_sec:
                source = target_cell if current_idx == 0 else neighbor_cells[(current_idx - 1) % len(neighbor_cells)]
                target = neighbor_cells[current_idx % len(neighbor_cells)]
                
                self._trigger_handover('attack_node', source, target, "Forced handover storm")
                
                current_idx += 1
                time.sleep(0.5)  # Handover every 500ms
            
            self.logger.warning(f"Handover storm completed: {current_idx} handovers in {duration_sec}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Handover storm failed: {e}")
            return False
    
    def query_ue_context(self, ue_rnti: int) -> Optional[Dict[str, Any]]:
        """
        Query UE context via E2SM-RC
        Returns: UE state, QoS flows, bearers
        """
        # In production: E2AP UE Context Request
        return {
            'rnti': ue_rnti,
            'state': 'RRC_CONNECTED',
            'serving_cell': 1,
            'qos_flows': [
                {'qfi': 1, '5qi': 9, 'gfbr_mbps': 10.0},
                {'qfi': 2, '5qi': 5, 'gfbr_mbps': 50.0},
            ],
            'drbs': [{'drb_id': 1, 'pdcp_sn': 12345}],
        }
    
    def get_cell_statistics(self, cell_id: int) -> Dict[str, Any]:
        """Get aggregated statistics for a cell"""
        cell_reports = [r for r in self.kpi_history if r.cell_id == cell_id]
        
        if not cell_reports:
            return {'error': 'No data for cell'}
        
        return {
            'cell_id': cell_id,
            'total_reports': len(cell_reports),
            'avg_prb_util_dl': np.mean([r.prb_utilization_dl for r in cell_reports]),
            'avg_prb_util_ul': np.mean([r.prb_utilization_ul for r in cell_reports]),
            'avg_throughput_dl_mbps': np.mean([r.throughput_dl_mbps for r in cell_reports]),
            'avg_active_ues': np.mean([r.active_ues for r in cell_reports]),
            'handover_success_rate': np.mean([r.handover_success_rate for r in cell_reports]),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall RIC xApp statistics"""
        return {
            'connected_e2_nodes': len(self.e2_nodes),
            'total_cells': sum(len(node.cell_ids) for node in self.e2_nodes.values()),
            'total_kpi_reports': len(self.kpi_history),
            'total_handover_decisions': len(self.handover_decisions),
            'dqn_enabled': self.dqn_agent is not None,
        }
    
    # ===== O-RAN/NTN Interoperability Boost (v1.6) =====
    
    def subscribe_e2sm_ni(self, node_id: str, reporting_period_ms: int = 1000) -> bool:
        """
        Subscribe to E2SM-NI (Network Intelligence) service model
        Enables AI/ML model deployment and inference control via RIC
        
        Args:
            node_id: E2 node identifier
            reporting_period_ms: Reporting period for NI indications
        
        Returns:
            True if subscription successful
        
        E2SM-NI Features:
        - Model deployment to gNB
        - Inference triggering and data collection
        - Performance monitoring (inference latency, accuracy)
        - Model updates and retraining signals
        
        Reference: O-RAN.WG3.E2SM-NI-v01.00
        """
        if node_id not in self.e2_nodes:
            self.logger.error(f"E2 node {node_id} not connected")
            return False
        
        try:
            # E2SM-NI subscription request (simulated)
            ni_subscription = {
                'service_model': 'E2SM-NI',
                'version': '1.0.0',
                'node_id': node_id,
                'reporting_period_ms': reporting_period_ms,
                'event_triggers': [
                    'MODEL_INFERENCE_COMPLETE',
                    'MODEL_PERFORMANCE_DEGRADATION',
                    'TRAINING_DATA_AVAILABLE',
                ],
                'actions': [
                    {'action_id': 1, 'action_type': 'DEPLOY_MODEL'},
                    {'action_id': 2, 'action_type': 'TRIGGER_INFERENCE'},
                    {'action_id': 3, 'action_type': 'UPDATE_MODEL'},
                ],
            }
            
            # Store subscription
            if not hasattr(self, 'ni_subscriptions'):
                self.ni_subscriptions = {}
            
            self.ni_subscriptions[node_id] = ni_subscription
            
            self.logger.info(f"E2SM-NI subscription active",
                           node=node_id,
                           period_ms=reporting_period_ms,
                           events=len(ni_subscription['event_triggers']))
            
            return True
            
        except Exception as e:
            self.logger.error(f"E2SM-NI subscription failed: {e}")
            return False
    
    def deploy_model_to_gnb(self, node_id: str, model_type: str, 
                            model_data: bytes) -> Dict[str, Any]:
        """
        Deploy AI/ML model to gNB via E2SM-NI
        
        Args:
            node_id: Target E2 node
            model_type: 'tflite', 'onnx', 'torchscript'
            model_data: Serialized model bytes
        
        Returns:
            Deployment status
        
        Use cases:
        - Edge inference for signal classification
        - Real-time beam prediction
        - Resource allocation optimization
        - Anomaly detection at gNB
        """
        if node_id not in self.e2_nodes:
            return {'success': False, 'error': 'node_not_found'}
        
        try:
            model_size_kb = len(model_data) / 1024
            
            # Validate model size (gNB memory constraints)
            max_size_mb = 50  # Typical gNB model size limit
            if model_size_kb > max_size_mb * 1024:
                self.logger.error(f"Model too large: {model_size_kb:.1f}KB (max: {max_size_mb}MB)")
                return {'success': False, 'error': 'model_too_large'}
            
            # E2SM-NI Control Request: DEPLOY_MODEL (simulated)
            deployment_result = {
                'success': True,
                'node_id': node_id,
                'model_type': model_type,
                'model_size_kb': model_size_kb,
                'deployment_time_ms': np.random.uniform(500, 2000),  # Simulated
                'model_id': hashlib.md5(model_data).hexdigest()[:8],
            }
            
            self.logger.info(f"Model deployed to gNB",
                           node=node_id,
                           type=model_type,
                           size_kb=f"{model_size_kb:.1f}",
                           model_id=deployment_result['model_id'])
            
            return deployment_result
            
        except Exception as e:
            self.logger.error(f"Model deployment failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def control_ntn_handover(self, ue_id: str, target_satellite_id: int, 
                            beam_id: int = None) -> Dict[str, Any]:
        """
        Control NTN (Non-Terrestrial Network) handover via RIC
        Handles satellite-to-satellite and satellite-to-terrestrial handovers
        
        Args:
            ue_id: UE identifier
            target_satellite_id: Target LEO/GEO satellite ID
            beam_id: Target beam ID (for multi-beam satellites)
        
        Returns:
            Handover execution result
        
        NTN Challenges:
        - High Doppler shift (±7 kHz for LEO)
        - Long propagation delay (25-270ms for LEO/GEO)
        - Frequent handovers (LEO visibility ~5-15 min)
        - Timing advance adjustments
        
        Target: +25% handover success rate vs baseline
        """
        try:
            # Calculate NTN-specific parameters
            satellite_altitude_km = 550 if target_satellite_id < 1000 else 35786  # LEO vs GEO
            propagation_delay_ms = self._calculate_ntn_delay(satellite_altitude_km)
            doppler_khz = self._calculate_ntn_doppler(satellite_altitude_km, ue_id)
            
            # Timing advance calculation (3GPP TS 38.213)
            timing_advance_us = propagation_delay_ms * 2 * 1000  # Round-trip
            
            # Handover preparation with NTN compensation
            handover_params = {
                'ue_id': ue_id,
                'target_satellite': target_satellite_id,
                'target_beam': beam_id,
                'propagation_delay_ms': propagation_delay_ms,
                'doppler_shift_khz': doppler_khz,
                'timing_advance_us': timing_advance_us,
                'frequency_offset_hz': doppler_khz * 1000,
            }
            
            # Pre-compensate Doppler before handover
            self._apply_doppler_compensation(ue_id, doppler_khz)
            
            # Execute handover via E2SM-RC
            handover_success = np.random.random() < 0.90  # 90% success with RIC control
            
            result = {
                'success': handover_success,
                'ue_id': ue_id,
                'target_satellite': target_satellite_id,
                'handover_latency_ms': propagation_delay_ms + np.random.uniform(50, 150),
                'doppler_compensated': True,
                'timing_advance_applied': True,
                **handover_params,
            }
            
            if handover_success:
                self.logger.info(f"NTN handover successful",
                               ue=ue_id,
                               satellite=target_satellite_id,
                               delay_ms=f"{propagation_delay_ms:.1f}",
                               doppler_khz=f"{doppler_khz:.2f}")
            else:
                self.logger.warning(f"NTN handover failed", ue=ue_id, satellite=target_satellite_id)
            
            return result
            
        except Exception as e:
            self.logger.error(f"NTN handover control failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def integrate_oai_ntn(self, oai_config: Dict[str, Any]) -> bool:
        """
        Integrate with OpenAirInterface (OAI) Rel-20 NTN branch
        Enables real NTN handover manipulation with OAI gNB/UE
        
        Args:
            oai_config: OAI configuration {gnb_ip, ue_ip, ntn_enabled}
        
        Returns:
            True if integration successful
        
        OAI NTN Features (Rel-20):
        - LEO/GEO satellite channel models
        - Doppler compensation
        - Timing advance for long delays
        - Beam management
        
        Integration: gRPC API for RIC-to-gNB control
        """
        try:
            gnb_ip = oai_config.get('gnb_ip', '192.168.1.100')
            ue_ip = oai_config.get('ue_ip', '192.168.1.101')
            ntn_enabled = oai_config.get('ntn_enabled', True)
            
            if not ntn_enabled:
                self.logger.warning("OAI NTN not enabled in configuration")
                return False
            
            # Initialize OAI gRPC connection (simulated)
            oai_connection = {
                'gnb_ip': gnb_ip,
                'ue_ip': ue_ip,
                'grpc_port': 50051,
                'connected': True,
                'ntn_capable': True,
                'oai_version': 'Rel-20 NTN',
            }
            
            # Store OAI integration state
            if not hasattr(self, 'oai_integrations'):
                self.oai_integrations = {}
            
            self.oai_integrations[gnb_ip] = oai_connection
            
            self.logger.info(f"OAI NTN integration active",
                           gnb=gnb_ip,
                           ue=ue_ip,
                           version='Rel-20')
            
            return True
            
        except Exception as e:
            self.logger.error(f"OAI NTN integration failed: {e}")
            return False
    
    def optimize_ntn_timing(self, satellite_id: int, ephemeris_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize NTN timing parameters using satellite ephemeris
        
        Args:
            satellite_id: Satellite identifier
            ephemeris_data: {'lat': float, 'lon': float, 'alt_km': float, 'velocity_kmps': float}
        
        Returns:
            Optimized timing parameters
        
        Adjustments:
        - Timing advance (TA) based on altitude
        - Doppler pre-compensation
        - Beam steering angles
        - Handover trigger thresholds
        """
        try:
            lat = ephemeris_data['lat']
            lon = ephemeris_data['lon']
            alt_km = ephemeris_data['alt_km']
            velocity_kmps = ephemeris_data['velocity_kmps']
            
            # Calculate propagation delay
            speed_of_light = 299792.458  # km/s
            propagation_delay_ms = (alt_km / speed_of_light) * 1000
            
            # Calculate Doppler shift
            # Simplified: assumes worst-case angle
            max_doppler_hz = (velocity_kmps / speed_of_light) * 2e9  # 2 GHz carrier
            
            # Timing advance (round-trip)
            timing_advance_symbols = int((2 * propagation_delay_ms * 1e-3) * 30720000 / 2048)  # 30.72 MHz sampling
            
            # Handover trigger optimization
            # Longer delays → earlier handover triggers
            handover_offset_db = -3 if alt_km > 1000 else -1
            
            optimized_params = {
                'satellite_id': satellite_id,
                'propagation_delay_ms': propagation_delay_ms,
                'max_doppler_hz': max_doppler_hz,
                'timing_advance_symbols': timing_advance_symbols,
                'handover_offset_db': handover_offset_db,
                'beam_dwell_time_sec': 10 if alt_km < 2000 else 60,  # LEO vs GEO
            }
            
            self.logger.info(f"NTN timing optimized",
                           satellite=satellite_id,
                           delay_ms=f"{propagation_delay_ms:.2f}",
                           doppler_hz=f"{max_doppler_hz:.0f}",
                           ta_symbols=timing_advance_symbols)
            
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"NTN timing optimization failed: {e}")
            return {}
    
    def _calculate_ntn_delay(self, altitude_km: float) -> float:
        """Calculate one-way propagation delay for NTN"""
        speed_of_light = 299792.458  # km/s
        return (altitude_km / speed_of_light) * 1000  # ms
    
    def _calculate_ntn_doppler(self, altitude_km: float, ue_id: str) -> float:
        """Calculate Doppler shift for NTN (simplified)"""
        # LEO: high Doppler (~7 kHz at 2 GHz)
        # GEO: low Doppler (~0.1 kHz)
        if altitude_km < 2000:  # LEO
            return np.random.uniform(-7.0, 7.0)  # kHz
        else:  # MEO/GEO
            return np.random.uniform(-0.5, 0.5)  # kHz
    
    def _apply_doppler_compensation(self, ue_id: str, doppler_khz: float):
        """Apply Doppler pre-compensation for NTN UE"""
        # In production: send frequency offset correction to gNB
        self.logger.debug(f"Doppler compensation applied",
                        ue=ue_id,
                        offset_khz=f"{doppler_khz:.2f}")
    
    def get_ntn_statistics(self) -> Dict[str, Any]:
        """Get NTN-specific statistics"""
        return {
            'e2sm_ni_enabled': hasattr(self, 'ni_subscriptions'),
            'ni_subscriptions': len(getattr(self, 'ni_subscriptions', {})),
            'oai_integrations': len(getattr(self, 'oai_integrations', {})),
            'ntn_handovers_supported': True,
            'model_deployment_supported': True,
        }


