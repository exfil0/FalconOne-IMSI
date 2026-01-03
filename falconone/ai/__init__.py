"""FalconOne AI/ML Package

Provides intelligent signal processing and analysis capabilities:
- Signal classification (CNN/LSTM or lightweight statistical)
- KPI monitoring and prediction
- SUCI de-concealment attempts
- Federated learning coordination
- O-RAN RIC optimization
- GNN topology inference
- Payload generation

All modules gracefully degrade when ML frameworks unavailable.
"""

from .signal_classifier import SignalClassifier
from .suci_deconcealment import SUCIDeconcealmentEngine
from .kpi_monitor import KPIMonitor
from .ric_optimizer import RICOptimizer
from .graph_topology import GNNTopologyInference
from .federated_coordinator import FederatedCoordinator
from .payload_generator import PayloadGenerator

__all__ = [
    'SignalClassifier',
    'SUCIDeconcealmentEngine', 
    'KPIMonitor',
    'RICOptimizer',
    'GNNTopologyInference',
    'FederatedCoordinator',
    'PayloadGenerator'
]

__version__ = '1.9.0'
