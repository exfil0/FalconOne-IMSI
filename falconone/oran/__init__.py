"""
FalconOne O-RAN Integration Package (v3.0)
Complete O-RAN Near-RT RIC and E2 interface implementation

Components:
- E2 Interface: E2AP protocol for RIC-RAN communication
- Near-RT RIC: xApp platform with SDL and A1 policy
- xApps: Reference implementations (Traffic Steering, Anomaly Detection, Resource Optimization)

Version: 3.0.0
"""

from .e2_interface import (
    E2Interface,
    E2MessageType,
    E2ServiceModel,
    E2Node,
    RICSubscription,
    E2Indication,
)

from .near_rt_ric import (
    NearRTRIC,
    XAppState,
    XAppDescriptor,
)

from .ric_xapp import (
    XAppBase,
    TrafficSteeringXApp,
    AnomalyDetectionXApp,
    ResourceOptimizationXApp,
    RICxAppController,  # Legacy
    E2NodeInfo,
    KPIReport,
    HandoverDecision,
    AnomalyAlert,
)

__all__ = [
    # E2 Interface
    'E2Interface',
    'E2MessageType',
    'E2ServiceModel',
    'E2Node',
    'RICSubscription',
    'E2Indication',
    # Near-RT RIC
    'NearRTRIC',
    'XAppState',
    'XAppDescriptor',
    # xApps
    'XAppBase',
    'TrafficSteeringXApp',
    'AnomalyDetectionXApp',
    'ResourceOptimizationXApp',
    'RICxAppController',
    'E2NodeInfo',
    'KPIReport',
    'HandoverDecision',
    'AnomalyAlert',
]

__version__ = '3.0.0'

