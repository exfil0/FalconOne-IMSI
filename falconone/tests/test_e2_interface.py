"""
Unit tests for O-RAN E2 Interface
Tests E2AP protocol, subscriptions, indications, and control messages
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json
import threading
import time


@pytest.fixture
def e2_interface():
    """Create E2Interface instance"""
    from falconone.oran.e2_interface import E2Interface
    return E2Interface(ric_endpoint='localhost:36421', logger=Mock())


@pytest.fixture
def sample_e2_node():
    """Create sample E2 node info"""
    from falconone.oran.e2_interface import E2Node
    return E2Node(
        global_e2_node_id='test_gnb_001',
        ran_function_id=1,
        service_models=['E2SM-KPM', 'E2SM-RC'],
        plmn_id='00101',
        gnb_id='001',
        cell_ids=[1, 2, 3],
        connected=True
    )


class TestE2Interface:
    """Test suite for E2Interface class"""
    
    def test_initialization(self, e2_interface):
        """Test E2Interface initialization"""
        assert e2_interface is not None
        assert e2_interface.ric_endpoint == 'localhost:36421'
        assert e2_interface.connected == False
        assert len(e2_interface.e2_nodes) == 0
    
    @patch('socket.socket')
    def test_connect(self, mock_socket, e2_interface):
        """Test E2 connection establishment"""
        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock
        
        result = e2_interface.connect()
        
        assert result == True
        assert e2_interface.connected == True
        mock_sock.connect.assert_called_once()
    
    def test_register_e2_node(self, e2_interface, sample_e2_node):
        """Test E2 node registration"""
        e2_interface.connected = True
        
        result = e2_interface.register_e2_node(
            node_id=sample_e2_node.global_e2_node_id,
            ran_function_id=sample_e2_node.ran_function_id,
            service_models=sample_e2_node.service_models,
            plmn_id=sample_e2_node.plmn_id,
            gnb_id=sample_e2_node.gnb_id,
            cell_ids=sample_e2_node.cell_ids
        )
        
        assert result == True
        assert sample_e2_node.global_e2_node_id in e2_interface.e2_nodes
        node = e2_interface.e2_nodes[sample_e2_node.global_e2_node_id]
        assert node.connected == True
    
    def test_subscribe_kpm(self, e2_interface, sample_e2_node):
        """Test E2SM-KPM subscription"""
        e2_interface.connected = True
        e2_interface.e2_nodes[sample_e2_node.global_e2_node_id] = sample_e2_node
        
        event_triggers = {'kpi_reporting_period_ms': 1000}
        actions = [
            {'action_id': 1, 'action_type': 'report'},
            {'action_id': 2, 'action_type': 'insert'}
        ]
        
        request_id = e2_interface.subscribe(
            ran_function_id=2,  # KPM
            service_model='E2SM-KPM',
            event_triggers=event_triggers,
            actions=actions,
            report_period_ms=1000
        )
        
        assert request_id is not None
        assert request_id in e2_interface.subscriptions
        subscription = e2_interface.subscriptions[request_id]
        assert subscription.service_model == 'E2SM-KPM'
        assert subscription.report_period_ms == 1000
    
    def test_unsubscribe(self, e2_interface):
        """Test unsubscribe from subscription"""
        e2_interface.connected = True
        
        # Create subscription first
        request_id = e2_interface.subscribe(
            ran_function_id=2,
            service_model='E2SM-KPM',
            event_triggers={},
            actions=[],
            report_period_ms=1000
        )
        
        # Unsubscribe
        result = e2_interface.unsubscribe(request_id)
        
        assert result == True
        assert request_id not in e2_interface.subscriptions
    
    def test_send_control_message(self, e2_interface, sample_e2_node):
        """Test sending RIC control message"""
        e2_interface.connected = True
        e2_interface.e2_nodes[sample_e2_node.global_e2_node_id] = sample_e2_node
        
        control_header = {'control_type': 'handover'}
        control_message = {
            'target_cell': 2,
            'ue_rnti': 12345
        }
        
        result = e2_interface.send_control(
            ran_function_id=3,  # E2SM-RC
            control_header=control_header,
            control_message=control_message
        )
        
        assert result is not None
        assert result['success'] == True
        assert e2_interface.statistics['control_requests_sent'] == 1
    
    def test_get_indications(self, e2_interface):
        """Test retrieving E2 indications"""
        e2_interface.connected = True
        
        # Add mock indication to queue
        from falconone.oran.e2_interface import E2Indication
        indication = E2Indication(
            request_id=1,
            ran_function_id=2,
            action_id=1,
            indication_type='report',
            indication_header=b'header',
            indication_message=b'message'
        )
        e2_interface.indication_queue.put(indication)
        
        indications = e2_interface.get_indications(timeout=0.1)
        
        assert len(indications) == 1
        assert indications[0].request_id == 1
    
    def test_message_serialization(self, e2_interface):
        """Test message serialization"""
        from falconone.oran.e2_interface import E2MessageType
        
        message = {
            'type': E2MessageType.RIC_SUBSCRIPTION_REQUEST.value,
            'request_id': 1,
            'ran_function_id': 2,
            'service_model': 'E2SM-KPM'
        }
        
        serialized = e2_interface._serialize_message(message)
        
        assert serialized is not None
        assert len(serialized) > 4  # Length prefix + payload
    
    def test_message_deserialization(self, e2_interface):
        """Test message deserialization"""
        original_message = {
            'type': 1,
            'data': {'key': 'value'}
        }
        
        serialized = e2_interface._serialize_message(original_message)
        deserialized = e2_interface._deserialize_message(serialized)
        
        assert deserialized['type'] == original_message['type']
        assert deserialized['data'] == original_message['data']
    
    def test_statistics_tracking(self, e2_interface):
        """Test statistics tracking"""
        e2_interface.connected = True
        
        # Simulate some activity
        e2_interface.statistics['messages_sent'] = 10
        e2_interface.statistics['messages_received'] = 15
        e2_interface.statistics['indications_processed'] = 8
        
        stats = e2_interface.get_statistics()
        
        assert stats['messages_sent'] == 10
        assert stats['messages_received'] == 15
        assert stats['indications_processed'] == 8
    
    def test_disconnect(self, e2_interface):
        """Test disconnection"""
        e2_interface.connected = True
        e2_interface.socket = Mock()
        
        e2_interface.disconnect()
        
        assert e2_interface.connected == False
        assert e2_interface.socket is None
    
    def test_subscription_with_handler(self, e2_interface):
        """Test subscription with indication handler"""
        e2_interface.connected = True
        
        handler = Mock()
        
        request_id = e2_interface.subscribe(
            ran_function_id=2,
            service_model='E2SM-KPM',
            event_triggers={},
            actions=[],
            report_period_ms=1000,
            indication_handler=handler
        )
        
        assert request_id in e2_interface.indication_handlers
        assert e2_interface.indication_handlers[request_id] == handler
    
    def test_e2_node_disconnect(self, e2_interface, sample_e2_node):
        """Test E2 node disconnection"""
        e2_interface.connected = True
        e2_interface.e2_nodes[sample_e2_node.global_e2_node_id] = sample_e2_node
        
        e2_interface.disconnect_e2_node(sample_e2_node.global_e2_node_id)
        
        node = e2_interface.e2_nodes.get(sample_e2_node.global_e2_node_id)
        if node:
            assert node.connected == False


class TestE2MessageTypes:
    """Test E2MessageType enum"""
    
    def test_message_type_values(self):
        """Test message type enum values"""
        from falconone.oran.e2_interface import E2MessageType
        
        assert E2MessageType.E2_SETUP_REQUEST.value == 1
        assert E2MessageType.RIC_SUBSCRIPTION_REQUEST.value == 4
        assert E2MessageType.RIC_INDICATION.value == 7
    
    def test_message_type_names(self):
        """Test message type names"""
        from falconone.oran.e2_interface import E2MessageType
        
        assert E2MessageType.E2_SETUP_REQUEST.name == 'E2_SETUP_REQUEST'
        assert E2MessageType.RIC_CONTROL_REQUEST.name == 'RIC_CONTROL_REQUEST'


class TestE2ServiceModel:
    """Test E2ServiceModel enum"""
    
    def test_service_model_values(self):
        """Test service model values"""
        from falconone.oran.e2_interface import E2ServiceModel
        
        assert E2ServiceModel.KPM.value == 'E2SM-KPM'
        assert E2ServiceModel.RC.value == 'E2SM-RC'
        assert E2ServiceModel.NI.value == 'E2SM-NI'


class TestE2Node:
    """Test E2Node dataclass"""
    
    def test_e2_node_creation(self, sample_e2_node):
        """Test E2Node creation"""
        assert sample_e2_node.global_e2_node_id == 'test_gnb_001'
        assert sample_e2_node.plmn_id == '00101'
        assert len(sample_e2_node.cell_ids) == 3
        assert 'E2SM-KPM' in sample_e2_node.service_models


class TestRICSubscription:
    """Test RICSubscription dataclass"""
    
    def test_subscription_creation(self):
        """Test RICSubscription creation"""
        from falconone.oran.e2_interface import RICSubscription
        
        subscription = RICSubscription(
            request_id=1,
            ran_function_id=2,
            service_model='E2SM-KPM',
            event_triggers={'period_ms': 1000},
            actions=[{'action_id': 1}],
            report_period_ms=1000
        )
        
        assert subscription.request_id == 1
        assert subscription.service_model == 'E2SM-KPM'
        assert subscription.report_period_ms == 1000


class TestE2Indication:
    """Test E2Indication dataclass"""
    
    def test_indication_creation(self):
        """Test E2Indication creation"""
        from falconone.oran.e2_interface import E2Indication
        
        indication = E2Indication(
            request_id=1,
            ran_function_id=2,
            action_id=1,
            indication_type='report',
            indication_header=b'test_header',
            indication_message=b'test_message'
        )
        
        assert indication.request_id == 1
        assert indication.indication_type == 'report'
        assert indication.indication_header == b'test_header'


@pytest.mark.slow
class TestE2InterfaceIntegration:
    """Integration tests for E2 interface"""
    
    def test_subscription_indication_flow(self, e2_interface, sample_e2_node):
        """Test complete subscription->indication flow"""
        e2_interface.connected = True
        e2_interface.e2_nodes[sample_e2_node.global_e2_node_id] = sample_e2_node
        
        indications_received = []
        
        def handler(indication):
            indications_received.append(indication)
        
        # Subscribe
        request_id = e2_interface.subscribe(
            ran_function_id=2,
            service_model='E2SM-KPM',
            event_triggers={},
            actions=[],
            report_period_ms=100,
            indication_handler=handler
        )
        
        # Simulate indication
        from falconone.oran.e2_interface import E2Indication
        indication = E2Indication(
            request_id=request_id,
            ran_function_id=2,
            action_id=1,
            indication_type='report',
            indication_header=b'header',
            indication_message=b'kpi_data'
        )
        
        # Process indication (simulate)
        if request_id in e2_interface.indication_handlers:
            e2_interface.indication_handlers[request_id](indication)
        
        assert len(indications_received) == 1
        assert indications_received[0].request_id == request_id
    
    @patch('socket.socket')
    def test_connection_error_handling(self, mock_socket, e2_interface):
        """Test error handling during connection"""
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = ConnectionRefusedError
        mock_socket.return_value = mock_sock
        
        result = e2_interface.connect()
        
        assert result == False
        assert e2_interface.connected == False
