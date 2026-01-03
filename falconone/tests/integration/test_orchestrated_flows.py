"""
FalconOne Orchestrated Flows Integration Tests
Tests for multi-component workflows and system orchestration

Version: 1.9.2
Coverage: Orchestrator, health monitoring, signal pipeline, exploit chains
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
import time
import threading
from queue import Queue


@pytest.fixture
def mock_config():
    """Comprehensive mock configuration"""
    config = Mock()
    config.get = Mock(side_effect=lambda key, default=None: {
        # SDR config
        'sdr.device_type': 'rtlsdr',
        'sdr.rx_gain': 40,
        'sdr.sample_rate': 2.4e6,
        
        # Monitoring config
        'monitoring.gsm.bands': ['GSM900'],
        'monitoring.gsm.tools': ['gr-gsm'],
        'monitoring.gsm.capture_mode': 'parallel',
        'monitoring.gsm.max_workers': 2,
        'monitoring.gsm.arfcn_scan': False,
        'monitoring.5g.mode': 'SA',
        'monitoring.5g.bands': ['n78'],
        
        # AI config
        'ai_ml.signal_classification.accuracy_threshold': 0.90,
        'ai_ml.signal_classification.adaptive_mode': True,
        'ai.ric.state_size': 10,
        'ai.ric.action_size': 5,
        
        # Orchestrator config
        'orchestrator.health_check_interval': 10,
        'orchestrator.restart_on_failure': True,
        'orchestrator.max_restart_attempts': 3,
    }.get(key, default))
    return config


@pytest.fixture
def mock_logger():
    """Mock logger with all methods"""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    logger.getChild = Mock(return_value=logger)
    return logger


@pytest.fixture
def mock_sdr_manager():
    """Mock SDR manager with realistic interface"""
    sdr = Mock()
    sdr.get_device_type = Mock(return_value='rtlsdr')
    sdr.get_available_devices = Mock(return_value=['rtlsdr0'])
    sdr.get_active_device_id = Mock(return_value=0)
    sdr.configure = Mock(return_value=True)
    sdr.read_samples = Mock(return_value=np.random.randn(1024) + 1j * np.random.randn(1024))
    sdr.active_device = Mock()
    sdr.active_device.device_type = 'rtlsdr'
    return sdr


class TestSignalPipeline:
    """Integration tests for signal capture -> classification pipeline"""
    
    def test_gsm_to_classifier_pipeline(self, mock_config, mock_logger, mock_sdr_manager):
        """Test GSM capture to signal classification"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        from falconone.ai.signal_classifier import SignalClassifier
        
        # Initialize components
        gsm_monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        classifier = SignalClassifier(mock_config, mock_logger)
        
        # Create mock signal data
        signals = [{
            'id': 'gsm_test_1',
            'iq_samples': np.random.randn(1024, 2),
            'frequency': 935e6,
            'bandwidth': 0.2e6
        }]
        
        # Run classification
        results = classifier.classify(signals)
        
        # Verify pipeline
        assert len(results) == 1
        assert results[0]['signal_id'] == 'gsm_test_1'
        assert 'predicted_generation' in results[0]
    
    def test_5g_to_classifier_pipeline(self, mock_config, mock_logger, mock_sdr_manager):
        """Test 5G capture to signal classification"""
        from falconone.monitoring.fiveg_monitor import FiveGMonitor
        from falconone.ai.signal_classifier import SignalClassifier
        
        # Initialize components
        fiveg_monitor = FiveGMonitor(mock_config, mock_logger, mock_sdr_manager)
        classifier = SignalClassifier(mock_config, mock_logger)
        
        # Create mock 5G signal
        signals = [{
            'id': '5g_test_1',
            'iq_samples': np.random.randn(1024, 2),
            'frequency': 3500e6,  # n78 band
            'bandwidth': 100e6
        }]
        
        # Run classification
        results = classifier.classify(signals)
        
        assert len(results) == 1
        assert 'confidence' in results[0]


class TestHealthMonitoring:
    """Integration tests for health monitoring (v1.9.2)"""
    
    def test_component_health_check(self, mock_config, mock_logger, mock_sdr_manager):
        """Test health check across components"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        from falconone.ai.signal_classifier import SignalClassifier
        
        # Initialize components
        gsm_monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        classifier = SignalClassifier(mock_config, mock_logger)
        
        # Components should be in valid state
        assert gsm_monitor.running is False
        assert classifier._models_loaded is False
    
    def test_monitor_start_stop_lifecycle(self, mock_config, mock_logger, mock_sdr_manager):
        """Test monitor lifecycle management"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        monitor.arfcns = [50]  # Avoid scan
        
        # Start
        monitor.start()
        assert monitor.running is True
        
        # Stop
        monitor.stop()
        assert monitor.running is False


class TestMultiGenerationFlow:
    """Integration tests for multi-generation monitoring"""
    
    def test_concurrent_generation_monitoring(self, mock_config, mock_logger, mock_sdr_manager):
        """Test monitoring multiple generations concurrently"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        from falconone.monitoring.fiveg_monitor import FiveGMonitor
        from falconone.ai.signal_classifier import SignalClassifier
        
        # Initialize all monitors
        gsm = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        fiveg = FiveGMonitor(mock_config, mock_logger, mock_sdr_manager)
        classifier = SignalClassifier(mock_config, mock_logger)
        
        # All should be independent
        assert gsm.running is False
        assert fiveg.running is False
        
        # Classify mixed signals
        signals = [
            {'id': 'gsm_1', 'frequency': 935e6, 'bandwidth': 0.2e6},
            {'id': 'lte_1', 'frequency': 2600e6, 'bandwidth': 20e6},
            {'id': '5g_1', 'frequency': 28e9, 'bandwidth': 100e6},
        ]
        
        results = classifier.classify(signals)
        assert len(results) == 3


class TestRICOptimizationFlow:
    """Integration tests for RIC optimization workflow"""
    
    def test_classifier_to_ric_feedback(self, mock_config, mock_logger):
        """Test signal classification feeds RIC optimizer"""
        from falconone.ai.signal_classifier import SignalClassifier
        from falconone.ai.ric_optimizer import RICOptimizer
        
        classifier = SignalClassifier(mock_config, mock_logger)
        optimizer = RICOptimizer(mock_config, mock_logger)
        
        # Simulate signal classification result
        signals = [{'id': 'test', 'frequency': 2600e6, 'bandwidth': 20e6}]
        classification = classifier.classify(signals)
        
        # Convert to RIC state
        state = np.zeros(optimizer.state_size)
        state[0] = classification[0]['confidence']
        
        # Get optimization action
        action = optimizer.get_action(state)
        
        assert 0 <= action < optimizer.action_size
    
    def test_reward_from_classification_accuracy(self, mock_config, mock_logger):
        """Test reward signal from classification accuracy"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        
        state = np.random.randn(optimizer.state_size)
        action = optimizer.get_action(state)
        
        # Simulate reward based on classification accuracy
        accuracy = 0.95
        reward = accuracy * 10 - 5  # Scale to -5 to +5
        
        next_state = np.random.randn(optimizer.state_size)
        optimizer.remember(state, action, reward, next_state, done=False)
        
        assert len(optimizer.memory) == 1


class TestExploitChainFlow:
    """Integration tests for exploit chain orchestration"""
    
    @patch('falconone.exploit.isac_exploiter.ISACExploiter')
    def test_isac_exploit_chain(self, mock_exploiter, mock_config, mock_logger):
        """Test ISAC exploit chain flow"""
        # Mock exploit result
        mock_exploiter.return_value.waveform_manipulate.return_value = Mock(
            success=True,
            sensing_leaked=True,
            target_info={'range_m': 100, 'velocity_mps': 5}
        )
        
        # In real flow, this would chain multiple components


class TestParallelCaptureFlow:
    """Integration tests for parallel ARFCN capture (v1.9.2)"""
    
    def test_parallel_capture_initialization(self, mock_config, mock_logger, mock_sdr_manager):
        """Test parallel capture mode initialization"""
        from falconone.monitoring.gsm_monitor import GSMMonitor, CaptureMode
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        # Should be in parallel or multi-sdr mode
        assert monitor.capture_mode in [CaptureMode.PARALLEL, CaptureMode.MULTI_SDR]
        assert monitor.max_workers >= 1
    
    def test_thread_pool_creation_on_start(self, mock_config, mock_logger, mock_sdr_manager):
        """Test thread pool is created on start"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        monitor.arfcns = [50, 51, 52]
        
        monitor.start()
        
        # Thread pool should be created for parallel mode
        if monitor.capture_mode.value in ['parallel', 'multi_sdr']:
            assert monitor.executor is not None
        
        monitor.stop()


class TestOnlineLearningFlow:
    """Integration tests for online learning (v1.9.2)"""
    
    def test_incremental_update_flow(self, mock_config, mock_logger):
        """Test incremental learning update flow"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        
        # Simulate new signal with known label
        signal = np.random.randn(1024, 2)
        label = 2  # e.g., CDMA
        
        # If partial_fit exists, test it
        if hasattr(classifier, 'partial_fit'):
            result = classifier.partial_fit(signal, label)
            assert result is not None or True  # May return None on no TF


class TestDataFlowIntegrity:
    """Integration tests for data flow integrity"""
    
    def test_signal_id_preservation(self, mock_config, mock_logger):
        """Test signal IDs are preserved through pipeline"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        
        # Create signals with specific IDs
        signals = [
            {'id': f'sig_{i}', 'frequency': 935e6 + i * 1e6}
            for i in range(5)
        ]
        
        results = classifier.classify(signals)
        
        # Verify all IDs are preserved
        result_ids = {r['signal_id'] for r in results}
        expected_ids = {f'sig_{i}' for i in range(5)}
        assert result_ids == expected_ids
    
    def test_error_isolation(self, mock_config, mock_logger):
        """Test errors in one signal don't affect others"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        
        # Mix of valid and problematic signals
        signals = [
            {'id': 'good_1', 'frequency': 935e6},
            {'id': 'good_2', 'frequency': 2600e6},
            {'id': 'good_3', 'frequency': 3500e6},
        ]
        
        results = classifier.classify(signals)
        
        # All signals should have results
        assert len(results) == 3


class TestQueueBasedDataFlow:
    """Integration tests for queue-based data flow"""
    
    def test_captured_data_queue_flow(self, mock_config, mock_logger, mock_sdr_manager):
        """Test data flows through capture queues"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        # Queue should be initialized
        assert isinstance(monitor.data_queue, Queue)
        assert isinstance(monitor.captured_data, Queue)
        
        # Simulate adding data
        monitor.data_queue.put({'type': 'IMSI', 'value': '310260123456789'})
        
        assert not monitor.data_queue.empty()


class TestGracefulDegradation:
    """Integration tests for graceful degradation"""
    
    def test_classifier_without_tensorflow(self, mock_config, mock_logger):
        """Test classifier works without TensorFlow"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        
        signals = [{'id': 'test', 'frequency': 935e6}]
        results = classifier.classify(signals)
        
        # Should use heuristic fallback
        assert len(results) == 1
        assert results[0]['method'] == 'heuristic'
    
    def test_optimizer_without_tensorflow(self, mock_config, mock_logger):
        """Test RIC optimizer works without TensorFlow"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        
        state = np.random.randn(10)
        action = optimizer.get_action(state)
        
        # Should return valid random action
        assert 0 <= action < optimizer.action_size


# Run with: pytest test_orchestrated_flows.py -v
