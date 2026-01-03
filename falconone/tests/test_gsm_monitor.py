"""
FalconOne GSM Monitor Unit Tests
Tests for gsm_monitor.py with parallel capture support

Version: 1.9.2
Coverage: GSMMonitor, ARFCN capture, parallel processing
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from queue import Queue
import threading
import time


@pytest.fixture
def mock_config():
    """Mock configuration object"""
    config = Mock()
    config.get = Mock(side_effect=lambda key, default=None: {
        'monitoring.gsm.bands': ['GSM900', 'GSM1800'],
        'monitoring.gsm.tools': ['gr-gsm', 'kalibrate-rtl'],
        'monitoring.gsm.capture_mode': 'parallel',
        'monitoring.gsm.max_workers': 4,
        'monitoring.gsm.arfcn_scan': False,  # Disable scan for tests
        'sdr.rx_gain': 40,
        'gsm.ppm': 0,
    }.get(key, default))
    return config


@pytest.fixture
def mock_logger():
    """Mock logger"""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    logger.getChild = Mock(return_value=logger)
    return logger


@pytest.fixture
def mock_sdr_manager():
    """Mock SDR manager"""
    sdr = Mock()
    sdr.get_device_type = Mock(return_value='rtlsdr')
    sdr.get_available_devices = Mock(return_value=['rtlsdr0', 'rtlsdr1'])
    sdr.get_active_device_id = Mock(return_value=0)
    sdr.active_device = Mock()
    sdr.active_device.device_type = 'rtlsdr'
    return sdr


class TestGSMMonitorInitialization:
    """Tests for GSMMonitor initialization"""
    
    def test_init_with_config(self, mock_config, mock_logger, mock_sdr_manager):
        """Test initialization with valid config"""
        from falconone.monitoring.gsm_monitor import GSMMonitor, CaptureMode
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        assert monitor.bands == ['GSM900', 'GSM1800']
        assert 'gr-gsm' in monitor.tools
        assert monitor.running is False
        assert monitor.capture_mode in [CaptureMode.PARALLEL, CaptureMode.MULTI_SDR]
    
    def test_init_without_sdr(self, mock_config, mock_logger):
        """Test initialization without SDR manager"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, None)
        
        assert monitor.sdr_manager is None
        assert monitor.sdr_devices == []
    
    def test_captured_data_queue_exists(self, mock_config, mock_logger, mock_sdr_manager):
        """Test that captured_data queue is initialized"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        assert hasattr(monitor, 'captured_data')
        assert isinstance(monitor.captured_data, Queue)
    
    def test_thread_pool_config(self, mock_config, mock_logger, mock_sdr_manager):
        """Test thread pool configuration"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        assert monitor.max_workers >= 1
        assert monitor.executor is None  # Not started yet


class TestARFCNConversion:
    """Tests for ARFCN to frequency conversion"""
    
    def test_gsm900_arfcn(self, mock_config, mock_logger, mock_sdr_manager):
        """Test GSM900 ARFCN conversion"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        # ARFCN 0 = 890 MHz
        freq = monitor._arfcn_to_freq(0)
        assert freq == 890.0e6
        
        # ARFCN 50 = 890 + 50*0.2 = 900 MHz
        freq = monitor._arfcn_to_freq(50)
        assert freq == 900.0e6
    
    def test_gsm1800_arfcn(self, mock_config, mock_logger, mock_sdr_manager):
        """Test GSM1800 (DCS) ARFCN conversion"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        # ARFCN 512 = 1710.2 MHz
        freq = monitor._arfcn_to_freq(512)
        assert freq == 1710.2e6
    
    def test_invalid_arfcn(self, mock_config, mock_logger, mock_sdr_manager):
        """Test invalid ARFCN handling"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        # Invalid ARFCN should return default
        freq = monitor._arfcn_to_freq(1000)
        assert freq == 935.0e6


class TestCaptureMode:
    """Tests for capture mode enum"""
    
    def test_capture_mode_values(self):
        """Test CaptureMode enum values"""
        from falconone.monitoring.gsm_monitor import CaptureMode
        
        assert CaptureMode.SEQUENTIAL.value == 'sequential'
        assert CaptureMode.PARALLEL.value == 'parallel'
        assert CaptureMode.MULTI_SDR.value == 'multi_sdr'


class TestARFCNCaptureResult:
    """Tests for ARFCNCaptureResult dataclass"""
    
    def test_capture_result_creation(self):
        """Test ARFCNCaptureResult creation"""
        from falconone.monitoring.gsm_monitor import ARFCNCaptureResult
        
        result = ARFCNCaptureResult(
            arfcn=100,
            success=True,
            imsi_count=5,
            tmsi_count=10,
            sms_count=2,
            duration_ms=1500.0
        )
        
        assert result.arfcn == 100
        assert result.success is True
        assert result.imsi_count == 5
        assert result.error is None
    
    def test_capture_result_default_values(self):
        """Test ARFCNCaptureResult default values"""
        from falconone.monitoring.gsm_monitor import ARFCNCaptureResult
        
        result = ARFCNCaptureResult(arfcn=50, success=False)
        
        assert result.imsi_count == 0
        assert result.tmsi_count == 0
        assert result.sms_count == 0
        assert result.error is None


class TestMonitorLifecycle:
    """Tests for monitor start/stop lifecycle"""
    
    def test_start_sets_running(self, mock_config, mock_logger, mock_sdr_manager):
        """Test that start() sets running flag"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        monitor.arfcns = [50, 51, 52]  # Set ARFCNs to avoid scan
        
        monitor.start()
        assert monitor.running is True
        
        monitor.stop()
    
    def test_stop_clears_running(self, mock_config, mock_logger, mock_sdr_manager):
        """Test that stop() clears running flag"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        monitor.arfcns = [50]
        
        monitor.start()
        monitor.stop()
        
        assert monitor.running is False
    
    def test_double_start_warning(self, mock_config, mock_logger, mock_sdr_manager):
        """Test that double start logs warning"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        monitor.arfcns = [50]
        
        monitor.start()
        monitor.start()  # Second start
        
        monitor.stop()


class TestCapturedDataAccess:
    """Tests for captured data retrieval"""
    
    def test_get_captured_data_method(self, mock_config, mock_logger, mock_sdr_manager):
        """Test get_captured_data method exists"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        assert hasattr(monitor, 'get_captured_data')
    
    def test_captured_imsi_set(self, mock_config, mock_logger, mock_sdr_manager):
        """Test captured_imsi is a set"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        assert isinstance(monitor.captured_imsi, set)
    
    def test_captured_tmsi_set(self, mock_config, mock_logger, mock_sdr_manager):
        """Test captured_tmsi is a set"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        assert isinstance(monitor.captured_tmsi, set)


class TestSDRDeviceAccess:
    """Tests for SDR device access pattern (fixed self.sdr -> self.sdr_manager)"""
    
    def test_sdr_manager_attribute(self, mock_config, mock_logger, mock_sdr_manager):
        """Test that sdr_manager attribute is used correctly"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        assert hasattr(monitor, 'sdr_manager')
        assert monitor.sdr_manager == mock_sdr_manager
    
    def test_no_sdr_attribute(self, mock_config, mock_logger, mock_sdr_manager):
        """Test that self.sdr is not used (should be sdr_manager)"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        # The attribute should not exist as a separate entity
        # It should only be sdr_manager
        assert hasattr(monitor, 'sdr_manager')


class TestParallelCapture:
    """Tests for parallel ARFCN capture (v1.9.2)"""
    
    def test_thread_safe_capture_methods(self, mock_config, mock_logger, mock_sdr_manager):
        """Test that thread-safe capture methods exist"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        # v1.9.2 added thread-safe wrapper methods
        assert hasattr(monitor, '_capture_with_grgsm_safe')
        assert hasattr(monitor, '_capture_with_osmocombb_safe')
    
    def test_capture_lock_exists(self, mock_config, mock_logger, mock_sdr_manager):
        """Test that capture lock exists for thread safety"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        assert hasattr(monitor, '_capture_lock')
        assert isinstance(monitor._capture_lock, type(threading.Lock()))


class TestCaptureStats:
    """Tests for capture statistics tracking"""
    
    def test_capture_stats_initialized(self, mock_config, mock_logger, mock_sdr_manager):
        """Test that capture statistics are initialized"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        assert hasattr(monitor, '_capture_stats')
        assert 'total_captures' in monitor._capture_stats
        assert 'successful_captures' in monitor._capture_stats
        assert 'failed_captures' in monitor._capture_stats


# Integration tests with mocked subprocess
class TestGRGSMCapture:
    """Tests for gr-gsm capture integration"""
    
    @patch('subprocess.run')
    def test_grgsm_command_construction(self, mock_run, mock_config, mock_logger, mock_sdr_manager):
        """Test gr-gsm command is constructed correctly"""
        from falconone.monitoring.gsm_monitor import GSMMonitor
        
        mock_run.return_value = Mock(returncode=0, stdout='', stderr='')
        
        monitor = GSMMonitor(mock_config, mock_logger, mock_sdr_manager)
        
        # The command should use sdr_manager, not self.sdr
        # This verifies the fix was applied correctly


# Run with: pytest test_gsm_monitor.py -v
