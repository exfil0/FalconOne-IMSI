"""
Unit tests for SDR failover and recovery (Phase 1.4)
Tests device failover, health monitoring, and automatic restart
"""

import pytest
import time
import threading
from unittest.mock import Mock, MagicMock, patch

from falconone.sdr.sdr_layer import SDRManager, SDRDevice
from falconone.core.config import Config


@pytest.fixture
def mock_config():
    """Create mock configuration"""
    config = Mock(spec=Config)
    config.get = MagicMock(side_effect=lambda key, default=None: {
        'sdr.devices': ['USRP', 'BladeRF', 'HackRF'],
        'sdr.priority': 'USRP',
        'sdr.failover_enabled': True,
        'sdr.failover_threshold_ms': 10000,
        'sdr.health_check_interval_s': 5
    }.get(key, default))
    return config


@pytest.fixture
def mock_logger():
    """Create mock logger"""
    return Mock()


@pytest.fixture
@patch('falconone.sdr.sdr_layer.SoapySDR')
def sdr_manager(mock_soapy, mock_config, mock_logger):
    """Create SDR manager instance with mocked devices"""
    # Mock SoapySDR device enumeration
    mock_soapy.Device.enumerate.return_value = [
        {'driver': 'usrp', 'serial': 'ABC123'},
        {'driver': 'bladerf', 'serial': 'DEF456'},
        {'driver': 'hackrf', 'serial': 'GHI789'}
    ]
    
    manager = SDRManager(mock_config, mock_logger)
    return manager


class TestSDRFailover:
    """Test automatic device failover"""
    
    def test_failover_enabled_by_default(self, sdr_manager):
        """Test that failover is enabled by configuration"""
        assert sdr_manager.failover_enabled == True
    
    def test_backup_devices_configured(self, sdr_manager):
        """Test that backup devices are configured"""
        sdr_manager.enable_failover()
        
        assert hasattr(sdr_manager, 'backup_devices')
        assert len(sdr_manager.backup_devices) > 0
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_failover_on_device_failure(self, mock_device_class, sdr_manager):
        """Test automatic failover when active device fails"""
        # Setup mock devices
        mock_primary = Mock()
        mock_primary.device_type = 'USRP'
        mock_primary.open.return_value = True
        mock_primary.close.return_value = True
        
        mock_backup = Mock()
        mock_backup.device_type = 'BladeRF'
        mock_backup.open.return_value = True
        mock_backup.close.return_value = True
        
        # Set active device
        sdr_manager.active_device = mock_primary
        sdr_manager.devices = {
            'USRP': mock_primary,
            'BladeRF': mock_backup
        }
        
        # Enable failover with backup
        sdr_manager.enable_failover(['BladeRF'])
        
        # Trigger failover
        success = sdr_manager._failover_to_backup()
        
        assert success == True
        assert sdr_manager.active_device.device_type == 'BladeRF'
    
    def test_failover_time_under_threshold(self, sdr_manager):
        """Test that failover completes within <10s target"""
        # Setup mock devices
        mock_primary = Mock()
        mock_primary.device_type = 'USRP'
        mock_primary.close = Mock()
        
        mock_backup = Mock()
        mock_backup.device_type = 'BladeRF'
        mock_backup.open = Mock(return_value=True)
        
        sdr_manager.active_device = mock_primary
        sdr_manager.devices = {
            'USRP': mock_primary,
            'BladeRF': mock_backup
        }
        sdr_manager.enable_failover(['BladeRF'])
        
        # Measure failover time
        start_time = time.time()
        sdr_manager._failover_to_backup()
        failover_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should be under 10s (10000ms) target
        assert failover_time < 10000
    
    def test_get_active_device_id(self, sdr_manager):
        """Test getting active device identifier"""
        mock_device = Mock()
        mock_device.device_type = 'USRP'
        sdr_manager.active_device = mock_device
        
        device_id = sdr_manager.get_active_device_id()
        assert device_id == 'USRP'
    
    def test_no_active_device_returns_none(self, sdr_manager):
        """Test that None is returned when no device is active"""
        sdr_manager.active_device = None
        
        device_id = sdr_manager.get_active_device_id()
        assert device_id is None
    
    def test_failover_priority_order(self, sdr_manager):
        """Test that failover respects device priority"""
        # Setup multiple backup devices
        mock_primary = Mock(device_type='USRP')
        mock_backup1 = Mock(device_type='BladeRF', open=Mock(return_value=True))
        mock_backup2 = Mock(device_type='HackRF', open=Mock(return_value=True))
        
        sdr_manager.active_device = mock_primary
        sdr_manager.devices = {
            'USRP': mock_primary,
            'BladeRF': mock_backup1,
            'HackRF': mock_backup2
        }
        
        # BladeRF has higher priority than HackRF
        sdr_manager.enable_failover(['BladeRF', 'HackRF'])
        sdr_manager._failover_to_backup()
        
        # Should failover to BladeRF (higher priority)
        assert sdr_manager.active_device.device_type == 'BladeRF'


class TestHealthMonitoring:
    """Test device health monitoring"""
    
    def test_health_monitoring_starts(self, sdr_manager):
        """Test that health monitoring thread starts"""
        sdr_manager.start_health_monitoring(interval=1)
        
        assert sdr_manager.health_monitoring_active == True
        assert sdr_manager.health_check_thread is not None
        assert sdr_manager.health_check_thread.is_alive()
        
        # Stop monitoring
        sdr_manager.health_monitoring_active = False
        sdr_manager.health_check_thread.join(timeout=2)
    
    def test_health_monitoring_interval(self, sdr_manager):
        """Test configurable health check interval"""
        interval = 10
        sdr_manager.start_health_monitoring(interval=interval)
        
        assert sdr_manager.health_check_interval == interval
        
        # Cleanup
        sdr_manager.health_monitoring_active = False
    
    def test_get_device_health_no_device(self, sdr_manager):
        """Test health check with no active device"""
        sdr_manager.active_device = None
        
        health = sdr_manager.get_device_health()
        
        assert health['healthy'] == False
        assert 'No active device' in health['issues']
    
    @patch('falconone.sdr.sdr_layer.SoapySDR')
    def test_get_device_health_success(self, mock_soapy, sdr_manager):
        """Test successful device health check"""
        # Setup mock device
        mock_sdr_device = Mock()
        mock_sdr_device.getSampleRate = Mock(return_value=10e6)  # 10 MHz
        mock_sdr_device.getFrequency = Mock(return_value=900e6)  # 900 MHz
        
        mock_device = Mock()
        mock_device.device_type = 'USRP'
        mock_device.sdr = mock_sdr_device
        mock_device.stream = None
        
        sdr_manager.active_device = mock_device
        
        health = sdr_manager.get_device_health()
        
        assert health['healthy'] == True
        assert len(health['issues']) == 0
        assert health['metrics']['sample_rate'] == 10e6
        assert health['metrics']['frequency'] == 900e6
    
    @patch('falconone.sdr.sdr_layer.SoapySDR')
    def test_get_device_health_invalid_sample_rate(self, mock_soapy, sdr_manager):
        """Test health check detects invalid sample rate"""
        mock_sdr_device = Mock()
        mock_sdr_device.getSampleRate = Mock(return_value=0)  # Invalid
        mock_sdr_device.getFrequency = Mock(return_value=900e6)
        
        mock_device = Mock()
        mock_device.device_type = 'USRP'
        mock_device.sdr = mock_sdr_device
        mock_device.stream = None
        
        sdr_manager.active_device = mock_device
        
        health = sdr_manager.get_device_health()
        
        assert health['healthy'] == False
        assert any('sample rate' in issue.lower() for issue in health['issues'])
    
    def test_health_monitoring_triggers_failover(self, sdr_manager):
        """Test that health monitoring triggers failover on failure"""
        # Setup unhealthy device
        mock_primary = Mock()
        mock_primary.device_type = 'USRP'
        mock_primary.sdr = None  # Unhealthy
        mock_primary.close = Mock()
        
        mock_backup = Mock()
        mock_backup.device_type = 'BladeRF'
        mock_backup.open = Mock(return_value=True)
        
        sdr_manager.active_device = mock_primary
        sdr_manager.devices = {
            'USRP': mock_primary,
            'BladeRF': mock_backup
        }
        
        # Enable failover and health monitoring
        sdr_manager.enable_failover(['BladeRF'])
        sdr_manager.start_health_monitoring(interval=1)
        
        # Wait for health check to trigger failover
        time.sleep(2)
        
        # Should have failed over to backup
        assert sdr_manager.active_device.device_type == 'BladeRF'
        
        # Cleanup
        sdr_manager.health_monitoring_active = False


class TestAutomaticRestart:
    """Test automatic device restart"""
    
    def test_auto_restart_enabled(self, sdr_manager):
        """Test enabling automatic restart"""
        sdr_manager.enable_auto_restart()
        
        assert sdr_manager.auto_restart_enabled == True
    
    def test_restart_device_success(self, sdr_manager):
        """Test successful device restart"""
        mock_device = Mock()
        mock_device.device_type = 'USRP'
        mock_device.close = Mock()
        mock_device.open = Mock(return_value=True)
        
        sdr_manager.active_device = mock_device
        
        success = sdr_manager._restart_device()
        
        assert success == True
        assert mock_device.close.called
        assert mock_device.open.called
    
    def test_restart_device_failure_triggers_failover(self, sdr_manager):
        """Test that failed restart triggers failover"""
        mock_primary = Mock()
        mock_primary.device_type = 'USRP'
        mock_primary.close = Mock()
        mock_primary.open = Mock(return_value=False)  # Restart fails
        
        mock_backup = Mock()
        mock_backup.device_type = 'BladeRF'
        mock_backup.open = Mock(return_value=True)
        
        sdr_manager.active_device = mock_primary
        sdr_manager.devices = {
            'USRP': mock_primary,
            'BladeRF': mock_backup
        }
        
        # Enable failover
        sdr_manager.enable_failover(['BladeRF'])
        
        # Attempt restart (should failover)
        sdr_manager._restart_device()
        
        # Should have failed over to backup
        assert sdr_manager.active_device.device_type == 'BladeRF'
    
    def test_detect_device_hang(self, sdr_manager):
        """Test device hang detection"""
        mock_device = Mock()
        mock_device.device_type = 'USRP'
        mock_device.stream = Mock()
        mock_device.read_samples = Mock(return_value=None)  # Hung device returns no samples
        
        sdr_manager.active_device = mock_device
        
        is_hung = sdr_manager._detect_device_hang()
        
        assert is_hung == True
    
    def test_auto_restart_on_hang(self, sdr_manager):
        """Test automatic restart when device hangs"""
        mock_device = Mock()
        mock_device.device_type = 'USRP'
        mock_device.sdr = None  # Hung device
        mock_device.close = Mock()
        mock_device.open = Mock(return_value=True)
        
        sdr_manager.active_device = mock_device
        
        # Enable auto-restart
        sdr_manager.enable_auto_restart()
        
        # Health monitoring should detect hang and restart
        sdr_manager.start_health_monitoring(interval=1)
        
        # Wait for health check
        time.sleep(2)
        
        # Device should have been restarted
        assert mock_device.close.called or mock_device.open.called
        
        # Cleanup
        sdr_manager.health_monitoring_active = False


class TestSDRManagerStatus:
    """Test SDR manager status reporting"""
    
    def test_get_status(self, sdr_manager):
        """Test getting SDR manager status"""
        mock_device = Mock()
        mock_device.device_type = 'USRP'
        sdr_manager.active_device = mock_device
        
        status = sdr_manager.get_status()
        
        assert 'active_device' in status
        assert status['active_device'] == 'USRP'
        assert 'failover_enabled' in status
        assert 'health_monitoring' in status
        assert 'auto_restart' in status
    
    def test_status_includes_failover_state(self, sdr_manager):
        """Test status includes failover state"""
        sdr_manager.enable_failover()
        
        status = sdr_manager.get_status()
        
        assert status['failover_enabled'] == True
    
    def test_status_includes_monitoring_state(self, sdr_manager):
        """Test status includes health monitoring state"""
        sdr_manager.start_health_monitoring()
        
        status = sdr_manager.get_status()
        
        assert status['health_monitoring'] == True
        
        # Cleanup
        sdr_manager.health_monitoring_active = False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
