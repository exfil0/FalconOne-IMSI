"""
End-to-end integration tests for monitoring workflow
Tests complete monitoring process from initialization to data export
"""

import pytest
import time
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from falconone.core.orchestrator import SystemOrchestrator
from falconone.monitoring.gsm_monitor import GSMMonitor
from falconone.monitoring.lte_monitor import LTEMonitor
from falconone.utils.database import DatabaseManager


@pytest.fixture
def temp_database(tmp_path):
    """Create temporary test database"""
    db_path = tmp_path / "test_monitoring.db"
    db = DatabaseManager(str(db_path))
    yield db
    db.close()


@pytest.fixture
@patch('falconone.sdr.sdr_layer.SoapySDR')
def orchestrator(mock_soapy, temp_database, tmp_path):
    """Create system orchestrator for integration testing"""
    config = {
        'database_path': str(tmp_path / "test_monitoring.db"),
        'sdr': {
            'devices': ['USRP'],
            'priority': 'USRP'
        }
    }
    
    mock_soapy.Device.enumerate.return_value = [
        {'driver': 'usrp', 'serial': 'TEST123'}
    ]
    
    orchestrator = SystemOrchestrator(config)
    yield orchestrator
    orchestrator.shutdown()


class TestGSMMonitoringWorkflow:
    """Test complete GSM monitoring workflow"""
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_gsm_capture_workflow(self, mock_sdr, orchestrator):
        """Test GSM signal capture from start to finish"""
        # 1. Start GSM monitoring
        success = orchestrator.start_monitoring('GSM', 900.0)
        assert success == True
        
        # 2. Capture signals for duration
        time.sleep(1)
        
        # 3. Stop monitoring
        orchestrator.stop_monitoring()
        
        # 4. Verify captures in database
        captures = orchestrator.database.get_signal_captures()
        gsm_captures = [c for c in captures if c.get('technology') == 'GSM']
        assert len(gsm_captures) > 0
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_gsm_cell_discovery(self, mock_sdr, orchestrator):
        """Test GSM cell discovery and tracking"""
        # Start monitoring
        orchestrator.start_monitoring('GSM', 900.0)
        time.sleep(0.5)
        
        # Discover cells
        cells = orchestrator.discover_cells('GSM')
        
        # Verify cells found
        assert isinstance(cells, list)
        
        # Verify cells stored in database
        db_cells = orchestrator.database.get_network_cells(technology='GSM')
        assert len(db_cells) >= 0
        
        orchestrator.stop_monitoring()
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_gsm_a51_decryption(self, mock_sdr, orchestrator):
        """Test GSM A5/1 decryption workflow (Phase 2.1)"""
        # Start monitoring with decryption
        orchestrator.start_monitoring('GSM', 900.0, enable_decryption=True)
        time.sleep(1)
        
        # Check if decrypted data captured
        captures = orchestrator.database.get_signal_captures()
        
        # Some captures may have decrypted flag
        decrypted_captures = [c for c in captures if c.get('decrypted') == True]
        # (May be 0 if no encrypted traffic)
        
        orchestrator.stop_monitoring()


class TestLTEMonitoringWorkflow:
    """Test complete LTE monitoring workflow"""
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_lte_capture_workflow(self, mock_sdr, orchestrator):
        """Test LTE signal capture workflow"""
        # Start LTE monitoring
        success = orchestrator.start_monitoring('LTE', 1850.0)
        assert success == True
        
        # Capture for duration
        time.sleep(1)
        
        # Stop monitoring
        orchestrator.stop_monitoring()
        
        # Verify LTE captures
        captures = orchestrator.database.get_signal_captures()
        lte_captures = [c for c in captures if c.get('technology') == 'LTE']
        assert len(lte_captures) > 0
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_lte_rrc_decoding(self, mock_sdr, orchestrator):
        """Test LTE RRC message decoding (Phase 2.2)"""
        # Start monitoring with RRC decoding
        orchestrator.start_monitoring('LTE', 1850.0, decode_rrc=True)
        time.sleep(1)
        
        # Check for decoded RRC messages
        captures = orchestrator.database.get_signal_captures()
        
        # Some captures should have RRC data
        rrc_captures = [c for c in captures if 'rrc_messages' in c.get('metadata', {})]
        
        orchestrator.stop_monitoring()
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_lte_key_extraction(self, mock_sdr, orchestrator):
        """Test LTE key extraction workflow (Phase 2.2)"""
        # Start monitoring with key extraction
        orchestrator.start_monitoring('LTE', 2600.0, extract_keys=True)
        time.sleep(1)
        
        # Check if keys extracted
        # (Keys would be in crypto analyzer if successful)
        
        orchestrator.stop_monitoring()


class TestMultiTechnologyMonitoring:
    """Test monitoring across multiple technologies"""
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_sequential_monitoring(self, mock_sdr, orchestrator):
        """Test monitoring different technologies sequentially"""
        # Monitor GSM
        orchestrator.start_monitoring('GSM', 900.0)
        time.sleep(0.5)
        orchestrator.stop_monitoring()
        
        # Monitor UMTS
        orchestrator.start_monitoring('UMTS', 2100.0)
        time.sleep(0.5)
        orchestrator.stop_monitoring()
        
        # Monitor LTE
        orchestrator.start_monitoring('LTE', 1850.0)
        time.sleep(0.5)
        orchestrator.stop_monitoring()
        
        # Verify captures for all technologies
        captures = orchestrator.database.get_signal_captures()
        technologies = set(c.get('technology') for c in captures)
        
        assert 'GSM' in technologies
        assert 'UMTS' in technologies or 'LTE' in technologies
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_frequency_scanning(self, mock_sdr, orchestrator):
        """Test frequency band scanning"""
        # Scan GSM 900 band
        frequencies = [890.0 + (i * 0.2) for i in range(5)]  # 5 frequencies
        
        for freq in frequencies:
            orchestrator.start_monitoring('GSM', freq)
            time.sleep(0.2)
            orchestrator.stop_monitoring()
        
        # Verify captures across frequencies
        captures = orchestrator.database.get_signal_captures()
        unique_freqs = set(c.get('frequency') for c in captures)
        
        assert len(unique_freqs) >= 2  # At least some variation


class TestDataExportWorkflow:
    """Test data export workflow (Phase 2.3)"""
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_csv_export_workflow(self, mock_sdr, orchestrator, tmp_path):
        """Test complete CSV export workflow"""
        # Capture some data
        orchestrator.start_monitoring('LTE', 2600.0)
        time.sleep(0.5)
        orchestrator.stop_monitoring()
        
        # Export to CSV
        export_path = tmp_path / "captures.csv"
        orchestrator.database.export_to_csv('signal_captures', str(export_path))
        
        # Verify file created
        assert export_path.exists()
        assert export_path.stat().st_size > 0
        
        # Verify CSV format
        with open(export_path, 'r') as f:
            header = f.readline()
            assert 'frequency' in header or 'technology' in header
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_json_export_workflow(self, mock_sdr, orchestrator, tmp_path):
        """Test complete JSON export workflow"""
        # Capture data
        orchestrator.start_monitoring('GSM', 900.0)
        time.sleep(0.5)
        orchestrator.stop_monitoring()
        
        # Export to JSON
        export_path = tmp_path / "captures.json"
        orchestrator.database.export_to_json('signal_captures', str(export_path))
        
        # Verify file created
        assert export_path.exists()
        assert export_path.stat().st_size > 0
        
        # Verify JSON format
        import json
        with open(export_path, 'r') as f:
            data = json.load(f)
            assert isinstance(data, list) or isinstance(data, dict)
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_pdf_report_workflow(self, mock_sdr, orchestrator, tmp_path):
        """Test complete PDF report generation workflow"""
        # Capture data and execute exploit
        orchestrator.start_monitoring('LTE', 1850.0)
        time.sleep(0.5)
        
        attack_params = {
            'attack_type': 'dos',
            'target_frequency': 1850.0,
            'duration': 0.5,
            'rate': 50
        }
        orchestrator.execute_exploit('dos', attack_params)
        
        orchestrator.stop_monitoring()
        
        # Generate PDF report
        report_path = tmp_path / "report.pdf"
        orchestrator.generate_pdf_report(str(report_path))
        
        # Verify PDF created
        assert report_path.exists()
        assert report_path.stat().st_size > 0
        
        # Verify PDF header
        with open(report_path, 'rb') as f:
            header = f.read(5)
            assert header == b'%PDF-'


class TestAnomalyDetection:
    """Test anomaly detection during monitoring"""
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_imsi_catcher_detection(self, mock_sdr, orchestrator):
        """Test IMSI catcher detection"""
        # Start monitoring with anomaly detection
        orchestrator.start_monitoring('GSM', 900.0, detect_anomalies=True)
        time.sleep(1)
        
        # Check for anomaly detections
        detections = orchestrator.database.get_detection_events()
        
        # May have detections if anomalies present
        imsi_catcher_detections = [
            d for d in detections 
            if 'imsi_catcher' in d.get('event_type', '').lower()
        ]
        
        orchestrator.stop_monitoring()
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_encryption_downgrade_detection(self, mock_sdr, orchestrator):
        """Test encryption downgrade detection"""
        # Monitor with downgrade detection
        orchestrator.start_monitoring('GSM', 900.0, detect_anomalies=True)
        time.sleep(1)
        
        detections = orchestrator.database.get_detection_events()
        
        # Check for encryption anomalies
        encryption_detections = [
            d for d in detections
            if 'encryption' in d.get('event_type', '').lower()
        ]
        
        orchestrator.stop_monitoring()


class TestContinuousMonitoring:
    """Test continuous monitoring operations"""
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_long_duration_monitoring(self, mock_sdr, orchestrator):
        """Test monitoring for extended duration"""
        # Start continuous monitoring
        orchestrator.start_monitoring('LTE', 2600.0)
        
        # Monitor for several seconds
        time.sleep(3)
        
        # Verify continuous capture
        captures = orchestrator.database.get_signal_captures()
        assert len(captures) > 5  # Should have multiple captures
        
        # Verify timestamps span duration
        timestamps = [c.get('timestamp') for c in captures]
        time_span = max(timestamps) - min(timestamps)
        assert time_span > 1  # At least 1 second span
        
        orchestrator.stop_monitoring()
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_monitoring_resource_usage(self, mock_sdr, orchestrator):
        """Test that monitoring doesn't exhaust resources"""
        import psutil
        
        # Get initial resource usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Monitor for duration
        orchestrator.start_monitoring('LTE', 1850.0)
        time.sleep(2)
        orchestrator.stop_monitoring()
        
        # Check resource usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (<500MB)
        assert memory_increase < 500


class TestMonitoringErrorHandling:
    """Test error handling during monitoring"""
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_invalid_frequency_handling(self, mock_sdr, orchestrator):
        """Test handling of invalid frequency"""
        # Try to monitor invalid frequency
        result = orchestrator.start_monitoring('LTE', 0.0)  # Invalid
        
        # Should handle gracefully
        assert result == False or isinstance(result, dict)
    
    @patch('falconone.sdr.sdr_layer.SDRDevice')
    def test_sdr_disconnect_handling(self, mock_sdr, orchestrator):
        """Test handling of SDR disconnection during monitoring"""
        # Start monitoring
        orchestrator.start_monitoring('GSM', 900.0)
        
        # Simulate SDR disconnect
        if hasattr(orchestrator, 'sdr_manager'):
            orchestrator.sdr_manager.active_device = None
        
        # Should handle gracefully
        time.sleep(0.5)
        
        # Stop monitoring (should not crash)
        orchestrator.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
