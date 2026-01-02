"""
Unit Tests for 6G NTN Monitor and Exploiter
FalconOne v1.9.0

Tests:
- NTN monitoring (Doppler, ISAC, detection)
- Beam hijacking exploits
- Quantum attack simulations
- Integration with LE mode
- Ephemeris calculations

Author: FalconOne Team
Date: January 2026
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from falconone.monitoring.ntn_6g_monitor import NTN6GMonitor, create_ntn_6g_monitor
from falconone.exploit.ntn_6g_exploiter import NTN6GExploiter


class TestNTN6GMonitor:
    """Test 6G NTN monitoring functionality"""
    
    @pytest.fixture
    def mock_sdr(self):
        """Mock SDR manager"""
        sdr = Mock()
        sdr.set_frequency = Mock()
        sdr.set_sample_rate = Mock()
        sdr.set_gain = Mock()
        sdr.capture_iq = Mock(return_value=self._generate_test_signal())
        return sdr
    
    @pytest.fixture
    def mock_ai_classifier(self):
        """Mock AI classifier"""
        classifier = Mock()
        classifier.classify = Mock(return_value={
            'technology': '6G_NTN',
            'rsrp': -95.5,
            'confidence': 0.92
        })
        return classifier
    
    @pytest.fixture
    def ntn_monitor(self, mock_sdr, mock_ai_classifier):
        """Create NTN monitor instance"""
        config = {
            'sub_thz_freq': 150e9,
            'doppler_threshold': 10e3,
            'isac_enabled': True,
            'use_ephemeris': False,  # Disable astropy for unit tests
            'latitude': -26.2041,
            'longitude': 28.0473,
            'altitude_m': 1753,
        }
        return NTN6GMonitor(mock_sdr, mock_ai_classifier, config)
    
    def _generate_test_signal(self, n_samples=10000):
        """Generate synthetic 6G NTN signal"""
        t = np.arange(n_samples) / 1e6
        # Signal with Doppler shift
        doppler = 5000 * np.sin(2 * np.pi * 0.01 * t)
        signal = np.exp(1j * 2 * np.pi * doppler * t)
        # Add noise
        noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * 0.1
        return signal + noise
    
    def test_monitor_initialization(self, ntn_monitor):
        """Test monitor initializes correctly"""
        assert ntn_monitor is not None
        assert ntn_monitor.sub_thz_freq == 150e9
        assert ntn_monitor.isac_enabled is True
        assert ntn_monitor.doppler_threshold == 10e3
    
    def test_satellite_types(self):
        """Test satellite type parameters"""
        assert 'LEO' in NTN6GMonitor.SATELLITE_TYPES
        assert 'MEO' in NTN6GMonitor.SATELLITE_TYPES
        assert 'GEO' in NTN6GMonitor.SATELLITE_TYPES
        assert 'HAPS' in NTN6GMonitor.SATELLITE_TYPES
        assert 'UAV' in NTN6GMonitor.SATELLITE_TYPES
        
        leo_params = NTN6GMonitor.SATELLITE_TYPES['LEO']
        assert leo_params['altitude_km'] == 550
        assert leo_params['velocity_ms'] == 7500
    
    def test_start_monitoring_leo(self, ntn_monitor):
        """Test LEO satellite monitoring"""
        results = ntn_monitor.start_monitoring(sat_type='LEO', duration_sec=1)
        
        assert results is not None
        assert results['satellite_type'] == 'LEO'
        assert results['signal_detected'] is True
        assert results['technology'] == '6G_NTN'
        assert 'doppler_shift_hz' in results
        assert 'isac_data' in results
    
    def test_doppler_compensation(self, ntn_monitor):
        """Test Doppler shift compensation"""
        test_signal = self._generate_test_signal()
        
        compensated, doppler_shift = ntn_monitor.compensate_doppler(test_signal, 'LEO')
        
        assert compensated is not None
        assert len(compensated) == len(test_signal)
        assert doppler_shift != 0
        assert abs(doppler_shift) < 100e3  # Should be < 100 kHz for 150 GHz
    
    def test_doppler_compensation_geo(self, ntn_monitor):
        """Test GEO satellite (zero Doppler)"""
        test_signal = self._generate_test_signal()
        
        compensated, doppler_shift = ntn_monitor.compensate_doppler(test_signal, 'GEO')
        
        # GEO should have minimal Doppler
        assert abs(doppler_shift) < ntn_monitor.doppler_threshold
    
    def test_isac_sensing(self, ntn_monitor):
        """Test ISAC/JCS sensing capabilities"""
        test_signal = self._generate_test_signal()
        
        isac_data = ntn_monitor.perform_isac(test_signal, 'LEO')
        
        assert isac_data is not None
        assert 'range_m' in isac_data
        assert 'velocity_mps' in isac_data
        assert 'angle_deg' in isac_data
        assert 'snr_db' in isac_data
        
        # LEO range should be ~550 km
        assert 400e3 < isac_data['range_m'] < 700e3
    
    def test_6g_feature_detection(self, ntn_monitor):
        """Test 6G vs 5G feature detection"""
        # Generate high-PAPR signal (6G characteristic)
        n_samples = 10000
        signal = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        signal[100:200] = 10 * signal[100:200]  # Create peaks for high PAPR
        
        is_6g = ntn_monitor._detect_6g_features(signal)
        assert is_6g is True  # Should detect as 6G due to high PAPR
    
    def test_monitoring_all_satellite_types(self, ntn_monitor):
        """Test monitoring for all satellite types"""
        sat_types = ['LEO', 'MEO', 'GEO', 'HAPS', 'UAV']
        
        for sat_type in sat_types:
            results = ntn_monitor.start_monitoring(sat_type=sat_type, duration_sec=1)
            assert results['satellite_type'] == sat_type
            assert results['signal_detected'] is True
    
    def test_invalid_satellite_type(self, ntn_monitor):
        """Test error handling for invalid satellite type"""
        with pytest.raises(ValueError, match="Invalid satellite type"):
            ntn_monitor.start_monitoring(sat_type='INVALID')
    
    def test_le_mode_evidence_logging(self, ntn_monitor):
        """Test LE mode evidence chain integration"""
        # Enable LE mode
        ntn_monitor.le_mode_enabled = True
        ntn_monitor.warrant_validated = True
        
        with patch('falconone.monitoring.ntn_6g_monitor.EvidenceChain') as mock_chain:
            mock_instance = MagicMock()
            mock_instance.add_block.return_value = 'abc123hash'
            mock_chain.return_value = mock_instance
            
            results = ntn_monitor.start_monitoring(sat_type='LEO', duration_sec=1)
            
            assert 'evidence_hash' in results
            assert results['evidence_hash'] == 'abc123hash'
    
    def test_get_statistics(self, ntn_monitor):
        """Test statistics retrieval"""
        # Run a few monitoring sessions
        ntn_monitor.start_monitoring('LEO', duration_sec=1)
        ntn_monitor.start_monitoring('MEO', duration_sec=1)
        
        stats = ntn_monitor.get_statistics()
        
        assert stats['total_sessions'] == 2
        assert stats['satellites_tracked'] == 2
        assert 'doppler_stats' in stats
        assert 'isac_stats' in stats
    
    def test_handover_analysis(self, ntn_monitor):
        """Test NTN handover analysis"""
        # Generate monitoring history
        ntn_monitor.start_monitoring('LEO', duration_sec=1)
        
        analysis = ntn_monitor.analyze_ntn_handover('sat1', 'sat2')
        
        assert analysis is not None
        assert analysis['handover_feasible'] is True
        assert 'optimal_time_sec' in analysis
        assert 'handover_probability' in analysis
        assert 0 <= analysis['handover_probability'] <= 1
    
    def test_factory_function(self, mock_sdr, mock_ai_classifier):
        """Test factory function"""
        config = {'sub_thz_freq': 200e9}
        monitor = create_ntn_6g_monitor(mock_sdr, mock_ai_classifier, config)
        
        assert monitor is not None
        assert monitor.sub_thz_freq == 200e9


class TestNTN6GExploiter:
    """Test 6G NTN exploitation functionality"""
    
    @pytest.fixture
    def mock_payload_gen(self):
        """Mock payload generator"""
        gen = Mock()
        gen.generate = Mock(return_value={'type': 'beam_hijack', 'payload': 'test_payload'})
        return gen
    
    @pytest.fixture
    def mock_xapp_manager(self):
        """Mock O-RAN xApp manager"""
        manager = Mock()
        manager.deploy_xapp = Mock(return_value=True)
        return manager
    
    @pytest.fixture
    def ntn_exploiter(self, mock_payload_gen):
        """Create NTN exploiter instance"""
        return NTN6GExploiter(mock_payload_gen, config={'le_mode_enabled': False})
    
    def test_exploiter_initialization(self, ntn_exploiter):
        """Test exploiter initializes correctly"""
        assert ntn_exploiter is not None
        assert ntn_exploiter.payload_gen is not None
    
    def test_beam_hijack_basic(self, ntn_exploiter, mock_xapp_manager):
        """Test beam hijacking exploit"""
        ntn_exploiter.xapp_manager = mock_xapp_manager
        
        with patch('falconone.exploit.ntn_6g_exploiter.requests') as mock_requests:
            mock_response = Mock()
            mock_response.ok = True
            mock_requests.post.return_value = mock_response
            
            with patch.object(ntn_exploiter, '_enhance_listening') as mock_listen:
                mock_listen.return_value = {'status': 'listening'}
                
                result = ntn_exploiter.beam_hijack('sat_123')
                
                assert result is not None
                mock_xapp_manager.deploy_xapp.assert_called_once()
    
    def test_beam_hijack_with_quantum(self, ntn_exploiter):
        """Test beam hijacking with quantum attack"""
        with patch('falconone.exploit.ntn_6g_exploiter.requests') as mock_requests:
            mock_response = Mock()
            mock_response.ok = True
            mock_requests.post.return_value = mock_response
            
            with patch.object(ntn_exploiter, '_enhance_listening') as mock_listen:
                mock_listen.return_value = {'status': 'listening'}
                
                result = ntn_exploiter.beam_hijack('sat_123', use_quantum=True)
                
                assert result is not None
    
    def test_quantum_resistant_payload(self, ntn_exploiter):
        """Test quantum attack payload generation"""
        payload = {'data': 'test'}
        
        with patch('falconone.exploit.ntn_6g_exploiter.QuantumCircuit') as mock_qc:
            modified = ntn_exploiter._quantum_attack_payload(payload)
            
            assert modified is not None
            assert 'quantum_modified' in modified or modified != payload
    
    def test_ris_manipulation(self, ntn_exploiter):
        """Test RIS (Reconfigurable Intelligent Surface) manipulation"""
        result = ntn_exploiter.manipulate_ris('ris_001', 'beam_redirect')
        
        assert result is not None
        assert 'ris_id' in result
        assert result['ris_id'] == 'ris_001'
    
    def test_handover_poisoning(self, ntn_exploiter):
        """Test handover poisoning attack"""
        result = ntn_exploiter.poison_handover('source_sat', 'target_sat')
        
        assert result is not None
        assert 'attack_type' in result
        assert result['attack_type'] == 'handover_poison'
    
    def test_le_mode_enforcement(self, mock_payload_gen):
        """Test LE mode warrant enforcement"""
        config = {
            'le_mode_enabled': True,
            'warrant_validated': False,  # No warrant
        }
        exploiter = NTN6GExploiter(mock_payload_gen, config)
        
        result = exploiter.beam_hijack('sat_123')
        
        # Should be denied without warrant
        assert result is not None
        assert 'denied' in str(result).lower() or 'warrant' in str(result).lower()
    
    def test_ai_evasion_integration(self, ntn_exploiter):
        """Test AI-based evasion techniques"""
        payload = {'type': 'test'}
        
        evaded = ntn_exploiter._apply_ai_evasion(payload)
        
        assert evaded is not None
        # Should have polymorphic modifications
    
    def test_exploit_chain_dos_to_intercept(self, ntn_exploiter):
        """Test DoS → Intercept exploit chain"""
        with patch.object(ntn_exploiter, 'beam_hijack') as mock_hijack:
            mock_hijack.return_value = {'status': 'success'}
            
            result = ntn_exploiter.execute_chain('dos_intercept', 'sat_123')
            
            assert result is not None
            assert 'chain' in result
    
    def test_cve_payload_generation(self, ntn_exploiter):
        """Test CVE-specific payload generation for NTN"""
        cve_ids = ['CVE-2026-NTN-001', 'CVE-2026-NTN-002']
        
        for cve_id in cve_ids:
            payload = ntn_exploiter.generate_cve_payload(cve_id)
            assert payload is not None
            assert 'cve_id' in payload


class TestNTNIntegration:
    """Integration tests for NTN monitor and exploiter"""
    
    def test_end_to_end_monitoring_exploitation(self):
        """Test complete workflow: Monitor → Exploit → Listen"""
        # Create mock components
        mock_sdr = Mock()
        mock_sdr.capture_iq = Mock(return_value=np.random.randn(10000) + 1j * np.random.randn(10000))
        
        mock_ai = Mock()
        mock_ai.classify = Mock(return_value={'technology': '6G_NTN', 'rsrp': -90})
        
        # Step 1: Monitor
        monitor = NTN6GMonitor(mock_sdr, mock_ai, {'use_ephemeris': False})
        monitor_results = monitor.start_monitoring('LEO', duration_sec=1, use_isac=True)
        
        assert monitor_results['signal_detected'] is True
        
        # Step 2: Exploit (if warranted)
        mock_payload_gen = Mock()
        mock_payload_gen.generate = Mock(return_value={'payload': 'test'})
        
        exploiter = NTN6GExploiter(mock_payload_gen, {'le_mode_enabled': False})
        
        with patch('falconone.exploit.ntn_6g_exploiter.requests') as mock_req:
            mock_req.post.return_value = Mock(ok=True)
            
            # Should succeed without LE restrictions
            exploit_result = exploiter.beam_hijack('sat_123')
            
            assert exploit_result is not None
    
    def test_doppler_compensation_accuracy(self):
        """Test Doppler compensation reduces shift"""
        # Create signal with known Doppler
        n_samples = 10000
        sample_rate = 1e6
        t = np.arange(n_samples) / sample_rate
        doppler_hz = 10000  # 10 kHz shift
        signal = np.exp(1j * 2 * np.pi * doppler_hz * t)
        
        monitor = NTN6GMonitor(None, None, {'use_ephemeris': False})
        compensated, measured_doppler = monitor.compensate_doppler(signal, 'LEO')
        
        # Measured Doppler should be non-zero
        assert abs(measured_doppler) > 1000  # Should detect some shift
    
    def test_isac_range_accuracy(self):
        """Test ISAC ranging accuracy for known satellite"""
        monitor = NTN6GMonitor(None, None, {'use_ephemeris': False})
        
        # LEO at 550 km
        test_signal = np.random.randn(10000) + 1j * np.random.randn(10000)
        isac_data = monitor.perform_isac(test_signal, 'LEO')
        
        # Range should be close to 550 km (±20%)
        expected_range = 550e3  # meters
        assert 0.8 * expected_range < isac_data['range_m'] < 1.2 * expected_range


# Performance benchmarks
@pytest.mark.benchmark
class TestNTNPerformance:
    """Performance tests for NTN operations"""
    
    def test_doppler_compensation_speed(self, benchmark):
        """Benchmark Doppler compensation speed"""
        monitor = NTN6GMonitor(None, None, {'use_ephemeris': False})
        signal = np.random.randn(100000) + 1j * np.random.randn(100000)
        
        result = benchmark(monitor.compensate_doppler, signal, 'LEO')
        
        # Should complete in < 100ms for 100k samples
        assert result is not None
    
    def test_isac_processing_speed(self, benchmark):
        """Benchmark ISAC processing speed"""
        monitor = NTN6GMonitor(None, None, {'use_ephemeris': False})
        signal = np.random.randn(10000) + 1j * np.random.randn(10000)
        
        result = benchmark(monitor.perform_isac, signal, 'LEO')
        
        # Should complete in < 50ms
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
