"""
ISAC (Integrated Sensing and Communications) Test Suite - FalconOne v1.9.0

Tests for ISAC monitoring and exploitation modules.
Coverage: waveform exploits, AI poisoning, quantum attacks, sensing accuracy.

Author: FalconOne Team
License: Proprietary
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock, patch

from falconone.monitoring.isac_monitor import ISACMonitor, ISACSensingResult, WaveformAnalysis, create_isac_monitor
from falconone.exploit.isac_exploiter import ISACExploiter, ISACExploitResult, create_isac_exploiter


# Fixtures

@pytest.fixture
def mock_sdr():
    """Mock SDR manager"""
    sdr = Mock()
    sdr.set_frequency = Mock()
    sdr.set_sample_rate = Mock()
    sdr.receive = Mock(return_value=np.random.randn(1000000) + 1j * np.random.randn(1000000))
    sdr.transmit = Mock()
    return sdr


@pytest.fixture
def mock_payload_gen():
    """Mock AI payload generator"""
    gen = Mock()
    gen.generate = Mock(return_value=np.random.randn(10000))
    return gen


@pytest.fixture
def mock_signal_bus():
    """Mock signal bus"""
    bus = Mock()
    bus.emit = Mock()
    return bus


@pytest.fixture
def mock_evidence_mgr():
    """Mock evidence manager"""
    mgr = Mock()
    mgr.log_event = Mock(return_value='evidence_hash_123')
    return mgr


@pytest.fixture
def isac_config():
    """ISAC configuration"""
    return {
        'isac_enabled': True,
        'modes': ['monostatic', 'bistatic', 'cooperative'],
        'frequency_default': 150e9,
        'sensing_resolution': 1.0,
        'max_targets': 10
    }


@pytest.fixture
def isac_monitor(mock_sdr, isac_config, mock_signal_bus, mock_evidence_mgr):
    """ISACMonitor instance"""
    return ISACMonitor(mock_sdr, isac_config, mock_signal_bus, mock_evidence_mgr)


@pytest.fixture
def isac_exploiter(mock_sdr, mock_payload_gen, mock_signal_bus, mock_evidence_mgr):
    """ISACExploiter instance"""
    return ISACExploiter(mock_sdr, mock_payload_gen, None, None, mock_signal_bus, mock_evidence_mgr)


# ISACMonitor Tests

class TestISACMonitor:
    """Tests for ISACMonitor"""
    
    def test_initialization(self, isac_monitor):
        """Test monitor initialization"""
        assert isac_monitor.enabled == True
        assert 'monostatic' in isac_monitor.modes
        assert isac_monitor.sensing_resolution == 1.0
        assert isac_monitor.stats['total_sessions'] == 0
    
    def test_sensing_modes_defined(self, isac_monitor):
        """Test that all sensing modes are defined"""
        assert 'monostatic' in ISACMonitor.SENSING_MODES
        assert 'bistatic' in ISACMonitor.SENSING_MODES
        assert 'cooperative' in ISACMonitor.SENSING_MODES
        assert ISACMonitor.SENSING_MODES['monostatic']['accuracy'] > 0.9
    
    def test_waveform_types_defined(self, isac_monitor):
        """Test that waveform types are defined"""
        assert 'OFDM' in ISACMonitor.WAVEFORM_TYPES
        assert 'DFT-s-OFDM' in ISACMonitor.WAVEFORM_TYPES
        assert 'FMCW' in ISACMonitor.WAVEFORM_TYPES
    
    def test_monostatic_sensing(self, isac_monitor):
        """Test monostatic sensing mode"""
        result = isac_monitor.start_sensing(
            mode='monostatic',
            duration_sec=5,
            frequency_ghz=150.0
        )
        
        assert isinstance(result, ISACSensingResult)
        assert result.mode == 'monostatic'
        assert result.range_m > 0
        assert result.snr_db > 0
        assert result.sensing_accuracy > 0
    
    def test_bistatic_sensing(self, isac_monitor):
        """Test bistatic sensing mode"""
        result = isac_monitor.start_sensing(
            mode='bistatic',
            duration_sec=5,
            frequency_ghz=150.0
        )
        
        assert result.mode == 'bistatic'
        assert result.angle_deg != 0.0  # Bistatic has angle estimation
        assert result.target_count >= 1
    
    def test_cooperative_sensing(self, isac_monitor):
        """Test cooperative sensing mode (multi-node)"""
        result = isac_monitor.start_sensing(
            mode='cooperative',
            duration_sec=5,
            frequency_ghz=150.0
        )
        
        assert result.mode == 'cooperative'
        assert result.sensing_accuracy > 0.95  # Higher accuracy
        assert result.target_count >= 1  # Can detect multiple targets
    
    def test_range_estimation(self, isac_monitor):
        """Test range estimation accuracy"""
        samples = np.random.randn(100000) + 1j * np.random.randn(100000)
        mode_params = ISACMonitor.SENSING_MODES['monostatic']
        
        range_m = isac_monitor._estimate_range(samples, mode_params)
        
        assert range_m >= mode_params['min_range_m']
        assert range_m <= mode_params['max_range_m']
    
    def test_velocity_estimation(self, isac_monitor):
        """Test velocity estimation via Doppler"""
        samples = np.random.randn(100000) + 1j * np.random.randn(100000)
        frequency = 150e9
        
        velocity_mps, doppler_hz = isac_monitor._estimate_velocity(samples, frequency)
        
        assert isinstance(velocity_mps, (int, float))
        assert isinstance(doppler_hz, (int, float))
    
    def test_angle_estimation_bistatic(self, isac_monitor):
        """Test angle estimation for bistatic mode"""
        samples = np.random.randn(10000) + 1j * np.random.randn(10000)
        
        angle_deg = isac_monitor._estimate_angle(samples, 'bistatic')
        
        assert -180 <= angle_deg <= 180
    
    def test_angle_estimation_cooperative(self, isac_monitor):
        """Test angle estimation for cooperative mode"""
        samples = np.random.randn(10000) + 1j * np.random.randn(10000)
        
        angle_deg = isac_monitor._estimate_angle(samples, 'cooperative')
        
        assert -90 <= angle_deg <= 90
    
    def test_target_detection_monostatic(self, isac_monitor):
        """Test single target detection (monostatic)"""
        samples = np.random.randn(10000) + 1j * np.random.randn(10000)
        
        target_count = isac_monitor._detect_targets(samples, 'monostatic')
        
        assert target_count == 1  # Monostatic detects single target
    
    def test_target_detection_cooperative(self, isac_monitor):
        """Test multi-target detection (cooperative)"""
        samples = np.random.randn(10000) + 1j * np.random.randn(10000)
        
        target_count = isac_monitor._detect_targets(samples, 'cooperative')
        
        assert 1 <= target_count <= isac_monitor.max_targets
    
    def test_waveform_analysis(self, isac_monitor):
        """Test waveform analysis for anomalies"""
        samples = np.random.randn(10000) + 1j * np.random.randn(10000)
        
        analysis = isac_monitor._analyze_waveform(samples, 'OFDM', 150e9)
        
        assert isinstance(analysis, WaveformAnalysis)
        assert analysis.waveform_type == 'OFDM'
        assert 0 <= analysis.sensing_overhead <= 1
        assert 0 <= analysis.comms_integrity <= 1
    
    def test_privacy_breach_detection_high_overhead(self, isac_monitor):
        """Test privacy breach detection (high sensing overhead)"""
        result = ISACSensingResult(
            mode='monostatic',
            range_m=500.0,
            velocity_mps=10.0,
            angle_deg=0.0,
            doppler_hz=100.0,
            snr_db=15.0,
            target_count=1,
            sensing_accuracy=0.95,
            timestamp=time.time()
        )
        
        waveform = WaveformAnalysis(
            waveform_type='OFDM',
            carrier_freq_ghz=150.0,
            bandwidth_mhz=100.0,
            pilot_density=0.1,
            sensing_overhead=0.6,  # >50% = breach
            comms_integrity=0.7,
            anomalies=[]
        )
        
        breach_detected = isac_monitor._detect_privacy_breach(result, waveform)
        
        assert breach_detected == True
    
    def test_privacy_breach_detection_fine_ranging(self, isac_monitor):
        """Test privacy breach detection (sub-meter ranging)"""
        result = ISACSensingResult(
            mode='monostatic',
            range_m=0.5,  # <1m = surveillance
            velocity_mps=2.0,
            angle_deg=0.0,
            doppler_hz=50.0,
            snr_db=20.0,
            target_count=1,
            sensing_accuracy=0.98,  # >0.95 = high precision
            timestamp=time.time()
        )
        
        waveform = WaveformAnalysis(
            waveform_type='OFDM',
            carrier_freq_ghz=150.0,
            bandwidth_mhz=100.0,
            pilot_density=0.1,
            sensing_overhead=0.2,
            comms_integrity=0.85,
            anomalies=[]
        )
        
        breach_detected = isac_monitor._detect_privacy_breach(result, waveform)
        
        assert breach_detected == True
    
    def test_le_mode_evidence_logging(self, isac_monitor, mock_evidence_mgr):
        """Test LE mode evidence logging"""
        result = isac_monitor.start_sensing(
            mode='monostatic',
            duration_sec=5,
            le_mode=True,
            warrant_id='WARRANT-12345'
        )
        
        assert result.evidence_hash is not None
        mock_evidence_mgr.log_event.assert_called_once()
    
    def test_statistics_update(self, isac_monitor):
        """Test statistics tracking"""
        initial_count = isac_monitor.stats['total_sessions']
        
        isac_monitor.start_sensing(mode='monostatic', duration_sec=3)
        
        assert isac_monitor.stats['total_sessions'] == initial_count + 1
        assert isac_monitor.stats['monostatic_count'] == 1
        assert isac_monitor.stats['avg_range_m'] > 0
    
    def test_cooperative_network_analysis(self, isac_monitor):
        """Test cooperative network topology analysis"""
        node_ids = ['gnb_001', 'gnb_002', 'gnb_003']
        
        topology = isac_monitor.analyze_cooperative_network(node_ids)
        
        assert topology['node_count'] == 3
        assert topology['sensing_coverage_km2'] == 75.0  # 3 * 25
        assert topology['cooperative_accuracy'] == 0.98
        assert len(topology['nodes']) == 3
    
    def test_factory_function(self, mock_sdr, isac_config):
        """Test create_isac_monitor factory"""
        monitor = create_isac_monitor(mock_sdr, isac_config)
        
        assert isinstance(monitor, ISACMonitor)
    
    def test_invalid_mode_raises_error(self, isac_monitor):
        """Test that invalid sensing mode raises ValueError"""
        with pytest.raises(ValueError, match="Invalid mode"):
            isac_monitor.start_sensing(mode='invalid_mode', duration_sec=5)
    
    def test_le_mode_without_warrant_raises_error(self, isac_monitor):
        """Test that LE mode without warrant raises ValueError"""
        with pytest.raises(ValueError, match="LE mode requires warrant_id"):
            isac_monitor.start_sensing(mode='monostatic', duration_sec=5, le_mode=True)


# ISACExploiter Tests

class TestISACExploiter:
    """Tests for ISACExploiter"""
    
    def test_initialization(self, isac_exploiter):
        """Test exploiter initialization"""
        assert isac_exploiter.stats['total_exploits'] == 0
        assert len(ISACExploiter.ISAC_CVES) == 8
    
    def test_cve_database(self, isac_exploiter):
        """Test ISAC CVE database"""
        assert 'CVE-2026-ISAC-001' in ISACExploiter.ISAC_CVES
        assert 'CVE-2026-ISAC-008' in ISACExploiter.ISAC_CVES
        
        cve_001 = ISACExploiter.ISAC_CVES['CVE-2026-ISAC-001']
        assert cve_001['name'] == 'Waveform DoS Attack'
        assert cve_001['severity'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        assert 0 < cve_001['success_rate'] <= 1.0
    
    def test_waveform_manipulation_basic(self, isac_exploiter):
        """Test basic waveform manipulation exploit"""
        result = isac_exploiter.waveform_manipulate(
            target_freq=150e9,
            mode='monostatic',
            waveform_type='OFDM'
        )
        
        assert isinstance(result, ISACExploitResult)
        assert result.exploit_type == 'waveform_manipulation'
        assert isinstance(result.success, bool)
    
    def test_waveform_manipulation_with_le_mode(self, isac_exploiter, mock_evidence_mgr):
        """Test waveform manipulation with LE evidence"""
        result = isac_exploiter.waveform_manipulate(
            target_freq=150e9,
            mode='monostatic',
            le_mode=True,
            warrant_id='WARRANT-12345'
        )
        
        assert result.evidence_hash is not None
        mock_evidence_mgr.log_event.assert_called_once()
    
    def test_pilot_corruption_attack(self, isac_exploiter):
        """Test pilot corruption (CVE-2026-ISAC-007)"""
        result = isac_exploiter.waveform_manipulate(
            target_freq=150e9,
            mode='bistatic',
            cve_id='CVE-2026-ISAC-007'
        )
        
        assert result.exploit_type == 'waveform_manipulation'
    
    def test_ai_poisoning_basic(self, isac_exploiter):
        """Test AI model poisoning exploit"""
        training_data = np.random.randn(1000, 64)
        
        result = isac_exploiter.ai_poison(
            training_data=training_data,
            target_system='oran_rapp',
            poisoning_rate=0.1
        )
        
        assert isinstance(result, ISACExploitResult)
        assert result.exploit_type == 'ai_poisoning'
        assert result.target_info['poisoned_samples'] == 100  # 10% of 1000
    
    def test_ai_poisoning_handover(self, isac_exploiter):
        """Test handover poisoning (CVE-2026-ISAC-008)"""
        training_data = np.random.randn(500, 32)
        
        result = isac_exploiter.ai_poison(
            training_data=training_data,
            target_system='oran_rapp',
            cve_id='CVE-2026-ISAC-008'
        )
        
        assert result.target_info['target_system'] == 'oran_rapp'
    
    def test_control_plane_hijack_monostatic_dos(self, isac_exploiter):
        """Test E2SM-RC hijack for monostatic DoS"""
        result = isac_exploiter.control_plane_hijack(
            target_node='gnb_001',
            exploit_goal='monostatic_dos'
        )
        
        assert isinstance(result, ISACExploitResult)
        assert result.exploit_type == 'control_plane_hijack'
        assert result.target_info['exploit_goal'] == 'monostatic_dos'
    
    def test_control_plane_hijack_beam_redirect(self, isac_exploiter):
        """Test E2SM-RC hijack for beam redirection"""
        result = isac_exploiter.control_plane_hijack(
            target_node='gnb_002',
            exploit_goal='beam_redirect'
        )
        
        assert result.target_info['e2_interface'] == 'E2SM-RC'
    
    @pytest.mark.skipif(not hasattr(isac_exploiter, 'QISKIT_AVAILABLE') or not isac_exploiter.__class__.__module__.startswith('falconone'), reason="Qiskit test")
    def test_quantum_attack_pns(self, isac_exploiter):
        """Test quantum PNS attack"""
        result = isac_exploiter.quantum_attack(
            target_link='qkd_link_001',
            attack_type='pns'
        )
        
        assert isinstance(result, ISACExploitResult)
        assert result.exploit_type == 'quantum_attack'
    
    @pytest.mark.skipif(not hasattr(isac_exploiter, 'QISKIT_AVAILABLE') or not isac_exploiter.__class__.__module__.startswith('falconone'), reason="Qiskit test")
    def test_quantum_attack_shor(self, isac_exploiter):
        """Test quantum Shor's algorithm attack"""
        result = isac_exploiter.quantum_attack(
            target_link='qkd_link_002',
            attack_type='shor'
        )
        
        assert result.target_info['attack_type'] == 'shor'
    
    def test_ntn_isac_doppler_manipulation(self, isac_exploiter):
        """Test NTN Doppler manipulation"""
        result = isac_exploiter.ntn_isac_exploit(
            satellite_id='LEO-001',
            exploit_type='doppler_manipulation'
        )
        
        assert isinstance(result, ISACExploitResult)
        assert result.exploit_type == 'ntn_isac_exploit'
        assert result.target_info['exploit_type'] == 'doppler_manipulation'
    
    def test_ntn_isac_handover_poison(self, isac_exploiter):
        """Test NTN handover poisoning"""
        result = isac_exploiter.ntn_isac_exploit(
            satellite_id='MEO-002',
            exploit_type='handover_poison'
        )
        
        assert result.target_info['satellite_id'] == 'MEO-002'
    
    def test_ntn_isac_cooperative_dos(self, isac_exploiter):
        """Test NTN cooperative ISAC DoS"""
        result = isac_exploiter.ntn_isac_exploit(
            satellite_id='GEO-003',
            exploit_type='cooperative_dos'
        )
        
        assert result.target_info['exploit_type'] == 'cooperative_dos'
    
    def test_generate_cve_payload_waveform(self, isac_exploiter):
        """Test CVE payload generation (waveform)"""
        payload = isac_exploiter.generate_cve_payload('CVE-2026-ISAC-001')
        
        assert payload['cve_id'] == 'CVE-2026-ISAC-001'
        assert payload['type'] == 'waveform'
        assert 'distortion_level' in payload
    
    def test_generate_cve_payload_ai_poison(self, isac_exploiter):
        """Test CVE payload generation (AI poisoning)"""
        payload = isac_exploiter.generate_cve_payload('CVE-2026-ISAC-003')
        
        assert payload['type'] == 'ai_poison'
        assert 'poisoning_rate' in payload
    
    def test_generate_cve_payload_control_plane(self, isac_exploiter):
        """Test CVE payload generation (control plane)"""
        payload = isac_exploiter.generate_cve_payload('CVE-2026-ISAC-004')
        
        assert payload['type'] == 'e2_control'
        assert payload['e2_interface'] == 'E2SM-RC'
    
    def test_generate_cve_payload_quantum(self, isac_exploiter):
        """Test CVE payload generation (quantum)"""
        payload = isac_exploiter.generate_cve_payload('CVE-2026-ISAC-005')
        
        assert payload['type'] == 'quantum'
        assert 'attack_type' in payload
    
    def test_generate_cve_payload_ntn(self, isac_exploiter):
        """Test CVE payload generation (NTN)"""
        payload = isac_exploiter.generate_cve_payload('CVE-2026-ISAC-006')
        
        assert payload['type'] == 'ntn_isac'
        assert 'satellite_id' in payload
    
    def test_generate_cve_payload_privacy(self, isac_exploiter):
        """Test CVE payload generation (privacy breach)"""
        payload = isac_exploiter.generate_cve_payload('CVE-2026-ISAC-002')
        
        assert payload['type'] == 'privacy'
        assert payload['unauthorized'] == True
    
    def test_le_mode_enforcement(self, isac_exploiter):
        """Test LE mode enforcement (requires warrant)"""
        with pytest.raises(ValueError, match="LE mode requires warrant_id"):
            isac_exploiter.waveform_manipulate(
                target_freq=150e9,
                le_mode=True
            )
    
    def test_statistics_tracking(self, isac_exploiter):
        """Test exploit statistics tracking"""
        initial_count = isac_exploiter.stats['total_exploits']
        
        isac_exploiter.waveform_manipulate(target_freq=150e9)
        
        assert isac_exploiter.stats['total_exploits'] == initial_count + 1
        assert isac_exploiter.stats['waveform_attacks'] >= 1
    
    def test_success_rate_calculation(self, isac_exploiter):
        """Test success rate calculation in statistics"""
        # Run multiple exploits
        for _ in range(5):
            isac_exploiter.waveform_manipulate(target_freq=150e9)
        
        stats = isac_exploiter.get_statistics()
        
        assert 'success_rate' in stats
        assert 0 <= stats['success_rate'] <= 1.0
    
    def test_factory_function(self, mock_sdr, mock_payload_gen):
        """Test create_isac_exploiter factory"""
        exploiter = create_isac_exploiter(mock_sdr, mock_payload_gen)
        
        assert isinstance(exploiter, ISACExploiter)


# Integration Tests

class TestISACIntegration:
    """Integration tests for ISAC monitor + exploiter"""
    
    def test_exploit_then_listen_chain(self, isac_exploiter, isac_monitor):
        """Test exploit-listen chain (waveform manipulation -> sensing leakage)"""
        # Set up exploiter with monitor
        isac_exploiter.isac_monitor = isac_monitor
        
        # Execute waveform exploit
        exploit_result = isac_exploiter.waveform_manipulate(
            target_freq=150e9,
            mode='bistatic'
        )
        
        # Verify listening enhancement
        if exploit_result.success:
            assert exploit_result.listening_enhanced or exploit_result.sensing_leaked
    
    def test_ntn_isac_integration(self, isac_exploiter):
        """Test NTN + ISAC integration"""
        # Mock NTN monitor
        mock_ntn = Mock()
        mock_ntn.get_satellite_ephemeris = Mock(return_value={'altitude_km': 550})
        isac_exploiter.ntn_monitor = mock_ntn
        
        # Execute NTN ISAC exploit
        result = isac_exploiter.ntn_isac_exploit(
            satellite_id='LEO-001',
            exploit_type='doppler_manipulation'
        )
        
        assert 'satellite_context' in result.target_info
    
    def test_end_to_end_isac_workflow(self, isac_monitor, isac_exploiter):
        """Test end-to-end ISAC workflow (monitor -> exploit -> listen)"""
        # Step 1: Initial sensing
        initial_sense = isac_monitor.start_sensing(mode='cooperative', duration_sec=3)
        assert initial_sense.sensing_accuracy > 0
        
        # Step 2: Exploit waveform
        isac_exploiter.isac_monitor = isac_monitor
        exploit_result = isac_exploiter.waveform_manipulate(target_freq=150e9, mode='cooperative')
        
        # Step 3: Enhanced listening (captured in exploit)
        assert exploit_result.exploit_type == 'waveform_manipulation'


# Performance Benchmarks

class TestISACPerformance:
    """Performance benchmarks for ISAC operations"""
    
    def test_sensing_speed(self, isac_monitor, benchmark):
        """Benchmark sensing speed (<50ms target)"""
        def sensing():
            return isac_monitor.start_sensing(mode='monostatic', duration_sec=1)
        
        result = benchmark(sensing)
        # Note: Benchmark will report timing automatically
    
    def test_waveform_exploit_speed(self, isac_exploiter, benchmark):
        """Benchmark waveform exploit speed (<30ms target)"""
        def exploit():
            return isac_exploiter.waveform_manipulate(target_freq=150e9)
        
        result = benchmark(exploit)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=falconone.monitoring.isac_monitor', 
                 '--cov=falconone.exploit.isac_exploiter', '--cov-report=term'])
