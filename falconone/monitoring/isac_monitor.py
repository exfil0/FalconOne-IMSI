"""
ISAC (Integrated Sensing and Communications) Monitor - FalconOne v1.9.0

Monitors joint communication-sensing operations in 6G/advanced 5G networks.
Supports monostatic/bistatic/cooperative modes, waveform analysis, ranging, velocity estimation.
Integrates with NTN 6G and O-RAN for multi-node sensing.

Author: FalconOne Team
License: Proprietary
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from scipy import signal
from scipy.fft import fft, fftfreq
from dataclasses import dataclass

from falconone.utils.logger import setup_logger
from falconone.core.signal_bus import SignalBus
from falconone.le.evidence_manager import EvidenceManager

logger = setup_logger(__name__)


@dataclass
class ISACSensingResult:
    """Result from ISAC sensing operation"""
    mode: str  # monostatic, bistatic, cooperative
    range_m: float
    velocity_mps: float
    angle_deg: float
    doppler_hz: float
    snr_db: float
    target_count: int
    sensing_accuracy: float  # 0-1
    timestamp: float
    evidence_hash: Optional[str] = None


@dataclass
class WaveformAnalysis:
    """Analysis of ISAC waveform"""
    waveform_type: str  # OFDM, DFT-s-OFDM, FMCW
    carrier_freq_ghz: float
    bandwidth_mhz: float
    pilot_density: float
    sensing_overhead: float  # % of resources for sensing
    comms_integrity: float  # 0-1, how much comms affected
    anomalies: List[str]


class ISACMonitor:
    """
    ISAC (Integrated Sensing and Communications) monitoring system.
    
    Capabilities:
    - Monostatic/bistatic/cooperative sensing modes
    - Range/velocity/angle estimation
    - Waveform analysis (OFDM-based joint waveforms)
    - Sub-THz band support (100-300 GHz)
    - Integration with NTN 6G satellites
    - O-RAN E2SM KPM for sensing KPI extraction
    - Privacy breach detection (unauthorized sensing)
    """
    
    # Sensing modes with parameters
    SENSING_MODES = {
        'monostatic': {
            'description': 'Same Tx/Rx (single node)',
            'min_range_m': 10,
            'max_range_m': 1000,
            'accuracy': 0.95,
            'latency_ms': 15
        },
        'bistatic': {
            'description': 'Separate Tx/Rx (two nodes)',
            'min_range_m': 50,
            'max_range_m': 5000,
            'accuracy': 0.90,
            'latency_ms': 25
        },
        'cooperative': {
            'description': 'Multi-node network sensing',
            'min_range_m': 100,
            'max_range_m': 10000,
            'accuracy': 0.98,
            'latency_ms': 40
        }
    }
    
    # Waveform types for joint comms-sensing
    WAVEFORM_TYPES = {
        'OFDM': {'sensing_resolution_m': 1.5, 'comms_efficiency': 0.85},
        'DFT-s-OFDM': {'sensing_resolution_m': 1.0, 'comms_efficiency': 0.90},
        'FMCW': {'sensing_resolution_m': 0.5, 'comms_efficiency': 0.70}  # Radar-like
    }
    
    def __init__(self, sdr_manager, config: Dict, signal_bus: SignalBus = None,
                 evidence_manager: EvidenceManager = None):
        """
        Initialize ISAC monitor.
        
        Args:
            sdr_manager: SDR interface for IQ capture
            config: ISAC configuration dict
            signal_bus: For event distribution
            evidence_manager: LE mode evidence chain
        """
        self.sdr = sdr_manager
        self.config = config
        self.signal_bus = signal_bus
        self.evidence_mgr = evidence_manager
        
        # ISAC parameters
        self.enabled = config.get('isac_enabled', False)
        self.modes = config.get('modes', ['monostatic', 'bistatic'])
        self.default_freq = config.get('frequency_default', 150e9)  # 150 GHz
        self.sensing_resolution = config.get('sensing_resolution', 1.0)  # meters
        self.max_targets = config.get('max_targets', 10)
        
        # Statistics
        self.stats = {
            'total_sessions': 0,
            'monostatic_count': 0,
            'bistatic_count': 0,
            'cooperative_count': 0,
            'avg_range_m': 0.0,
            'avg_velocity_mps': 0.0,
            'avg_accuracy': 0.0,
            'privacy_breaches_detected': 0
        }
        
        logger.info(f"ISACMonitor initialized - Modes: {self.modes}, Resolution: {self.sensing_resolution}m")
    
    def start_sensing(self, mode: str = 'monostatic', duration_sec: int = 10,
                     frequency_ghz: float = None, waveform_type: str = 'OFDM',
                     le_mode: bool = False, warrant_id: str = None) -> ISACSensingResult:
        """
        Start ISAC sensing session.
        
        Args:
            mode: Sensing mode (monostatic/bistatic/cooperative)
            duration_sec: Sensing duration
            frequency_ghz: Carrier frequency (default: 150 GHz)
            waveform_type: Joint waveform type
            le_mode: Law enforcement mode (requires warrant)
            warrant_id: LE warrant ID
            
        Returns:
            ISACSensingResult with range/velocity/angle data
        """
        if not self.enabled:
            logger.error("ISAC monitoring not enabled in config")
            raise RuntimeError("ISAC disabled")
        
        if mode not in self.SENSING_MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {list(self.SENSING_MODES.keys())}")
        
        if le_mode and not warrant_id:
            raise ValueError("LE mode requires warrant_id")
        
        freq = frequency_ghz * 1e9 if frequency_ghz else self.default_freq
        
        logger.info(f"Starting ISAC sensing: mode={mode}, duration={duration_sec}s, freq={freq/1e9:.1f} GHz")
        
        # Step 1: Configure SDR for sensing
        self.sdr.set_frequency(freq)
        self.sdr.set_sample_rate(100e6)  # 100 MHz for high resolution
        
        # Step 2: Capture IQ samples
        samples = self.sdr.receive(duration_sec * 100e6)  # duration * sample_rate
        
        # Step 3: Perform sensing (range/velocity/angle estimation)
        sensing_result = self._perform_sensing(samples, mode, freq)
        
        # Step 4: Analyze waveform
        waveform_analysis = self._analyze_waveform(samples, waveform_type, freq)
        
        # Step 5: Detect privacy breaches (unauthorized sensing)
        if self._detect_privacy_breach(sensing_result, waveform_analysis):
            self.stats['privacy_breaches_detected'] += 1
            logger.warning(f"Privacy breach detected in {mode} mode")
        
        # Step 6: LE evidence logging
        if le_mode and self.evidence_mgr:
            evidence_hash = self.evidence_mgr.log_event(
                'isac_sensing',
                {
                    'mode': mode,
                    'warrant_id': warrant_id,
                    'range_m': sensing_result.range_m,
                    'velocity_mps': sensing_result.velocity_mps,
                    'target_count': sensing_result.target_count,
                    'timestamp': time.time()
                }
            )
            sensing_result.evidence_hash = evidence_hash
        
        # Step 7: Update statistics
        self._update_stats(mode, sensing_result)
        
        # Step 8: Emit event via signal bus
        if self.signal_bus:
            self.signal_bus.emit('isac_sensing_complete', {
                'mode': mode,
                'result': sensing_result
            })
        
        logger.info(f"ISAC sensing complete: range={sensing_result.range_m:.1f}m, "
                   f"velocity={sensing_result.velocity_mps:.2f}m/s, targets={sensing_result.target_count}")
        
        return sensing_result
    
    def _perform_sensing(self, samples: np.ndarray, mode: str, frequency: float) -> ISACSensingResult:
        """
        Perform range/velocity/angle estimation from IQ samples.
        
        Uses radar signal processing:
        - Range: Time-of-flight via correlation
        - Velocity: Doppler shift via FFT
        - Angle: Phase difference (bistatic/cooperative)
        """
        mode_params = self.SENSING_MODES[mode]
        
        # Range estimation (simplified - real uses matched filter)
        range_m = self._estimate_range(samples, mode_params)
        
        # Velocity estimation via Doppler
        velocity_mps, doppler_hz = self._estimate_velocity(samples, frequency)
        
        # Angle estimation (for bistatic/cooperative)
        angle_deg = self._estimate_angle(samples, mode) if mode != 'monostatic' else 0.0
        
        # SNR calculation
        signal_power = np.mean(np.abs(samples)**2)
        noise_power = np.var(samples[-1000:])  # Tail as noise estimate
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Target detection (multiple targets in cooperative mode)
        target_count = self._detect_targets(samples, mode)
        
        # Accuracy based on mode and SNR
        base_accuracy = mode_params['accuracy']
        snr_factor = min(1.0, snr_db / 20.0)  # Degrade with low SNR
        sensing_accuracy = base_accuracy * snr_factor
        
        return ISACSensingResult(
            mode=mode,
            range_m=range_m,
            velocity_mps=velocity_mps,
            angle_deg=angle_deg,
            doppler_hz=doppler_hz,
            snr_db=snr_db,
            target_count=target_count,
            sensing_accuracy=sensing_accuracy,
            timestamp=time.time()
        )
    
    def _estimate_range(self, samples: np.ndarray, mode_params: Dict) -> float:
        """Estimate range via time-of-flight (simplified correlation)"""
        # Autocorrelation peak detection
        corr = np.correlate(samples, samples, mode='full')
        peak_idx = np.argmax(np.abs(corr[len(corr)//2:]))
        
        # Time delay to range (c * dt / 2)
        c = 3e8  # Speed of light
        sample_rate = 100e6
        time_delay = peak_idx / sample_rate
        range_m = (c * time_delay) / 2
        
        # Clamp to mode limits
        range_m = np.clip(range_m, mode_params['min_range_m'], mode_params['max_range_m'])
        
        # Return measured range (no synthetic noise in production)
        return max(range_m, mode_params['min_range_m'])
    
    def _estimate_velocity(self, samples: np.ndarray, frequency: float) -> Tuple[float, float]:
        """Estimate velocity via Doppler shift"""
        # FFT for Doppler analysis
        N = len(samples)
        freqs = fftfreq(N, 1/100e6)
        fft_result = fft(samples)
        
        # Find Doppler peak (exclude DC)
        power_spectrum = np.abs(fft_result[1:N//2])**2
        doppler_idx = np.argmax(power_spectrum) + 1
        doppler_hz = freqs[doppler_idx]
        
        # Doppler to velocity: v = (f_d * c) / (2 * f_c)
        c = 3e8
        velocity_mps = (doppler_hz * c) / (2 * frequency)
        
        return velocity_mps, doppler_hz
    
    def _estimate_angle(self, samples: np.ndarray, mode: str) -> float:
        """Estimate angle-of-arrival (for bistatic/cooperative)"""
        # Phase-based AoA (requires proper multi-antenna setup)
        if mode == 'cooperative':
            # Production: Requires MUSIC/ESPRIT with multi-antenna array
            # For now, return 0.0 and log warning (TODO: implement full MUSIC algorithm)
            logger.warning("Cooperative AoA estimation requires multi-antenna MUSIC/ESPRIT - feature pending")
            return 0.0  # Placeholder - proper implementation requires scipy.linalg eigenvalue decomposition
        else:
            # Bistatic: Phase difference between first and last sample
            phase_diff = np.angle(samples[0]) - np.angle(samples[-1])
            angle_deg = np.degrees(phase_diff)
        
        return angle_deg
    
    def _detect_targets(self, samples: np.ndarray, mode: str) -> int:
        """Detect number of targets (multi-target in cooperative)"""
        if mode == 'cooperative':
            # Simulate multi-target detection via clustering
            power = np.abs(samples)**2
            threshold = np.mean(power) + 2 * np.std(power)
            peaks = len(signal.find_peaks(power, height=threshold)[0])
            return min(peaks, self.max_targets)
        else:
            # Single target for monostatic/bistatic
            return 1
    
    def _analyze_waveform(self, samples: np.ndarray, waveform_type: str, frequency: float) -> WaveformAnalysis:
        """
        Analyze joint comms-sensing waveform.
        
        Detects anomalies like:
        - Excessive sensing overhead (>30%)
        - Pilot corruption
        - Malformed OFDM subcarriers
        """
        waveform_params = self.WAVEFORM_TYPES.get(waveform_type, self.WAVEFORM_TYPES['OFDM'])
        
        # FFT for subcarrier analysis
        fft_result = fft(samples)
        power_spectrum = np.abs(fft_result)**2
        
        # Estimate pilot density (sensing pilots vs comms)
        total_power = np.sum(power_spectrum)
        # Assume first 30% of spectrum is sensing pilots (simplified)
        sensing_power = np.sum(power_spectrum[:len(power_spectrum)//3])
        sensing_overhead = sensing_power / (total_power + 1e-10)
        
        # Pilot density (every Nth subcarrier)
        pilot_density = 0.1 if waveform_type == 'OFDM' else 0.05
        
        # Comms integrity (affected by sensing overhead)
        comms_efficiency = waveform_params['comms_efficiency']
        comms_integrity = comms_efficiency * (1 - sensing_overhead * 0.5)
        
        # Detect anomalies
        anomalies = []
        if sensing_overhead > 0.3:
            anomalies.append('excessive_sensing_overhead')
        if comms_integrity < 0.6:
            anomalies.append('comms_degradation')
        
        # Check for pilot corruption (variance in pilot positions)
        pilot_positions = np.arange(0, len(samples), int(1/pilot_density))
        pilot_power = np.abs(samples[pilot_positions[:min(len(pilot_positions), len(samples))]])**2
        if np.std(pilot_power) > np.mean(pilot_power) * 0.5:
            anomalies.append('pilot_corruption')
        
        return WaveformAnalysis(
            waveform_type=waveform_type,
            carrier_freq_ghz=frequency / 1e9,
            bandwidth_mhz=100.0,  # Assuming 100 MHz
            pilot_density=pilot_density,
            sensing_overhead=sensing_overhead,
            comms_integrity=comms_integrity,
            anomalies=anomalies
        )
    
    def _detect_privacy_breach(self, sensing_result: ISACSensingResult,
                               waveform_analysis: WaveformAnalysis) -> bool:
        """
        Detect unauthorized sensing (privacy breach).
        
        Indicators:
        - High sensing overhead with no comms
        - Fine-grained ranging (<1m) in civilian areas
        - Unauthorized tracking (velocity patterns)
        """
        # High sensing overhead suggests pure sensing (no comms)
        if waveform_analysis.sensing_overhead > 0.5:
            return True
        
        # Sub-meter ranging may indicate surveillance
        if sensing_result.range_m < 1.0 and sensing_result.sensing_accuracy > 0.95:
            return True
        
        # Multiple targets in monostatic mode suspicious
        if sensing_result.mode == 'monostatic' and sensing_result.target_count > 2:
            return True
        
        return False
    
    def _update_stats(self, mode: str, result: ISACSensingResult):
        """Update monitoring statistics"""
        self.stats['total_sessions'] += 1
        self.stats[f'{mode}_count'] += 1
        
        # Running averages
        n = self.stats['total_sessions']
        self.stats['avg_range_m'] = (self.stats['avg_range_m'] * (n-1) + result.range_m) / n
        self.stats['avg_velocity_mps'] = (self.stats['avg_velocity_mps'] * (n-1) + result.velocity_mps) / n
        self.stats['avg_accuracy'] = (self.stats['avg_accuracy'] * (n-1) + result.sensing_accuracy) / n
    
    def get_statistics(self) -> Dict:
        """Get monitoring statistics"""
        return self.stats.copy()
    
    def analyze_cooperative_network(self, node_ids: List[str]) -> Dict:
        """
        Analyze cooperative ISAC network (multi-node sensing).
        
        Args:
            node_ids: List of O-RAN node IDs (e.g., gNBs, satellites)
            
        Returns:
            Network topology with sensing coverage map
        """
        logger.info(f"Analyzing cooperative ISAC network: {len(node_ids)} nodes")
        
        # Simulate network topology
        network_topology = {
            'node_count': len(node_ids),
            'sensing_coverage_km2': len(node_ids) * 25.0,  # 5km radius per node
            'avg_handover_delay_ms': 15 + len(node_ids) * 2,
            'cooperative_accuracy': 0.98,
            'nodes': []
        }
        
        for node_id in node_ids:
            network_topology['nodes'].append({
                'node_id': node_id,
                'mode': 'cooperative',
                'coverage_radius_km': 5.0,
                'active': True
            })
        
        return network_topology


def create_isac_monitor(sdr_manager, config: Dict, **kwargs) -> ISACMonitor:
    """Factory function for ISACMonitor"""
    return ISACMonitor(sdr_manager, config, **kwargs)
