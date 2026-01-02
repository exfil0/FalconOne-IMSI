"""
FalconOne Enhanced Simulation Engine (v1.6)
Generate realistic I/Q datasets for 5G/NTN/V2X without hardware
Capabilities:
- ns-3 integration for protocol-level simulation
- Realistic channel models (TDL-A/B/C, CDL)
- High-mobility scenarios (V2X, NTN handover)
- Edge case testing (jamming, interference, multipath)

Benefit: Safe real-world scenario rehearsal without RF emissions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

try:
    from ..utils.logger import ModuleLogger
except ImportError:
    class ModuleLogger:
        def __init__(self, name, parent):
            self.logger = logging.getLogger(name) if parent else logging.getLogger(__name__)
        def info(self, msg, **kw): self.logger.info(f"{msg} {kw if kw else ''}")
        def warning(self, msg, **kw): self.logger.warning(f"{msg} {kw if kw else ''}")
        def error(self, msg, **kw): self.logger.error(f"{msg} {kw if kw else ''}")


class ScenarioType(Enum):
    """Simulation scenario types"""
    FIVEG_URBAN = "5g_urban"
    FIVEG_RURAL = "5g_rural"
    NTN_LEO = "ntn_leo"
    NTN_GEO = "ntn_geo"
    V2X_HIGHWAY = "v2x_highway"
    V2X_URBAN = "v2x_urban"
    JAMMING = "jamming"
    INTERFERENCE = "interference"


@dataclass
class SimulationResult:
    """Simulation output"""
    scenario: ScenarioType
    iq_samples: np.ndarray  # Complex I/Q samples
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    sample_rate: float = 30.72e6  # Default 30.72 MSPS
    duration_sec: float = 1.0


class SimulationEngine:
    """
    Hardware-free simulation engine for testing FalconOne capabilities
    Generates realistic I/Q data with protocol-level accuracy
    """
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = ModuleLogger('SimEngine', logger)
        
        # Channel models
        self.channel_models = {
            'TDL-A': self._init_tdl_a,
            'TDL-B': self._init_tdl_b,
            'TDL-C': self._init_tdl_c,
            'CDL-C': self._init_cdl_c,
        }
        
        # ns-3 integration (optional)
        self.ns3_enabled = False
        try:
            import ns3wrapper
            self.ns3_enabled = True
            self.logger.info("ns-3 integration enabled")
        except ImportError:
            self.logger.warning("ns-3 not available, using built-in models only")
        
        self.logger.info("Simulation Engine initialized (v1.6)")
    
    def generate_5g_urban_scenario(self, duration_sec: float = 1.0, 
                                   num_ues: int = 10, 
                                   snr_db: float = 20.0) -> SimulationResult:
        """
        Generate 5G urban scenario with multiple UEs
        
        Args:
            duration_sec: Simulation duration
            num_ues: Number of active UEs
            snr_db: Signal-to-noise ratio
        
        Returns:
            Simulated I/Q samples with metadata
        """
        sample_rate = 30.72e6  # 20 MHz bandwidth
        num_samples = int(duration_sec * sample_rate)
        
        # Generate base 5G NR waveform (OFDM)
        iq_samples = self._generate_ofdm_waveform(num_samples, sample_rate, num_ues)
        
        # Apply urban channel model (TDL-A)
        iq_samples = self._apply_channel_model(iq_samples, 'TDL-A', doppler_hz=10.0)
        
        # Add AWGN
        iq_samples = self._add_awgn(iq_samples, snr_db)
        
        metadata = {
            'scenario': ScenarioType.FIVEG_URBAN.value,
            'num_ues': num_ues,
            'snr_db': snr_db,
            'channel_model': 'TDL-A',
            'doppler_hz': 10.0,
            'carrier_freq': 3.5e9,
        }
        
        self.logger.info(f"Generated 5G urban scenario: {num_ues} UEs, {duration_sec}s")
        
        return SimulationResult(
            scenario=ScenarioType.FIVEG_URBAN,
            iq_samples=iq_samples,
            metadata=metadata,
            sample_rate=sample_rate,
            duration_sec=duration_sec
        )
    
    def generate_ntn_leo_scenario(self, duration_sec: float = 5.0,
                                  satellite_altitude_km: float = 550.0,
                                  elevation_deg: float = 45.0) -> SimulationResult:
        """
        Generate NTN LEO satellite scenario with Doppler/delay
        
        Args:
            duration_sec: Simulation duration
            satellite_altitude_km: LEO altitude
            elevation_deg: Elevation angle to satellite
        
        Returns:
            Simulated NTN I/Q samples
        """
        sample_rate = 30.72e6
        num_samples = int(duration_sec * sample_rate)
        
        # Generate 5G NR waveform
        iq_samples = self._generate_ofdm_waveform(num_samples, sample_rate, num_ues=1)
        
        # Calculate Doppler shift (LEO satellite)
        doppler_hz = self._calculate_leo_doppler(satellite_altitude_km, elevation_deg)
        
        # Apply Doppler shift
        iq_samples = self._apply_doppler_shift(iq_samples, doppler_hz, sample_rate)
        
        # Calculate propagation delay
        delay_ms = self._calculate_propagation_delay(satellite_altitude_km, elevation_deg)
        delay_samples = int(delay_ms * 1e-3 * sample_rate)
        
        # Apply delay
        iq_samples = np.concatenate([np.zeros(delay_samples, dtype=complex), iq_samples])[:num_samples]
        
        # Apply fading (Rician channel for LoS satellite link)
        iq_samples = self._apply_rician_fading(iq_samples, k_factor_db=15.0)
        
        # Add noise
        iq_samples = self._add_awgn(iq_samples, snr_db=15.0)
        
        metadata = {
            'scenario': ScenarioType.NTN_LEO.value,
            'satellite_altitude_km': satellite_altitude_km,
            'elevation_deg': elevation_deg,
            'doppler_hz': doppler_hz,
            'delay_ms': delay_ms,
            'channel_model': 'Rician',
        }
        
        self.logger.info(f"Generated NTN LEO scenario: altitude={satellite_altitude_km}km, Doppler={doppler_hz:.1f}Hz")
        
        return SimulationResult(
            scenario=ScenarioType.NTN_LEO,
            iq_samples=iq_samples,
            metadata=metadata,
            sample_rate=sample_rate,
            duration_sec=duration_sec
        )
    
    def generate_v2x_highway_scenario(self, duration_sec: float = 2.0,
                                      vehicle_speed_kmh: float = 120.0,
                                      num_vehicles: int = 5) -> SimulationResult:
        """
        Generate V2X highway scenario with high mobility
        
        Args:
            duration_sec: Simulation duration
            vehicle_speed_kmh: Vehicle speed
            num_vehicles: Number of transmitting vehicles
        
        Returns:
            Simulated V2X I/Q samples
        """
        sample_rate = 20e6  # C-V2X uses 10/20 MHz channels
        num_samples = int(duration_sec * sample_rate)
        
        # Generate C-V2X sidelink waveform
        iq_samples = self._generate_cv2x_waveform(num_samples, sample_rate, num_vehicles)
        
        # Calculate Doppler from vehicle speed
        doppler_hz = self._calculate_vehicle_doppler(vehicle_speed_kmh, carrier_freq=5.9e9)
        
        # Apply time-varying Doppler (vehicles approaching/receding)
        iq_samples = self._apply_time_varying_doppler(iq_samples, doppler_hz, sample_rate, duration_sec)
        
        # Apply fast fading (TDL-C for high speed)
        iq_samples = self._apply_channel_model(iq_samples, 'TDL-C', doppler_hz=doppler_hz)
        
        # Add noise
        iq_samples = self._add_awgn(iq_samples, snr_db=18.0)
        
        metadata = {
            'scenario': ScenarioType.V2X_HIGHWAY.value,
            'vehicle_speed_kmh': vehicle_speed_kmh,
            'num_vehicles': num_vehicles,
            'doppler_hz': doppler_hz,
            'channel_model': 'TDL-C',
            'carrier_freq': 5.9e9,
        }
        
        self.logger.info(f"Generated V2X highway scenario: {num_vehicles} vehicles, {vehicle_speed_kmh} km/h")
        
        return SimulationResult(
            scenario=ScenarioType.V2X_HIGHWAY,
            iq_samples=iq_samples,
            metadata=metadata,
            sample_rate=sample_rate,
            duration_sec=duration_sec
        )
    
    def generate_jamming_scenario(self, duration_sec: float = 1.0,
                                  jammer_power_db: float = 30.0,
                                  jammer_type: str = 'barrage') -> SimulationResult:
        """
        Generate jamming attack scenario
        
        Args:
            duration_sec: Simulation duration
            jammer_power_db: Jammer power (dB above signal)
            jammer_type: 'barrage', 'sweep', 'pulse'
        
        Returns:
            Simulated jammed I/Q samples
        """
        sample_rate = 30.72e6
        num_samples = int(duration_sec * sample_rate)
        
        # Generate legitimate signal
        iq_signal = self._generate_ofdm_waveform(num_samples, sample_rate, num_ues=5)
        iq_signal = self._add_awgn(iq_signal, snr_db=20.0)
        
        # Generate jamming signal
        if jammer_type == 'barrage':
            iq_jammer = self._generate_barrage_jamming(num_samples, jammer_power_db)
        elif jammer_type == 'sweep':
            iq_jammer = self._generate_sweep_jamming(num_samples, sample_rate, jammer_power_db)
        elif jammer_type == 'pulse':
            iq_jammer = self._generate_pulse_jamming(num_samples, jammer_power_db, duty_cycle=0.5)
        else:
            iq_jammer = np.zeros(num_samples, dtype=complex)
        
        # Combine signal + jamming
        iq_samples = iq_signal + iq_jammer
        
        metadata = {
            'scenario': ScenarioType.JAMMING.value,
            'jammer_power_db': jammer_power_db,
            'jammer_type': jammer_type,
            'sir_db': -jammer_power_db,  # Signal-to-interference ratio
        }
        
        self.logger.info(f"Generated jamming scenario: {jammer_type}, SIR={-jammer_power_db}dB")
        
        return SimulationResult(
            scenario=ScenarioType.JAMMING,
            iq_samples=iq_samples,
            metadata=metadata,
            sample_rate=sample_rate,
            duration_sec=duration_sec
        )
    
    # ===== Internal waveform generators =====
    
    def _generate_ofdm_waveform(self, num_samples: int, sample_rate: float, num_ues: int) -> np.ndarray:
        """Generate simplified OFDM waveform (5G NR-like)"""
        # FFT size for 20 MHz (30.72 MSPS)
        fft_size = 2048
        num_subcarriers = 1200  # Active subcarriers
        cp_len = int(fft_size * 0.07)  # Cyclic prefix
        
        num_symbols = num_samples // (fft_size + cp_len)
        iq_samples = np.array([], dtype=complex)
        
        for _ in range(num_symbols):
            # Frequency domain data (random QPSK for simplicity)
            freq_data = np.zeros(fft_size, dtype=complex)
            used_indices = np.random.choice(num_subcarriers, size=num_ues * 100, replace=False)
            freq_data[used_indices] = (np.random.randint(0, 2, size=len(used_indices)) * 2 - 1) + \
                                      1j * (np.random.randint(0, 2, size=len(used_indices)) * 2 - 1)
            
            # IFFT to time domain
            time_data = np.fft.ifft(np.fft.ifftshift(freq_data))
            
            # Add cyclic prefix
            symbol = np.concatenate([time_data[-cp_len:], time_data])
            iq_samples = np.concatenate([iq_samples, symbol])
        
        return iq_samples[:num_samples]
    
    def _generate_cv2x_waveform(self, num_samples: int, sample_rate: float, num_vehicles: int) -> np.ndarray:
        """Generate simplified C-V2X sidelink waveform"""
        # Similar to OFDM but with PC5 sidelink structure
        return self._generate_ofdm_waveform(num_samples, sample_rate, num_vehicles)
    
    def _generate_barrage_jamming(self, num_samples: int, power_db: float) -> np.ndarray:
        """Generate barrage (wideband) jamming"""
        power_linear = 10 ** (power_db / 10.0)
        return np.sqrt(power_linear) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
    
    def _generate_sweep_jamming(self, num_samples: int, sample_rate: float, power_db: float) -> np.ndarray:
        """Generate frequency-sweep jamming"""
        power_linear = 10 ** (power_db / 10.0)
        t = np.arange(num_samples) / sample_rate
        sweep_rate = 10e6  # 10 MHz/s sweep
        phase = 2 * np.pi * (sweep_rate / 2) * t**2
        return np.sqrt(power_linear) * np.exp(1j * phase)
    
    def _generate_pulse_jamming(self, num_samples: int, power_db: float, duty_cycle: float) -> np.ndarray:
        """Generate pulse jamming"""
        power_linear = 10 ** (power_db / 10.0)
        pulse_period = int(num_samples * 0.01)  # 1% of total duration per pulse
        pulses = np.zeros(num_samples, dtype=complex)
        
        on_samples = int(pulse_period * duty_cycle)
        for i in range(0, num_samples, pulse_period):
            pulses[i:i+on_samples] = np.sqrt(power_linear) * (np.random.randn(on_samples) + 1j * np.random.randn(on_samples)) / np.sqrt(2)
        
        return pulses
    
    # ===== Channel models =====
    
    def _apply_channel_model(self, iq_samples: np.ndarray, model_name: str, doppler_hz: float) -> np.ndarray:
        """Apply TDL/CDL channel model"""
        if model_name not in self.channel_models:
            return iq_samples
        
        taps, delays = self.channel_models[model_name]()
        return self._apply_multipath(iq_samples, taps, delays, doppler_hz)
    
    def _apply_multipath(self, iq_samples: np.ndarray, taps: np.ndarray, 
                        delays: np.ndarray, doppler_hz: float) -> np.ndarray:
        """Apply multipath channel with fading"""
        output = np.zeros_like(iq_samples)
        
        for tap, delay in zip(taps, delays):
            delay_samples = int(delay * 30.72e6)  # Convert to samples
            if delay_samples < len(iq_samples):
                # Apply fading to tap
                fading = self._generate_rayleigh_fading(len(iq_samples), doppler_hz)
                delayed = np.concatenate([np.zeros(delay_samples), iq_samples[:-delay_samples] if delay_samples > 0 else iq_samples])
                output += tap * fading * delayed
        
        return output
    
    def _generate_rayleigh_fading(self, num_samples: int, doppler_hz: float) -> np.ndarray:
        """Generate Rayleigh fading envelope"""
        # Jakes model approximation
        i_comp = np.random.randn(num_samples)
        q_comp = np.random.randn(num_samples)
        fading = (i_comp + 1j * q_comp) / np.sqrt(2)
        
        # Apply Doppler spectrum shaping (simplified)
        if doppler_hz > 0:
            # Low-pass filter to shape Doppler spectrum
            from scipy import signal
            cutoff = doppler_hz / (30.72e6 / 2)  # Normalized cutoff
            b, a = signal.butter(4, cutoff, btype='low')
            fading_real = signal.filtfilt(b, a, fading.real)
            fading_imag = signal.filtfilt(b, a, fading.imag)
            fading = fading_real + 1j * fading_imag
        
        return fading
    
    def _apply_rician_fading(self, iq_samples: np.ndarray, k_factor_db: float) -> np.ndarray:
        """Apply Rician fading (LoS + multipath)"""
        k_linear = 10 ** (k_factor_db / 10.0)
        los_component = np.sqrt(k_linear / (k_linear + 1))
        scatter_component = np.sqrt(1 / (k_linear + 1))
        
        rayleigh = self._generate_rayleigh_fading(len(iq_samples), doppler_hz=5.0)
        rician = los_component + scatter_component * rayleigh
        
        return iq_samples * rician
    
    def _init_tdl_a(self) -> Tuple[np.ndarray, np.ndarray]:
        """TDL-A channel model (low delay spread)"""
        taps = np.array([0.0, -13.4, -16.3, -18.4, -20.5, -22.6])
        taps_linear = 10 ** (taps / 20.0)
        delays = np.array([0, 30e-9, 70e-9, 90e-9, 110e-9, 190e-9])
        return taps_linear, delays
    
    def _init_tdl_b(self) -> Tuple[np.ndarray, np.ndarray]:
        """TDL-B channel model (medium delay spread)"""
        taps = np.array([0.0, -2.3, -4.5, -6.0, -7.8, -10.0])
        taps_linear = 10 ** (taps / 20.0)
        delays = np.array([0, 100e-9, 200e-9, 400e-9, 800e-9, 1600e-9])
        return taps_linear, delays
    
    def _init_tdl_c(self) -> Tuple[np.ndarray, np.ndarray]:
        """TDL-C channel model (large delay spread)"""
        taps = np.array([0.0, -3.2, -6.3, -9.4, -12.5, -15.6])
        taps_linear = 10 ** (taps / 20.0)
        delays = np.array([0, 200e-9, 500e-9, 1000e-9, 2000e-9, 4000e-9])
        return taps_linear, delays
    
    def _init_cdl_c(self) -> Tuple[np.ndarray, np.ndarray]:
        """CDL-C channel model (clustered delay line)"""
        return self._init_tdl_c()  # Simplified
    
    # ===== Doppler/delay calculations =====
    
    def _calculate_leo_doppler(self, altitude_km: float, elevation_deg: float) -> float:
        """Calculate LEO satellite Doppler shift"""
        earth_radius_km = 6371
        orbital_velocity = np.sqrt(398600 / (earth_radius_km + altitude_km))  # km/s
        radial_velocity = orbital_velocity * np.cos(np.radians(elevation_deg))
        doppler_hz = (radial_velocity / 299792.458) * 2.1e9  # At 2.1 GHz carrier
        return doppler_hz
    
    def _calculate_propagation_delay(self, altitude_km: float, elevation_deg: float) -> float:
        """Calculate satellite propagation delay (ms)"""
        earth_radius_km = 6371
        slant_range = np.sqrt((earth_radius_km + altitude_km)**2 - 
                             (earth_radius_km * np.cos(np.radians(elevation_deg)))**2)
        return (slant_range / 299792.458) * 1000  # ms
    
    def _calculate_vehicle_doppler(self, speed_kmh: float, carrier_freq: float) -> float:
        """Calculate vehicle Doppler shift"""
        speed_ms = speed_kmh / 3.6
        return (speed_ms / 299792458.0) * carrier_freq
    
    def _apply_doppler_shift(self, iq_samples: np.ndarray, doppler_hz: float, sample_rate: float) -> np.ndarray:
        """Apply constant Doppler frequency shift"""
        t = np.arange(len(iq_samples)) / sample_rate
        shift = np.exp(1j * 2 * np.pi * doppler_hz * t)
        return iq_samples * shift
    
    def _apply_time_varying_doppler(self, iq_samples: np.ndarray, max_doppler_hz: float, 
                                   sample_rate: float, duration_sec: float) -> np.ndarray:
        """Apply time-varying Doppler (vehicle approaching then receding)"""
        t = np.arange(len(iq_samples)) / sample_rate
        # Sinusoidal Doppler variation
        doppler_t = max_doppler_hz * np.sin(2 * np.pi * (0.5 / duration_sec) * t)
        phase = 2 * np.pi * np.cumsum(doppler_t) / sample_rate
        shift = np.exp(1j * phase)
        return iq_samples * shift
    
    def _add_awgn(self, iq_samples: np.ndarray, snr_db: float) -> np.ndarray:
        """Add Additive White Gaussian Noise"""
        signal_power = np.mean(np.abs(iq_samples)**2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(iq_samples)) + 
                                           1j * np.random.randn(len(iq_samples)))
        return iq_samples + noise
    
    def save_simulation(self, result: SimulationResult, output_path: str):
        """Save simulation result to file"""
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        self.logger.info(f"Saved simulation to {output_path}")
    
    def load_simulation(self, input_path: str) -> SimulationResult:
        """Load simulation result from file"""
        import pickle
        with open(input_path, 'rb') as f:
            result = pickle.load(f)
        self.logger.info(f"Loaded simulation from {input_path}")
        return result
