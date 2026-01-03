"""
FalconOne Hardware-in-Loop Tests (v1.9.3)
=========================================
Hardware-in-loop (HIL) tests for SDR integration validation.
Tests include:
- SDR mock interfaces with realistic behavior
- Timing validation for real-time constraints
- Signal injection and verification
- Hardware abstraction layer testing
- End-to-end signal flow validation

Requirements:
- pytest
- numpy
- scipy (for signal processing)

Run with:
    pytest falconone/tests/test_hardware.py -v --tb=short

Note: These tests use mock SDR interfaces. For actual hardware testing,
set FALCONONE_HIL_HARDWARE=1 environment variable.
"""

import pytest
import numpy as np
import time
import threading
import queue
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
import struct
import cmath
import math


# ==================== SDR MOCK INTERFACES ====================

class SDRType(Enum):
    """Supported SDR types"""
    RTLSDR = "rtlsdr"
    HACKRF = "hackrf"
    USRP = "usrp"
    LIMESDR = "limesdr"
    BLADERF = "bladerf"
    PLUTO = "pluto"


@dataclass
class SDRConfig:
    """SDR configuration parameters"""
    sdr_type: SDRType = SDRType.RTLSDR
    center_frequency: float = 935e6  # 935 MHz (GSM 900)
    sample_rate: float = 2.4e6  # 2.4 MSPS
    bandwidth: float = 2e6  # 2 MHz
    gain: float = 40.0  # dB
    ppm_correction: int = 0
    dc_offset_mode: bool = True
    iq_balance: Tuple[float, float] = (1.0, 0.0)
    agc_enabled: bool = False
    buffer_size: int = 16384


@dataclass
class SDRStatus:
    """SDR status information"""
    connected: bool = False
    streaming: bool = False
    samples_received: int = 0
    samples_dropped: int = 0
    overflow_count: int = 0
    underflow_count: int = 0
    current_frequency: float = 0.0
    current_gain: float = 0.0
    temperature_c: Optional[float] = None
    rssi_dbm: float = -100.0
    last_error: Optional[str] = None


class MockSDRDevice:
    """Mock SDR device for hardware-in-loop testing"""
    
    def __init__(self, sdr_type: SDRType = SDRType.RTLSDR, config: Optional[SDRConfig] = None):
        self.sdr_type = sdr_type
        self.config = config or SDRConfig(sdr_type=sdr_type)
        self.status = SDRStatus()
        
        self._sample_queue: queue.Queue = queue.Queue(maxsize=100)
        self._streaming_thread: Optional[threading.Thread] = None
        self._stop_streaming = threading.Event()
        
        # Signal injection
        self._injected_signals: List[Dict[str, Any]] = []
        
        # Timing
        self._start_time: Optional[float] = None
        self._sample_count: int = 0
        
        # Callbacks
        self._sample_callback: Optional[Callable] = None
    
    def connect(self) -> bool:
        """Connect to SDR device"""
        # Simulate connection delay
        time.sleep(0.05)
        
        self.status.connected = True
        self.status.current_frequency = self.config.center_frequency
        self.status.current_gain = self.config.gain
        self.status.last_error = None
        
        return True
    
    def disconnect(self) -> None:
        """Disconnect from SDR device"""
        self.stop_streaming()
        self.status.connected = False
    
    def set_frequency(self, freq_hz: float) -> bool:
        """Set center frequency"""
        if not self.status.connected:
            self.status.last_error = "Device not connected"
            return False
        
        # Validate frequency range
        freq_ranges = {
            SDRType.RTLSDR: (24e6, 1.766e9),
            SDRType.HACKRF: (1e6, 6e9),
            SDRType.USRP: (70e6, 6e9),
            SDRType.LIMESDR: (100e3, 3.8e9),
            SDRType.BLADERF: (300e6, 3.8e9),
            SDRType.PLUTO: (325e6, 3.8e9),
        }
        
        min_freq, max_freq = freq_ranges.get(self.sdr_type, (1e6, 6e9))
        
        if not (min_freq <= freq_hz <= max_freq):
            self.status.last_error = f"Frequency {freq_hz/1e6:.2f} MHz out of range"
            return False
        
        # Simulate tuning delay
        time.sleep(0.01)
        
        self.config.center_frequency = freq_hz
        self.status.current_frequency = freq_hz
        return True
    
    def set_gain(self, gain_db: float) -> bool:
        """Set gain"""
        if not self.status.connected:
            return False
        
        gain_ranges = {
            SDRType.RTLSDR: (0, 49.6),
            SDRType.HACKRF: (0, 62),
            SDRType.USRP: (0, 76),
            SDRType.LIMESDR: (0, 73),
            SDRType.BLADERF: (0, 60),
            SDRType.PLUTO: (-3, 71),
        }
        
        min_gain, max_gain = gain_ranges.get(self.sdr_type, (0, 60))
        gain_db = max(min_gain, min(max_gain, gain_db))
        
        self.config.gain = gain_db
        self.status.current_gain = gain_db
        return True
    
    def set_sample_rate(self, rate_sps: float) -> bool:
        """Set sample rate"""
        if not self.status.connected:
            return False
        
        # Validate sample rate
        if not (250e3 <= rate_sps <= 20e6):
            self.status.last_error = "Invalid sample rate"
            return False
        
        self.config.sample_rate = rate_sps
        return True
    
    def start_streaming(self, callback: Optional[Callable] = None) -> bool:
        """Start streaming samples"""
        if not self.status.connected:
            return False
        
        if self.status.streaming:
            return True
        
        self._sample_callback = callback
        self._stop_streaming.clear()
        self._start_time = time.time()
        self._sample_count = 0
        
        self._streaming_thread = threading.Thread(
            target=self._streaming_loop,
            daemon=True
        )
        self._streaming_thread.start()
        
        self.status.streaming = True
        return True
    
    def stop_streaming(self) -> None:
        """Stop streaming samples"""
        self._stop_streaming.set()
        
        if self._streaming_thread:
            self._streaming_thread.join(timeout=1.0)
            self._streaming_thread = None
        
        self.status.streaming = False
    
    def _streaming_loop(self) -> None:
        """Internal streaming loop"""
        samples_per_buffer = self.config.buffer_size
        sample_period = 1.0 / self.config.sample_rate
        buffer_period = samples_per_buffer * sample_period
        
        while not self._stop_streaming.is_set():
            # Generate samples
            samples = self._generate_samples(samples_per_buffer)
            
            self._sample_count += samples_per_buffer
            self.status.samples_received += samples_per_buffer
            
            # Call callback or queue samples
            if self._sample_callback:
                try:
                    self._sample_callback(samples)
                except Exception as e:
                    self.status.last_error = str(e)
            else:
                try:
                    self._sample_queue.put_nowait(samples)
                except queue.Full:
                    self.status.samples_dropped += samples_per_buffer
                    self.status.overflow_count += 1
            
            # Simulate real-time behavior
            time.sleep(buffer_period * 0.8)  # Slightly faster than real-time
    
    def _generate_samples(self, count: int) -> np.ndarray:
        """Generate mock samples with injected signals"""
        t = np.arange(count) / self.config.sample_rate
        t += self._sample_count / self.config.sample_rate
        
        # Start with noise floor
        noise_power = 10 ** (-100 / 10)  # -100 dBm noise floor
        samples = np.sqrt(noise_power / 2) * (
            np.random.randn(count) + 1j * np.random.randn(count)
        )
        
        # Add injected signals
        for signal in self._injected_signals:
            freq_offset = signal['frequency'] - self.config.center_frequency
            power_linear = 10 ** (signal['power_dbm'] / 10)
            amplitude = np.sqrt(power_linear)
            
            # Apply signal type
            if signal['type'] == 'tone':
                tone = amplitude * np.exp(2j * np.pi * freq_offset * t)
                samples += tone
            elif signal['type'] == 'gsm':
                # Simplified GSM burst
                gsm = amplitude * self._generate_gsm_burst(count, freq_offset, t)
                samples += gsm
            elif signal['type'] == 'lte':
                # Simplified LTE signal
                lte = amplitude * self._generate_lte_signal(count, freq_offset, t)
                samples += lte
        
        # Apply gain
        gain_linear = 10 ** (self.config.gain / 20)
        samples *= gain_linear / 1000  # Scale to reasonable levels
        
        # Apply I/Q imbalance if configured
        if self.config.iq_balance != (1.0, 0.0):
            amp_imb, phase_imb = self.config.iq_balance
            samples = np.real(samples) * amp_imb + 1j * np.imag(samples) * np.exp(1j * phase_imb)
        
        # Update RSSI estimate
        self.status.rssi_dbm = 10 * np.log10(np.mean(np.abs(samples) ** 2) + 1e-20)
        
        return samples.astype(np.complex64)
    
    def _generate_gsm_burst(self, count: int, freq_offset: float, t: np.ndarray) -> np.ndarray:
        """Generate simplified GSM burst signal"""
        # GSM uses GMSK modulation, simplified here as phase modulation
        burst_len = int(0.577e-3 * self.config.sample_rate)  # 577 μs burst
        
        signal = np.zeros(count, dtype=np.complex64)
        
        if count >= burst_len:
            # Random data symbols
            symbols = np.random.choice([-1, 1], burst_len // 4)
            phase = np.cumsum(symbols * np.pi / 2)
            phase_interp = np.repeat(phase, 4)[:burst_len]
            
            burst = np.exp(1j * phase_interp) * np.exp(2j * np.pi * freq_offset * t[:burst_len])
            signal[:burst_len] = burst
        
        return signal
    
    def _generate_lte_signal(self, count: int, freq_offset: float, t: np.ndarray) -> np.ndarray:
        """Generate simplified LTE OFDM signal"""
        # Simplified OFDM with random subcarriers
        n_subcarriers = 64
        fft_size = 64
        
        signal = np.zeros(count, dtype=np.complex64)
        
        for i in range(0, count - fft_size, fft_size):
            # Random QPSK symbols on subcarriers
            symbols = (np.random.choice([-1, 1], n_subcarriers) + 
                      1j * np.random.choice([-1, 1], n_subcarriers)) / np.sqrt(2)
            
            # OFDM symbol via IFFT
            ofdm_symbol = np.fft.ifft(symbols, fft_size)
            signal[i:i+fft_size] = ofdm_symbol
        
        # Apply frequency offset
        signal *= np.exp(2j * np.pi * freq_offset * t[:len(signal)])
        
        return signal
    
    def inject_signal(self, frequency: float, power_dbm: float, 
                     signal_type: str = 'tone') -> None:
        """Inject a signal for testing"""
        self._injected_signals.append({
            'frequency': frequency,
            'power_dbm': power_dbm,
            'type': signal_type
        })
    
    def clear_injected_signals(self) -> None:
        """Clear all injected signals"""
        self._injected_signals.clear()
    
    def get_samples(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get samples from queue"""
        try:
            return self._sample_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ==================== TIMING VALIDATION ====================

class TimingValidator:
    """Validate real-time timing constraints"""
    
    def __init__(self, tolerance_percent: float = 5.0):
        self.tolerance = tolerance_percent / 100.0
        self.measurements: List[Dict[str, Any]] = []
    
    def measure(self, operation: Callable, expected_duration: float, 
                name: str = "operation") -> Dict[str, Any]:
        """Measure operation timing"""
        start = time.perf_counter()
        result = operation()
        elapsed = time.perf_counter() - start
        
        deviation = abs(elapsed - expected_duration) / expected_duration if expected_duration > 0 else 0
        passed = deviation <= self.tolerance
        
        measurement = {
            'name': name,
            'expected': expected_duration,
            'actual': elapsed,
            'deviation_percent': deviation * 100,
            'passed': passed,
            'result': result
        }
        
        self.measurements.append(measurement)
        return measurement
    
    def validate_sample_rate(self, sdr: MockSDRDevice, duration: float = 1.0) -> Dict[str, Any]:
        """Validate sample rate accuracy"""
        sdr.start_streaming()
        
        start_time = time.perf_counter()
        total_samples = 0
        
        while time.perf_counter() - start_time < duration:
            samples = sdr.get_samples(timeout=0.1)
            if samples is not None:
                total_samples += len(samples)
        
        sdr.stop_streaming()
        
        actual_duration = time.perf_counter() - start_time
        actual_rate = total_samples / actual_duration
        expected_rate = sdr.config.sample_rate
        
        deviation = abs(actual_rate - expected_rate) / expected_rate
        
        return {
            'expected_rate': expected_rate,
            'actual_rate': actual_rate,
            'total_samples': total_samples,
            'duration': actual_duration,
            'deviation_percent': deviation * 100,
            'passed': deviation <= self.tolerance
        }
    
    def validate_latency(self, sdr: MockSDRDevice, n_measurements: int = 10) -> Dict[str, Any]:
        """Validate sample latency"""
        latencies = []
        
        sdr.start_streaming()
        
        for _ in range(n_measurements):
            start = time.perf_counter()
            samples = sdr.get_samples(timeout=1.0)
            if samples is not None:
                latency = time.perf_counter() - start
                latencies.append(latency)
        
        sdr.stop_streaming()
        
        if not latencies:
            return {'error': 'No samples received', 'passed': False}
        
        expected_latency = sdr.config.buffer_size / sdr.config.sample_rate
        
        return {
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'avg_latency': sum(latencies) / len(latencies),
            'expected_latency': expected_latency,
            'jitter': max(latencies) - min(latencies),
            'passed': max(latencies) < expected_latency * 3
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get timing validation summary"""
        if not self.measurements:
            return {'total': 0, 'passed': 0, 'failed': 0}
        
        passed = sum(1 for m in self.measurements if m['passed'])
        
        return {
            'total': len(self.measurements),
            'passed': passed,
            'failed': len(self.measurements) - passed,
            'pass_rate': passed / len(self.measurements) * 100
        }


# ==================== SIGNAL INJECTION FRAMEWORK ====================

class SignalInjector:
    """Framework for injecting test signals"""
    
    def __init__(self, sdr: MockSDRDevice):
        self.sdr = sdr
        self.injected_signals: List[Dict[str, Any]] = []
    
    def inject_cw_tone(self, frequency: float, power_dbm: float = -50) -> str:
        """Inject continuous wave tone"""
        signal_id = f"cw_{int(frequency)}_{int(power_dbm)}"
        self.sdr.inject_signal(frequency, power_dbm, 'tone')
        self.injected_signals.append({
            'id': signal_id,
            'type': 'tone',
            'frequency': frequency,
            'power': power_dbm
        })
        return signal_id
    
    def inject_gsm_burst(self, frequency: float, power_dbm: float = -60) -> str:
        """Inject GSM burst"""
        signal_id = f"gsm_{int(frequency)}_{int(power_dbm)}"
        self.sdr.inject_signal(frequency, power_dbm, 'gsm')
        self.injected_signals.append({
            'id': signal_id,
            'type': 'gsm',
            'frequency': frequency,
            'power': power_dbm
        })
        return signal_id
    
    def inject_lte_signal(self, frequency: float, power_dbm: float = -70) -> str:
        """Inject LTE signal"""
        signal_id = f"lte_{int(frequency)}_{int(power_dbm)}"
        self.sdr.inject_signal(frequency, power_dbm, 'lte')
        self.injected_signals.append({
            'id': signal_id,
            'type': 'lte',
            'frequency': frequency,
            'power': power_dbm
        })
        return signal_id
    
    def inject_interference(self, center_freq: float, bandwidth: float, 
                           power_dbm: float = -40) -> List[str]:
        """Inject wideband interference"""
        signal_ids = []
        n_tones = int(bandwidth / 100e3)  # One tone per 100 kHz
        
        for i in range(n_tones):
            freq = center_freq - bandwidth/2 + i * (bandwidth / n_tones)
            signal_id = self.inject_cw_tone(freq, power_dbm - 10)  # Spread power
            signal_ids.append(signal_id)
        
        return signal_ids
    
    def clear_all(self) -> None:
        """Clear all injected signals"""
        self.sdr.clear_injected_signals()
        self.injected_signals.clear()


# ==================== SIGNAL ANALYZER ====================

class SignalAnalyzer:
    """Analyze received signals"""
    
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
    
    def measure_power(self, samples: np.ndarray) -> float:
        """Measure signal power in dBm"""
        power_linear = np.mean(np.abs(samples) ** 2)
        return 10 * np.log10(power_linear + 1e-20)
    
    def find_peaks(self, samples: np.ndarray, threshold_db: float = -80) -> List[Dict[str, float]]:
        """Find spectral peaks"""
        # Compute FFT
        fft = np.fft.fftshift(np.fft.fft(samples))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/self.sample_rate))
        
        # Power spectrum in dB
        power_db = 10 * np.log10(np.abs(fft) ** 2 + 1e-20)
        
        # Find peaks above threshold
        peaks = []
        for i in range(1, len(power_db) - 1):
            if (power_db[i] > threshold_db and 
                power_db[i] > power_db[i-1] and 
                power_db[i] > power_db[i+1]):
                peaks.append({
                    'frequency': freqs[i],
                    'power_db': power_db[i],
                    'bin_index': i
                })
        
        # Sort by power
        peaks.sort(key=lambda p: p['power_db'], reverse=True)
        return peaks[:10]  # Top 10 peaks
    
    def estimate_snr(self, samples: np.ndarray, signal_bw: float) -> float:
        """Estimate signal-to-noise ratio"""
        fft = np.fft.fftshift(np.fft.fft(samples))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/self.sample_rate))
        power = np.abs(fft) ** 2
        
        # Signal band
        signal_mask = np.abs(freqs) < signal_bw / 2
        signal_power = np.mean(power[signal_mask])
        
        # Noise band (edges of spectrum)
        noise_mask = np.abs(freqs) > signal_bw
        noise_power = np.mean(power[noise_mask]) if np.any(noise_mask) else 1e-20
        
        return 10 * np.log10(signal_power / (noise_power + 1e-20))
    
    def detect_signal_type(self, samples: np.ndarray) -> str:
        """Attempt to identify signal type"""
        # Simple heuristic based on signal characteristics
        
        # Check for CW tone (very narrow spectrum)
        fft = np.fft.fft(samples)
        power = np.abs(fft) ** 2
        peak_power = np.max(power)
        total_power = np.sum(power)
        
        peak_ratio = peak_power / total_power
        
        if peak_ratio > 0.5:
            return 'cw_tone'
        elif peak_ratio > 0.1:
            return 'narrowband'
        else:
            return 'wideband'


# ==================== TEST FIXTURES ====================

@pytest.fixture
def mock_sdr():
    """Provide mock SDR device"""
    sdr = MockSDRDevice(SDRType.RTLSDR)
    sdr.connect()
    yield sdr
    sdr.disconnect()


@pytest.fixture
def mock_hackrf():
    """Provide mock HackRF device"""
    sdr = MockSDRDevice(SDRType.HACKRF)
    sdr.connect()
    yield sdr
    sdr.disconnect()


@pytest.fixture
def timing_validator():
    """Provide timing validator"""
    return TimingValidator(tolerance_percent=10.0)


@pytest.fixture
def signal_injector(mock_sdr):
    """Provide signal injector"""
    injector = SignalInjector(mock_sdr)
    yield injector
    injector.clear_all()


@pytest.fixture
def signal_analyzer(mock_sdr):
    """Provide signal analyzer"""
    return SignalAnalyzer(mock_sdr.config.sample_rate)


# ==================== SDR INTERFACE TESTS ====================

class TestSDRInterface:
    """Tests for SDR interface operations"""
    
    def test_sdr_connection(self):
        """Test SDR connection and disconnection"""
        sdr = MockSDRDevice(SDRType.RTLSDR)
        
        assert not sdr.status.connected
        
        result = sdr.connect()
        assert result
        assert sdr.status.connected
        
        sdr.disconnect()
        assert not sdr.status.connected
    
    def test_frequency_setting(self, mock_sdr):
        """Test frequency setting"""
        # Valid frequency
        result = mock_sdr.set_frequency(935e6)
        assert result
        assert mock_sdr.status.current_frequency == 935e6
        
        # Out of range frequency
        result = mock_sdr.set_frequency(10e9)  # Too high for RTL-SDR
        assert not result
        assert "out of range" in mock_sdr.status.last_error.lower()
    
    def test_gain_setting(self, mock_sdr):
        """Test gain setting"""
        result = mock_sdr.set_gain(30.0)
        assert result
        assert mock_sdr.status.current_gain == 30.0
        
        # Test gain clamping
        mock_sdr.set_gain(100.0)
        assert mock_sdr.status.current_gain <= 49.6  # RTL-SDR max
    
    def test_sample_rate_setting(self, mock_sdr):
        """Test sample rate setting"""
        result = mock_sdr.set_sample_rate(2.048e6)
        assert result
        assert mock_sdr.config.sample_rate == 2.048e6
        
        # Invalid sample rate
        result = mock_sdr.set_sample_rate(100e6)
        assert not result
    
    def test_multiple_sdr_types(self):
        """Test different SDR types"""
        for sdr_type in SDRType:
            sdr = MockSDRDevice(sdr_type)
            assert sdr.connect()
            assert sdr.status.connected
            sdr.disconnect()


# ==================== STREAMING TESTS ====================

class TestSDRStreaming:
    """Tests for SDR streaming operations"""
    
    def test_start_stop_streaming(self, mock_sdr):
        """Test streaming lifecycle"""
        assert not mock_sdr.status.streaming
        
        result = mock_sdr.start_streaming()
        assert result
        assert mock_sdr.status.streaming
        
        time.sleep(0.1)
        
        mock_sdr.stop_streaming()
        assert not mock_sdr.status.streaming
    
    def test_sample_reception(self, mock_sdr):
        """Test sample reception"""
        mock_sdr.start_streaming()
        
        samples = mock_sdr.get_samples(timeout=1.0)
        
        mock_sdr.stop_streaming()
        
        assert samples is not None
        assert len(samples) == mock_sdr.config.buffer_size
        assert samples.dtype == np.complex64
    
    def test_continuous_streaming(self, mock_sdr):
        """Test continuous streaming"""
        mock_sdr.start_streaming()
        
        received_buffers = 0
        total_samples = 0
        
        start_time = time.time()
        while time.time() - start_time < 0.5:
            samples = mock_sdr.get_samples(timeout=0.1)
            if samples is not None:
                received_buffers += 1
                total_samples += len(samples)
        
        mock_sdr.stop_streaming()
        
        assert received_buffers > 0
        assert total_samples > 0
        assert mock_sdr.status.samples_received > 0
    
    def test_callback_streaming(self, mock_sdr):
        """Test streaming with callback"""
        received_samples = []
        
        def callback(samples):
            received_samples.append(samples)
        
        mock_sdr.start_streaming(callback=callback)
        time.sleep(0.3)
        mock_sdr.stop_streaming()
        
        assert len(received_samples) > 0


# ==================== TIMING VALIDATION TESTS ====================

class TestTimingValidation:
    """Tests for timing accuracy"""
    
    def test_operation_timing(self, timing_validator, mock_sdr):
        """Test operation timing measurement"""
        measurement = timing_validator.measure(
            lambda: mock_sdr.set_frequency(935e6),
            expected_duration=0.01,
            name="set_frequency"
        )
        
        assert 'actual' in measurement
        assert measurement['actual'] > 0
    
    def test_sample_rate_validation(self, timing_validator, mock_sdr):
        """Test sample rate accuracy"""
        result = timing_validator.validate_sample_rate(mock_sdr, duration=0.5)
        
        assert 'actual_rate' in result
        assert 'deviation_percent' in result
        # Allow larger deviation for mock
        assert result['deviation_percent'] < 50  # 50% tolerance for mock
    
    def test_latency_validation(self, timing_validator, mock_sdr):
        """Test latency measurement"""
        result = timing_validator.validate_latency(mock_sdr, n_measurements=5)
        
        assert 'avg_latency' in result
        assert result['avg_latency'] > 0
    
    def test_timing_summary(self, timing_validator, mock_sdr):
        """Test timing summary"""
        timing_validator.measure(
            lambda: time.sleep(0.01),
            expected_duration=0.01,
            name="sleep_10ms"
        )
        
        summary = timing_validator.get_summary()
        
        assert summary['total'] >= 1
        assert 'pass_rate' in summary


# ==================== SIGNAL INJECTION TESTS ====================

class TestSignalInjection:
    """Tests for signal injection"""
    
    def test_inject_cw_tone(self, signal_injector, mock_sdr, signal_analyzer):
        """Test CW tone injection"""
        # Inject tone at center frequency + 100 kHz
        signal_injector.inject_cw_tone(
            frequency=mock_sdr.config.center_frequency + 100e3,
            power_dbm=-50
        )
        
        mock_sdr.start_streaming()
        samples = mock_sdr.get_samples(timeout=1.0)
        mock_sdr.stop_streaming()
        
        assert samples is not None
        
        # Analyze signal
        peaks = signal_analyzer.find_peaks(samples, threshold_db=-80)
        
        # Should find the injected tone
        assert len(peaks) > 0
    
    def test_inject_gsm_burst(self, signal_injector, mock_sdr):
        """Test GSM burst injection"""
        signal_injector.inject_gsm_burst(
            frequency=mock_sdr.config.center_frequency,
            power_dbm=-60
        )
        
        mock_sdr.start_streaming()
        samples = mock_sdr.get_samples(timeout=1.0)
        mock_sdr.stop_streaming()
        
        assert samples is not None
        assert len(samples) > 0
    
    def test_inject_lte_signal(self, signal_injector, mock_sdr):
        """Test LTE signal injection"""
        signal_injector.inject_lte_signal(
            frequency=mock_sdr.config.center_frequency,
            power_dbm=-70
        )
        
        mock_sdr.start_streaming()
        samples = mock_sdr.get_samples(timeout=1.0)
        mock_sdr.stop_streaming()
        
        assert samples is not None
    
    def test_inject_interference(self, signal_injector, mock_sdr, signal_analyzer):
        """Test wideband interference injection"""
        signal_injector.inject_interference(
            center_freq=mock_sdr.config.center_frequency,
            bandwidth=500e3,
            power_dbm=-40
        )
        
        mock_sdr.start_streaming()
        samples = mock_sdr.get_samples(timeout=1.0)
        mock_sdr.stop_streaming()
        
        # Power should be higher with interference
        power = signal_analyzer.measure_power(samples)
        assert power > -100  # Should have significant power
    
    def test_clear_signals(self, signal_injector, mock_sdr):
        """Test clearing injected signals"""
        signal_injector.inject_cw_tone(935e6, -50)
        signal_injector.inject_gsm_burst(936e6, -60)
        
        assert len(signal_injector.injected_signals) == 2
        
        signal_injector.clear_all()
        
        assert len(signal_injector.injected_signals) == 0


# ==================== SIGNAL ANALYSIS TESTS ====================

class TestSignalAnalysis:
    """Tests for signal analysis"""
    
    def test_power_measurement(self, signal_analyzer):
        """Test power measurement"""
        # Generate known signal
        samples = 0.1 * (np.random.randn(1024) + 1j * np.random.randn(1024))
        
        power = signal_analyzer.measure_power(samples)
        
        # Power should be around -20 dBm for 0.1 amplitude
        assert -30 < power < -10
    
    def test_peak_detection(self, signal_analyzer):
        """Test spectral peak detection"""
        # Generate tone + noise
        t = np.arange(1024) / 2.4e6
        tone = 0.5 * np.exp(2j * np.pi * 100e3 * t)  # 100 kHz tone
        noise = 0.01 * (np.random.randn(1024) + 1j * np.random.randn(1024))
        samples = tone + noise
        
        peaks = signal_analyzer.find_peaks(samples, threshold_db=-50)
        
        assert len(peaks) > 0
        # Strongest peak should be near 100 kHz
        assert abs(peaks[0]['frequency'] - 100e3) < 10e3
    
    def test_snr_estimation(self, signal_analyzer):
        """Test SNR estimation"""
        # High SNR signal
        t = np.arange(1024) / 2.4e6
        signal = 1.0 * np.exp(2j * np.pi * 50e3 * t)
        noise = 0.01 * (np.random.randn(1024) + 1j * np.random.randn(1024))
        samples = signal + noise
        
        snr = signal_analyzer.estimate_snr(samples, signal_bw=100e3)
        
        # Should have high SNR
        assert snr > 20  # At least 20 dB
    
    def test_signal_type_detection(self, signal_analyzer):
        """Test signal type detection"""
        # CW tone
        t = np.arange(1024) / 2.4e6
        cw = np.exp(2j * np.pi * 100e3 * t)
        
        detected = signal_analyzer.detect_signal_type(cw)
        assert detected == 'cw_tone'
        
        # Wideband noise
        noise = np.random.randn(1024) + 1j * np.random.randn(1024)
        detected = signal_analyzer.detect_signal_type(noise)
        assert detected in ['wideband', 'narrowband']


# ==================== END-TO-END TESTS ====================

class TestEndToEnd:
    """End-to-end hardware-in-loop tests"""
    
    def test_full_capture_cycle(self, mock_sdr, signal_injector, signal_analyzer):
        """Test complete capture and analysis cycle"""
        # Setup
        mock_sdr.set_frequency(935e6)
        mock_sdr.set_gain(40)
        mock_sdr.set_sample_rate(2.4e6)
        
        # Inject test signal
        signal_injector.inject_cw_tone(935.1e6, -40)
        
        # Capture
        mock_sdr.start_streaming()
        
        all_samples = []
        for _ in range(5):
            samples = mock_sdr.get_samples(timeout=1.0)
            if samples is not None:
                all_samples.append(samples)
        
        mock_sdr.stop_streaming()
        
        # Analyze
        assert len(all_samples) > 0
        combined = np.concatenate(all_samples)
        
        power = signal_analyzer.measure_power(combined)
        peaks = signal_analyzer.find_peaks(combined)
        
        assert power > -100
        assert len(peaks) > 0
    
    def test_frequency_scan(self, mock_sdr, signal_injector):
        """Test frequency scanning"""
        frequencies = [935e6, 936e6, 937e6, 938e6]
        
        # Inject signal at 937 MHz
        signal_injector.inject_cw_tone(937e6, -30)
        
        power_readings = {}
        
        for freq in frequencies:
            mock_sdr.set_frequency(freq)
            mock_sdr.start_streaming()
            
            samples = mock_sdr.get_samples(timeout=0.5)
            if samples is not None:
                power = 10 * np.log10(np.mean(np.abs(samples) ** 2) + 1e-20)
                power_readings[freq] = power
            
            mock_sdr.stop_streaming()
        
        assert len(power_readings) > 0
        # 937 MHz should have highest power
        max_freq = max(power_readings, key=power_readings.get)
        assert max_freq == 937e6
    
    def test_multi_sdr_operation(self):
        """Test multiple SDR operation"""
        sdr1 = MockSDRDevice(SDRType.RTLSDR)
        sdr2 = MockSDRDevice(SDRType.HACKRF)
        
        sdr1.connect()
        sdr2.connect()
        
        sdr1.set_frequency(935e6)
        sdr2.set_frequency(1800e6)
        
        sdr1.start_streaming()
        sdr2.start_streaming()
        
        samples1 = sdr1.get_samples(timeout=0.5)
        samples2 = sdr2.get_samples(timeout=0.5)
        
        sdr1.stop_streaming()
        sdr2.stop_streaming()
        sdr1.disconnect()
        sdr2.disconnect()
        
        assert samples1 is not None
        assert samples2 is not None


# ==================== MAIN ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
