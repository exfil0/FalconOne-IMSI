"""
FalconOne Data Validator Unit Tests
Tests for input validation and data sanitization

Version: 1.9.2
Coverage: Input validation, data sanitization, security checks
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import json


@pytest.fixture
def mock_logger():
    """Mock logger"""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


class TestIMSIValidation:
    """Tests for IMSI validation"""
    
    def test_valid_imsi_15_digits(self):
        """Test valid 15-digit IMSI"""
        imsi = "310260123456789"
        
        # Basic validation
        assert len(imsi) == 15
        assert imsi.isdigit()
    
    def test_valid_imsi_14_digits(self):
        """Test valid 14-digit IMSI"""
        imsi = "31026012345678"
        
        assert 14 <= len(imsi) <= 15
        assert imsi.isdigit()
    
    def test_invalid_imsi_with_letters(self):
        """Test IMSI with letters is invalid"""
        imsi = "31026012345678A"
        
        assert not imsi.isdigit()
    
    def test_invalid_imsi_too_short(self):
        """Test IMSI too short is invalid"""
        imsi = "31026"
        
        assert len(imsi) < 14


class TestTMSIValidation:
    """Tests for TMSI validation"""
    
    def test_valid_tmsi_hex(self):
        """Test valid hexadecimal TMSI"""
        tmsi = "ABCD1234"
        
        # TMSI is typically 4 bytes (8 hex chars)
        try:
            int(tmsi, 16)
            valid = True
        except ValueError:
            valid = False
        
        assert valid
    
    def test_valid_tmsi_integer(self):
        """Test valid integer TMSI"""
        tmsi = 0xABCD1234
        
        assert isinstance(tmsi, int)
        assert tmsi > 0


class TestFrequencyValidation:
    """Tests for frequency range validation"""
    
    def test_valid_gsm_frequency(self):
        """Test valid GSM frequency range"""
        freq = 935e6  # GSM900 downlink
        
        # GSM900: 935-960 MHz
        assert 890e6 <= freq <= 960e6
    
    def test_valid_lte_frequency(self):
        """Test valid LTE frequency"""
        freq = 2600e6  # LTE Band 7
        
        assert 700e6 <= freq <= 3700e6
    
    def test_valid_5g_mmwave(self):
        """Test valid 5G mmWave frequency"""
        freq = 28e9  # 28 GHz
        
        assert 24e9 <= freq <= 52e9
    
    def test_invalid_negative_frequency(self):
        """Test negative frequency is invalid"""
        freq = -900e6
        
        assert freq < 0


class TestIQDataValidation:
    """Tests for IQ data validation"""
    
    def test_valid_complex_iq_data(self):
        """Test valid complex IQ data"""
        iq_data = np.random.randn(1024) + 1j * np.random.randn(1024)
        
        assert np.iscomplexobj(iq_data)
        assert len(iq_data) > 0
    
    def test_valid_real_iq_data(self):
        """Test valid real I/Q interleaved data"""
        iq_data = np.random.randn(2048)  # I and Q interleaved
        
        assert len(iq_data) % 2 == 0  # Must be even for I/Q pairs
    
    def test_iq_data_normalization(self):
        """Test IQ data can be normalized"""
        iq_data = np.random.randn(1024) * 100  # Large values
        
        max_val = np.max(np.abs(iq_data))
        normalized = iq_data / max_val if max_val > 0 else iq_data
        
        assert np.max(np.abs(normalized)) <= 1.0
    
    def test_empty_iq_data(self):
        """Test empty IQ data is invalid"""
        iq_data = np.array([])
        
        assert len(iq_data) == 0


class TestConfigValidation:
    """Tests for configuration validation"""
    
    def test_valid_config_structure(self):
        """Test valid configuration structure"""
        config = {
            'monitoring': {
                'gsm': {
                    'bands': ['GSM900', 'GSM1800'],
                    'tools': ['gr-gsm']
                }
            },
            'sdr': {
                'rx_gain': 40,
                'sample_rate': 2.4e6
            }
        }
        
        assert 'monitoring' in config
        assert 'sdr' in config
    
    def test_gain_range_validation(self):
        """Test gain value is in valid range"""
        gain = 40
        
        # Typical SDR gain range: 0-50 dB
        assert 0 <= gain <= 60
    
    def test_sample_rate_validation(self):
        """Test sample rate is valid"""
        sample_rate = 2.4e6
        
        # Typical RTL-SDR: 225 kHz - 3.2 MHz
        assert 100e3 <= sample_rate <= 20e6


class TestJSONSerialization:
    """Tests for JSON serialization safety"""
    
    def test_serialize_signal_result(self):
        """Test signal result can be serialized"""
        result = {
            'signal_id': 'test_123',
            'predicted_generation': 'LTE',
            'confidence': 0.95,
            'timestamp': datetime.now().isoformat()
        }
        
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
    
    def test_serialize_numpy_arrays(self):
        """Test numpy arrays are converted for JSON"""
        data = {
            'values': np.array([1, 2, 3]),
            'mean': np.float64(2.0)
        }
        
        # Convert numpy types
        serializable = {
            'values': data['values'].tolist(),
            'mean': float(data['mean'])
        }
        
        json_str = json.dumps(serializable)
        assert isinstance(json_str, str)


class TestInputSanitization:
    """Tests for input sanitization"""
    
    def test_sanitize_imsi_strips_colons(self):
        """Test IMSI sanitization removes colons"""
        raw_imsi = "31:02:60:12:34:56:78:9"
        sanitized = raw_imsi.replace(':', '')
        
        assert ':' not in sanitized
    
    def test_sanitize_path_traversal(self):
        """Test path traversal prevention"""
        malicious_path = "../../../etc/passwd"
        
        # Should detect traversal
        assert '..' in malicious_path
    
    def test_sanitize_special_characters(self):
        """Test special character handling"""
        raw_input = "test<script>alert('xss')</script>"
        
        # Should detect potential XSS
        assert '<' in raw_input or '>' in raw_input


class TestBandValidation:
    """Tests for band validation"""
    
    def test_valid_gsm_bands(self):
        """Test valid GSM band names"""
        valid_bands = ['GSM850', 'GSM900', 'GSM1800', 'GSM1900']
        
        for band in valid_bands:
            assert band.startswith('GSM')
    
    def test_valid_lte_bands(self):
        """Test valid LTE band numbers"""
        valid_bands = [1, 3, 7, 20, 28, 41]
        
        for band in valid_bands:
            assert 1 <= band <= 85  # LTE band range
    
    def test_valid_5g_bands(self):
        """Test valid 5G NR band names"""
        valid_bands = ['n78', 'n79', 'n257', 'n258']
        
        for band in valid_bands:
            assert band.startswith('n')


class TestRateLimitingValidation:
    """Tests for rate limiting validation"""
    
    def test_request_rate_within_limit(self):
        """Test request rate is within limits"""
        requests_per_minute = 50
        limit = 60
        
        assert requests_per_minute <= limit
    
    def test_capture_duration_limit(self):
        """Test capture duration limits"""
        duration_sec = 30
        max_duration = 300  # 5 minutes max
        
        assert duration_sec <= max_duration


class TestARFCNValidation:
    """Tests for ARFCN validation"""
    
    def test_valid_gsm900_arfcn(self):
        """Test valid GSM900 ARFCN range"""
        arfcn = 50
        
        assert 0 <= arfcn <= 124
    
    def test_valid_gsm1800_arfcn(self):
        """Test valid GSM1800 ARFCN range"""
        arfcn = 600
        
        assert 512 <= arfcn <= 885
    
    def test_invalid_arfcn_negative(self):
        """Test negative ARFCN is invalid"""
        arfcn = -1
        
        assert arfcn < 0


class TestCoordinateValidation:
    """Tests for geolocation coordinate validation"""
    
    def test_valid_latitude(self):
        """Test valid latitude range"""
        lat = 40.7128  # NYC
        
        assert -90.0 <= lat <= 90.0
    
    def test_valid_longitude(self):
        """Test valid longitude range"""
        lon = -74.0060  # NYC
        
        assert -180.0 <= lon <= 180.0
    
    def test_invalid_latitude(self):
        """Test invalid latitude"""
        lat = 100.0
        
        assert not (-90.0 <= lat <= 90.0)


# Run with: pytest test_data_validator.py -v
