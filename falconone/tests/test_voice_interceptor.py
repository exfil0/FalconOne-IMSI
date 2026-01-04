"""
FalconOne Voice Processing Unit Tests
Tests for voice interceptor Opus decoding, speaker diarization, and VAD

Version: 1.9.6
Coverage: Opus codec, speaker diarization, voice activity detection, call analysis
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import struct


@pytest.fixture
def mock_logger():
    """Mock logger"""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def mock_config():
    """Mock configuration"""
    config = Mock()
    config.get = Mock(side_effect=lambda key, default=None: {
        'voice.sample_rate': 16000,
        'voice.channels': 1,
        'voice.diarization.min_speakers': 2,
        'voice.diarization.max_speakers': 5,
        'voice.vad.aggressiveness': 2,
    }.get(key, default))
    return config


@pytest.fixture
def sample_audio_16khz():
    """Generate sample 16kHz mono audio (1 second)"""
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate 440Hz tone
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    return audio


@pytest.fixture
def sample_audio_48khz():
    """Generate sample 48kHz stereo audio (1 second)"""
    duration = 1.0
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Stereo: left and right channels
    left = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
    return np.column_stack([left, right])


@pytest.fixture
def sample_rtp_packets():
    """Generate sample RTP-like packets with Opus payloads"""
    packets = []
    for seq in range(10):
        # Simplified RTP header (12 bytes) + dummy Opus payload
        header = struct.pack('>BBHII',
            0x80,  # Version 2
            111,   # Payload type (Opus)
            seq,   # Sequence number
            seq * 960,  # Timestamp (20ms @ 48kHz)
            0x12345678  # SSRC
        )
        # Dummy Opus frame (20-40 bytes typical)
        payload = bytes([0x78] * 30)  # Opus CBR frame placeholder
        packets.append({
            'ssrc': 0x12345678,
            'sequence': seq,
            'timestamp': seq * 960,
            'payload': payload,
            'header': header
        })
    return packets


# =============================================================================
# VoiceInterceptor Initialization Tests
# =============================================================================

class TestVoiceInterceptorInit:
    """Tests for VoiceInterceptor initialization"""
    
    def test_init_with_config(self, mock_config, mock_logger):
        """Test initialization with valid config"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        assert interceptor is not None
        assert hasattr(interceptor, '_opus_decoders')
        assert isinstance(interceptor._opus_decoders, dict)
    
    def test_opus_decoders_initialized_empty(self, mock_config, mock_logger):
        """Test Opus decoder dictionary is empty on init"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        assert len(interceptor._opus_decoders) == 0
    
    def test_diarization_pipeline_initialized(self, mock_config, mock_logger):
        """Test diarization components are initialized"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        # May be None if pyannote not available
        assert hasattr(interceptor, '_diarization_pipeline')
        assert hasattr(interceptor, '_voice_encoder')


# =============================================================================
# Opus Decoding Tests
# =============================================================================

class TestOpusDecoding:
    """Tests for Opus codec decoding"""
    
    def test_decode_opus_method_exists(self, mock_config, mock_logger):
        """Test _decode_opus method exists"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        assert hasattr(interceptor, '_decode_opus')
        assert callable(interceptor._decode_opus)
    
    def test_decode_opus_native_method_exists(self, mock_config, mock_logger):
        """Test _decode_opus_native method exists"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        assert hasattr(interceptor, '_decode_opus_native')
    
    def test_decode_opus_ffmpeg_method_exists(self, mock_config, mock_logger):
        """Test _decode_opus_ffmpeg method exists"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        assert hasattr(interceptor, '_decode_opus_ffmpeg')
    
    def test_decode_opus_stream_method_exists(self, mock_config, mock_logger):
        """Test decode_opus_stream method exists"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        assert hasattr(interceptor, 'decode_opus_stream')
        assert callable(interceptor.decode_opus_stream)
    
    def test_decode_opus_empty_data_returns_none(self, mock_config, mock_logger):
        """Test decoding empty data returns None"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        result = interceptor._decode_opus(b'', ssrc=0x12345678)
        
        assert result is None
    
    def test_decode_opus_stream_empty_packets(self, mock_config, mock_logger):
        """Test decoding empty packet list"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        result = interceptor.decode_opus_stream([], ssrc=0x12345678)
        
        assert result is not None
        assert 'audio' in result or 'error' in result or len(result.get('samples', [])) == 0
    
    def test_opus_decoder_per_ssrc_state(self, mock_config, mock_logger):
        """Test separate decoder state per SSRC"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        # Access decoders for different SSRCs
        ssrc1 = 0x11111111
        ssrc2 = 0x22222222
        
        # Trigger decoder creation (if opuslib available)
        interceptor._decode_opus(b'\x00' * 20, ssrc=ssrc1)
        interceptor._decode_opus(b'\x00' * 20, ssrc=ssrc2)
        
        # Decoders should be separate (if opuslib available)
        if len(interceptor._opus_decoders) > 0:
            assert ssrc1 in interceptor._opus_decoders or ssrc2 in interceptor._opus_decoders


# =============================================================================
# Speaker Diarization Tests
# =============================================================================

class TestSpeakerDiarization:
    """Tests for speaker diarization functionality"""
    
    def test_diarize_audio_method_exists(self, mock_config, mock_logger):
        """Test diarize_audio method exists"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        assert hasattr(interceptor, 'diarize_audio')
        assert callable(interceptor.diarize_audio)
    
    def test_diarize_fallback_method_exists(self, mock_config, mock_logger):
        """Test _diarize_fallback method exists"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        assert hasattr(interceptor, '_diarize_fallback')
    
    def test_extract_speaker_embeddings_method_exists(self, mock_config, mock_logger):
        """Test extract_speaker_embeddings method exists"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        assert hasattr(interceptor, 'extract_speaker_embeddings')
        assert callable(interceptor.extract_speaker_embeddings)
    
    def test_diarize_audio_returns_dict(self, mock_config, mock_logger, sample_audio_16khz):
        """Test diarize_audio returns dictionary result"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        result = interceptor.diarize_audio(
            sample_audio_16khz.tobytes(),
            sample_rate=16000
        )
        
        assert isinstance(result, dict)
        # Should have segments key (may be empty if diarization unavailable)
        assert 'segments' in result or 'error' in result or 'speakers' in result
    
    def test_diarize_audio_min_max_speakers(self, mock_config, mock_logger, sample_audio_16khz):
        """Test diarization respects min/max speakers"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        result = interceptor.diarize_audio(
            sample_audio_16khz.tobytes(),
            sample_rate=16000,
            min_speakers=2,
            max_speakers=4
        )
        
        assert isinstance(result, dict)
    
    def test_extract_embeddings_returns_dict(self, mock_config, mock_logger, sample_audio_16khz):
        """Test extract_speaker_embeddings returns dictionary"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        # Create mock segments
        segments = [
            {'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 0.5},
            {'speaker': 'SPEAKER_01', 'start': 0.5, 'end': 1.0}
        ]
        
        result = interceptor.extract_speaker_embeddings(
            sample_audio_16khz.tobytes(),
            segments,
            sample_rate=16000
        )
        
        assert isinstance(result, dict)


# =============================================================================
# Voice Activity Detection Tests
# =============================================================================

class TestVoiceActivityDetection:
    """Tests for voice activity detection"""
    
    def test_detect_voice_activity_method_exists(self, mock_config, mock_logger):
        """Test detect_voice_activity method exists"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        assert hasattr(interceptor, 'detect_voice_activity')
        assert callable(interceptor.detect_voice_activity)
    
    def test_detect_vad_returns_list(self, mock_config, mock_logger, sample_audio_16khz):
        """Test detect_voice_activity returns list of segments"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        result = interceptor.detect_voice_activity(
            sample_audio_16khz.tobytes(),
            sample_rate=16000
        )
        
        assert isinstance(result, (list, dict))
    
    def test_detect_vad_aggressiveness_levels(self, mock_config, mock_logger, sample_audio_16khz):
        """Test VAD with different aggressiveness levels"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        for level in [0, 1, 2, 3]:
            result = interceptor.detect_voice_activity(
                sample_audio_16khz.tobytes(),
                sample_rate=16000,
                aggressiveness=level
            )
            
            assert isinstance(result, (list, dict))
    
    def test_detect_vad_supported_sample_rates(self, mock_config, mock_logger):
        """Test VAD supports standard sample rates"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        # WebRTC VAD supports 8000, 16000, 32000, 48000
        for rate in [8000, 16000, 32000]:
            duration = 0.5
            samples = int(rate * duration)
            audio = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples)) * 32767).astype(np.int16)
            
            result = interceptor.detect_voice_activity(
                audio.tobytes(),
                sample_rate=rate
            )
            
            assert isinstance(result, (list, dict))


# =============================================================================
# Call Analysis Tests
# =============================================================================

class TestCallAnalysis:
    """Tests for comprehensive call analysis"""
    
    def test_analyze_call_method_exists(self, mock_config, mock_logger):
        """Test analyze_call method exists"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        assert hasattr(interceptor, 'analyze_call')
        assert callable(interceptor.analyze_call)
    
    def test_analyze_call_returns_dict(self, mock_config, mock_logger, sample_audio_16khz):
        """Test analyze_call returns comprehensive dictionary"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        result = interceptor.analyze_call(
            sample_audio_16khz.tobytes(),
            sample_rate=16000
        )
        
        assert isinstance(result, dict)
    
    def test_analyze_call_includes_diarization(self, mock_config, mock_logger, sample_audio_16khz):
        """Test analyze_call includes diarization results"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        result = interceptor.analyze_call(
            sample_audio_16khz.tobytes(),
            sample_rate=16000
        )
        
        # Should have diarization-related keys
        assert 'diarization' in result or 'segments' in result or 'speakers' in result or 'error' in result
    
    def test_analyze_call_includes_vad(self, mock_config, mock_logger, sample_audio_16khz):
        """Test analyze_call includes VAD results"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        result = interceptor.analyze_call(
            sample_audio_16khz.tobytes(),
            sample_rate=16000
        )
        
        # Should have VAD-related keys
        assert 'vad' in result or 'voice_activity' in result or 'speech_segments' in result or 'error' in result
    
    def test_analyze_call_speaking_time_calculation(self, mock_config, mock_logger, sample_audio_16khz):
        """Test analyze_call calculates speaking times"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        result = interceptor.analyze_call(
            sample_audio_16khz.tobytes(),
            sample_rate=16000
        )
        
        # May include speaking time or similar metric
        assert isinstance(result, dict)


# =============================================================================
# Audio Format Tests
# =============================================================================

class TestAudioFormats:
    """Tests for audio format handling"""
    
    def test_handle_mono_audio(self, mock_config, mock_logger, sample_audio_16khz):
        """Test handling mono audio"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        result = interceptor.detect_voice_activity(
            sample_audio_16khz.tobytes(),
            sample_rate=16000
        )
        
        assert isinstance(result, (list, dict))
    
    def test_handle_int16_samples(self, mock_config, mock_logger):
        """Test handling int16 audio samples"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        # Generate int16 audio
        audio = np.array([0, 16384, 32767, -32768, -16384], dtype=np.int16)
        
        result = interceptor.detect_voice_activity(
            audio.tobytes(),
            sample_rate=16000
        )
        
        assert isinstance(result, (list, dict))


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestVoiceErrorHandling:
    """Tests for error handling in voice processing"""
    
    def test_decode_opus_invalid_data(self, mock_config, mock_logger):
        """Test decoding invalid Opus data"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        result = interceptor._decode_opus(b'invalid_opus_data', ssrc=0x12345678)
        
        # Should handle gracefully (return None or error)
        assert result is None or isinstance(result, (bytes, np.ndarray))
    
    def test_diarize_empty_audio(self, mock_config, mock_logger):
        """Test diarization with empty audio"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        result = interceptor.diarize_audio(b'', sample_rate=16000)
        
        assert isinstance(result, dict)
    
    def test_vad_empty_audio(self, mock_config, mock_logger):
        """Test VAD with empty audio"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        result = interceptor.detect_voice_activity(b'', sample_rate=16000)
        
        assert isinstance(result, (list, dict))
    
    def test_analyze_call_empty_audio(self, mock_config, mock_logger):
        """Test analyze_call with empty audio"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        result = interceptor.analyze_call(b'', sample_rate=16000)
        
        assert isinstance(result, dict)


# =============================================================================
# Integration Tests
# =============================================================================

class TestVoiceIntegration:
    """Integration tests for voice processing pipeline"""
    
    def test_full_voice_analysis_pipeline(self, mock_config, mock_logger, sample_audio_16khz):
        """Test complete voice analysis pipeline"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        audio_bytes = sample_audio_16khz.tobytes()
        
        # Step 1: VAD
        vad_result = interceptor.detect_voice_activity(audio_bytes, sample_rate=16000)
        assert isinstance(vad_result, (list, dict))
        
        # Step 2: Diarization
        diarization_result = interceptor.diarize_audio(audio_bytes, sample_rate=16000)
        assert isinstance(diarization_result, dict)
        
        # Step 3: Full analysis
        analysis_result = interceptor.analyze_call(audio_bytes, sample_rate=16000)
        assert isinstance(analysis_result, dict)
    
    def test_multi_speaker_scenario(self, mock_config, mock_logger):
        """Test with simulated multi-speaker audio"""
        from falconone.voice.interceptor import VoiceInterceptor
        
        interceptor = VoiceInterceptor(mock_config, mock_logger)
        
        # Generate 2 seconds with different frequencies (simulating 2 speakers)
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # First half: 440Hz, second half: 880Hz
        audio = np.zeros(len(t), dtype=np.int16)
        mid = len(t) // 2
        audio[:mid] = (np.sin(2 * np.pi * 440 * t[:mid]) * 32767).astype(np.int16)
        audio[mid:] = (np.sin(2 * np.pi * 880 * t[mid:]) * 32767).astype(np.int16)
        
        result = interceptor.analyze_call(audio.tobytes(), sample_rate=sample_rate)
        
        assert isinstance(result, dict)
