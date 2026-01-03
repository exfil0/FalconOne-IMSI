"""
FalconOne Voice Codec Support Module
v1.9.4: Extended codec support for VoLTE/VoNR interception

Supported Codecs:
- AMR-NB: Adaptive Multi-Rate Narrowband (8kHz, 4.75-12.2 kbps)
- AMR-WB: Adaptive Multi-Rate Wideband (16kHz, 6.6-23.85 kbps)
- EVS: Enhanced Voice Services (8-48kHz, 5.9-128 kbps)
- OPUS: Internet standard codec (8-48kHz, 6-510 kbps)
- SILK: Skype/proprietary codec (8-24kHz, 6-40 kbps)
- G.711: PCM A-law/μ-law (64 kbps)
- G.722: Sub-band ADPCM (48-64 kbps)
- G.729: CS-ACELP (8 kbps)

Features:
- Automatic codec detection from RTP payload type
- Transcoding between codecs
- Real-time decoding for streaming
- Quality metrics calculation
"""

import subprocess
import os
import struct
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import io

from ..utils.logger import ModuleLogger


class AudioCodec(Enum):
    """Supported audio codecs"""
    # Narrowband (8kHz)
    AMR_NB = "AMR-NB"
    G711_ALAW = "G.711-A"
    G711_ULAW = "G.711-U"
    G729 = "G.729"
    
    # Wideband (16kHz)
    AMR_WB = "AMR-WB"
    G722 = "G.722"
    
    # Super-Wideband/Fullband (24-48kHz)
    EVS = "EVS"
    OPUS = "OPUS"
    SILK = "SILK"
    
    # Raw
    PCM_S16LE = "PCM-S16LE"
    PCM_S16BE = "PCM-S16BE"
    
    # Unknown
    UNKNOWN = "UNKNOWN"


@dataclass
class AudioFormat:
    """Audio format specification"""
    codec: AudioCodec
    sample_rate: int  # Hz
    channels: int
    bits_per_sample: int = 16
    bitrate_kbps: Optional[float] = None
    
    @property
    def bytes_per_sample(self) -> int:
        return (self.bits_per_sample * self.channels) // 8
    
    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.bytes_per_sample


@dataclass
class DecodedAudio:
    """Decoded audio result"""
    pcm_data: bytes
    format: AudioFormat
    duration_ms: float
    original_codec: AudioCodec
    quality_metrics: Dict[str, float] = field(default_factory=dict)


# RTP Payload Type to Codec mapping (RFC 3551)
RTP_PAYLOAD_TYPES = {
    0: AudioCodec.G711_ULAW,
    8: AudioCodec.G711_ALAW,
    9: AudioCodec.G722,
    18: AudioCodec.G729,
    # Dynamic payload types (typically 96-127)
    96: AudioCodec.AMR_NB,
    97: AudioCodec.AMR_WB,
    98: AudioCodec.EVS,
    99: AudioCodec.OPUS,
    100: AudioCodec.SILK,
}


class CodecDecoder(ABC):
    """Abstract base class for codec decoders"""
    
    @abstractmethod
    def decode(self, data: bytes) -> Optional[bytes]:
        """Decode audio data to PCM"""
        pass
    
    @abstractmethod
    def get_output_format(self) -> AudioFormat:
        """Get output PCM format"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if decoder is available"""
        pass


class AMRDecoder(CodecDecoder):
    """AMR-NB and AMR-WB decoder"""
    
    # AMR-NB frame sizes (without CMR byte)
    AMR_NB_FRAME_SIZES = {
        0: 13,   # 4.75 kbps
        1: 14,   # 5.15 kbps
        2: 16,   # 5.90 kbps
        3: 18,   # 6.70 kbps
        4: 20,   # 7.40 kbps
        5: 21,   # 7.95 kbps
        6: 27,   # 10.2 kbps
        7: 32,   # 12.2 kbps
        15: 1,   # NO_DATA
    }
    
    # AMR-WB frame sizes
    AMR_WB_FRAME_SIZES = {
        0: 18,   # 6.60 kbps
        1: 24,   # 8.85 kbps
        2: 33,   # 12.65 kbps
        3: 37,   # 14.25 kbps
        4: 41,   # 15.85 kbps
        5: 47,   # 18.25 kbps
        6: 51,   # 19.85 kbps
        7: 59,   # 23.05 kbps
        8: 61,   # 23.85 kbps
        15: 1,   # NO_DATA
    }
    
    def __init__(self, wideband: bool = False, logger: logging.Logger = None):
        self.wideband = wideband
        self.logger = ModuleLogger('AMR-Decoder', logger) if logger else None
        self._ffmpeg_available = self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def decode(self, data: bytes) -> Optional[bytes]:
        """Decode AMR to PCM"""
        if not data:
            return None
        
        # Try native decoder first
        pcm = self._decode_native(data)
        if pcm:
            return pcm
        
        # Fallback to ffmpeg
        return self._decode_ffmpeg(data)
    
    def _decode_native(self, data: bytes) -> Optional[bytes]:
        """Native AMR decoding"""
        try:
            import pyamr
            mode = 'wb' if self.wideband else 'nb'
            decoder = pyamr.Decoder(mode=mode)
            return decoder.decode_bytes(data)
        except ImportError:
            return None
        except Exception:
            return None
    
    def _decode_ffmpeg(self, data: bytes) -> Optional[bytes]:
        """Decode using ffmpeg"""
        if not self._ffmpeg_available:
            return None
        
        try:
            # Prepare AMR data with header
            header = b'#!AMR-WB\n' if self.wideband else b'#!AMR\n'
            if not data.startswith(header[:5]):
                data = header + data
            
            sample_rate = 16000 if self.wideband else 8000
            
            with tempfile.NamedTemporaryFile(suffix='.amr', delete=False) as amr_file:
                amr_file.write(data)
                amr_path = amr_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as pcm_file:
                pcm_path = pcm_file.name
            
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'amr',
                '-i', amr_path,
                '-f', 's16le',
                '-ar', str(sample_rate),
                '-ac', '1',
                pcm_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(pcm_path):
                with open(pcm_path, 'rb') as f:
                    pcm_data = f.read()
                return pcm_data
            
            return None
            
        finally:
            for path in [amr_path, pcm_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    def get_output_format(self) -> AudioFormat:
        return AudioFormat(
            codec=AudioCodec.PCM_S16LE,
            sample_rate=16000 if self.wideband else 8000,
            channels=1
        )
    
    def is_available(self) -> bool:
        try:
            import pyamr
            return True
        except ImportError:
            return self._ffmpeg_available


class EVSDecoder(CodecDecoder):
    """Enhanced Voice Services (EVS) decoder"""
    
    EVS_FRAME_SIZES_BITS = {
        # Primary mode (no AMR-WB IO)
        2.8: 56,
        7.2: 144,
        8.0: 160,
        9.6: 192,
        13.2: 264,
        16.4: 328,
        24.4: 488,
        32.0: 640,
        48.0: 960,
        64.0: 1280,
        96.0: 1920,
        128.0: 2560,
    }
    
    def __init__(self, sample_rate: int = 16000, logger: logging.Logger = None):
        self.sample_rate = sample_rate
        self.logger = ModuleLogger('EVS-Decoder', logger) if logger else None
        self._evs_available = self._check_evs()
        self._ffmpeg_available = self._check_ffmpeg()
    
    def _check_evs(self) -> bool:
        """Check if EVS decoder is available"""
        try:
            result = subprocess.run(
                ['EVS_dec', '--help'],
                capture_output=True,
                timeout=5
            )
            return True
        except:
            return False
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg has EVS support"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-decoders'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return 'evs' in result.stdout.lower()
        except:
            return False
    
    def decode(self, data: bytes) -> Optional[bytes]:
        """Decode EVS to PCM"""
        if not data:
            return None
        
        # Try 3GPP reference decoder
        if self._evs_available:
            pcm = self._decode_evs_ref(data)
            if pcm:
                return pcm
        
        # Try ffmpeg
        if self._ffmpeg_available:
            return self._decode_ffmpeg(data)
        
        # Last resort: try to parse as AMR-WB IO mode
        return self._decode_amr_wb_io(data)
    
    def _decode_evs_ref(self, data: bytes) -> Optional[bytes]:
        """Decode using 3GPP EVS reference decoder"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as evs_file:
                evs_file.write(data)
                evs_path = evs_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as pcm_file:
                pcm_path = pcm_file.name
            
            cmd = [
                'EVS_dec',
                '-mime',  # MIME storage format
                evs_path,
                pcm_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(pcm_path):
                with open(pcm_path, 'rb') as f:
                    return f.read()
            
            return None
            
        finally:
            for path in [evs_path, pcm_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    def _decode_ffmpeg(self, data: bytes) -> Optional[bytes]:
        """Decode using ffmpeg"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.evs', delete=False) as evs_file:
                evs_file.write(data)
                evs_path = evs_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as pcm_file:
                pcm_path = pcm_file.name
            
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 'evs',
                '-i', evs_path,
                '-f', 's16le',
                '-ar', str(self.sample_rate),
                '-ac', '1',
                pcm_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(pcm_path):
                with open(pcm_path, 'rb') as f:
                    return f.read()
            
            return None
            
        finally:
            for path in [evs_path, pcm_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    def _decode_amr_wb_io(self, data: bytes) -> Optional[bytes]:
        """Try to decode as AMR-WB IO mode (backward compatible)"""
        amr_decoder = AMRDecoder(wideband=True)
        return amr_decoder.decode(data)
    
    def get_output_format(self) -> AudioFormat:
        return AudioFormat(
            codec=AudioCodec.PCM_S16LE,
            sample_rate=self.sample_rate,
            channels=1
        )
    
    def is_available(self) -> bool:
        return self._evs_available or self._ffmpeg_available


class OpusDecoder(CodecDecoder):
    """OPUS codec decoder"""
    
    def __init__(self, sample_rate: int = 48000, channels: int = 1, logger: logging.Logger = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.logger = ModuleLogger('OPUS-Decoder', logger) if logger else None
        self._opus_available = self._check_opus()
    
    def _check_opus(self) -> bool:
        """Check if opus decoder is available"""
        try:
            import opuslib
            return True
        except ImportError:
            pass
        
        try:
            result = subprocess.run(
                ['opusdec', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def decode(self, data: bytes) -> Optional[bytes]:
        """Decode OPUS to PCM"""
        if not data:
            return None
        
        # Try native Python decoder
        pcm = self._decode_native(data)
        if pcm:
            return pcm
        
        # Try opusdec command line tool
        return self._decode_opusdec(data)
    
    def _decode_native(self, data: bytes) -> Optional[bytes]:
        """Native OPUS decoding using opuslib"""
        try:
            import opuslib
            
            decoder = opuslib.Decoder(self.sample_rate, self.channels)
            
            # OPUS frames are typically 20ms
            frame_size = int(self.sample_rate * 0.02)
            
            # Decode frame by frame
            pcm_data = b''
            offset = 0
            
            while offset < len(data):
                # Try to find frame boundaries (simplified)
                # Real implementation would parse packet headers
                try:
                    # Assume each packet is a complete frame
                    frame_len = min(len(data) - offset, 640)  # Max frame size
                    frame = data[offset:offset + frame_len]
                    
                    pcm_frame = decoder.decode(frame, frame_size)
                    pcm_data += pcm_frame
                    
                    offset += frame_len
                except:
                    break
            
            return pcm_data if pcm_data else None
            
        except ImportError:
            return None
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Native OPUS decode failed: {e}")
            return None
    
    def _decode_opusdec(self, data: bytes) -> Optional[bytes]:
        """Decode using opusdec command"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as opus_file:
                opus_file.write(data)
                opus_path = opus_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as pcm_file:
                pcm_path = pcm_file.name
            
            cmd = [
                'opusdec',
                '--rate', str(self.sample_rate),
                '--force-wav',
                opus_path,
                pcm_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(pcm_path):
                with open(pcm_path, 'rb') as f:
                    return f.read()
            
            return None
            
        finally:
            for path in [opus_path, pcm_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    def get_output_format(self) -> AudioFormat:
        return AudioFormat(
            codec=AudioCodec.PCM_S16LE,
            sample_rate=self.sample_rate,
            channels=self.channels
        )
    
    def is_available(self) -> bool:
        return self._opus_available


class SILKDecoder(CodecDecoder):
    """SILK codec decoder (used by Skype, WeChat, etc.)"""
    
    def __init__(self, sample_rate: int = 24000, logger: logging.Logger = None):
        self.sample_rate = sample_rate
        self.logger = ModuleLogger('SILK-Decoder', logger) if logger else None
        self._silk_available = self._check_silk()
    
    def _check_silk(self) -> bool:
        """Check if SILK decoder is available"""
        try:
            result = subprocess.run(
                ['silk_v3_decoder', '-h'],
                capture_output=True,
                timeout=5
            )
            return True
        except:
            return False
    
    def decode(self, data: bytes) -> Optional[bytes]:
        """Decode SILK to PCM"""
        if not data:
            return None
        
        # Check for SILK header
        if not data.startswith(b'#!SILK_V3'):
            # Add header if missing
            data = b'#!SILK_V3' + data
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.silk', delete=False) as silk_file:
                silk_file.write(data)
                silk_path = silk_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as pcm_file:
                pcm_path = pcm_file.name
            
            cmd = [
                'silk_v3_decoder',
                silk_path,
                pcm_path,
                '-Fs_API', str(self.sample_rate)
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(pcm_path):
                with open(pcm_path, 'rb') as f:
                    return f.read()
            
            return None
            
        finally:
            for path in [silk_path, pcm_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    def get_output_format(self) -> AudioFormat:
        return AudioFormat(
            codec=AudioCodec.PCM_S16LE,
            sample_rate=self.sample_rate,
            channels=1
        )
    
    def is_available(self) -> bool:
        return self._silk_available


class G711Decoder(CodecDecoder):
    """G.711 A-law and μ-law decoder"""
    
    # A-law decompression table
    ALAW_TABLE = [
        -5504, -5248, -6016, -5760, -4480, -4224, -4992, -4736,
        -7552, -7296, -8064, -7808, -6528, -6272, -7040, -6784,
        -2752, -2624, -3008, -2880, -2240, -2112, -2496, -2368,
        -3776, -3648, -4032, -3904, -3264, -3136, -3520, -3392,
        -22016, -20992, -24064, -23040, -17920, -16896, -19968, -18944,
        -30208, -29184, -32256, -31232, -26112, -25088, -28160, -27136,
        -11008, -10496, -12032, -11520, -8960, -8448, -9984, -9472,
        -15104, -14592, -16128, -15616, -13056, -12544, -14080, -13568,
        -344, -328, -376, -360, -280, -264, -312, -296,
        -472, -456, -504, -488, -408, -392, -440, -424,
        -88, -72, -120, -104, -24, -8, -56, -40,
        -216, -200, -248, -232, -152, -136, -184, -168,
        -1376, -1312, -1504, -1440, -1120, -1056, -1248, -1184,
        -1888, -1824, -2016, -1952, -1632, -1568, -1760, -1696,
        -688, -656, -752, -720, -560, -528, -624, -592,
        -944, -912, -1008, -976, -816, -784, -880, -848,
        5504, 5248, 6016, 5760, 4480, 4224, 4992, 4736,
        7552, 7296, 8064, 7808, 6528, 6272, 7040, 6784,
        2752, 2624, 3008, 2880, 2240, 2112, 2496, 2368,
        3776, 3648, 4032, 3904, 3264, 3136, 3520, 3392,
        22016, 20992, 24064, 23040, 17920, 16896, 19968, 18944,
        30208, 29184, 32256, 31232, 26112, 25088, 28160, 27136,
        11008, 10496, 12032, 11520, 8960, 8448, 9984, 9472,
        15104, 14592, 16128, 15616, 13056, 12544, 14080, 13568,
        344, 328, 376, 360, 280, 264, 312, 296,
        472, 456, 504, 488, 408, 392, 440, 424,
        88, 72, 120, 104, 24, 8, 56, 40,
        216, 200, 248, 232, 152, 136, 184, 168,
        1376, 1312, 1504, 1440, 1120, 1056, 1248, 1184,
        1888, 1824, 2016, 1952, 1632, 1568, 1760, 1696,
        688, 656, 752, 720, 560, 528, 624, 592,
        944, 912, 1008, 976, 816, 784, 880, 848
    ]
    
    def __init__(self, law: str = 'ulaw', logger: logging.Logger = None):
        """
        Initialize G.711 decoder.
        
        Args:
            law: 'alaw' or 'ulaw'
        """
        self.law = law.lower()
        self.logger = ModuleLogger('G711-Decoder', logger) if logger else None
    
    def decode(self, data: bytes) -> Optional[bytes]:
        """Decode G.711 to PCM"""
        if not data:
            return None
        
        try:
            import audioop
            
            if self.law == 'alaw':
                return audioop.alaw2lin(data, 2)
            else:  # ulaw
                return audioop.ulaw2lin(data, 2)
                
        except ImportError:
            # Manual decoding fallback
            return self._decode_manual(data)
    
    def _decode_manual(self, data: bytes) -> bytes:
        """Manual G.711 decoding without audioop"""
        pcm_data = []
        
        for byte in data:
            if self.law == 'alaw':
                sample = self.ALAW_TABLE[byte]
            else:
                # μ-law decoding
                byte = ~byte & 0xFF
                sign = (byte & 0x80) >> 7
                exponent = (byte & 0x70) >> 4
                mantissa = byte & 0x0F
                
                sample = ((mantissa << 3) + 0x84) << exponent
                sample = -sample if sign else sample
            
            pcm_data.append(struct.pack('<h', sample))
        
        return b''.join(pcm_data)
    
    def get_output_format(self) -> AudioFormat:
        return AudioFormat(
            codec=AudioCodec.PCM_S16LE,
            sample_rate=8000,
            channels=1
        )
    
    def is_available(self) -> bool:
        return True  # Pure Python implementation always available


class VoiceCodecManager:
    """
    Manager for voice codec operations.
    
    Provides unified interface for decoding various voice codecs.
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = ModuleLogger('Codec-Manager', logger) if logger else None
        
        # Initialize decoders
        self.decoders: Dict[AudioCodec, CodecDecoder] = {
            AudioCodec.AMR_NB: AMRDecoder(wideband=False, logger=logger),
            AudioCodec.AMR_WB: AMRDecoder(wideband=True, logger=logger),
            AudioCodec.EVS: EVSDecoder(logger=logger),
            AudioCodec.OPUS: OpusDecoder(logger=logger),
            AudioCodec.SILK: SILKDecoder(logger=logger),
            AudioCodec.G711_ALAW: G711Decoder(law='alaw', logger=logger),
            AudioCodec.G711_ULAW: G711Decoder(law='ulaw', logger=logger),
        }
        
        # Check availability
        self._available_codecs = self._check_available_codecs()
        
        if self.logger:
            self.logger.info(f"Codec manager initialized",
                           available=list(self._available_codecs))
    
    def _check_available_codecs(self) -> List[AudioCodec]:
        """Check which codecs are available"""
        available = []
        for codec, decoder in self.decoders.items():
            if decoder.is_available():
                available.append(codec)
        return available
    
    def get_available_codecs(self) -> List[AudioCodec]:
        """Get list of available codecs"""
        return self._available_codecs.copy()
    
    def detect_codec(self, data: bytes, rtp_payload_type: int = None) -> AudioCodec:
        """
        Detect codec from data or RTP payload type.
        
        Args:
            data: Audio data bytes
            rtp_payload_type: RTP payload type (if known)
            
        Returns:
            Detected codec
        """
        # Check RTP payload type first
        if rtp_payload_type is not None:
            if rtp_payload_type in RTP_PAYLOAD_TYPES:
                return RTP_PAYLOAD_TYPES[rtp_payload_type]
        
        # Check for known headers
        if data:
            if data.startswith(b'#!AMR-WB'):
                return AudioCodec.AMR_WB
            elif data.startswith(b'#!AMR'):
                return AudioCodec.AMR_NB
            elif data.startswith(b'#!SILK'):
                return AudioCodec.SILK
            elif data.startswith(b'OggS'):
                return AudioCodec.OPUS
        
        return AudioCodec.UNKNOWN
    
    def decode(self, data: bytes, codec: AudioCodec = None, 
               rtp_payload_type: int = None) -> Optional[DecodedAudio]:
        """
        Decode audio data.
        
        Args:
            data: Encoded audio data
            codec: Codec to use (auto-detect if None)
            rtp_payload_type: RTP payload type for auto-detection
            
        Returns:
            DecodedAudio or None on failure
        """
        if not data:
            return None
        
        # Auto-detect codec if not specified
        if codec is None:
            codec = self.detect_codec(data, rtp_payload_type)
        
        if codec == AudioCodec.UNKNOWN:
            if self.logger:
                self.logger.warning("Unable to detect codec")
            return None
        
        # Get decoder
        decoder = self.decoders.get(codec)
        if not decoder:
            if self.logger:
                self.logger.error(f"No decoder for codec: {codec}")
            return None
        
        if not decoder.is_available():
            if self.logger:
                self.logger.error(f"Decoder not available: {codec}")
            return None
        
        # Decode
        import time
        start_time = time.time()
        
        pcm_data = decoder.decode(data)
        
        if pcm_data is None:
            if self.logger:
                self.logger.error(f"Decoding failed for codec: {codec}")
            return None
        
        decode_time = (time.time() - start_time) * 1000
        
        output_format = decoder.get_output_format()
        duration_ms = len(pcm_data) / output_format.bytes_per_second * 1000
        
        if self.logger:
            self.logger.debug(f"Decoded {codec.value}: {len(data)} -> {len(pcm_data)} bytes, "
                            f"{duration_ms:.0f}ms audio, {decode_time:.1f}ms decode time")
        
        return DecodedAudio(
            pcm_data=pcm_data,
            format=output_format,
            duration_ms=duration_ms,
            original_codec=codec,
            quality_metrics={
                'decode_time_ms': decode_time,
                'compression_ratio': len(data) / len(pcm_data) if pcm_data else 0
            }
        )
    
    def transcode(self, data: bytes, from_codec: AudioCodec, 
                  to_codec: AudioCodec) -> Optional[bytes]:
        """
        Transcode audio from one codec to another.
        
        Args:
            data: Source audio data
            from_codec: Source codec
            to_codec: Target codec
            
        Returns:
            Transcoded audio data or None on failure
        """
        # First decode to PCM
        decoded = self.decode(data, from_codec)
        if not decoded:
            return None
        
        # For now, only support transcoding to PCM
        if to_codec in (AudioCodec.PCM_S16LE, AudioCodec.PCM_S16BE):
            if to_codec == AudioCodec.PCM_S16BE:
                # Convert endianness
                samples = struct.unpack(f'<{len(decoded.pcm_data)//2}h', decoded.pcm_data)
                return struct.pack(f'>{len(samples)}h', *samples)
            return decoded.pcm_data
        
        if self.logger:
            self.logger.warning(f"Transcoding to {to_codec} not yet supported")
        return None
    
    def save_wav(self, pcm_data: bytes, output_path: str, 
                 sample_rate: int = 16000, channels: int = 1) -> bool:
        """
        Save PCM data as WAV file.
        
        Args:
            pcm_data: PCM audio data (16-bit signed LE)
            output_path: Output file path
            sample_rate: Sample rate in Hz
            channels: Number of channels
            
        Returns:
            True if successful
        """
        try:
            import wave
            
            with wave.open(output_path, 'wb') as wav:
                wav.setnchannels(channels)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(sample_rate)
                wav.writeframes(pcm_data)
            
            if self.logger:
                self.logger.info(f"Saved WAV file: {output_path}")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save WAV: {e}")
            return False
    
    def get_codec_info(self, codec: AudioCodec) -> Dict[str, Any]:
        """Get information about a codec"""
        info = {
            'codec': codec.value,
            'available': codec in self._available_codecs,
        }
        
        decoder = self.decoders.get(codec)
        if decoder:
            output_format = decoder.get_output_format()
            info.update({
                'output_sample_rate': output_format.sample_rate,
                'output_channels': output_format.channels,
                'output_bits': output_format.bits_per_sample,
            })
        
        # Codec-specific info
        codec_details = {
            AudioCodec.AMR_NB: {
                'bandwidth': 'Narrowband',
                'sample_rate': 8000,
                'bitrates': '4.75-12.2 kbps'
            },
            AudioCodec.AMR_WB: {
                'bandwidth': 'Wideband',
                'sample_rate': 16000,
                'bitrates': '6.6-23.85 kbps'
            },
            AudioCodec.EVS: {
                'bandwidth': 'Super-Wideband/Fullband',
                'sample_rate': '8-48 kHz',
                'bitrates': '5.9-128 kbps'
            },
            AudioCodec.OPUS: {
                'bandwidth': 'Fullband',
                'sample_rate': '8-48 kHz',
                'bitrates': '6-510 kbps'
            },
            AudioCodec.SILK: {
                'bandwidth': 'Wideband',
                'sample_rate': '8-24 kHz',
                'bitrates': '6-40 kbps'
            },
        }
        
        if codec in codec_details:
            info.update(codec_details[codec])
        
        return info
