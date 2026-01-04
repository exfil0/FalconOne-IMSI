"""
FalconOne Voice Interception Module
VoLTE/VoNR/VoWiFi capture and reassembly

Version 1.9.4: Added Opus codec for VoWiFi and speaker diarization
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import subprocess
import os
from collections import defaultdict
import struct
import tempfile
import numpy as np

from ..utils.logger import ModuleLogger

# Optional imports for advanced features
try:
    import opuslib
    OPUS_AVAILABLE = True
except ImportError:
    OPUS_AVAILABLE = False

try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False


class VoiceInterceptor:
    """Voice interception for VoLTE, VoNR, and VoWiFi"""
    
    def __init__(self, config, logger: logging.Logger):
        """Initialize voice interceptor"""
        self.config = config
        self.logger = ModuleLogger('VoiceInterceptor', logger)
        
        self.protocols = config.get('voice_interception.protocols', ['VoLTE', 'VoNR', 'VoWiFi'])
        self.codecs = config.get('voice_interception.codecs', ['AMR', 'EVS', 'OPUS'])
        
        # RTP stream reassembly
        self.rtp_streams = defaultdict(list)
        
        # Opus decoder state
        self._opus_decoders: Dict[int, Any] = {}  # SSRC -> decoder
        
        # Speaker diarization pipeline
        self._diarization_pipeline = None
        self._voice_encoder = None
        self._init_diarization()
        
        self.logger.info("Voice interceptor initialized", 
                        protocols=self.protocols,
                        opus_available=OPUS_AVAILABLE,
                        diarization_available=PYANNOTE_AVAILABLE)
    
    def _init_diarization(self):
        """Initialize speaker diarization pipeline"""
        if PYANNOTE_AVAILABLE:
            try:
                # Use pretrained diarization model
                # Requires HuggingFace token for pyannote models
                hf_token = self.config.get('voice_interception.huggingface_token', None)
                
                if hf_token:
                    self._diarization_pipeline = DiarizationPipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token
                    )
                    self.logger.info("Pyannote diarization pipeline initialized")
                else:
                    self.logger.warning("HuggingFace token not configured, diarization limited")
            except Exception as e:
                self.logger.warning(f"Failed to initialize pyannote diarization: {e}")
        
        if RESEMBLYZER_AVAILABLE:
            try:
                self._voice_encoder = VoiceEncoder()
                self.logger.info("Resemblyzer voice encoder initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize voice encoder: {e}")
    
    def capture_voice(self, pcap_file: str) -> List[Dict[str, Any]]:
        """
        Capture and reassemble voice traffic from PCAP
        
        Args:
            pcap_file: Path to PCAP file containing VoLTE/VoNR traffic
            
        Returns:
            List of voice call records with audio
        """
        try:
            # Parse PCAP for SIP and RTP packets
            sip_sessions = self._parse_sip_sessions(pcap_file)
            
            # Extract and reassemble RTP streams
            voice_calls = []
            
            for session in sip_sessions:
                call_id = session.get('call_id')
                rtp_packets = self._extract_rtp_stream(pcap_file, session)
                
                if rtp_packets:
                    # Reassemble audio
                    audio_data = self._reassemble_rtp(rtp_packets)
                    
                    # Decode based on codec
                    codec = session.get('codec', 'AMR')
                    decoded_audio = self.decode_audio(audio_data, codec)
                    
                    # Save to file
                    output_file = f"/tmp/voice_call_{call_id}.wav"
                    self._save_wav(decoded_audio, output_file)
                    
                    voice_calls.append({
                        'call_id': call_id,
                        'from': session.get('from'),
                        'to': session.get('to'),
                        'codec': codec,
                        'duration': session.get('duration'),
                        'audio_file': output_file
                    })
                    
                    self.logger.info(f"Voice call captured: {call_id}")
            
            return voice_calls
            
        except Exception as e:
            self.logger.error(f"Voice capture error: {e}")
            return []
    
    def _parse_sip_sessions(self, pcap_file: str) -> List[Dict[str, Any]]:
        """Parse SIP signaling to identify voice sessions"""
        try:
            import pyshark
            
            cap = pyshark.FileCapture(pcap_file, display_filter='sip')
            
            sessions = []
            session_map = {}
            
            for pkt in cap:
                try:
                    if hasattr(pkt, 'sip'):
                        sip = pkt.sip
                        
                        # Extract Call-ID
                        call_id = str(sip.call_id) if hasattr(sip, 'call_id') else None
                        
                        if not call_id:
                            continue
                        
                        # INVITE - start new session
                        if hasattr(sip, 'method') and sip.method == 'INVITE':
                            session_map[call_id] = {
                                'call_id': call_id,
                                'from': str(sip.from_user) if hasattr(sip, 'from_user') else 'unknown',
                                'to': str(sip.to_user) if hasattr(sip, 'to_user') else 'unknown',
                                'start_time': float(pkt.sniff_timestamp),
                                'codec': 'AMR',  # Default
                                'rtp_port': None
                            }
                            
                            # Extract codec from SDP
                            if hasattr(sip, 'sdp_media_format'):
                                codec_str = str(sip.sdp_media_format)
                                if 'EVS' in codec_str:
                                    session_map[call_id]['codec'] = 'EVS'
                                elif 'AMR' in codec_str:
                                    session_map[call_id]['codec'] = 'AMR'
                            
                            # Extract RTP port from SDP
                            if hasattr(sip, 'sdp_media_port'):
                                session_map[call_id]['rtp_port'] = int(sip.sdp_media_port)
                        
                        # BYE - end session
                        elif hasattr(sip, 'method') and sip.method == 'BYE':
                            if call_id in session_map:
                                session_map[call_id]['end_time'] = float(pkt.sniff_timestamp)
                                session_map[call_id]['duration'] = \
                                    session_map[call_id]['end_time'] - session_map[call_id]['start_time']
                                sessions.append(session_map[call_id])
                                
                except AttributeError:
                    continue
            
            cap.close()
            
            # Add any sessions without BYE
            for session in session_map.values():
                if 'end_time' not in session:
                    session['duration'] = 0
                    sessions.append(session)
            
            self.logger.debug(f"Found {len(sessions)} SIP sessions")
            return sessions
            
        except ImportError:
            self.logger.error("pyshark not installed")
            return []
        except Exception as e:
            self.logger.error(f"SIP parsing error: {e}")
            return []
    
    def _extract_rtp_stream(self, pcap_file: str, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract RTP packets for a specific session"""
        try:
            import pyshark
            
            rtp_port = session.get('rtp_port')
            if not rtp_port:
                return []
            
            # Filter for RTP on specific port
            cap = pyshark.FileCapture(
                pcap_file,
                display_filter=f'rtp && udp.port == {rtp_port}'
            )
            
            rtp_packets = []
            
            for pkt in cap:
                try:
                    if hasattr(pkt, 'rtp'):
                        rtp = pkt.rtp
                        rtp_packets.append({
                            'seq': int(rtp.seq) if hasattr(rtp, 'seq') else 0,
                            'timestamp': int(rtp.timestamp) if hasattr(rtp, 'timestamp') else 0,
                            'payload': bytes.fromhex(rtp.payload.replace(':', '')) if hasattr(rtp, 'payload') else b'',
                            'ssrc': int(rtp.ssrc, 16) if hasattr(rtp, 'ssrc') else 0
                        })
                except AttributeError:
                    continue
            
            cap.close()
            
            # Sort by sequence number
            rtp_packets.sort(key=lambda x: x['seq'])
            
            self.logger.debug(f"Extracted {len(rtp_packets)} RTP packets")
            return rtp_packets
            
        except Exception as e:
            self.logger.error(f"RTP extraction error: {e}")
            return []
    
    def _reassemble_rtp(self, rtp_packets: List[Dict[str, Any]]) -> bytes:
        """Reassemble RTP packets into continuous audio stream"""
        audio_data = b''
        
        for packet in rtp_packets:
            audio_data += packet['payload']
        
        return audio_data
    
    def decode_audio(self, encoded_data: bytes, codec: str, ssrc: int = 0) -> bytes:
        """
        Decode voice audio using external codec tools
        
        Args:
            encoded_data: Encoded audio data
            codec: Codec name ('AMR', 'EVS', 'OPUS')
            ssrc: RTP SSRC for stateful decoders (Opus)
            
        Returns:
            PCM audio data
        """
        try:
            if codec == 'AMR':
                return self._decode_amr(encoded_data)
            elif codec == 'EVS':
                return self._decode_evs(encoded_data)
            elif codec == 'OPUS' or codec.upper() == 'OPUS':
                return self._decode_opus(encoded_data, ssrc)
            else:
                self.logger.warning(f"Unknown codec: {codec}")
                return encoded_data
                
        except Exception as e:
            self.logger.error(f"Audio decoding error: {e}")
            return b''
    
    def _decode_opus(self, opus_data: bytes, ssrc: int = 0, 
                     sample_rate: int = 48000, channels: int = 1) -> bytes:
        """
        Decode Opus audio (VoWiFi/WebRTC standard).
        
        Opus is the mandatory codec for WebRTC and commonly used in VoWiFi.
        
        Args:
            opus_data: Opus-encoded audio frames
            ssrc: RTP SSRC for maintaining decoder state
            sample_rate: Output sample rate (8000, 12000, 16000, 24000, 48000)
            channels: Number of output channels (1 or 2)
            
        Returns:
            PCM audio data (16-bit signed, native endian)
        """
        if OPUS_AVAILABLE:
            return self._decode_opus_native(opus_data, ssrc, sample_rate, channels)
        else:
            return self._decode_opus_ffmpeg(opus_data, sample_rate, channels)
    
    def _decode_opus_native(self, opus_data: bytes, ssrc: int,
                            sample_rate: int, channels: int) -> bytes:
        """
        Native Opus decoding using opuslib.
        
        Maintains per-SSRC decoder state for proper PLC (packet loss concealment).
        """
        try:
            # Get or create decoder for this SSRC
            if ssrc not in self._opus_decoders:
                self._opus_decoders[ssrc] = opuslib.Decoder(sample_rate, channels)
                self.logger.debug(f"Created Opus decoder for SSRC {ssrc}")
            
            decoder = self._opus_decoders[ssrc]
            
            # Opus frame size is variable, typical: 2.5, 5, 10, 20, 40, 60 ms
            # For WebRTC, typically 20ms @ 48kHz = 960 samples
            frame_size = 960  # Default 20ms @ 48kHz
            
            # Decode opus data
            pcm_data = decoder.decode(opus_data, frame_size)
            
            self.logger.debug(f"Opus decoded: {len(opus_data)} bytes -> {len(pcm_data)} bytes PCM")
            return pcm_data
            
        except Exception as e:
            self.logger.error(f"Native Opus decode error: {e}")
            return self._decode_opus_ffmpeg(opus_data, sample_rate, channels)
    
    def _decode_opus_ffmpeg(self, opus_data: bytes, sample_rate: int, channels: int) -> bytes:
        """
        Decode Opus using ffmpeg fallback.
        """
        try:
            opus_file = '/tmp/audio_opus.ogg'
            pcm_file = '/tmp/audio_pcm.raw'
            
            # Opus data might need Ogg container for ffmpeg
            with open(opus_file, 'wb') as f:
                f.write(opus_data)
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'ogg',  # Ogg container with Opus
                '-i', opus_file,
                '-f', 's16le',  # 16-bit PCM
                '-ar', str(sample_rate),
                '-ac', str(channels),
                '-acodec', 'pcm_s16le',
                pcm_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=10, text=True)
            
            if result.returncode == 0 and os.path.exists(pcm_file):
                with open(pcm_file, 'rb') as f:
                    pcm_data = f.read()
                
                # Cleanup
                if os.path.exists(opus_file):
                    os.remove(opus_file)
                if os.path.exists(pcm_file):
                    os.remove(pcm_file)
                
                self.logger.debug(f"Opus ffmpeg decoded: {len(pcm_data)} bytes PCM ({sample_rate}Hz)")
                return pcm_data
            else:
                self.logger.error(f"Opus ffmpeg decoding failed: {result.stderr}")
                return b''
                
        except FileNotFoundError:
            self.logger.error("ffmpeg not found for Opus decoding")
            return b''
        except Exception as e:
            self.logger.error(f"Opus ffmpeg decode error: {e}")
            return b''
    
    def decode_opus_stream(self, rtp_packets: List[Dict[str, Any]], 
                           sample_rate: int = 48000) -> bytes:
        """
        Decode a complete Opus RTP stream with proper frame handling.
        
        Handles:
        - Packet reordering
        - Packet loss (PLC)
        - Variable bitrate frames
        
        Args:
            rtp_packets: List of RTP packets with 'payload' and 'ssrc'
            sample_rate: Target sample rate
            
        Returns:
            Decoded PCM audio
        """
        if not rtp_packets:
            return b''
        
        # Group by SSRC and sort by sequence
        ssrc = rtp_packets[0].get('ssrc', 0)
        sorted_packets = sorted(rtp_packets, key=lambda x: x.get('seq', 0))
        
        pcm_frames = []
        last_seq = None
        
        for packet in sorted_packets:
            seq = packet.get('seq', 0)
            payload = packet.get('payload', b'')
            
            # Handle packet loss with PLC
            if last_seq is not None and seq > last_seq + 1:
                lost_packets = seq - last_seq - 1
                self.logger.debug(f"Opus: {lost_packets} packets lost, applying PLC")
                
                # Generate PLC frames (pass None to decoder)
                if OPUS_AVAILABLE and ssrc in self._opus_decoders:
                    for _ in range(min(lost_packets, 10)):  # Limit PLC
                        try:
                            plc_frame = self._opus_decoders[ssrc].decode(None, 960)
                            pcm_frames.append(plc_frame)
                        except Exception:
                            pass
            
            # Decode actual frame
            pcm = self._decode_opus(payload, ssrc, sample_rate, 1)
            if pcm:
                pcm_frames.append(pcm)
            
            last_seq = seq
        
        return b''.join(pcm_frames)
    
    def _decode_amr(self, amr_data: bytes) -> bytes:
        """
        Decode AMR audio using opencore-amr or ffmpeg.
        
        Supports:
        - AMR-NB (Narrowband): 8kHz, 4.75-12.2 kbps
        - AMR-WB (Wideband): 16kHz, 6.6-23.85 kbps
        """
        try:
            # Write AMR data to temp file
            amr_file = '/tmp/audio_amr.amr'
            pcm_file = '/tmp/audio_pcm.raw'
            wav_file = '/tmp/audio.wav'
            
            # Add AMR header if not present
            amr_header_nb = b'#!AMR\n'
            amr_header_wb = b'#!AMR-WB\n'
            
            # Detect AMR type
            is_wideband = False
            if not amr_data.startswith(amr_header_nb) and not amr_data.startswith(amr_header_wb):
                # Default to narrowband
                amr_data = amr_header_nb + amr_data
            elif amr_data.startswith(amr_header_wb):
                is_wideband = True
            
            with open(amr_file, 'wb') as f:
                f.write(amr_data)
            
            # Try native Python decoding first (faster)
            try:
                import audioop
                # Use opencore-amr if available
                pcm_data = self._decode_amr_native(amr_data, is_wideband)
                if pcm_data:
                    return pcm_data
            except Exception:
                pass
            
            # Fallback to ffmpeg (slower but more reliable)
            sample_rate = '16000' if is_wideband else '8000'
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'amr',  # Force AMR format
                '-i', amr_file,
                '-f', 's16le',  # 16-bit PCM
                '-ar', sample_rate,
                '-ac', '1',  # Mono
                '-acodec', 'pcm_s16le',
                pcm_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=10, text=True)
            
            if result.returncode == 0 and os.path.exists(pcm_file):
                with open(pcm_file, 'rb') as f:
                    pcm_data = f.read()
                
                # Cleanup
                if os.path.exists(amr_file):
                    os.remove(amr_file)
                if os.path.exists(pcm_file):
                    os.remove(pcm_file)
                
                self.logger.debug(f"AMR decoded: {len(pcm_data)} bytes PCM ({sample_rate}Hz)")
                return pcm_data
            else:
                self.logger.error(f"AMR decoding failed: {result.stderr}")
                return b''
                
        except FileNotFoundError:
            self.logger.error("ffmpeg not found. Install with: apt-get install ffmpeg")
            return b''
        except Exception as e:
            self.logger.error(f"AMR decode error: {e}")
            return b''
    
    def _decode_amr_native(self, amr_data: bytes, is_wideband: bool = False) -> Optional[bytes]:
        """
        Native AMR decoding using opencore-amr library (if available).
        
        Much faster than ffmpeg for real-time processing.
        """
        try:
            import pyamr
            
            # Remove header
            if amr_data.startswith(b'#!AMR-WB\\n'):
                amr_data = amr_data[9:]
            elif amr_data.startswith(b'#!AMR\\n'):
                amr_data = amr_data[6:]
            
            # Decode frames
            if is_wideband:
                # AMR-WB: 320 samples per 20ms frame @ 16kHz
                decoder = pyamr.Decoder(mode='wb')
            else:
                # AMR-NB: 160 samples per 20ms frame @ 8kHz
                decoder = pyamr.Decoder(mode='nb')
            
            pcm_data = decoder.decode_bytes(amr_data)
            
            self.logger.debug(f"Native AMR decoding: {len(pcm_data)} bytes")
            return pcm_data
            
        except ImportError:
            # pyamr not available, will fallback to ffmpeg
            return None
        except Exception as e:
            self.logger.debug(f"Native AMR decode failed: {e}, falling back to ffmpeg")
            return None
    
    def _decode_evs(self, evs_data: bytes) -> bytes:
        """
        Decode EVS audio (Enhanced Voice Services).
        
        Supports:
        - EVS Primary: 5.9-128 kbps, up to 48kHz Super-Wideband/Fullband
        - EVS AMR-WB IO: Backward compatible with AMR-WB
        - Channel-aware mode for VoLTE
        """
        try:
            # EVS decoding requires proprietary codec
            # Use 3GPP EVS codec reference implementation
            
            evs_file = '/tmp/audio_evs.bin'
            pcm_file = '/tmp/audio_pcm.raw'
            
            with open(evs_file, 'wb') as f:
                f.write(evs_data)
            
            # Try native EVS decoder first
            pcm_data = self._decode_evs_native(evs_data)
            if pcm_data:
                return pcm_data
            
            # Use EVS decoder (if available)
            cmd = [
                'EVS_dec',  # 3GPP EVS decoder
                '-mime',  # MIME storage format (VoLTE)
                evs_file,
                pcm_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=10, text=True)
            
            if result.returncode == 0 and os.path.exists(pcm_file):
                with open(pcm_file, 'rb') as f:
                    pcm_data = f.read()
                
                if os.path.exists(evs_file):
                    os.remove(evs_file)
                if os.path.exists(pcm_file):
                    os.remove(pcm_file)
                
                self.logger.debug(f"EVS decoded: {len(pcm_data)} bytes PCM")
                return pcm_data
            else:
                # Fallback: Try ffmpeg with EVS support
                return self._decode_evs_ffmpeg(evs_data, evs_file, pcm_file)
                
        except FileNotFoundError:
            self.logger.warning("EVS decoder not found, trying ffmpeg fallback")
            return self._decode_evs_ffmpeg(evs_data, '/tmp/audio_evs.bin', '/tmp/audio_pcm.raw')
        except Exception as e:
            self.logger.error(f"EVS decode error: {e}")
            return b''
    
    def _decode_evs_native(self, evs_data: bytes) -> Optional[bytes]:
        """
        Native EVS decoding using Python library (if available).
        
        EVS is more complex than AMR, native decoders are rare.
        """
        try:
            # Check for Python EVS bindings (uncommon)
            import pyevs
            
            decoder = pyevs.Decoder()
            pcm_data = decoder.decode(evs_data)
            
            self.logger.debug(f"Native EVS decoding: {len(pcm_data)} bytes")
            return pcm_data
            
        except ImportError:
            # No native EVS available
            return None
        except Exception as e:
            self.logger.debug(f"Native EVS decode failed: {e}")
            return None
    
    def _decode_evs_ffmpeg(self, evs_data: bytes, evs_file: str, pcm_file: str) -> bytes:
        """
        Decode EVS using ffmpeg (if it has EVS support).
        """
        try:
            with open(evs_file, 'wb') as f:
                f.write(evs_data)
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'evs',  # EVS format
                '-i', evs_file,
                '-f', 's16le',  # 16-bit PCM
                '-ar', '16000',  # 16kHz default
                '-ac', '1',  # Mono
                pcm_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=10, text=True)
            
            if result.returncode == 0 and os.path.exists(pcm_file):
                with open(pcm_file, 'rb') as f:
                    pcm_data = f.read()
                
                if os.path.exists(evs_file):
                    os.remove(evs_file)
                if os.path.exists(pcm_file):
                    os.remove(pcm_file)
                
                return pcm_data
            else:
                self.logger.warning("EVS decoder not available, returning raw data")
                return evs_data
                
        except Exception as e:
            self.logger.warning(f"EVS ffmpeg decode failed: {e}, returning raw data")
            return evs_data
    
    def decode_realtime_stream(self, rtp_packets: List[Dict[str, Any]], codec: str) -> bytes:
        """
        Decode RTP stream in real-time mode (low latency).
        
        Process packets as they arrive without waiting for full stream.
        Useful for live monitoring.
        """
        decoded_frames = []
        
        for packet in rtp_packets:
            payload = packet['payload']
            
            # Decode individual frame
            if codec == 'AMR':
                # AMR frame decoding (single frame)
                decoded = self._decode_amr_frame(payload)
            elif codec == 'EVS':
                # EVS frame decoding
                decoded = self._decode_evs_frame(payload)
            else:
                decoded = payload
            
            if decoded:
                decoded_frames.append(decoded)
        
        # Concatenate all decoded frames
        return b''.join(decoded_frames)
    
    def _decode_amr_frame(self, frame_data: bytes) -> bytes:
        """Decode single AMR frame (20ms)."""
        try:
            # AMR frame is self-contained, can decode independently
            return self._decode_amr(frame_data)
        except Exception as e:
            self.logger.debug(f"AMR frame decode error: {e}")
            return b''
    
    def _decode_evs_frame(self, frame_data: bytes) -> bytes:
        """Decode single EVS frame (20ms)."""
        try:
            # EVS frame decoding
            return self._decode_evs(frame_data)
        except Exception as e:
            self.logger.debug(f"EVS frame decode error: {e}")
            return b''
    
    def export_audio_formats(self, pcm_data: bytes, base_filename: str, sample_rate: int = 8000):
        """
        Export audio in multiple formats: WAV, MP3, FLAC.
        
        Args:
            pcm_data: Raw PCM audio data
            base_filename: Base filename without extension
            sample_rate: Sample rate in Hz
            
        Returns:
            Dict with paths to exported files
        """
        exports = {}
        
        # WAV format (lossless)
        wav_file = f"{base_filename}.wav"
        self._save_wav(pcm_data, wav_file, sample_rate)
        exports['wav'] = wav_file
        
        # MP3 format (compressed, widely compatible)
        try:
            mp3_file = f"{base_filename}.mp3"
            cmd = [
                'ffmpeg', '-y',
                '-f', 's16le',
                '-ar', str(sample_rate),
                '-ac', '1',
                '-i', 'pipe:0',
                '-codec:a', 'libmp3lame',
                '-qscale:a', '2',  # High quality MP3
                mp3_file
            ]
            
            result = subprocess.run(cmd, input=pcm_data, capture_output=True, timeout=10)
            
            if result.returncode == 0 and os.path.exists(mp3_file):
                exports['mp3'] = mp3_file
                self.logger.debug(f"MP3 export: {mp3_file}")
        except Exception as e:
            self.logger.debug(f"MP3 export failed: {e}")
        
        # FLAC format (lossless compression)
        try:
            flac_file = f"{base_filename}.flac"
            cmd = [
                'ffmpeg', '-y',
                '-f', 's16le',
                '-ar', str(sample_rate),
                '-ac', '1',
                '-i', 'pipe:0',
                '-codec:a', 'flac',
                flac_file
            ]
            
            result = subprocess.run(cmd, input=pcm_data, capture_output=True, timeout=10)
            
            if result.returncode == 0 and os.path.exists(flac_file):
                exports['flac'] = flac_file
                self.logger.debug(f"FLAC export: {flac_file}")
        except Exception as e:
            self.logger.debug(f"FLAC export failed: {e}")
        
        return exports
    
    def _save_wav(self, pcm_data: bytes, output_file: str, sample_rate: int = 8000):
        """Save PCM data as WAV file"""
        try:
            import wave
            
            with wave.open(output_file, 'wb') as wav:
                wav.setnchannels(1)  # Mono
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(sample_rate)
                wav.writeframes(pcm_data)
            
            self.logger.debug(f"Audio saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"WAV save error: {e}")
    
    # =========================================================================
    # Speaker Diarization & Voice Analysis (v1.9.4)
    # =========================================================================
    
    def diarize_audio(self, wav_file: str, 
                      num_speakers: Optional[int] = None,
                      min_speakers: int = 1,
                      max_speakers: int = 10) -> Dict[str, Any]:
        """
        Perform speaker diarization on audio file.
        
        Identifies who spoke when in a multi-speaker audio recording.
        Uses pyannote.audio for state-of-the-art diarization.
        
        Args:
            wav_file: Path to WAV audio file
            num_speakers: Known number of speakers (None for auto-detect)
            min_speakers: Minimum expected speakers (if num_speakers is None)
            max_speakers: Maximum expected speakers (if num_speakers is None)
            
        Returns:
            Dict containing:
            - segments: List of (start, end, speaker) tuples
            - speakers: Set of unique speaker labels
            - timeline: Time-aligned speaker annotations
        """
        if not PYANNOTE_AVAILABLE or self._diarization_pipeline is None:
            return self._diarize_fallback(wav_file, num_speakers)
        
        try:
            # Run diarization pipeline
            if num_speakers:
                diarization = self._diarization_pipeline(
                    wav_file,
                    num_speakers=num_speakers
                )
            else:
                diarization = self._diarization_pipeline(
                    wav_file,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
            
            # Extract segments
            segments = []
            speakers = set()
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'duration': turn.end - turn.start,
                    'speaker': speaker
                })
                speakers.add(speaker)
            
            self.logger.info(f"Diarization complete: {len(segments)} segments, "
                           f"{len(speakers)} speakers")
            
            return {
                'segments': segments,
                'speakers': list(speakers),
                'num_speakers': len(speakers),
                'total_segments': len(segments),
                'source': 'pyannote'
            }
            
        except Exception as e:
            self.logger.error(f"Diarization error: {e}")
            return self._diarize_fallback(wav_file, num_speakers)
    
    def _diarize_fallback(self, wav_file: str, 
                          num_speakers: Optional[int] = None) -> Dict[str, Any]:
        """
        Fallback diarization using resemblyzer embeddings and clustering.
        
        Less accurate than pyannote but works without pretrained models.
        """
        if not RESEMBLYZER_AVAILABLE or self._voice_encoder is None:
            self.logger.warning("No diarization backend available")
            return {
                'segments': [],
                'speakers': [],
                'num_speakers': 0,
                'error': 'No diarization backend available'
            }
        
        try:
            from sklearn.cluster import AgglomerativeClustering
            import librosa
            
            # Load and preprocess audio
            wav, sr = librosa.load(wav_file, sr=16000)
            wav = preprocess_wav(wav)
            
            # Segment audio (500ms windows with 250ms overlap)
            window_samples = int(0.5 * sr)
            hop_samples = int(0.25 * sr)
            
            segments = []
            embeddings = []
            
            for i in range(0, len(wav) - window_samples, hop_samples):
                segment = wav[i:i + window_samples]
                
                # Get voice embedding
                embed = self._voice_encoder.embed_utterance(segment)
                
                segments.append({
                    'start': i / sr,
                    'end': (i + window_samples) / sr,
                    'duration': window_samples / sr
                })
                embeddings.append(embed)
            
            if not embeddings:
                return {'segments': [], 'speakers': [], 'num_speakers': 0}
            
            # Cluster embeddings
            embeddings_array = np.array(embeddings)
            
            n_clusters = num_speakers or min(4, len(embeddings))
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings_array)
            
            # Assign speakers to segments
            for i, label in enumerate(labels):
                segments[i]['speaker'] = f"SPEAKER_{label:02d}"
            
            speakers = list(set(f"SPEAKER_{l:02d}" for l in labels))
            
            self.logger.info(f"Fallback diarization: {len(segments)} segments, "
                           f"{len(speakers)} speakers")
            
            return {
                'segments': segments,
                'speakers': speakers,
                'num_speakers': len(speakers),
                'total_segments': len(segments),
                'source': 'resemblyzer'
            }
            
        except ImportError as e:
            self.logger.error(f"Missing dependency for fallback diarization: {e}")
            return {'segments': [], 'speakers': [], 'num_speakers': 0, 'error': str(e)}
        except Exception as e:
            self.logger.error(f"Fallback diarization error: {e}")
            return {'segments': [], 'speakers': [], 'num_speakers': 0, 'error': str(e)}
    
    def extract_speaker_embeddings(self, wav_file: str) -> Dict[str, np.ndarray]:
        """
        Extract voice embeddings for each speaker in audio.
        
        Useful for speaker identification and voice fingerprinting.
        
        Args:
            wav_file: Path to WAV audio file
            
        Returns:
            Dict mapping speaker labels to their embedding vectors
        """
        if not RESEMBLYZER_AVAILABLE or self._voice_encoder is None:
            self.logger.warning("Resemblyzer not available for embedding extraction")
            return {}
        
        try:
            import librosa
            
            # First diarize to get speaker segments
            diarization = self.diarize_audio(wav_file)
            
            if not diarization.get('segments'):
                return {}
            
            # Load audio
            wav, sr = librosa.load(wav_file, sr=16000)
            
            # Collect segments per speaker
            speaker_segments: Dict[str, List[np.ndarray]] = defaultdict(list)
            
            for segment in diarization['segments']:
                speaker = segment['speaker']
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)
                
                segment_audio = wav[start_sample:end_sample]
                
                if len(segment_audio) >= sr * 0.5:  # At least 500ms
                    speaker_segments[speaker].append(segment_audio)
            
            # Compute average embedding per speaker
            speaker_embeddings = {}
            
            for speaker, segments in speaker_segments.items():
                embeddings = []
                for segment_audio in segments[:10]:  # Limit to 10 segments
                    try:
                        processed = preprocess_wav(segment_audio)
                        embed = self._voice_encoder.embed_utterance(processed)
                        embeddings.append(embed)
                    except Exception:
                        continue
                
                if embeddings:
                    speaker_embeddings[speaker] = np.mean(embeddings, axis=0)
            
            self.logger.info(f"Extracted embeddings for {len(speaker_embeddings)} speakers")
            return speaker_embeddings
            
        except Exception as e:
            self.logger.error(f"Speaker embedding extraction error: {e}")
            return {}
    
    def detect_voice_activity(self, pcm_data: bytes, sample_rate: int = 16000,
                              frame_duration_ms: int = 30,
                              aggressiveness: int = 2) -> List[Dict[str, Any]]:
        """
        Detect voice activity in audio using WebRTC VAD.
        
        Args:
            pcm_data: Raw PCM audio (16-bit, mono)
            sample_rate: Sample rate (8000, 16000, 32000, or 48000)
            frame_duration_ms: Frame duration (10, 20, or 30 ms)
            aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
            
        Returns:
            List of voice activity segments with start/end times
        """
        if not WEBRTCVAD_AVAILABLE:
            self.logger.warning("webrtcvad not available")
            return []
        
        try:
            vad = webrtcvad.Vad(aggressiveness)
            
            # Calculate frame size in bytes
            frame_samples = int(sample_rate * frame_duration_ms / 1000)
            frame_bytes = frame_samples * 2  # 16-bit = 2 bytes per sample
            
            # Process frames
            frames = []
            for i in range(0, len(pcm_data) - frame_bytes, frame_bytes):
                frame = pcm_data[i:i + frame_bytes]
                is_speech = vad.is_speech(frame, sample_rate)
                frames.append({
                    'start': i / (sample_rate * 2),  # Convert bytes to seconds
                    'end': (i + frame_bytes) / (sample_rate * 2),
                    'is_speech': is_speech
                })
            
            # Merge consecutive speech frames into segments
            segments = []
            current_segment = None
            
            for frame in frames:
                if frame['is_speech']:
                    if current_segment is None:
                        current_segment = {'start': frame['start'], 'end': frame['end']}
                    else:
                        current_segment['end'] = frame['end']
                else:
                    if current_segment is not None:
                        current_segment['duration'] = current_segment['end'] - current_segment['start']
                        segments.append(current_segment)
                        current_segment = None
            
            # Don't forget last segment
            if current_segment is not None:
                current_segment['duration'] = current_segment['end'] - current_segment['start']
                segments.append(current_segment)
            
            self.logger.debug(f"VAD: {len(segments)} speech segments detected")
            return segments
            
        except Exception as e:
            self.logger.error(f"VAD error: {e}")
            return []
    
    def analyze_call(self, wav_file: str, perform_diarization: bool = True,
                     extract_embeddings: bool = True) -> Dict[str, Any]:
        """
        Comprehensive voice call analysis.
        
        Performs:
        - Speaker diarization (who spoke when)
        - Speaker embedding extraction (voice fingerprints)
        - Voice activity detection
        - Speaking time statistics per speaker
        
        Args:
            wav_file: Path to WAV audio file
            perform_diarization: Whether to run diarization
            extract_embeddings: Whether to extract voice embeddings
            
        Returns:
            Comprehensive analysis results
        """
        import wave
        
        analysis = {
            'file': wav_file,
            'duration': 0.0,
            'sample_rate': 0,
            'diarization': None,
            'embeddings': {},
            'speaking_times': {},
            'vad_segments': []
        }
        
        try:
            # Get audio info
            with wave.open(wav_file, 'rb') as wav:
                analysis['sample_rate'] = wav.getframerate()
                analysis['duration'] = wav.getnframes() / wav.getframerate()
                analysis['channels'] = wav.getnchannels()
        except Exception as e:
            self.logger.error(f"Could not read WAV file: {e}")
            return analysis
        
        # Speaker diarization
        if perform_diarization:
            analysis['diarization'] = self.diarize_audio(wav_file)
            
            # Calculate speaking times
            if analysis['diarization'].get('segments'):
                speaking_times: Dict[str, float] = defaultdict(float)
                for segment in analysis['diarization']['segments']:
                    speaking_times[segment['speaker']] += segment['duration']
                
                analysis['speaking_times'] = dict(speaking_times)
        
        # Speaker embeddings
        if extract_embeddings:
            analysis['embeddings'] = self.extract_speaker_embeddings(wav_file)
            # Convert numpy arrays to lists for JSON serialization
            analysis['embeddings'] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in analysis['embeddings'].items()
            }
        
        # VAD on raw audio
        try:
            with open(wav_file, 'rb') as f:
                # Skip WAV header (44 bytes typically)
                f.seek(44)
                pcm_data = f.read()
            
            analysis['vad_segments'] = self.detect_voice_activity(
                pcm_data, 
                sample_rate=analysis['sample_rate']
            )
        except Exception as e:
            self.logger.debug(f"VAD analysis skipped: {e}")
        
        self.logger.info(f"Call analysis complete: {analysis['duration']:.1f}s, "
                        f"{len(analysis.get('diarization', {}).get('speakers', []))} speakers")
        
        return analysis




