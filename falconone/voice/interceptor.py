"""
FalconOne Voice Interception Module
VoLTE/VoNR capture and reassembly
"""

from typing import Dict, List, Any, Optional
import logging
import subprocess
import os
from collections import defaultdict
import struct

from ..utils.logger import ModuleLogger


class VoiceInterceptor:
    """Voice interception for VoLTE and VoNR"""
    
    def __init__(self, config, logger: logging.Logger):
        """Initialize voice interceptor"""
        self.config = config
        self.logger = ModuleLogger('VoiceInterceptor', logger)
        
        self.protocols = config.get('voice_interception.protocols', ['VoLTE', 'VoNR'])
        self.codecs = config.get('voice_interception.codecs', ['AMR', 'EVS'])
        
        # RTP stream reassembly
        self.rtp_streams = defaultdict(list)
        
        self.logger.info("Voice interceptor initialized", protocols=self.protocols)
    
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
    
    def decode_audio(self, encoded_data: bytes, codec: str) -> bytes:
        """
        Decode voice audio using external codec tools
        
        Args:
            encoded_data: Encoded audio data
            codec: Codec name ('AMR', 'EVS')
            
        Returns:
            PCM audio data
        """
        try:
            if codec == 'AMR':
                return self._decode_amr(encoded_data)
            elif codec == 'EVS':
                return self._decode_evs(encoded_data)
            else:
                self.logger.warning(f"Unknown codec: {codec}")
                return encoded_data
                
        except Exception as e:
            self.logger.error(f"Audio decoding error: {e}")
            return b''
    
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




