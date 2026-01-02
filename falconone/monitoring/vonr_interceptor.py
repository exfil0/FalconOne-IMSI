"""
FalconOne Enhanced VoNR Interceptor Module (v1.5.1)
Voice over New Radio (VoNR) interception with SDAP/QoS flow parsing
Capabilities:
- SDAP header parsing for QoS Flow ID extraction
- RTP payload extraction from QoS flows
- EVS/AMR-WB codec decoding
- Call metadata extraction (setup, duration, participants)

References:
- 3GPP TS 38.323 (SDAP protocol)
- 3GPP TS 26.445 (EVS codec)
- RFC 3550 (RTP)
- RFC 4867 (RTP payload format for AMR and AMR-WB)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import struct
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


@dataclass
class QoSFlowInfo:
    """QoS Flow information"""
    qfi: int  # QoS Flow ID
    five_qi: int  # 5QI value
    gfbr_kbps: Optional[int] = None  # Guaranteed Flow Bit Rate
    mfbr_kbps: Optional[int] = None  # Maximum Flow Bit Rate
    is_voice: bool = False


@dataclass
class RTPPacket:
    """Parsed RTP packet"""
    version: int
    payload_type: int
    sequence_number: int
    timestamp: int
    ssrc: int
    payload: bytes
    marker: bool = False


@dataclass
class VoiceCall:
    """Voice call session"""
    call_id: str
    qfi: int
    codec: str  # EVS, AMR-WB, AMR-WB+
    start_time: datetime
    end_time: Optional[datetime] = None
    packets_captured: int = 0
    audio_bytes: int = 0


class VoNRInterceptor:
    """
    Enhanced VoNR interceptor with SDAP/QoS flow parsing
    Intercepts voice calls over 5G NR with QoS awareness
    """
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = ModuleLogger('VoNR-Intercept', logger)
        
        # Active calls
        self.active_calls: Dict[str, VoiceCall] = {}
        
        # QoS flow mappings
        self.qos_flows: Dict[int, QoSFlowInfo] = {}
        
        # Voice QFI detection (typically QFI 1-4 for voice)
        self.voice_qfis = {1, 2, 3, 4}
        
        # Codec configurations
        self.codec_configs = {
            'EVS': {
                'payload_types': [111, 112, 113],
                'sample_rate': 16000,
                'frame_size_ms': 20,
            },
            'AMR-WB': {
                'payload_types': [97, 98],
                'sample_rate': 16000,
                'frame_size_ms': 20,
            },
            'AMR-WB+': {
                'payload_types': [100, 101],
                'sample_rate': 24000,
                'frame_size_ms': 20,
            }
        }
        
        self.logger.info("VoNR Interceptor initialized")
    
    def register_qos_flow(self, qfi: int, five_qi: int, gfbr_kbps: Optional[int] = None) -> bool:
        """
        Register QoS flow for tracking
        
        Args:
            qfi: QoS Flow ID
            five_qi: 5G QoS Identifier
            gfbr_kbps: Guaranteed Flow Bit Rate
        
        Returns:
            True if flow is voice-related
        """
        # 5QI 1 = Conversational Voice (GBR)
        # 5QI 2 = Conversational Video (GBR)
        # 5QI 5 = IMS signaling
        is_voice = five_qi == 1 or (qfi in self.voice_qfis and gfbr_kbps and gfbr_kbps < 100)
        
        flow = QoSFlowInfo(
            qfi=qfi,
            five_qi=five_qi,
            gfbr_kbps=gfbr_kbps,
            is_voice=is_voice
        )
        
        self.qos_flows[qfi] = flow
        
        if is_voice:
            self.logger.info(f"Voice QoS flow registered: QFI={qfi}, 5QI={five_qi}, GFBR={gfbr_kbps} kbps")
        
        return is_voice
    
    def parse_sdap_header(self, packet: bytes) -> Tuple[Optional[int], bytes]:
        """
        Parse SDAP header to extract QFI
        
        SDAP header format (TS 38.323):
        - 1 byte: D/C | QFI (6 bits) | RQI | RDI
        
        Args:
            packet: SDAP PDU
        
        Returns:
            (QFI, payload) or (None, packet) if not SDAP
        """
        if len(packet) < 1:
            return None, packet
        
        try:
            header_byte = packet[0]
            
            # Check D/C bit (bit 7): 1 = Data, 0 = Control
            is_data = (header_byte & 0x80) != 0
            
            if not is_data:
                return None, packet  # Control packet, not voice data
            
            # Extract QFI (bits 6-1)
            qfi = (header_byte >> 1) & 0x3F
            
            # Payload starts at byte 1
            payload = packet[1:]
            
            return qfi, payload
            
        except Exception as e:
            self.logger.error(f"SDAP parsing failed: {e}")
            return None, packet
    
    def parse_rtp_packet(self, data: bytes) -> Optional[RTPPacket]:
        """
        Parse RTP packet (RFC 3550)
        
        RTP header:
        - 2 bytes: V(2) | P | X | CC | M | PT
        - 2 bytes: Sequence Number
        - 4 bytes: Timestamp
        - 4 bytes: SSRC
        """
        if len(data) < 12:
            return None
        
        try:
            # Parse fixed header
            byte0, byte1 = struct.unpack('BB', data[0:2])
            
            version = (byte0 >> 6) & 0x3
            padding = (byte0 >> 5) & 0x1
            extension = (byte0 >> 4) & 0x1
            csrc_count = byte0 & 0xF
            
            marker = (byte1 >> 7) & 0x1
            payload_type = byte1 & 0x7F
            
            sequence_number, timestamp, ssrc = struct.unpack('>HIL', data[2:12])
            
            # Skip CSRC identifiers
            header_len = 12 + (csrc_count * 4)
            
            # Skip extension if present
            if extension:
                if len(data) < header_len + 4:
                    return None
                ext_len = struct.unpack('>H', data[header_len+2:header_len+4])[0] * 4
                header_len += 4 + ext_len
            
            payload = data[header_len:]
            
            return RTPPacket(
                version=version,
                payload_type=payload_type,
                sequence_number=sequence_number,
                timestamp=timestamp,
                ssrc=ssrc,
                payload=payload,
                marker=marker == 1
            )
            
        except Exception as e:
            self.logger.error(f"RTP parsing failed: {e}")
            return None
    
    def intercept_voice_packet(self, pdcp_pdu: bytes, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Intercept and process voice packet from PDCP PDU
        
        Args:
            pdcp_pdu: PDCP PDU containing SDAP + IP + UDP + RTP
            context: Additional context (RNTI, timestamp, etc.)
        
        Returns:
            Intercepted call information
        """
        try:
            # Parse SDAP header
            qfi, ip_packet = self.parse_sdap_header(pdcp_pdu)
            
            if qfi is None:
                return None
            
            # Check if QFI is voice-related
            flow_info = self.qos_flows.get(qfi)
            if not flow_info or not flow_info.is_voice:
                return None
            
            # Parse IP header (simplified - skip to UDP)
            # In production: use scapy or dpkt for full IP/UDP parsing
            # Assuming UDP payload starts at offset 28 (20 IP + 8 UDP)
            if len(ip_packet) < 28:
                return None
            
            rtp_data = ip_packet[28:]
            
            # Parse RTP
            rtp_packet = self.parse_rtp_packet(rtp_data)
            if not rtp_packet:
                return None
            
            # Identify codec from RTP payload type
            codec = self._identify_codec(rtp_packet.payload_type)
            
            # Track call session
            call_id = f"{context.get('rnti', 0)}_{rtp_packet.ssrc}"
            
            if call_id not in self.active_calls:
                self.active_calls[call_id] = VoiceCall(
                    call_id=call_id,
                    qfi=qfi,
                    codec=codec,
                    start_time=datetime.now(),
                )
                self.logger.info(f"New VoNR call detected: {call_id}, codec={codec}, QFI={qfi}")
            
            # Update call statistics
            call = self.active_calls[call_id]
            call.packets_captured += 1
            call.audio_bytes += len(rtp_packet.payload)
            
            return {
                'call_id': call_id,
                'qfi': qfi,
                'codec': codec,
                'rtp_sequence': rtp_packet.sequence_number,
                'rtp_timestamp': rtp_packet.timestamp,
                'payload_size': len(rtp_packet.payload),
                'marker': rtp_packet.marker,
                'audio_payload': rtp_packet.payload,  # Raw audio data
            }
            
        except Exception as e:
            self.logger.error(f"Voice interception failed: {e}")
            return None
    
    def _identify_codec(self, payload_type: int) -> str:
        """Identify codec from RTP payload type"""
        for codec, config in self.codec_configs.items():
            if payload_type in config['payload_types']:
                return codec
        return 'Unknown'
    
    def end_call(self, call_id: str):
        """Mark call as ended"""
        if call_id in self.active_calls:
            call = self.active_calls[call_id]
            call.end_time = datetime.now()
            
            duration = (call.end_time - call.start_time).total_seconds()
            self.logger.info(f"Call ended: {call_id}, duration={duration:.1f}s, "
                           f"packets={call.packets_captured}, audio_bytes={call.audio_bytes}")
    
    def decode_evs_frame(self, evs_payload: bytes) -> Optional[np.ndarray]:
        """
        Decode EVS codec frame
        Note: Full EVS decoding requires 3GPP reference codec
        This is a placeholder for integration with external decoder
        
        Args:
            evs_payload: EVS-encoded audio frame
        
        Returns:
            Decoded PCM samples (16-bit, 16 kHz)
        """
        # In production: use 3GPP EVS decoder library
        # For now, return placeholder
        self.logger.warning("EVS decoding not implemented - requires 3GPP reference codec")
        return None
    
    def decode_amr_wb_frame(self, amr_payload: bytes) -> Optional[np.ndarray]:
        """
        Decode AMR-WB codec frame
        Requires opencore-amr or similar library
        
        Args:
            amr_payload: AMR-WB encoded audio frame
        
        Returns:
            Decoded PCM samples
        """
        try:
            # Placeholder for AMR-WB decoding
            # In production: use opencore-amr Python bindings
            self.logger.warning("AMR-WB decoding not implemented - requires opencore-amr")
            return None
            
        except Exception as e:
            self.logger.error(f"AMR-WB decoding failed: {e}")
            return None
    
    def export_call_audio(self, call_id: str, output_file: str) -> bool:
        """
        Export captured call audio to file
        Requires codec decoding
        
        Args:
            call_id: Call identifier
            output_file: Output file path (WAV format)
        
        Returns:
            True if successful
        """
        if call_id not in self.active_calls:
            self.logger.error(f"Call not found: {call_id}")
            return False
        
        call = self.active_calls[call_id]
        
        # In production: collect all RTP payloads, decode, and write to WAV
        self.logger.warning(f"Audio export not fully implemented for call {call_id}")
        self.logger.info(f"Call has {call.packets_captured} packets, {call.audio_bytes} bytes")
        
        return False
    
    def get_active_calls(self) -> List[Dict[str, Any]]:
        """Get list of active calls"""
        return [
            {
                'call_id': call.call_id,
                'qfi': call.qfi,
                'codec': call.codec,
                'start_time': call.start_time.isoformat(),
                'duration_sec': (datetime.now() - call.start_time).total_seconds(),
                'packets': call.packets_captured,
                'audio_bytes': call.audio_bytes,
            }
            for call in self.active_calls.values()
            if call.end_time is None
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get VoNR interception statistics"""
        active_count = sum(1 for c in self.active_calls.values() if c.end_time is None)
        total_calls = len(self.active_calls)
        total_packets = sum(c.packets_captured for c in self.active_calls.values())
        total_audio_mb = sum(c.audio_bytes for c in self.active_calls.values()) / (1024 * 1024)
        
        return {
            'active_calls': active_count,
            'total_calls': total_calls,
            'total_voice_packets': total_packets,
            'total_audio_mb': round(total_audio_mb, 2),
            'registered_qos_flows': len(self.qos_flows),
            'voice_qos_flows': sum(1 for f in self.qos_flows.values() if f.is_voice),
        }
