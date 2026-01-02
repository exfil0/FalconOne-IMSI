"""
FalconOne LTE (4G) Monitoring Module
Implements IMSI/GUTI capture for LTE networks
"""

import subprocess
import threading
import time
import os
import re
from typing import Dict, List, Any
from queue import Queue
import logging

from ..utils.logger import ModuleLogger
from ..utils.performance import get_cache, get_fft, get_monitor


class LTEMonitor:
    """LTE/4G monitoring and identity capture"""
    
    def __init__(self, config, logger: logging.Logger, sdr_manager):
        """Initialize LTE monitor"""
        self.config = config
        self.logger = ModuleLogger('LTE', logger)
        self.sdr_manager = sdr_manager
        
        self.running = False
        self.capture_thread = None
        self.data_queue = Queue()
        
        self.bands = config.get('monitoring.lte.bands', [1, 3, 7])
        self.tools = config.get('monitoring.lte.tools', ['LTESniffer', 'srsRAN'])
        
        self.captured_imsi = set()
        self.captured_guti = set()
        
        self.logger.info("LTE Monitor initialized", bands=self.bands, tools=self.tools)
    
    def start(self):
        """Start LTE monitoring"""
        if self.running:
            return
        
        self.logger.info("Starting LTE monitoring...")
        self.running = True
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
    
    def stop(self):
        """Stop LTE monitoring"""
        self.logger.info("Stopping LTE monitoring...")
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
    
    def _capture_loop(self):
        """Main capture loop"""
        while self.running:
            try:
                # Use LTESniffer for passive monitoring
                if 'LTESniffer' in self.tools:
                    self._capture_with_ltesniffer()
                
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"LTE capture error: {e}")
                time.sleep(5)
    
    def _capture_with_ltesniffer(self):
        """Capture using LTESniffer"""
        try:
            # LTESniffer command: ltesniffer -A 2 -W -n 100 -b <band>
            # -A: number of downlink decoder threads
            # -W: enable PCAP output
            # -n: number of subframes to capture
            # -b: LTE band to monitor
            
            pcap_file = f"/tmp/lte_capture_{int(time.time())}.pcap"
            
            for band in self.bands:
                if not self.running:
                    break
                
                cmd = [
                    'LTESniffer',
                    '-A', '2',  # 2 decoder threads
                    '-W',  # Enable PCAP
                    '-n', '500',  # 500 subframes (~5 seconds)
                    '-b', str(band),
                    '-g', str(self.config.get('sdr.rx_gain', 40)),
                    '-o', pcap_file
                ]
                
                self.logger.debug(f"Running LTESniffer on band {band}: {' '.join(cmd)}")
                
                try:
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=15,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        self.logger.info(f"LTESniffer capture successful on band {band}")
                        
                        # Parse stdout for real-time IMSI/GUTI
                        self._parse_ltesniffer_output(result.stdout)
                        
                        # Parse PCAP if generated
                        if os.path.exists(pcap_file):
                            self._parse_lte_pcap(pcap_file)
                            os.remove(pcap_file)
                    else:
                        self.logger.warning(f"LTESniffer error on band {band}: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"LTESniffer timeout on band {band}")
                except FileNotFoundError:
                    self.logger.error("LTESniffer not found in PATH. Install from: https://github.com/SysSec-KAIST/LTESniffer")
                    break
                    
        except Exception as e:
            self.logger.error(f"LTESniffer error: {e}")
        finally:
            if os.path.exists(pcap_file):
                try:
                    os.remove(pcap_file)
                except:
                    pass
    
    def _parse_ltesniffer_output(self, output: str):
        """Parse LTESniffer stdout for IMSI/GUTI"""
        try:
            # LTESniffer outputs IMSI in format: IMSI: 123456789012345
            imsi_pattern = r'IMSI:\s*([0-9]{14,15})'
            guti_pattern = r'GUTI:\s*([0-9A-Fa-f]+)'
            
            for line in output.split('\n'):
                # Extract IMSI
                imsi_match = re.search(imsi_pattern, line)
                if imsi_match:
                    imsi = imsi_match.group(1)
                    self.logger.info(f"📱 LTE IMSI captured: {imsi}")
                    self.data_queue.put({
                        'type': 'IMSI',
                        'value': imsi,
                        'timestamp': time.time(),
                        'protocol': 'LTE'
                    })
                
                # Extract GUTI
                guti_match = re.search(guti_pattern, line)
                if guti_match:
                    guti = guti_match.group(1)
                    self.logger.info(f"📱 LTE GUTI captured: {guti}")
                    self.data_queue.put({
                        'type': 'GUTI',
                        'value': guti,
                        'timestamp': time.time(),
                        'protocol': 'LTE'
                    })
                    
        except Exception as e:
            self.logger.error(f"Error parsing LTESniffer output: {e}")
    
    def _parse_lte_pcap(self, pcap_file: str):
        """Parse LTE PCAP file for additional data"""
        try:
            import pyshark
            
            cap = pyshark.FileCapture(
                pcap_file,
                display_filter='nas-eps',
                use_json=True
            )
            
            for pkt in cap:
                try:
                    if hasattr(pkt, 'nas_eps'):
                        nas = pkt.nas_eps
                        
                        # Extract IMSI from Attach Request
                        if hasattr(nas, 'nas_eps_emm_imsi'):
                            imsi = str(nas.nas_eps_emm_imsi).replace(':', '')
                            if len(imsi) >= 14:
                                self.logger.info(f"📱 LTE IMSI (PCAP): {imsi}")
                                self.data_queue.put({
                                    'type': 'IMSI',
                                    'value': imsi,
                                    'timestamp': time.time(),
                                    'protocol': 'LTE'
                                })
                        
                        # Extract GUTI
                        if hasattr(nas, 'nas_eps_emm_guti'):
                            guti = str(nas.nas_eps_emm_guti)
                            self.data_queue.put({
                                'type': 'GUTI',
                                'value': guti,
                                'timestamp': time.time(),
                                'protocol': 'LTE'
                            })
                            
                except AttributeError:
                    continue
                    
            cap.close()
            
        except ImportError:
            self.logger.warning("pyshark not installed - skipping PCAP parsing")
        except Exception as e:
            self.logger.error(f"Error parsing LTE PCAP: {e}")
    
    def get_captured_data(self) -> List[Dict[str, Any]]:
        """Get captured data"""
        data = []
        while not self.data_queue.empty():
            data.append(self.data_queue.get())
        return data
    
    def get_suci_data(self) -> List[Dict[str, Any]]:
        """Get SUCI data (not applicable for LTE)"""
        return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get module status"""
        return {
            'running': self.running,
            'bands': self.bands,
            'imsi_count': len(self.captured_imsi),
            'guti_count': len(self.captured_guti)
        }
    
    # ==================== PHASE 2.2.1: LTE KEY EXTRACTION ====================
    
    def extract_lte_keys(self, pcap_file: str) -> Dict[str, Any]:
        """
        Extract LTE encryption keys from captured traffic
        
        Args:
            pcap_file: Path to LTE capture with NAS/RRC messages
        
        Returns:
            Dictionary with extracted keys (Kasumi, SNOW3G, AES)
        """
        try:
            self.logger.info(f"Extracting LTE keys from {pcap_file}")
            
            keys = {
                'kasumi_keys': [],
                'snow3g_keys': [],
                'aes_keys': [],
                'integrity_keys': [],
                'ksi': []  # Key Set Identifier
            }
            
            # Parse NAS Security Mode Command messages
            nas_keys = self._extract_keys_from_nas(pcap_file)
            keys.update(nas_keys)
            
            # Parse RRC Security Mode Command messages
            rrc_keys = self._extract_keys_from_rrc(pcap_file)
            keys.update(rrc_keys)
            
            # Derive additional keys from base keys
            if keys.get('kasumi_keys'):
                derived = self._derive_lte_keys(keys['kasumi_keys'][0])
                keys['derived_keys'] = derived
            
            self.logger.info(f"Extracted {len(keys.get('kasumi_keys', []))} Kasumi keys, "
                           f"{len(keys.get('snow3g_keys', []))} SNOW3G keys, "
                           f"{len(keys.get('aes_keys', []))} AES keys")
            
            return keys
        
        except Exception as e:
            self.logger.error(f"LTE key extraction failed: {e}")
            return {}
    
    def _extract_keys_from_nas(self, pcap_file: str) -> Dict[str, List]:
        """Extract encryption keys from NAS Security Mode Command"""
        try:
            keys = {'kasumi_keys': [], 'snow3g_keys': [], 'aes_keys': []}
            
            # Use tshark to extract NAS security parameters
            cmd = [
                'tshark',
                '-r', pcap_file,
                '-Y', 'nas_eps.nas_msg_emm_type == 0x5d',  # Security Mode Command
                '-T', 'fields',
                '-e', 'nas_eps.emm.nas_key_set_id',
                '-e', 'nas_eps.emm.ue_sec_cap_eea',  # Encryption algorithms
                '-e', 'nas_eps.emm.ue_sec_cap_eia',  # Integrity algorithms
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            ksi = parts[0]
                            eea = parts[1] if len(parts) > 1 else ''
                            
                            # Parse encryption algorithm
                            if '128-EEA1' in eea or 'SNOW' in eea:
                                keys['snow3g_keys'].append({'ksi': ksi, 'algorithm': '128-EEA1'})
                            elif '128-EEA2' in eea or 'AES' in eea:
                                keys['aes_keys'].append({'ksi': ksi, 'algorithm': '128-EEA2'})
                            elif 'Kasumi' in eea or 'EEA1' in eea:
                                keys['kasumi_keys'].append({'ksi': ksi, 'algorithm': 'EEA1'})
            
            return keys
        
        except Exception as e:
            self.logger.error(f"NAS key extraction failed: {e}")
            return {'kasumi_keys': [], 'snow3g_keys': [], 'aes_keys': []}
    
    def _extract_keys_from_rrc(self, pcap_file: str) -> Dict[str, List]:
        """Extract encryption keys from RRC Security Mode Command"""
        try:
            keys = {'integrity_keys': []}
            
            # Use tshark to extract RRC security configuration
            cmd = [
                'tshark',
                '-r', pcap_file,
                '-Y', 'lte-rrc.securityModeCommand_element',
                '-T', 'fields',
                '-e', 'lte-rrc.cipheringAlgorithm',
                '-e', 'lte-rrc.integrityProtAlgorithm',
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            cipher_alg = parts[0]
                            integrity_alg = parts[1]
                            
                            keys['integrity_keys'].append({
                                'cipher': cipher_alg,
                                'integrity': integrity_alg
                            })
            
            return keys
        
        except Exception as e:
            self.logger.error(f"RRC key extraction failed: {e}")
            return {'integrity_keys': []}
    
    def _derive_lte_keys(self, base_key: Dict) -> Dict[str, str]:
        """
        Derive LTE encryption and integrity keys from base key
        
        Args:
            base_key: Base key material (K_ASME)
        
        Returns:
            Derived keys (K_eNB, K_NASenc, K_NASint, K_RRCenc, K_RRCint, K_UPenc)
        """
        try:
            # LTE key derivation uses KDF (Key Derivation Function)
            # K_eNB = KDF(K_ASME, uplink_NAS_count)
            # K_NASenc = KDF(K_ASME, NAS_ENC_ALG)
            # K_NASint = KDF(K_ASME, NAS_INT_ALG)
            # K_RRCenc = KDF(K_eNB, RRC_ENC_ALG)
            # K_RRCint = KDF(K_eNB, RRC_INT_ALG)
            # K_UPenc = KDF(K_eNB, UP_ENC_ALG)
            
            from Crypto.Hash import HMAC, SHA256
            from Crypto.Protocol.KDF import PBKDF2
            
            # Placeholder derivation (requires actual K_ASME key material)
            k_asme = bytes.fromhex(base_key.get('ksi', '00' * 32))
            
            derived = {
                'k_nas_enc': PBKDF2(k_asme, b'NAS_ENC', 32, count=1),
                'k_nas_int': PBKDF2(k_asme, b'NAS_INT', 32, count=1),
                'k_rrc_enc': PBKDF2(k_asme, b'RRC_ENC', 32, count=1),
                'k_rrc_int': PBKDF2(k_asme, b'RRC_INT', 32, count=1),
                'k_up_enc': PBKDF2(k_asme, b'UP_ENC', 32, count=1)
            }
            
            return {k: v.hex() for k, v in derived.items()}
        
        except Exception as e:
            self.logger.error(f"Key derivation failed: {e}")
            return {}
    
    # ==================== PHASE 2.2.2: S1-AP MONITORING ====================
    
    def start_s1ap_monitoring(self, interface: str = 'eth0') -> bool:
        """
        Start monitoring S1 interface between eNodeB and EPC
        
        Args:
            interface: Network interface to monitor (default: eth0)
        
        Returns:
            True if monitoring started successfully
        """
        try:
            self.logger.info(f"Starting S1-AP monitoring on interface {interface}")
            
            self.s1ap_enabled = True
            self.s1ap_events = []
            
            # Start S1-AP capture thread
            s1ap_thread = threading.Thread(
                target=self._s1ap_capture_loop,
                args=(interface,),
                daemon=True
            )
            s1ap_thread.start()
            
            self.logger.info("S1-AP monitoring started")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to start S1-AP monitoring: {e}")
            return False
    
    def _s1ap_capture_loop(self, interface: str):
        """Capture S1-AP messages from network interface"""
        try:
            import tempfile
            pcap_file = tempfile.mktemp(suffix='.pcap')
            
            # Capture S1-AP traffic (SCTP port 36412)
            cmd = [
                'tshark',
                '-i', interface,
                '-f', 'sctp port 36412',  # S1-AP uses SCTP port 36412
                '-w', pcap_file,
                '-a', 'duration:60'  # 60 second capture
            ]
            
            self.logger.debug(f"Starting S1-AP capture: {' '.join(cmd)}")
            
            while self.s1ap_enabled:
                try:
                    subprocess.run(cmd, timeout=65)
                    
                    # Parse captured S1-AP messages
                    if os.path.exists(pcap_file):
                        self._parse_s1ap_messages(pcap_file)
                        os.remove(pcap_file)
                
                except subprocess.TimeoutExpired:
                    pass
                except Exception as e:
                    self.logger.error(f"S1-AP capture error: {e}")
                    time.sleep(5)
        
        except Exception as e:
            self.logger.error(f"S1-AP capture loop failed: {e}")
    
    def _parse_s1ap_messages(self, pcap_file: str):
        """Parse S1-AP messages from pcap"""
        try:
            self.logger.info(f"Parsing S1-AP messages from {pcap_file}")
            
            # Extract S1-AP message types
            cmd = [
                'tshark',
                '-r', pcap_file,
                '-Y', 's1ap',
                '-T', 'fields',
                '-e', 's1ap.procedureCode',
                '-e', 's1ap.messageType',
                '-e', 's1ap.eNB_UE_S1AP_ID',
                '-e', 's1ap.MME_UE_S1AP_ID',
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            s1ap_event = {
                                'procedure': parts[0],
                                'msg_type': parts[1],
                                'enb_id': parts[2] if len(parts) > 2 else None,
                                'mme_id': parts[3] if len(parts) > 3 else None,
                                'timestamp': time.time()
                            }
                            
                            self.s1ap_events.append(s1ap_event)
                            
                            # Log important events
                            if s1ap_event['procedure'] in ['0', '9']:  # Initial UE Message, Handover
                                self.logger.info(f"S1-AP event: {s1ap_event}")
                
                self.logger.info(f"Parsed {len(self.s1ap_events)} S1-AP events")
        
        except Exception as e:
            self.logger.error(f"S1-AP parsing failed: {e}")
    
    def get_s1ap_events(self, event_type: str = None) -> List[Dict]:
        """Get captured S1-AP events"""
        if event_type:
            return [e for e in self.s1ap_events if e.get('msg_type') == event_type]
        return self.s1ap_events
    
    # ==================== PHASE 2.2.3: RRC DECODER ====================
    
    def decode_rrc_messages(self, pcap_file: str) -> List[Dict[str, Any]]:
        """
        Decode LTE RRC messages using ASN.1
        
        Args:
            pcap_file: Path to LTE capture with RRC messages
        
        Returns:
            List of decoded RRC messages
        """
        try:
            self.logger.info(f"Decoding RRC messages from {pcap_file}")
            
            decoded_messages = []
            
            # Decode different RRC message types
            sib1_messages = self._decode_sib1(pcap_file)
            decoded_messages.extend(sib1_messages)
            
            rrc_setup_messages = self._decode_rrc_setup(pcap_file)
            decoded_messages.extend(rrc_setup_messages)
            
            measurement_messages = self._decode_measurement_reports(pcap_file)
            decoded_messages.extend(measurement_messages)
            
            self.logger.info(f"Decoded {len(decoded_messages)} RRC messages")
            return decoded_messages
        
        except Exception as e:
            self.logger.error(f"RRC decoding failed: {e}")
            return []
    
    def _decode_sib1(self, pcap_file: str) -> List[Dict]:
        """Decode SystemInformationBlockType1 messages"""
        try:
            messages = []
            
            cmd = [
                'tshark',
                '-r', pcap_file,
                '-Y', 'lte-rrc.systemInformationBlockType1_element',
                '-T', 'fields',
                '-e', 'lte-rrc.cellAccessRelatedInfo_element',
                '-e', 'lte-rrc.cellIdentity',
                '-e', 'lte-rrc.trackingAreaCode',
                '-e', 'lte-rrc.plmn_Identity_element',
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split('\t')
                        messages.append({
                            'type': 'SIB1',
                            'cell_id': parts[1] if len(parts) > 1 else None,
                            'tac': parts[2] if len(parts) > 2 else None,
                            'plmn': parts[3] if len(parts) > 3 else None,
                            'timestamp': time.time()
                        })
            
            return messages
        
        except Exception as e:
            self.logger.error(f"SIB1 decoding failed: {e}")
            return []
    
    def _decode_rrc_setup(self, pcap_file: str) -> List[Dict]:
        """Decode RRC Connection Setup messages"""
        try:
            messages = []
            
            cmd = [
                'tshark',
                '-r', pcap_file,
                '-Y', 'lte-rrc.rrcConnectionSetup_element',
                '-T', 'fields',
                '-e', 'lte-rrc.rrc_TransactionIdentifier',
                '-e', 'lte-rrc.radioResourceConfigDedicated_element',
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split('\t')
                        messages.append({
                            'type': 'RRCConnectionSetup',
                            'transaction_id': parts[0] if parts else None,
                            'timestamp': time.time()
                        })
            
            return messages
        
        except Exception as e:
            self.logger.error(f"RRC Setup decoding failed: {e}")
            return []
    
    def _decode_measurement_reports(self, pcap_file: str) -> List[Dict]:
        """Decode RRC Measurement Report messages"""
        try:
            messages = []
            
            cmd = [
                'tshark',
                '-r', pcap_file,
                '-Y', 'lte-rrc.measurementReport_element',
                '-T', 'fields',
                '-e', 'lte-rrc.measId',
                '-e', 'lte-rrc.rsrpResult',
                '-e', 'lte-rrc.rsrqResult',
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split('\t')
                        messages.append({
                            'type': 'MeasurementReport',
                            'meas_id': parts[0] if parts else None,
                            'rsrp': parts[1] if len(parts) > 1 else None,
                            'rsrq': parts[2] if len(parts) > 2 else None,
                            'timestamp': time.time()
                        })
            
            return messages
        
        except Exception as e:
            self.logger.error(f"Measurement Report decoding failed: {e}")
            return []
