"""
FalconOne GSM (2G) Monitoring Module
Implements IMSI/TMSI capture and SMS interception for GSM networks
"""

import subprocess
import threading
import time
import re
import os
from typing import Dict, List, Optional, Any
from queue import Queue
import logging

from ..utils.logger import ModuleLogger


class GSMMonitor:
    """GSM/2G monitoring and IMSI catching"""
    
    def __init__(self, config, logger: logging.Logger, sdr_manager):
        """
        Initialize GSM monitor
        
        Args:
            config: Configuration object
            logger: Logger instance
            sdr_manager: SDR manager instance
        """
        self.config = config
        self.logger = ModuleLogger('GSM', logger)
        self.sdr_manager = sdr_manager
        
        # Monitoring state
        self.running = False
        self.capture_thread = None
        self.data_queue = Queue()
        
        # GSM configuration
        self.bands = config.get('monitoring.gsm.bands', ['GSM900', 'GSM1800'])
        self.tools = config.get('monitoring.gsm.tools', ['gr-gsm', 'kalibrate-rtl'])
        
        # Captured data
        self.captured_imsi = set()
        self.captured_tmsi = set()
        self.captured_sms = []
        
        # ARFCN (Absolute Radio Frequency Channel Number) list
        self.arfcns = []
        
        self.logger.info("GSM Monitor initialized", bands=self.bands, tools=self.tools)
    
    def start(self):
        """Start GSM monitoring"""
        if self.running:
            self.logger.warning("GSM monitor already running")
            return
        
        self.logger.info("Starting GSM monitoring...")
        self.running = True
        
        # First, scan for ARFCNs
        if self.config.get('monitoring.gsm.arfcn_scan', True):
            self._scan_arfcns()
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.logger.info("GSM monitoring started")
    
    def stop(self):
        """Stop GSM monitoring"""
        self.logger.info("Stopping GSM monitoring...")
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
        
        self.logger.info("GSM monitoring stopped")
    
    def _scan_arfcns(self):
        """Scan for active GSM ARFCNs using kalibrate-rtl"""
        self.logger.info("Scanning for GSM ARFCNs...")
        
        for band in self.bands:
            try:
                # Run kalibrate-rtl to find ARFCNs
                cmd = ['kal', '-s', band, '-g', '40']
                self.logger.debug(f"Running command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse output for ARFCN numbers
                arfcns = self._parse_kal_output(result.stdout)
                self.arfcns.extend(arfcns)
                
                self.logger.info(f"Found {len(arfcns)} ARFCNs in {band}", arfcns=arfcns)
                
            except subprocess.TimeoutExpired:
                self.logger.warning(f"ARFCN scan timeout for {band}")
            except FileNotFoundError:
                self.logger.error("kalibrate-rtl (kal) not found - install it first")
                break
            except Exception as e:
                self.logger.error(f"ARFCN scan error for {band}: {e}")
        
        if not self.arfcns:
            self.logger.warning("No ARFCNs found, using default channel")
            self.arfcns = [0]  # Default ARFCN
    
    def _parse_kal_output(self, output: str) -> List[int]:
        """
        Parse kalibrate-rtl output to extract ARFCNs
        
        Args:
            output: kalibrate-rtl stdout
            
        Returns:
            List of ARFCN numbers
        """
        arfcns = []
        
        # Look for lines like: "chan: 3 (937.6MHz - 1.640kHz)   power: 54321.45"
        pattern = r'chan:\s+(\d+)\s+\(.*?\)\s+power:\s+[\d.]+'
        
        for match in re.finditer(pattern, output):
            arfcn = int(match.group(1))
            arfcns.append(arfcn)
        
        return arfcns
    
    def _capture_loop(self):
        """Main capture loop"""
        self.logger.info("GSM capture loop started")
        
        while self.running:
            try:
                # Rotate through ARFCNs
                for arfcn in self.arfcns:
                    if not self.running:
                        break
                    
                    self.logger.debug(f"Monitoring ARFCN {arfcn}")
                    self._capture_arfcn(arfcn)
                    
                    # Brief pause between ARFCNs
                    time.sleep(2)
                
                # Pause before next scan cycle
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(5)
    
    def _capture_arfcn(self, arfcn: int):
        """
        Capture data from specific ARFCN
        
        Args:
            arfcn: ARFCN number to monitor
        """
        try:
            # Use grgsm_livemon or grgsm_scanner
            if 'gr-gsm' in self.tools:
                self._capture_with_grgsm(arfcn)
            elif 'OsmocomBB' in self.tools:
                self._capture_with_osmocombb(arfcn)
            
        except Exception as e:
            self.logger.error(f"Error capturing ARFCN {arfcn}: {e}")
    
    def _capture_with_grgsm(self, arfcn: int):
        """
        Capture GSM traffic using gr-gsm
        
        Args:
            arfcn: ARFCN to capture
        """
        freq = self._arfcn_to_freq(arfcn)
        pcap_file = f"/tmp/gsm_capture_{arfcn}_{int(time.time())}.pcap"
        
        try:
            # Launch grgsm_livemon with GSMTAP output
            cmd = [
                'grgsm_livemon',
                '-f', str(freq),  # Frequency in Hz
                '-a', str(arfcn),
                '-s', '2000000',  # Sample rate 2 MHz
                '-g', str(self.config.get('sdr.rx_gain', 40)),
                '-p', str(self.config.get('gsm.ppm', 0)),
                '--args', f"driver={self.sdr.get_device_type()}",
                '-o', pcap_file,
                '-T', '3',  # Capture for 3 seconds
            ]
            
            self.logger.debug(f"Launching gr-gsm: {' '.join(cmd)}")
            
            # Run with timeout
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info(f"gr-gsm capture successful for ARFCN {arfcn}")
                
                # Parse the captured PCAP
                if os.path.exists(pcap_file):
                    self._parse_gsm_pcap(pcap_file)
                    os.remove(pcap_file)  # Cleanup
            else:
                self.logger.warning(f"gr-gsm returned code {result.returncode}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"gr-gsm capture timeout for ARFCN {arfcn}")
        except FileNotFoundError:
            self.logger.error("grgsm_livemon not found in PATH")
        except Exception as e:
            self.logger.error(f"Error in gr-gsm capture: {e}")
        finally:
            # Cleanup
            if os.path.exists(pcap_file):
                try:
                    os.remove(pcap_file)
                except:
                    pass
    
    def _capture_with_osmocombb(self, arfcn: int):
        """
        Capture using OsmocomBB (v1.9.1 - Full Implementation)
        
        OsmocomBB is a free software implementation of the GSM baseband
        that runs on Calypso-based phones (Motorola C1xx series).
        
        Requirements:
            - OsmocomBB compiled and installed
            - Calypso phone connected via serial
            - osmocon running with firmware loaded
        
        Args:
            arfcn: ARFCN to monitor
        """
        from ..utils.circuit_breaker import subprocess_context, CircuitBreakerOpenError
        
        # OsmocomBB configuration
        osmocon_socket = self.config.get('monitoring.gsm.osmocon_socket', '/tmp/osmocom_l2')
        ccch_scan_path = self.config.get('monitoring.gsm.ccch_scan_path', 'ccch_scan')
        cell_log_path = self.config.get('monitoring.gsm.cell_log_path', 'cell_log')
        
        freq = self._arfcn_to_freq(arfcn)
        
        try:
            # Check if osmocon socket exists (firmware must be loaded)
            if not os.path.exists(osmocon_socket):
                self.logger.warning(f"OsmocomBB socket not found: {osmocon_socket}")
                self.logger.info("Ensure osmocon is running: osmocon -p /dev/ttyUSB0 layer1.compalram.bin")
                return
            
            # Use cell_log for comprehensive cell information capture
            cmd = [
                cell_log_path,
                '-O', osmocon_socket,  # OsmocomBB socket
                '-a', str(arfcn),      # Target ARFCN
                '-t', '5',             # Capture duration in seconds
                '-f', 'gsmtap',        # Output format
            ]
            
            self.logger.info(f"OsmocomBB capture on ARFCN {arfcn} ({freq/1e6:.2f} MHz)")
            
            # Use circuit breaker protected subprocess
            try:
                with subprocess_context(cmd, timeout=10.0, circuit_name='osmocombb') as proc:
                    stdout, stderr = proc.communicate(timeout=10.0)
                    
                    if proc.returncode == 0:
                        # Parse cell_log output for IMSI/TMSI
                        self._parse_osmocombb_output(stdout, arfcn)
                    else:
                        self.logger.warning(f"OsmocomBB returned code {proc.returncode}: {stderr}")
                        
            except CircuitBreakerOpenError:
                self.logger.warning("OsmocomBB circuit breaker is open, skipping capture")
                return
                
        except FileNotFoundError:
            self.logger.error(f"OsmocomBB tools not found. Install from https://osmocom.org/projects/baseband")
        except Exception as e:
            self.logger.error(f"OsmocomBB capture error: {e}")
    
    def _parse_osmocombb_output(self, output: str, arfcn: int):
        """
        Parse OsmocomBB cell_log output for identifiers
        
        Args:
            output: cell_log stdout
            arfcn: Current ARFCN for context
        """
        # Parse System Information messages
        si_pattern = r'SYSTEM INFORMATION TYPE (\d+)'
        for match in re.finditer(si_pattern, output):
            si_type = match.group(1)
            self.logger.debug(f"Captured SI Type {si_type} on ARFCN {arfcn}")
        
        # Parse IMSI from Paging Request or Identity Response
        imsi_pattern = r'IMSI[:\s]+([0-9]{14,15})'
        for match in re.finditer(imsi_pattern, output):
            imsi = match.group(1)
            self.logger.info(f"📱 IMSI captured (OsmocomBB): {imsi}")
            self.captured_imsi.add(imsi)
            self.data_queue.put({
                'type': 'IMSI',
                'value': imsi,
                'timestamp': time.time(),
                'arfcn': arfcn,
                'protocol': 'GSM',
                'source': 'OsmocomBB'
            })
        
        # Parse TMSI from Paging Request
        tmsi_pattern = r'TMSI[:\s]+([0-9A-Fa-f]{8})'
        for match in re.finditer(tmsi_pattern, output):
            tmsi = match.group(1).upper()
            self.logger.info(f"📱 TMSI captured (OsmocomBB): {tmsi}")
            self.captured_tmsi.add(tmsi)
            self.data_queue.put({
                'type': 'TMSI',
                'value': tmsi,
                'timestamp': time.time(),
                'arfcn': arfcn,
                'protocol': 'GSM',
                'source': 'OsmocomBB'
            })
        
        # Parse Cell Identity (CI), LAC, MCC, MNC from System Information
        cell_id_pattern = r'Cell ID[:\s]+(\d+)'
        lac_pattern = r'LAC[:\s]+(\d+)'
        mcc_pattern = r'MCC[:\s]+(\d{3})'
        mnc_pattern = r'MNC[:\s]+(\d{2,3})'
        
        for pattern, field in [(cell_id_pattern, 'cell_id'), (lac_pattern, 'lac'),
                               (mcc_pattern, 'mcc'), (mnc_pattern, 'mnc')]:
            match = re.search(pattern, output)
            if match:
                self.data_queue.put({
                    'type': field.upper(),
                    'value': match.group(1),
                    'timestamp': time.time(),
                    'arfcn': arfcn,
                    'protocol': 'GSM',
                    'source': 'OsmocomBB'
                })
    
    def _arfcn_to_freq(self, arfcn: int) -> float:
        """
        Convert ARFCN to frequency in Hz
        
        Args:
            arfcn: ARFCN number
            
        Returns:
            Frequency in Hz
        """
        # GSM900 uplink: 890-915 MHz (ARFCN 0-124)
        # GSM1800 uplink: 1710-1785 MHz (ARFCN 512-885)
        
        if 0 <= arfcn <= 124:
            # GSM900
            return (890.0 + 0.2 * arfcn) * 1e6
        elif 512 <= arfcn <= 885:
            # GSM1800 (DCS1800)
            return (1710.2 + 0.2 * (arfcn - 512)) * 1e6
        else:
            # Default to GSM900 base
            return 935.0e6
    
    def _parse_gsm_pcap(self, pcap_file: str):
        """
        Parse GSM PCAP file to extract IMSI/TMSI using pyshark
        
        Args:
            pcap_file: Path to PCAP file
        """
        try:
            import pyshark
            
            cap = pyshark.FileCapture(
                pcap_file,
                display_filter='gsmtap',
                use_json=True,
                include_raw=True
            )
            
            packets_parsed = 0
            
            for pkt in cap:
                packets_parsed += 1
                
                try:
                    # Check for GSMTAP layer
                    if hasattr(pkt, 'gsmtap'):
                        # Look for GSM A-I/F DTAP messages
                        if hasattr(pkt, 'gsm_a'):
                            gsm_layer = pkt.gsm_a
                            
                            # Extract IMSI from Identity Response or Location Update
                            if hasattr(gsm_layer, 'imsi'):
                                imsi = str(gsm_layer.imsi).replace(':', '')
                                if imsi and len(imsi) >= 14:
                                    self.logger.info(f"📱 IMSI captured: {imsi}")
                                    self.captured_data.put({
                                        'type': 'IMSI',
                                        'value': imsi,
                                        'timestamp': time.time(),
                                        'arfcn': getattr(pkt.gsmtap, 'arfcn', 'unknown'),
                                        'protocol': 'GSM'
                                    })
                            
                            # Extract TMSI
                            if hasattr(gsm_layer, 'tmsi'):
                                tmsi = str(gsm_layer.tmsi)
                                self.logger.info(f"📱 TMSI captured: {tmsi}")
                                self.captured_data.put({
                                    'type': 'TMSI',
                                    'value': tmsi,
                                    'timestamp': time.time(),
                                    'arfcn': getattr(pkt.gsmtap, 'arfcn', 'unknown'),
                                    'protocol': 'GSM'
                                })
                            
                            # Extract LAI (Location Area Identity)
                            if hasattr(gsm_layer, 'lai'):
                                lai = str(gsm_layer.lai)
                                self.captured_data.put({
                                    'type': 'LAI',
                                    'value': lai,
                                    'timestamp': time.time(),
                                    'protocol': 'GSM'
                                })
                
                except AttributeError:
                    continue  # Skip packets without required layers
                    
            cap.close()
            self.logger.debug(f"Parsed {packets_parsed} GSMTAP packets from {pcap_file}")
            
        except ImportError:
            self.logger.error("pyshark not installed. Install with: pip install pyshark")
            # Fallback to tshark command-line
            self._parse_with_tshark(pcap_file)
        except Exception as e:
            self.logger.error(f"Error parsing PCAP: {e}")
    
    def _parse_with_tshark(self, pcap_file: str):
        """
        Fallback: Parse PCAP using tshark command-line
        
        Args:
            pcap_file: Path to PCAP file
        """
        try:
            # Extract IMSI using tshark
            cmd = [
                'tshark', '-r', pcap_file,
                '-Y', 'gsm_a.imsi',
                '-T', 'fields',
                '-e', 'gsm_a.imsi',
                '-e', 'gsmtap.arfcn'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 1:
                            imsi = parts[0].replace(':', '')
                            arfcn = parts[1] if len(parts) > 1 else 'unknown'
                            
                            if len(imsi) >= 14:
                                self.logger.info(f"📱 IMSI captured (tshark): {imsi}")
                                self.captured_data.put({
                                    'type': 'IMSI',
                                    'value': imsi,
                                    'timestamp': time.time(),
                                    'arfcn': arfcn,
                                    'protocol': 'GSM'
                                })
            
            # Extract TMSI
            cmd_tmsi = [
                'tshark', '-r', pcap_file,
                '-Y', 'gsm_a.tmsi',
                '-T', 'fields',
                '-e', 'gsm_a.tmsi'
            ]
            
            result = subprocess.run(cmd_tmsi, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        tmsi = line.strip()
                        self.logger.info(f"📱 TMSI captured (tshark): {tmsi}")
                        self.captured_data.put({
                            'type': 'TMSI',
                            'value': tmsi,
                            'timestamp': time.time(),
                            'protocol': 'GSM'
                        })
                        
        except FileNotFoundError:
            self.logger.error("tshark not found in PATH")
        except Exception as e:
            self.logger.error(f"Error in tshark parsing: {e}")
    
    def _process_imsi(self, imsi: str):
        """
        Process captured IMSI
        
        Args:
            imsi: IMSI string
        """
        if imsi and imsi not in self.captured_imsi:
            self.captured_imsi.add(imsi)
            self.logger.info(f"New IMSI captured: {imsi}")
            
            # Add to data queue
            self.data_queue.put({
                'type': 'imsi',
                'value': imsi,
                'generation': '2G',
                'timestamp': time.time()
            })
    
    def _process_tmsi(self, tmsi: str):
        """
        Process captured TMSI
        
        Args:
            tmsi: TMSI string
        """
        if tmsi and tmsi not in self.captured_tmsi:
            self.captured_tmsi.add(tmsi)
            self.logger.info(f"New TMSI captured: {tmsi}")
            
            # Add to data queue
            self.data_queue.put({
                'type': 'tmsi',
                'value': tmsi,
                'generation': '2G',
                'timestamp': time.time()
            })
    
    def get_captured_data(self) -> List[Dict[str, Any]]:
        """
        Get captured data from queue
        
        Returns:
            List of captured data items
        """
        data = []
        while not self.data_queue.empty():
            data.append(self.data_queue.get())
        return data
    
    def get_suci_data(self) -> List[Dict[str, Any]]:
        """Get SUCI data (not applicable for GSM)"""
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            'imsi_count': len(self.captured_imsi),
            'tmsi_count': len(self.captured_tmsi),
            'sms_count': len(self.captured_sms),
            'arfcn_count': len(self.arfcns),
            'running': self.running
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get module status"""
        return {
            'running': self.running,
            'bands': self.bands,
            'arfcns': self.arfcns,
            'statistics': self.get_statistics()
        }    
    # ==================== PHASE 2.1.1: A5/1 DECRYPTION ====================
    
    def decrypt_a51(self, encrypted_data: bytes, key_material: bytes = None) -> Optional[bytes]:
        """
        Decrypt GSM A5/1 encrypted data using Kraken or similar
        
        Args:
            encrypted_data: Encrypted GSM burst data
            key_material: Optional key material (Kc) if already known
        
        Returns:
            Decrypted plaintext data or None if decryption fails
        """
        try:
            if key_material:
                # If we have the key, decrypt directly
                return self._decrypt_with_key(encrypted_data, key_material)
            else:
                # Try to crack the key using Kraken rainbow tables
                key = self._crack_a51_key(encrypted_data)
                if key:
                    return self._decrypt_with_key(encrypted_data, key)
                else:
                    self.logger.warning("Failed to crack A5/1 key")
                    return None
        
        except Exception as e:
            self.logger.error(f"A5/1 decryption failed: {e}")
            return None
    
    def _crack_a51_key(self, encrypted_bursts: bytes) -> Optional[bytes]:
        """
        Attempt to crack A5/1 key using Kraken rainbow tables
        
        Args:
            encrypted_bursts: Multiple encrypted bursts for correlation
        
        Returns:
            64-bit A5/1 key (Kc) or None if cracking fails
        """
        try:
            self.logger.info("Attempting A5/1 key cracking with Kraken...")
            
            # Check if Kraken is available
            kraken_path = self.config.get('monitoring.gsm.kraken_path', '/usr/bin/kraken')
            if not os.path.exists(kraken_path):
                self.logger.warning(f"Kraken not found at {kraken_path}. Install from https://github.com/rtlsdr-scanner/pyrtlsdr")
                return None
            
            # Save bursts to temporary file for Kraken
            import tempfile
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
                burst_file = f.name
                f.write(encrypted_bursts)
            
            try:
                # Run Kraken to crack the key
                # Format: kraken <burst_file> <rainbow_table_dir>
                rainbow_dir = self.config.get('monitoring.gsm.rainbow_tables', '/opt/kraken/tables')
                
                cmd = [kraken_path, burst_file, rainbow_dir]
                self.logger.debug(f"Running Kraken: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes max
                )
                
                # Parse key from output
                # Kraken output format: "Key found: 0x1234567890ABCDEF"
                key_match = re.search(r'Key found:\s*0x([0-9A-Fa-f]{16})', result.stdout)
                if key_match:
                    key_hex = key_match.group(1)
                    key = bytes.fromhex(key_hex)
                    self.logger.info(f"A5/1 key cracked: {key_hex}")
                    return key
                else:
                    self.logger.warning("Kraken did not find key")
                    return None
            
            finally:
                # Clean up temporary file
                if os.path.exists(burst_file):
                    os.remove(burst_file)
        
        except subprocess.TimeoutExpired:
            self.logger.error("Kraken key cracking timed out (5 minutes)")
            return None
        except Exception as e:
            self.logger.error(f"A5/1 key cracking failed: {e}")
            return None
    
    def _decrypt_with_key(self, encrypted_data: bytes, key: bytes) -> bytes:
        """
        Decrypt GSM data with known A5/1 key
        
        Args:
            encrypted_data: Encrypted burst data
            key: 64-bit A5/1 key (Kc)
        
        Returns:
            Decrypted plaintext data
        """
        try:
            # Implement A5/1 stream cipher decryption
            # A5/1 uses 3 LFSRs (Linear Feedback Shift Registers)
            # For production use, integrate with osmocom or similar
            
            self.logger.debug(f"Decrypting {len(encrypted_data)} bytes with A5/1 key")
            
            # Initialize A5/1 state with key
            lfsr1, lfsr2, lfsr3 = self._init_a51_state(key)
            
            # Generate keystream and XOR with encrypted data
            keystream = self._generate_a51_keystream(lfsr1, lfsr2, lfsr3, len(encrypted_data))
            
            # XOR decryption
            decrypted = bytes([e ^ k for e, k in zip(encrypted_data, keystream)])
            
            self.logger.debug("A5/1 decryption complete")
            return decrypted
        
        except Exception as e:
            self.logger.error(f"A5/1 decryption with key failed: {e}")
            return encrypted_data  # Return encrypted data on failure
    
    def _init_a51_state(self, key: bytes) -> tuple:
        """
        Initialize A5/1 LFSR state from key
        
        Args:
            key: 64-bit encryption key
        
        Returns:
            Tuple of (lfsr1, lfsr2, lfsr3) initial states
        """
        # Convert key to integer
        key_int = int.from_bytes(key, byteorder='big')
        
        # Initialize 3 LFSRs with key bits
        # LFSR1: 19 bits, LFSR2: 22 bits, LFSR3: 23 bits
        lfsr1 = (key_int >> 45) & 0x7FFFF  # 19 bits
        lfsr2 = (key_int >> 23) & 0x3FFFFF  # 22 bits
        lfsr3 = key_int & 0x7FFFFF  # 23 bits
        
        return lfsr1, lfsr2, lfsr3
    
    def _generate_a51_keystream(self, lfsr1: int, lfsr2: int, lfsr3: int, length: int) -> bytes:
        """
        Generate A5/1 keystream
        
        Args:
            lfsr1, lfsr2, lfsr3: LFSR states
            length: Number of keystream bytes to generate
        
        Returns:
            Keystream bytes
        """
        keystream = []
        
        for _ in range(length):
            # Get clocking bit from each LFSR (majority voting)
            clk1 = (lfsr1 >> 8) & 1
            clk2 = (lfsr2 >> 10) & 1
            clk3 = (lfsr3 >> 10) & 1
            
            majority = (clk1 + clk2 + clk3) >= 2
            
            # Clock LFSRs based on majority
            if clk1 == majority:
                feedback1 = ((lfsr1 >> 18) ^ (lfsr1 >> 17) ^ (lfsr1 >> 16) ^ (lfsr1 >> 13)) & 1
                lfsr1 = ((lfsr1 << 1) | feedback1) & 0x7FFFF
            
            if clk2 == majority:
                feedback2 = ((lfsr2 >> 21) ^ (lfsr2 >> 20)) & 1
                lfsr2 = ((lfsr2 << 1) | feedback2) & 0x3FFFFF
            
            if clk3 == majority:
                feedback3 = ((lfsr3 >> 22) ^ (lfsr3 >> 21) ^ (lfsr3 >> 20) ^ (lfsr3 >> 7)) & 1
                lfsr3 = ((lfsr3 << 1) | feedback3) & 0x7FFFFF
            
            # Output bit is XOR of MSBs
            output_bit = ((lfsr1 >> 18) ^ (lfsr2 >> 21) ^ (lfsr3 >> 22)) & 1
            keystream.append(output_bit)
        
        # Convert bits to bytes
        keystream_bytes = bytes([
            sum([keystream[i * 8 + j] << (7 - j) for j in range(8)])
            for i in range(len(keystream) // 8)
        ])
        
        return keystream_bytes
    
    def extract_encrypted_bursts(self, pcap_file: str) -> List[bytes]:
        """
        Extract encrypted bursts from GSM capture for decryption
        
        Args:
            pcap_file: Path to pcap file with GSM traffic
        
        Returns:
            List of encrypted burst data
        """
        try:
            self.logger.info(f"Extracting encrypted bursts from {pcap_file}")
            
            bursts = []
            
            # Use tshark to extract GSM bursts
            cmd = [
                'tshark',
                '-r', pcap_file,
                '-Y', 'gsm_a.dtap',  # Filter for GSM DTAP messages
                '-T', 'fields',
                '-e', 'data.data',  # Extract data field
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Parse hex data
            for line in result.stdout.strip().split('\n'):
                if line:
                    burst_data = bytes.fromhex(line.replace(':', ''))
                    bursts.append(burst_data)
            
            self.logger.info(f"Extracted {len(bursts)} encrypted bursts")
            return bursts
        
        except Exception as e:
            self.logger.error(f"Failed to extract encrypted bursts: {e}")
            return []
    
    # ==================== PHASE 2.1.2: GSM GPRS CAPTURE ====================
    
    def start_gprs_capture(self, arfcn: int = None) -> bool:
        """
        Start capturing GPRS (2.5G) data traffic
        
        Args:
            arfcn: Specific ARFCN to monitor, or None to scan all
        
        Returns:
            True if capture started successfully
        """
        try:
            self.logger.info(f"Starting GPRS capture on ARFCN {arfcn or 'all'}")
            
            # Check if gr-gsm is available
            try:
                subprocess.run(['grgsm_livemon', '--help'], capture_output=True, timeout=5)
            except FileNotFoundError:
                self.logger.error("gr-gsm not found. Install with: apt-get install gr-gsm")
                return False
            
            # Start GPRS capture thread
            gprs_thread = threading.Thread(
                target=self._gprs_capture_loop,
                args=(arfcn,),
                daemon=True
            )
            gprs_thread.start()
            
            self.logger.info("GPRS capture started")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to start GPRS capture: {e}")
            return False
    
    def _gprs_capture_loop(self, arfcn: Optional[int]):
        """
        GPRS capture loop using gr-gsm
        
        Args:
            arfcn: ARFCN to monitor
        """
        try:
            # Use gr-gsm to capture GPRS traffic
            import tempfile
            pcap_file = tempfile.mktemp(suffix='.pcap')
            
            if arfcn:
                freq = self._arfcn_to_freq(arfcn)
                cmd = [
                    'grgsm_livemon',
                    '-f', str(freq * 1e6),
                    '--args', f'rtl={self.sdr_manager.get_active_device_id() if self.sdr_manager else 0}',
                    '-g', '40',
                    '-s', '2e6',  # 2 MHz sample rate
                    '--output-file', pcap_file
                ]
            else:
                # Scan mode
                if not self.arfcns:
                    self._scan_arfcns()
                
                if not self.arfcns:
                    self.logger.warning("No ARFCNs found for GPRS capture")
                    return
                
                freq = self._arfcn_to_freq(self.arfcns[0])
                cmd = [
                    'grgsm_livemon',
                    '-f', str(freq * 1e6),
                    '-g', '40',
                    '--output-file', pcap_file
                ]
            
            self.logger.debug(f"Starting gr-gsm: {' '.join(cmd)}")
            
            # Run gr-gsm capture
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.logger.info(f"GPRS capture running, output: {pcap_file}")
            time.sleep(30)  # Accumulate data
            
            # Parse captured GPRS data
            self._parse_gprs_capture(pcap_file)
            
            # Cleanup
            process.terminate()
            process.wait(timeout=5)
            
            if os.path.exists(pcap_file):
                os.remove(pcap_file)
        
        except Exception as e:
            self.logger.error(f"GPRS capture loop failed: {e}")
    
    def _parse_gprs_capture(self, pcap_file: str):
        """Parse GPRS capture file"""
        try:
            if not os.path.exists(pcap_file):
                return
            
            self.logger.info(f"Parsing GPRS capture: {pcap_file}")
            
            # Use tshark to extract GPRS packets
            cmd = [
                'tshark',
                '-r', pcap_file,
                '-Y', 'gprs',
                '-T', 'json'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return
            
            # Parse JSON output
            import json
            try:
                packets = json.loads(result.stdout)
                
                for packet in packets:
                    gprs_data = self._extract_gprs_data(packet)
                    if gprs_data:
                        self.data_queue.put({
                            'type': 'gprs',
                            'data': gprs_data,
                            'generation': '2.5G',
                            'timestamp': time.time()
                        })
                
                self.logger.info(f"Parsed {len(packets)} GPRS packets")
            
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse tshark JSON output")
        
        except Exception as e:
            self.logger.error(f"Failed to parse GPRS capture: {e}")
    
    def _extract_gprs_data(self, packet: dict) -> Optional[dict]:
        """Extract useful data from GPRS packet"""
        try:
            layers = packet.get('_source', {}).get('layers', {})
            
            ip_layer = layers.get('ip', {})
            if ip_layer:
                return {
                    'src_ip': ip_layer.get('ip.src'),
                    'dst_ip': ip_layer.get('ip.dst'),
                    'protocol': ip_layer.get('ip.proto'),
                    'length': ip_layer.get('ip.len'),
                    'data': layers.get('data', {}).get('data.data', '')
                }
            
            llc_layer = layers.get('llc', {})
            if llc_layer:
                return {
                    'type': 'llc',
                    'control': llc_layer.get('llc.control'),
                    'sapi': llc_layer.get('llc.sapi'),
                    'data': layers.get('data', {}).get('data.data', '')
                }
            
            return None
        
        except Exception as e:
            return None
    
    # ==================== PHASE 2.1.3: GSM LAU TRACKING ====================
    
    def start_lau_tracking(self) -> bool:
        """Start tracking Location Area Updates (LAU)"""
        try:
            self.logger.info("Starting LAU tracking...")
            self.lau_tracking_enabled = True
            self.lau_history = []
            self.logger.info("LAU tracking enabled")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to start LAU tracking: {e}")
            return False
    
    def _parse_lau_message(self, gsm_msg: dict):
        """Parse GSM Layer 3 Location Area Update message"""
        try:
            msg_type = gsm_msg.get('gsm_a.dtap.msg_type', '')
            
            if msg_type in ['Location Updating Request', '0x08']:
                lai = gsm_msg.get('gsm_a.lai', {})
                location_area = {
                    'mcc': lai.get('gsm_a.mcc'),
                    'mnc': lai.get('gsm_a.mnc'),
                    'lac': lai.get('gsm_a.lac'),
                    'type': gsm_msg.get('gsm_a.dtap.upd_type', 'normal'),
                    'imsi': gsm_msg.get('gsm_a.imsi'),
                    'tmsi': gsm_msg.get('gsm_a.tmsi'),
                    'timestamp': time.time()
                }
                
                self.lau_history.append(location_area)
                self.logger.info(f"LAU detected: IMSI {location_area.get('imsi')}, LAC {location_area.get('lac')}")
                
                self.data_queue.put({
                    'type': 'lau',
                    'data': location_area,
                    'generation': '2G',
                    'timestamp': time.time()
                })
        
        except Exception as e:
            self.logger.debug(f"Failed to parse LAU message: {e}")
    
    def get_lau_history(self, imsi: str = None) -> List[dict]:
        """Get Location Area Update history"""
        if imsi:
            return [lau for lau in self.lau_history if lau.get('imsi') == imsi]
        return self.lau_history
    
    def track_target_mobility(self, imsi: str) -> dict:
        """Track mobility pattern of target IMSI"""
        lau_events = self.get_lau_history(imsi)
        
        if not lau_events:
            return {
                'imsi': imsi,
                'events': 0,
                'locations': []
            }
        
        unique_lacs = set([e['lac'] for e in lau_events if e.get('lac')])
        
        return {
            'imsi': imsi,
            'events': len(lau_events),
            'unique_locations': len(unique_lacs),
            'locations': list(unique_lacs),
            'first_seen': min([e['timestamp'] for e in lau_events]),
            'last_seen': max([e['timestamp'] for e in lau_events]),
            'mobility': 'high' if len(unique_lacs) > 5 else 'low'
        }
