"""
FalconOne Ambient IoT (A-IoT) Exploitation Module
3GPP Release 19 A-IoT surveillance, interception, and attack techniques
Version 1.6.1 - December 29, 2025

Capabilities:
- Passive eavesdropping on backscattered signals
- Energy depletion/jamming attacks
- Spoofing/impersonation of A-IoT tags
- Privacy tracking and metadata leakage
- Side-channel analysis on harvesting/reflection
- Multi-static topology manipulation

Reference: 3GPP TS 38.213 (A-IoT resource allocation), Rel-19 specs
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

try:
    from ..utils.logger import ModuleLogger
except ImportError:
    class ModuleLogger:
        def __init__(self, name, parent):
            self.logger = logging.getLogger(name) if parent is None else parent.getChild(name)
        def info(self, msg, **kw): self.logger.info(f"{msg} {kw if kw else ''}")
        def warning(self, msg, **kw): self.logger.warning(f"{msg} {kw if kw else ''}")
        def error(self, msg, **kw): self.logger.error(f"{msg} {kw if kw else ''}")
        def debug(self, msg, **kw): self.logger.debug(f"{msg} {kw if kw else ''}")


@dataclass
class AmbientIoTTag:
    """
    Represents an A-IoT device (tag)
    
    Attributes:
        tag_id: Unique identifier (from backscattered preamble)
        tag_type: 'type1' (zero-power) or 'type2' (stored-power)
        frequency_mhz: Operating frequency (e.g., 915 MHz ISM)
        topology: 'monostatic', 'bistatic', 'multistatic'
        modulation: 'OOK', 'FSK', 'PSK'
        power_budget_uj: Energy budget in microjoules
        rssi_dbm: Received signal strength
        last_seen: Timestamp of last detection
        location: Estimated coordinates (lat, lon)
        device_profile: Task 2.6.3 - Device classification
        behavior_signature: Task 2.6.3 - Behavioral fingerprint
    """
    tag_id: str
    tag_type: str  # 'type1' or 'type2'
    frequency_mhz: float
    topology: str
    modulation: str
    power_budget_uj: float
    rssi_dbm: float
    last_seen: datetime
    location: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = None
    device_profile: Optional[Dict[str, Any]] = None  # Task 2.6.3
    behavior_signature: Optional[Dict[str, Any]] = None  # Task 2.6.3


@dataclass
class BackscatteredSignal:
    """
    Captured backscattered signal from A-IoT tag
    
    Attributes:
        iq_samples: Complex I/Q samples
        tag_id: Associated tag identifier
        payload: Decoded data bits
        timestamp: Capture time
        rssi_dbm: Signal strength
        snr_db: Signal-to-noise ratio
        modulation: Detected modulation scheme
    """
    iq_samples: np.ndarray
    tag_id: str
    payload: Optional[bytes]
    timestamp: datetime
    rssi_dbm: float
    snr_db: float
    modulation: str


class AmbientIoTMonitor:
    """
    Ambient IoT (A-IoT) exploitation and surveillance system
    
    3GPP Release 19 A-IoT is ultra-low-power backscattering communication.
    This module enables:
    - Passive eavesdropping (<-60 dBm sensitivity)
    - Jamming/energy depletion (20 dBm attack power)
    - Spoofing with timing-precise injection
    - Privacy tracking via TDOA multilateration
    - Side-channel analysis on reflection patterns
    
    Typical usage:
        monitor = AmbientIoTMonitor(config, logger)
        monitor.start_passive_monitoring(frequency_mhz=915, bandwidth_mhz=1)
        tags = monitor.discover_tags(duration_sec=60)
        data = monitor.eavesdrop_tag('TAG_12345', duration_sec=10)
    """
    
    def __init__(self, config, logger: logging.Logger):
        """
        Initialize A-IoT monitor
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = ModuleLogger('AmbientIoTMonitor', logger)
        
        # SDR configuration
        self.sdr_sample_rate = config.get('aiot.sdr_sample_rate', 2e6)  # 2 MS/s
        self.sdr_gain = config.get('aiot.sdr_gain', 40)  # 40 dB
        self.center_freq_mhz = config.get('aiot.center_freq', 915.0)  # ISM band
        
        # Detection parameters
        self.energy_threshold_dbm = config.get('aiot.energy_threshold', -60)
        self.backscatter_detection_window = config.get('aiot.detection_window', 0.001)  # 1ms
        
        # Discovered tags
        self.discovered_tags: Dict[str, AmbientIoTTag] = {}
        self.intercepted_signals: List[BackscatteredSignal] = []
        
        # Attack state
        self.jamming_active = False
        self.spoofing_active = False
        
        # Tracking database
        self.tag_tracks: Dict[str, List[Tuple[datetime, Tuple[float, float]]]] = defaultdict(list)
        
        self.logger.info("A-IoT monitor initialized",
                       freq_mhz=self.center_freq_mhz,
                       sample_rate=f"{self.sdr_sample_rate/1e6:.1f}MS/s",
                       threshold_dbm=self.energy_threshold_dbm)
    
    # ===== 1. PASSIVE EAVESDROPPING =====
    
    def start_passive_monitoring(self, frequency_mhz: float = 915.0, 
                                 bandwidth_mhz: float = 1.0) -> bool:
        """
        Start passive monitoring of A-IoT backscattered signals
        Undetectable by tags/readers (pure reception)
        
        Args:
            frequency_mhz: Center frequency (ISM: 860-960 MHz typical)
            bandwidth_mhz: Monitoring bandwidth
        
        Returns:
            True if monitoring started
        
        Procedure:
        1. Tune SDR to frequency ± bandwidth/2
        2. Set gain for -60 dBm sensitivity
        3. Enable energy detector
        4. Start continuous I/Q capture
        
        Target: 90-95% detection rate, <1ms latency
        """
        try:
            self.center_freq_mhz = frequency_mhz
            
            # Configure SDR (simulated - in production use UHD/SoapySDR)
            self.logger.info(f"Starting passive A-IoT monitoring",
                           freq_mhz=frequency_mhz,
                           bw_mhz=bandwidth_mhz,
                           sensitivity_dbm=self.energy_threshold_dbm)
            
            # Initialize energy detector
            self._init_energy_detector()
            
            # Start background capture (simulated)
            self.monitoring_active = True
            
            self.logger.info("Passive monitoring active - eavesdropping undetectable")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            return False
    
    def discover_tags(self, duration_sec: int = 60) -> List[AmbientIoTTag]:
        """
        Discover A-IoT tags via passive scanning
        
        Args:
            duration_sec: Scan duration
        
        Returns:
            List of discovered tags
        
        Detection: Energy threshold + OOK/FSK demodulation
        """
        discovered = []
        scan_start = time.time()
        
        self.logger.info(f"Scanning for A-IoT tags ({duration_sec}s)")
        
        # Simulate tag discovery
        num_tags = np.random.randint(5, 20)
        
        for i in range(num_tags):
            # Simulate backscatter detection
            tag_id = f"TAG_{np.random.randint(10000, 99999)}"
            
            tag = AmbientIoTTag(
                tag_id=tag_id,
                tag_type='type1' if np.random.random() < 0.7 else 'type2',
                frequency_mhz=self.center_freq_mhz + np.random.uniform(-0.5, 0.5),
                topology='monostatic' if np.random.random() < 0.6 else 'bistatic',
                modulation='OOK' if np.random.random() < 0.8 else 'FSK',
                power_budget_uj=np.random.uniform(0.1, 1.0) if i % 3 == 0 else 0.0,
                rssi_dbm=np.random.uniform(-70, -40),
                last_seen=datetime.now(),
                metadata={
                    'manufacturer': f'Vendor_{chr(65 + i % 5)}',
                    'sensor_type': np.random.choice(['temp', 'humidity', 'motion', 'inventory']),
                }
            )
            
            self.discovered_tags[tag_id] = tag
            discovered.append(tag)
            
            self.logger.debug(f"Discovered tag {tag_id}",
                            type=tag.tag_type,
                            rssi=f"{tag.rssi_dbm:.1f}dBm",
                            mod=tag.modulation)
        
        self.logger.info(f"Discovery complete: {len(discovered)} tags found")
        return discovered
    
    def eavesdrop_tag(self, tag_id: str, duration_sec: int = 10) -> List[BackscatteredSignal]:
        """
        Eavesdrop on specific A-IoT tag's backscattered signals
        
        Args:
            tag_id: Target tag identifier
            duration_sec: Eavesdrop duration
        
        Returns:
            List of intercepted signals with decoded payloads
        
        Procedure:
        1. Isolate tag frequency (via prior scan)
        2. Detect backscattering events (energy spike)
        3. Demodulate (OOK/FSK)
        4. Decode preamble + payload
        5. Extract metadata (tag ID, sensor data)
        
        Success rate: 90-95% in low-interference
        """
        if tag_id not in self.discovered_tags:
            self.logger.error(f"Tag {tag_id} not in discovered list")
            return []
        
        tag = self.discovered_tags[tag_id]
        signals = []
        
        self.logger.info(f"Eavesdropping on {tag_id} ({duration_sec}s)")
        
        # Simulate interception
        num_transmissions = int(duration_sec / 2)  # Tag transmits every ~2s
        
        for i in range(num_transmissions):
            # Simulate backscattered I/Q capture
            iq_samples = self._capture_backscattered_iq(tag)
            
            # Demodulate and decode
            payload = self._demodulate_backscatter(iq_samples, tag.modulation)
            
            signal = BackscatteredSignal(
                iq_samples=iq_samples,
                tag_id=tag_id,
                payload=payload,
                timestamp=datetime.now(),
                rssi_dbm=tag.rssi_dbm + np.random.uniform(-3, 3),
                snr_db=np.random.uniform(10, 25),
                modulation=tag.modulation
            )
            
            signals.append(signal)
            self.intercepted_signals.append(signal)
            
            self.logger.debug(f"Intercepted transmission {i+1}/{num_transmissions}",
                            payload_bytes=len(payload) if payload else 0,
                            snr=f"{signal.snr_db:.1f}dB")
        
        self.logger.info(f"Eavesdropping complete: {len(signals)} transmissions intercepted")
        return signals
    
    def decode_payload(self, signal: BackscatteredSignal) -> Dict[str, Any]:
        """
        Decode A-IoT payload from backscattered signal
        
        Args:
            signal: Backscattered signal
        
        Returns:
            Decoded data (sensor readings, metadata)
        
        Format (typical):
        - Preamble: 8-16 bits (tag ID)
        - Data: 8-64 bits (sensor value, status)
        - CRC: 8-16 bits (optional)
        """
        if signal.payload is None:
            return {'error': 'no_payload'}
        
        # Parse payload (simplified)
        payload_bits = signal.payload
        
        decoded = {
            'tag_id': signal.tag_id,
            'preamble': payload_bits[:16] if len(payload_bits) >= 16 else payload_bits,
            'data': payload_bits[16:-8] if len(payload_bits) > 24 else b'',
            'crc': payload_bits[-8:] if len(payload_bits) >= 8 else b'',
            'timestamp': signal.timestamp.isoformat(),
        }
        
        # Infer sensor value (simulated)
        if len(decoded['data']) >= 8:
            sensor_value = int.from_bytes(decoded['data'][:8], byteorder='big') % 100
            decoded['sensor_reading'] = sensor_value
            decoded['sensor_unit'] = 'celsius' if sensor_value < 50 else 'percent'
        
        return decoded
    
    # ===== 2. JAMMING/DoS ATTACKS =====
    
    def jam_tag_energy_harvesting(self, tag_id: str, duration_sec: int = 30,
                                   power_dbm: float = 20) -> Dict[str, Any]:
        """
        Jam A-IoT tag by disrupting energy harvesting
        Transmit high-power CW or noise to prevent rectification
        
        Args:
            tag_id: Target tag
            duration_sec: Jam duration
            power_dbm: Jamming power (20 dBm typical)
        
        Returns:
            Attack results
        
        Procedure:
        1. Identify tag harvesting frequency
        2. Transmit CW or wideband noise at 20 dBm
        3. Monitor tag silence (confirms depletion)
        
        Success: >95% for Type 1 tags, 10-60s depletion
        """
        if tag_id not in self.discovered_tags:
            return {'success': False, 'error': 'tag_not_found'}
        
        tag = self.discovered_tags[tag_id]
        
        self.logger.warning(f"Jamming tag {tag_id} energy harvesting",
                          duration_sec=duration_sec,
                          power_dbm=power_dbm)
        
        self.jamming_active = True
        jam_start = time.time()
        
        # Simulate jamming (in production: transmit via SDR)
        while time.time() - jam_start < duration_sec:
            # Transmit jamming signal
            self._transmit_jamming_signal(tag.frequency_mhz, power_dbm)
            time.sleep(0.1)
        
        self.jamming_active = False
        
        # Verify tag silence
        tag_silent = self._check_tag_silence(tag_id, verify_duration_sec=5)
        
        result = {
            'success': tag_silent,
            'tag_id': tag_id,
            'jam_duration_sec': duration_sec,
            'power_dbm': power_dbm,
            'tag_type': tag.tag_type,
            'confirmed_depletion': tag_silent,
        }
        
        if tag_silent:
            self.logger.warning(f"Tag {tag_id} successfully jammed (energy depleted)")
        else:
            self.logger.info(f"Tag {tag_id} still active (Type 2 with stored power?)")
        
        return result
    
    def deplete_tag_capacitor(self, tag_id: str) -> Dict[str, Any]:
        """
        Deplete tag capacitor via excessive query flooding
        Force repeated responses to drain stored energy
        
        Args:
            tag_id: Target tag
        
        Returns:
            Depletion results
        
        Procedure:
        1. Spoof reader "select" commands
        2. Force tag to respond repeatedly (1-10 Hz)
        3. Monitor capacitor depletion (<1 μJ/bit)
        4. Confirm silence
        
        Target: Type 2 tags with stored power
        """
        if tag_id not in self.discovered_tags:
            return {'success': False, 'error': 'tag_not_found'}
        
        tag = self.discovered_tags[tag_id]
        
        if tag.tag_type == 'type1':
            self.logger.warning(f"Tag {tag_id} is Type 1 (zero-power), use energy jamming instead")
            return {'success': False, 'error': 'wrong_tag_type', 'recommended': 'jam_tag_energy_harvesting'}
        
        self.logger.warning(f"Depleting tag {tag_id} capacitor via query flooding")
        
        # Flood with queries
        query_count = 0
        depletion_start = time.time()
        
        while time.time() - depletion_start < 60:  # Max 60s
            # Spoof reader query
            self._spoof_reader_query(tag_id, query_type='select')
            query_count += 1
            time.sleep(0.1)  # 10 Hz query rate
            
            # Check if depleted
            if query_count % 50 == 0:
                if self._check_tag_silence(tag_id, verify_duration_sec=2):
                    break
        
        depletion_duration = time.time() - depletion_start
        tag_depleted = self._check_tag_silence(tag_id, verify_duration_sec=5)
        
        result = {
            'success': tag_depleted,
            'tag_id': tag_id,
            'queries_sent': query_count,
            'depletion_duration_sec': depletion_duration,
            'tag_type': tag.tag_type,
            'confirmed_depletion': tag_depleted,
        }
        
        if tag_depleted:
            self.logger.warning(f"Tag {tag_id} capacitor depleted after {query_count} queries")
        
        return result
    
    # ===== 3. SPOOFING/IMPERSONATION =====
    
    def spoof_tag_response(self, target_tag_id: str, fake_payload: bytes,
                          duration_sec: int = 10) -> Dict[str, Any]:
        """
        Spoof A-IoT tag by forging backscattered responses
        Inject fake data to mislead reader
        
        Args:
            target_tag_id: Tag to impersonate
            fake_payload: Forged data (e.g., modified sensor reading)
            duration_sec: Spoofing duration
        
        Returns:
            Spoofing results
        
        Procedure:
        1. Eavesdrop legitimate tag-reader exchange
        2. Emulate backscattering via SDR (modulate reflection)
        3. Inject fake payload during tag response window
        4. Verify reader acknowledgment
        
        Success: 70-85% (depends on anti-collision)
        """
        if target_tag_id not in self.discovered_tags:
            return {'success': False, 'error': 'tag_not_found'}
        
        tag = self.discovered_tags[target_tag_id]
        
        self.logger.warning(f"Spoofing tag {target_tag_id}",
                          payload_len=len(fake_payload),
                          duration=duration_sec)
        
        self.spoofing_active = True
        spoof_start = time.time()
        injections = 0
        successful_injections = 0
        
        while time.time() - spoof_start < duration_sec:
            # Wait for reader query (simulate timing)
            reader_query_detected = self._detect_reader_query(tag.frequency_mhz)
            
            if reader_query_detected:
                # Inject fake backscattered response
                success = self._inject_backscattered_response(
                    tag, fake_payload, modulation=tag.modulation
                )
                
                injections += 1
                if success:
                    successful_injections += 1
                
                time.sleep(0.5)  # Wait for next query cycle
        
        self.spoofing_active = False
        
        result = {
            'success': successful_injections > 0,
            'tag_id': target_tag_id,
            'total_injections': injections,
            'successful_injections': successful_injections,
            'success_rate': successful_injections / injections if injections > 0 else 0,
            'duration_sec': duration_sec,
        }
        
        self.logger.warning(f"Spoofing complete: {successful_injections}/{injections} successful",
                          rate=f"{result['success_rate']*100:.1f}%")
        
        return result
    
    def clone_tag(self, original_tag_id: str) -> str:
        """
        Clone A-IoT tag by capturing and replaying credentials
        
        Args:
            original_tag_id: Tag to clone
        
        Returns:
            Cloned tag ID
        
        Procedure:
        1. Eavesdrop tag responses for extended period
        2. Extract authentication pattern (if any)
        3. Create virtual clone with same credentials
        4. Test clone against reader
        """
        if original_tag_id not in self.discovered_tags:
            return None
        
        original_tag = self.discovered_tags[original_tag_id]
        
        self.logger.warning(f"Cloning tag {original_tag_id}")
        
        # Eavesdrop to capture credentials
        signals = self.eavesdrop_tag(original_tag_id, duration_sec=30)
        
        # Analyze for auth pattern
        auth_pattern = self._extract_auth_pattern(signals)
        
        # Create clone
        clone_id = f"CLONE_{original_tag_id}"
        clone_tag = AmbientIoTTag(
            tag_id=clone_id,
            tag_type=original_tag.tag_type,
            frequency_mhz=original_tag.frequency_mhz,
            topology=original_tag.topology,
            modulation=original_tag.modulation,
            power_budget_uj=original_tag.power_budget_uj,
            rssi_dbm=original_tag.rssi_dbm,
            last_seen=datetime.now(),
            metadata={'cloned_from': original_tag_id, 'auth_pattern': auth_pattern}
        )
        
        self.discovered_tags[clone_id] = clone_tag
        
        self.logger.warning(f"Tag cloned: {clone_id}",
                          original=original_tag_id,
                          has_auth=auth_pattern is not None)
        
        return clone_id
    
    # ===== 4. PRIVACY TRACKING =====
    
    def track_tag_location(self, tag_id: str, monitor_duration_sec: int = 300,
                          num_sensors: int = 4) -> List[Tuple[datetime, Tuple[float, float]]]:
        """
        Track A-IoT tag location via TDOA multilateration
        Multi-site passive monitoring for privacy analysis
        
        Args:
            tag_id: Tag to track
            monitor_duration_sec: Tracking duration
            num_sensors: Number of monitoring sites (3+ required)
        
        Returns:
            List of (timestamp, (lat, lon)) location estimates
        
        Procedure:
        1. Deploy num_sensors SDRs at known locations
        2. Passive monitoring for tag emissions
        3. TDOA calculation from time differences
        4. Multilateration (≥3 sensors)
        5. Graph topology for movement prediction
        
        Accuracy: <50m with 4 sensors
        Persistence: >90% tracking
        """
        if tag_id not in self.discovered_tags:
            return []
        
        if num_sensors < 3:
            self.logger.error("Need ≥3 sensors for TDOA multilateration")
            return []
        
        tag = self.discovered_tags[tag_id]
        
        self.logger.info(f"Tracking tag {tag_id}",
                       duration_sec=monitor_duration_sec,
                       sensors=num_sensors)
        
        # Simulate sensor deployment
        sensor_locations = self._deploy_virtual_sensors(num_sensors)
        
        track_start = time.time()
        locations = []
        
        while time.time() - track_start < monitor_duration_sec:
            # Capture at each sensor
            tdoa_measurements = []
            
            for sensor_id, sensor_loc in enumerate(sensor_locations):
                # Simulate RSSI/timing measurement
                rssi = tag.rssi_dbm + np.random.uniform(-5, 5)
                arrival_time = time.time() + np.random.uniform(0, 0.001)  # <1ms jitter
                
                tdoa_measurements.append({
                    'sensor_id': sensor_id,
                    'location': sensor_loc,
                    'rssi': rssi,
                    'arrival_time': arrival_time
                })
            
            # TDOA multilateration
            estimated_location = self._tdoa_multilateration(tdoa_measurements)
            
            if estimated_location:
                timestamp = datetime.now()
                locations.append((timestamp, estimated_location))
                self.tag_tracks[tag_id].append((timestamp, estimated_location))
                
                self.logger.debug(f"Location update",
                                tag=tag_id,
                                lat=f"{estimated_location[0]:.6f}",
                                lon=f"{estimated_location[1]:.6f}")
            
            time.sleep(2)  # Update every 2s
        
        self.logger.info(f"Tracking complete: {len(locations)} location updates",
                       tag=tag_id)
        
        return locations
    
    def extract_privacy_leaks(self, tag_id: str) -> Dict[str, Any]:
        """
        Extract privacy-sensitive metadata from A-IoT tag
        
        Args:
            tag_id: Target tag
        
        Returns:
            Leaked information
        
        Leaks:
        - Tag ID (persistent identifier)
        - Manufacturer (via length/format)
        - Sensor type (via payload patterns)
        - Movement patterns (via location history)
        - Association with UE (co-located NB-IoT)
        """
        if tag_id not in self.discovered_tags:
            return {'error': 'tag_not_found'}
        
        tag = self.discovered_tags[tag_id]
        
        # Analyze intercepted signals
        tag_signals = [s for s in self.intercepted_signals if s.tag_id == tag_id]
        
        leaks = {
            'tag_id': tag_id,
            'persistent_identifier': True,  # Tag ID is static
            'manufacturer': tag.metadata.get('manufacturer', 'Unknown'),
            'sensor_type': tag.metadata.get('sensor_type', 'Unknown'),
            'tracking_duration_hours': len(self.tag_tracks.get(tag_id, [])) * 2 / 3600,
            'total_locations': len(self.tag_tracks.get(tag_id, [])),
            'transmission_count': len(tag_signals),
        }
        
        # Fingerprinting via payload patterns
        if tag_signals:
            payload_lengths = [len(s.payload) if s.payload else 0 for s in tag_signals]
            leaks['payload_length_pattern'] = {
                'min': min(payload_lengths),
                'max': max(payload_lengths),
                'avg': np.mean(payload_lengths),
            }
        
        # Movement analysis
        if tag_id in self.tag_tracks and len(self.tag_tracks[tag_id]) > 2:
            locations = [loc for _, loc in self.tag_tracks[tag_id]]
            
            # Calculate movement speed (simplified)
            distances = []
            for i in range(1, len(locations)):
                dist = self._haversine_distance(locations[i-1], locations[i])
                distances.append(dist)
            
            if distances:
                leaks['movement_pattern'] = {
                    'max_distance_m': max(distances),
                    'avg_speed_mps': np.mean(distances) / 2,  # Assuming 2s intervals
                    'stationary': max(distances) < 10,  # <10m = stationary
                }
        
        self.logger.info(f"Privacy leaks extracted",
                       tag=tag_id,
                       leaks=len(leaks))
        
        return leaks
    
    # ===== 5. SIDE-CHANNEL ATTACKS =====
    
    def analyze_side_channels(self, tag_id: str, num_measurements: int = 1000) -> Dict[str, Any]:
        """
        Side-channel analysis on A-IoT tag reflection/harvesting
        
        Args:
            tag_id: Target tag
            num_measurements: Number of measurements
        
        Returns:
            Side-channel analysis results
        
        Techniques:
        - Timing analysis on backscatter modulation
        - Power analysis on reflection coefficient
        - Differential analysis for bit recovery
        
        Proximity: <10m required
        Success: 60-80% bit recovery
        """
        if tag_id not in self.discovered_tags:
            return {'error': 'tag_not_found'}
        
        tag = self.discovered_tags[tag_id]
        
        self.logger.warning(f"Side-channel analysis on tag {tag_id}",
                          measurements=num_measurements)
        
        # Collect timing measurements
        timing_measurements = []
        power_measurements = []
        
        for i in range(num_measurements):
            # Trigger tag response
            response = self._trigger_tag_response(tag_id)
            
            if response:
                # Measure reflection timing
                timing = response.get('response_time_us', 0)
                timing_measurements.append(timing)
                
                # Measure reflected power
                power = response.get('reflected_power_dbm', 0)
                power_measurements.append(power)
        
        # Statistical analysis
        timing_array = np.array(timing_measurements)
        power_array = np.array(power_measurements)
        
        results = {
            'tag_id': tag_id,
            'measurements': num_measurements,
            'timing_analysis': {
                'mean_us': float(np.mean(timing_array)),
                'std_us': float(np.std(timing_array)),
                'variance': float(np.var(timing_array)),
                'leak_detected': np.std(timing_array) > 5,  # >5μs variance = leak
            },
            'power_analysis': {
                'mean_dbm': float(np.mean(power_array)),
                'std_dbm': float(np.std(power_array)),
                'variance': float(np.var(power_array)),
                'leak_detected': np.std(power_array) > 2,  # >2dB variance = leak
            },
        }
        
        # Differential analysis
        if results['timing_analysis']['leak_detected'] or results['power_analysis']['leak_detected']:
            bit_recovery = self._differential_bit_recovery(timing_array, power_array)
            results['bit_recovery'] = bit_recovery
        
        self.logger.warning(f"Side-channel analysis complete",
                          timing_leak=results['timing_analysis']['leak_detected'],
                          power_leak=results['power_analysis']['leak_detected'])
        
        return results
    
    def fault_injection_attack(self, tag_id: str, voltage_variation_percent: float = 20) -> Dict[str, Any]:
        """
        Fault injection via variable-power exciter
        Induce bit flips in tag response
        
        Args:
            tag_id: Target tag
            voltage_variation_percent: Power variation (% of nominal)
        
        Returns:
            Fault injection results
        
        Technique:
        - Transmit under-voltage RF to cause harvesting faults
        - Monitor for bit flips in response
        - Correlate faults with data patterns
        """
        if tag_id not in self.discovered_tags:
            return {'error': 'tag_not_found'}
        
        tag = self.discovered_tags[tag_id]
        
        self.logger.warning(f"Fault injection attack on {tag_id}",
                          voltage_var=f"{voltage_variation_percent}%")
        
        # Baseline capture
        baseline_responses = []
        for _ in range(10):
            response = self._trigger_tag_response(tag_id, nominal_power=True)
            if response:
                baseline_responses.append(response['payload'])
        
        # Fault injection
        faulted_responses = []
        for _ in range(50):
            # Vary exciter power
            power_factor = 1.0 - (voltage_variation_percent / 100) * np.random.uniform(0, 1)
            response = self._trigger_tag_response(tag_id, nominal_power=False, power_factor=power_factor)
            
            if response:
                faulted_responses.append(response['payload'])
        
        # Analyze for bit flips
        bit_flips = 0
        for baseline, faulted in zip(baseline_responses[:len(faulted_responses)], faulted_responses):
            if baseline != faulted:
                bit_flips += 1
        
        results = {
            'tag_id': tag_id,
            'voltage_variation_percent': voltage_variation_percent,
            'baseline_captures': len(baseline_responses),
            'faulted_captures': len(faulted_responses),
            'bit_flips_detected': bit_flips,
            'fault_success_rate': bit_flips / len(faulted_responses) if faulted_responses else 0,
        }
        
        self.logger.warning(f"Fault injection complete",
                          bit_flips=bit_flips,
                          success_rate=f"{results['fault_success_rate']*100:.1f}%")
        
        return results
    
    # ===== HELPER METHODS =====
    
    def _init_energy_detector(self):
        """Initialize energy detector for backscatter detection"""
        self.energy_detector_active = True
        self.logger.debug("Energy detector initialized",
                        threshold=f"{self.energy_threshold_dbm}dBm")
    
    def _capture_backscattered_iq(self, tag: AmbientIoTTag) -> np.ndarray:
        """Capture backscattered I/Q samples"""
        # Simulate I/Q capture (in production: use SDR)
        num_samples = int(self.sdr_sample_rate * 0.01)  # 10ms
        iq = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        iq *= 10 ** ((tag.rssi_dbm - 30) / 20)  # Scale by RSSI
        return iq
    
    def _demodulate_backscatter(self, iq_samples: np.ndarray, modulation: str) -> Optional[bytes]:
        """Demodulate backscattered signal"""
        # Simplified demodulation (in production: use GNU Radio blocks)
        if modulation == 'OOK':
            # Envelope detection
            envelope = np.abs(iq_samples)
            threshold = np.mean(envelope)
            bits = (envelope > threshold).astype(int)
        elif modulation == 'FSK':
            # Frequency discrimination
            phase = np.angle(iq_samples)
            freq = np.diff(phase)
            bits = (freq > 0).astype(int)
        else:
            return None
        
        # Convert to bytes (simplified)
        payload_bits = bits[:80] if len(bits) >= 80 else bits  # 10 bytes max
        payload = bytes([int(''.join(map(str, payload_bits[i:i+8])), 2) 
                        for i in range(0, len(payload_bits), 8) if i+8 <= len(payload_bits)])
        
        return payload
    
    def _transmit_jamming_signal(self, frequency_mhz: float, power_dbm: float):
        """Transmit jamming signal (simulated)"""
        # In production: use SDR to transmit CW or noise
        pass
    
    def _check_tag_silence(self, tag_id: str, verify_duration_sec: int = 5) -> bool:
        """Check if tag is silent (depleted/jammed)"""
        # Simulate tag silence check
        return np.random.random() < 0.9  # 90% success rate
    
    def _spoof_reader_query(self, tag_id: str, query_type: str = 'select'):
        """Spoof reader query to trigger tag response"""
        # In production: forge and transmit reader command
        pass
    
    def _detect_reader_query(self, frequency_mhz: float) -> bool:
        """Detect reader query signal"""
        return np.random.random() < 0.8  # 80% detection rate
    
    def _inject_backscattered_response(self, tag: AmbientIoTTag, payload: bytes, modulation: str) -> bool:
        """Inject fake backscattered response"""
        # In production: modulate and transmit via SDR
        return np.random.random() < 0.75  # 75% injection success
    
    def _extract_auth_pattern(self, signals: List[BackscatteredSignal]) -> Optional[Dict]:
        """Extract authentication pattern from signals"""
        if not signals:
            return None
        
        # Analyze for repeating patterns
        payloads = [s.payload for s in signals if s.payload]
        
        if len(payloads) < 3:
            return None
        
        return {
            'pattern_type': 'static_id',
            'detected': True,
        }
    
    def _deploy_virtual_sensors(self, num_sensors: int) -> List[Tuple[float, float]]:
        """Deploy virtual sensor locations for TDOA"""
        # Generate sensor grid
        base_lat, base_lon = 45.0, -122.0
        sensors = []
        
        for i in range(num_sensors):
            lat = base_lat + (i % 2) * 0.001
            lon = base_lon + (i // 2) * 0.001
            sensors.append((lat, lon))
        
        return sensors
    
    def _tdoa_multilateration(self, measurements: List[Dict]) -> Optional[Tuple[float, float]]:
        """TDOA-based multilateration"""
        if len(measurements) < 3:
            return None
        
        # Simplified multilateration (in production: use proper TDOA solver)
        lats = [m['location'][0] for m in measurements]
        lons = [m['location'][1] for m in measurements]
        
        estimated_lat = np.mean(lats) + np.random.uniform(-0.0005, 0.0005)
        estimated_lon = np.mean(lons) + np.random.uniform(-0.0005, 0.0005)
        
        return (estimated_lat, estimated_lon)
    
    def _haversine_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate Haversine distance between two points"""
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        
        R = 6371000  # Earth radius in meters
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _trigger_tag_response(self, tag_id: str, nominal_power: bool = True, 
                             power_factor: float = 1.0) -> Optional[Dict]:
        """Trigger tag response for side-channel analysis"""
        if tag_id not in self.discovered_tags:
            return None
        
        tag = self.discovered_tags[tag_id]
        
        # Simulate response
        response_time_us = np.random.uniform(50, 150) * power_factor
        reflected_power_dbm = tag.rssi_dbm * power_factor + np.random.uniform(-2, 2)
        
        return {
            'response_time_us': response_time_us,
            'reflected_power_dbm': reflected_power_dbm,
            'payload': np.random.bytes(8),
        }
    
    def _differential_bit_recovery(self, timing_array: np.ndarray, power_array: np.ndarray) -> Dict:
        """Differential analysis for bit recovery"""
        # Simplified differential analysis
        # In production: correlate timing/power with bit values
        
        recovered_bits = int(len(timing_array) * 0.65)  # 65% recovery rate
        
        return {
            'recovered_bits': recovered_bits,
            'total_bits': len(timing_array),
            'recovery_rate': recovered_bits / len(timing_array),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get A-IoT monitoring statistics"""
        return {
            'discovered_tags': len(self.discovered_tags),
            'intercepted_signals': len(self.intercepted_signals),
            'tracked_tags': len(self.tag_tracks),
            'monitoring_active': getattr(self, 'monitoring_active', False),
            'jamming_active': self.jamming_active,
            'spoofing_active': self.spoofing_active,
            'tag_types': {
                'type1': sum(1 for t in self.discovered_tags.values() if t.tag_type == 'type1'),
                'type2': sum(1 for t in self.discovered_tags.values() if t.tag_type == 'type2'),
            },
        }
