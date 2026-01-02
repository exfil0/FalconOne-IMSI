"""
AI/IoT Device Profiling Module
Machine learning-based device classification and behavioral analysis

Version 1.0: Phase 2.6.3 - AI/IoT Device Profiling
- IoT device classification by traffic patterns
- NB-IoT/LTE-M fingerprinting
- Behavioral anomaly detection
- Device type prediction with ML models
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..utils.logger import ModuleLogger


class IoTDeviceProfiler:
    """ML-based IoT device profiling and classification"""
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize IoT device profiler
        
        Args:
            logger: Logger instance
        """
        self.logger = ModuleLogger('DeviceProfiler', logger or logging.getLogger(__name__))
        
        # Device database
        self.devices: Dict[str, Dict] = {}  # device_id -> profile
        
        # Traffic history for behavioral analysis
        self.traffic_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # ML model for classification
        self.classifier = None
        self.scaler = None
        
        if SKLEARN_AVAILABLE:
            self._initialize_ml_models()
        else:
            self.logger.warning("scikit-learn not available - ML classification disabled")
        
        # Device type signatures (pattern-based classification)
        self.device_signatures = self._load_device_signatures()
        
        self.logger.info("IoT Device Profiler initialized")
    
    def _initialize_ml_models(self):
        """Initialize machine learning models"""
        try:
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.scaler = StandardScaler()
            
            # Pre-train with synthetic data (in production: use real training data)
            self._pretrain_classifier()
            
            self.logger.info("ML models initialized and pre-trained")
            
        except Exception as e:
            self.logger.error(f"ML model initialization failed: {e}")
            self.classifier = None
    
    def _pretrain_classifier(self):
        """Pre-train classifier with synthetic training data"""
        # Synthetic training data for device types
        # Features: [packet_size_avg, packet_rate, duty_cycle, burst_size, inter_arrival_time]
        
        training_data = []
        labels = []
        
        # NB-IoT sensor (low rate, small packets, periodic)
        for _ in range(50):
            features = [
                np.random.normal(100, 20),  # Small packet size
                np.random.normal(0.1, 0.02),  # Low packet rate
                np.random.normal(0.05, 0.01),  # Low duty cycle
                np.random.normal(1, 0.2),  # Single packets
                np.random.normal(60, 10)  # ~1 minute intervals
            ]
            training_data.append(features)
            labels.append('nb_iot_sensor')
        
        # LTE-M tracker (moderate rate, GPS bursts)
        for _ in range(50):
            features = [
                np.random.normal(200, 50),  # Medium packet size
                np.random.normal(0.5, 0.1),  # Moderate packet rate
                np.random.normal(0.2, 0.05),  # Moderate duty cycle
                np.random.normal(5, 1),  # Burst of 5 packets
                np.random.normal(10, 2)  # ~10 second intervals
            ]
            training_data.append(features)
            labels.append('lte_m_tracker')
        
        # Smart meter (high rate, regular intervals)
        for _ in range(50):
            features = [
                np.random.normal(150, 30),  # Medium packet size
                np.random.normal(1.0, 0.2),  # Regular packet rate
                np.random.normal(0.3, 0.05),  # Moderate duty cycle
                np.random.normal(1, 0.1),  # Single packets
                np.random.normal(5, 1)  # ~5 second intervals
            ]
            training_data.append(features)
            labels.append('smart_meter')
        
        # Camera (high rate, large packets, continuous)
        for _ in range(50):
            features = [
                np.random.normal(1000, 200),  # Large packet size
                np.random.normal(10.0, 2),  # High packet rate
                np.random.normal(0.8, 0.1),  # High duty cycle
                np.random.normal(50, 10),  # Large bursts
                np.random.normal(0.1, 0.02)  # Continuous
            ]
            training_data.append(features)
            labels.append('iot_camera')
        
        # Wearable (moderate rate, small bursts)
        for _ in range(50):
            features = [
                np.random.normal(50, 10),  # Small packet size
                np.random.normal(0.2, 0.05),  # Low-moderate rate
                np.random.normal(0.1, 0.02),  # Low duty cycle
                np.random.normal(2, 0.5),  # Small bursts
                np.random.normal(30, 5)  # ~30 second intervals
            ]
            training_data.append(features)
            labels.append('wearable')
        
        # Scale features and train
        X = self.scaler.fit_transform(training_data)
        self.classifier.fit(X, labels)
        
        self.logger.info(f"Classifier pre-trained with {len(training_data)} samples")
    
    def _load_device_signatures(self) -> Dict[str, Dict]:
        """Load known device type signatures"""
        return {
            'nb_iot_sensor': {
                'packet_size_range': (50, 200),
                'packet_rate_range': (0.01, 0.2),  # packets/sec
                'duty_cycle_range': (0.01, 0.1),
                'burst_size_range': (1, 3),
                'periodicity': 'high',  # Regular intervals
                'modulation': ['QPSK', 'BPSK'],
                'frequency_bands': ['700MHz', '800MHz', '900MHz'],
                'typical_use_cases': ['sensors', 'meters', 'trackers']
            },
            'lte_m_tracker': {
                'packet_size_range': (100, 500),
                'packet_rate_range': (0.1, 1.0),
                'duty_cycle_range': (0.1, 0.3),
                'burst_size_range': (3, 10),
                'periodicity': 'medium',
                'modulation': ['QPSK', '16QAM'],
                'frequency_bands': ['700MHz', '800MHz', '1800MHz'],
                'typical_use_cases': ['asset_tracking', 'vehicle_tracking', 'logistics']
            },
            'smart_meter': {
                'packet_size_range': (100, 300),
                'packet_rate_range': (0.5, 2.0),
                'duty_cycle_range': (0.2, 0.4),
                'burst_size_range': (1, 2),
                'periodicity': 'very_high',  # Highly regular
                'modulation': ['QPSK'],
                'frequency_bands': ['900MHz', '1800MHz'],
                'typical_use_cases': ['electricity_meter', 'water_meter', 'gas_meter']
            },
            'iot_camera': {
                'packet_size_range': (500, 1500),
                'packet_rate_range': (5.0, 30.0),
                'duty_cycle_range': (0.5, 0.9),
                'burst_size_range': (20, 100),
                'periodicity': 'low',  # Continuous/irregular
                'modulation': ['16QAM', '64QAM'],
                'frequency_bands': ['1800MHz', '2100MHz', '2600MHz'],
                'typical_use_cases': ['surveillance', 'security_camera', 'doorbell']
            },
            'wearable': {
                'packet_size_range': (20, 100),
                'packet_rate_range': (0.1, 0.5),
                'duty_cycle_range': (0.05, 0.15),
                'burst_size_range': (1, 5),
                'periodicity': 'medium',
                'modulation': ['QPSK'],
                'frequency_bands': ['700MHz', '800MHz', '1800MHz'],
                'typical_use_cases': ['fitness_tracker', 'smartwatch', 'medical_device']
            },
            'industrial_sensor': {
                'packet_size_range': (50, 200),
                'packet_rate_range': (0.05, 0.3),
                'duty_cycle_range': (0.02, 0.1),
                'burst_size_range': (1, 2),
                'periodicity': 'high',
                'modulation': ['QPSK', 'BPSK'],
                'frequency_bands': ['700MHz', '800MHz', '900MHz'],
                'typical_use_cases': ['temperature_sensor', 'pressure_sensor', 'vibration_sensor']
            }
        }
    
    def profile_device(self, device_id: str, traffic_samples: List[Dict]) -> Dict[str, Any]:
        """
        Profile IoT device based on traffic patterns
        Task 2.6.3: IoT device classification
        
        Args:
            device_id: Device identifier
            traffic_samples: List of traffic samples with keys:
                           {timestamp, packet_size, rssi, frequency, modulation}
        
        Returns:
            Device profile with classification
        """
        try:
            self.logger.info(f"Profiling device: {device_id} ({len(traffic_samples)} samples)")
            
            if not traffic_samples:
                return {'success': False, 'reason': 'no_traffic_samples'}
            
            # Extract features from traffic patterns
            features = self._extract_traffic_features(traffic_samples)
            
            # Classify device type (ML-based)
            device_type = self._classify_device_ml(features) if self.classifier else None
            
            # Fallback to pattern-based classification
            if not device_type:
                device_type = self._classify_device_pattern(features)
            
            # Generate behavioral signature
            behavior_signature = self._generate_behavior_signature(traffic_samples)
            
            # Store device profile
            profile = {
                'device_id': device_id,
                'device_type': device_type,
                'confidence': features.get('classification_confidence', 0.7),
                'features': features,
                'behavior_signature': behavior_signature,
                'first_seen': traffic_samples[0]['timestamp'],
                'last_seen': traffic_samples[-1]['timestamp'],
                'num_samples': len(traffic_samples),
                'profiling_timestamp': datetime.utcnow().isoformat()
            }
            
            self.devices[device_id] = profile
            
            self.logger.info(f"Device profiled: {device_id} -> {device_type} "
                           f"(confidence={profile['confidence']:.2%})")
            
            return {
                'success': True,
                'profile': profile
            }
            
        except Exception as e:
            self.logger.error(f"Device profiling error: {e}")
            return {'success': False, 'error': str(e)}
    
    def detect_anomalies(self, device_id: str, recent_traffic: List[Dict]) -> Dict[str, Any]:
        """
        Detect behavioral anomalies in device traffic
        Task 2.6.3: Behavioral anomaly detection
        
        Args:
            device_id: Device identifier
            recent_traffic: Recent traffic samples
        
        Returns:
            Anomaly detection results
        """
        try:
            self.logger.info(f"Detecting anomalies for device: {device_id}")
            
            if device_id not in self.devices:
                return {'success': False, 'reason': 'device_not_profiled'}
            
            if not recent_traffic:
                return {'success': False, 'reason': 'no_traffic_samples'}
            
            device_profile = self.devices[device_id]
            baseline_features = device_profile['features']
            
            # Extract features from recent traffic
            current_features = self._extract_traffic_features(recent_traffic)
            
            # Compare against baseline
            anomalies = []
            anomaly_score = 0.0
            
            # Check packet size deviation
            if 'packet_size_avg' in current_features and 'packet_size_avg' in baseline_features:
                size_deviation = abs(current_features['packet_size_avg'] - 
                                   baseline_features['packet_size_avg'])
                size_threshold = baseline_features.get('packet_size_std', 50) * 2
                
                if size_deviation > size_threshold:
                    anomalies.append({
                        'type': 'packet_size_anomaly',
                        'severity': 'medium',
                        'current_value': current_features['packet_size_avg'],
                        'baseline_value': baseline_features['packet_size_avg'],
                        'deviation': size_deviation
                    })
                    anomaly_score += 0.3
            
            # Check packet rate deviation
            if 'packet_rate' in current_features and 'packet_rate' in baseline_features:
                rate_deviation = abs(current_features['packet_rate'] - 
                                   baseline_features['packet_rate'])
                rate_threshold = baseline_features['packet_rate'] * 0.5  # 50% threshold
                
                if rate_deviation > rate_threshold:
                    anomalies.append({
                        'type': 'packet_rate_anomaly',
                        'severity': 'high',
                        'current_value': current_features['packet_rate'],
                        'baseline_value': baseline_features['packet_rate'],
                        'deviation': rate_deviation
                    })
                    anomaly_score += 0.5
            
            # Check duty cycle deviation
            if 'duty_cycle' in current_features and 'duty_cycle' in baseline_features:
                duty_deviation = abs(current_features['duty_cycle'] - 
                                   baseline_features['duty_cycle'])
                
                if duty_deviation > 0.2:  # 20% threshold
                    anomalies.append({
                        'type': 'duty_cycle_anomaly',
                        'severity': 'medium',
                        'current_value': current_features['duty_cycle'],
                        'baseline_value': baseline_features['duty_cycle'],
                        'deviation': duty_deviation
                    })
                    anomaly_score += 0.3
            
            # Check for unexpected modulation change
            if 'modulation' in current_features and 'modulation' in device_profile:
                if current_features['modulation'] != device_profile.get('modulation'):
                    anomalies.append({
                        'type': 'modulation_change',
                        'severity': 'critical',
                        'current_value': current_features['modulation'],
                        'baseline_value': device_profile.get('modulation'),
                        'deviation': None
                    })
                    anomaly_score += 0.7
            
            # Check for frequency hopping
            frequencies = [sample.get('frequency') for sample in recent_traffic if sample.get('frequency')]
            if len(set(frequencies)) > 2:
                anomalies.append({
                    'type': 'frequency_hopping',
                    'severity': 'high',
                    'current_value': len(set(frequencies)),
                    'baseline_value': 1,
                    'deviation': len(set(frequencies)) - 1
                })
                anomaly_score += 0.6
            
            # Overall anomaly classification
            anomaly_score = min(anomaly_score, 1.0)
            anomalous = anomaly_score > 0.5
            
            result = {
                'success': True,
                'device_id': device_id,
                'anomalous': anomalous,
                'anomaly_score': float(anomaly_score),
                'num_anomalies': len(anomalies),
                'anomalies': anomalies,
                'severity': self._classify_severity(anomaly_score),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            if anomalous:
                self.logger.warning(f"Anomalies detected for {device_id}: "
                                  f"score={anomaly_score:.2%}, count={len(anomalies)}")
            else:
                self.logger.debug(f"No significant anomalies for {device_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Anomaly detection error: {e}")
            return {'success': False, 'error': str(e)}
    
    def fingerprint_nbiot_device(self, device_id: str, signal_data: Dict) -> Dict[str, Any]:
        """
        Generate NB-IoT device fingerprint
        Task 2.6.3: NB-IoT/LTE-M fingerprinting
        
        Args:
            device_id: Device identifier
            signal_data: Signal characteristics
        
        Returns:
            Device fingerprint
        """
        try:
            self.logger.info(f"Fingerprinting NB-IoT device: {device_id}")
            
            fingerprint = {
                'device_id': device_id,
                'technology': 'NB-IoT',
                'frequency_mhz': signal_data.get('frequency_mhz'),
                'bandwidth_khz': signal_data.get('bandwidth_khz', 180),  # NB-IoT = 180 kHz
                'modulation': signal_data.get('modulation', 'QPSK'),
                'repetition_level': signal_data.get('repetition_level', 1),
                'coverage_enhancement': signal_data.get('coverage_enhancement', 'CE_mode_A'),
                'power_class': signal_data.get('power_class', 3),  # Class 3 = 23 dBm
                'timing_advance': signal_data.get('timing_advance', 0),
                'rssi_dbm': signal_data.get('rssi_dbm'),
                'rsrp_dbm': signal_data.get('rsrp_dbm'),
                'sinr_db': signal_data.get('sinr_db'),
                'nprach_config': signal_data.get('nprach_config', {}),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Calculate fingerprint hash (for tracking)
            fingerprint_str = f"{fingerprint['frequency_mhz']}_{fingerprint['modulation']}_" \
                            f"{fingerprint['repetition_level']}_{fingerprint['timing_advance']}"
            fingerprint['hash'] = hash(fingerprint_str) % 100000
            
            self.logger.info(f"NB-IoT fingerprint generated: hash={fingerprint['hash']}")
            
            return {
                'success': True,
                'fingerprint': fingerprint
            }
            
        except Exception as e:
            self.logger.error(f"NB-IoT fingerprinting error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_traffic_features(self, traffic_samples: List[Dict]) -> Dict[str, Any]:
        """Extract statistical features from traffic samples"""
        if not traffic_samples:
            return {}
        
        # Packet sizes
        packet_sizes = [s.get('packet_size', 0) for s in traffic_samples]
        
        # Inter-arrival times
        timestamps = [s.get('timestamp') for s in traffic_samples if s.get('timestamp')]
        inter_arrival_times = []
        if len(timestamps) > 1:
            for i in range(1, len(timestamps)):
                if isinstance(timestamps[i], datetime) and isinstance(timestamps[i-1], datetime):
                    delta = (timestamps[i] - timestamps[i-1]).total_seconds()
                    inter_arrival_times.append(delta)
        
        # RSSI values
        rssi_values = [s.get('rssi', -100) for s in traffic_samples if s.get('rssi')]
        
        # Modulation types
        modulations = [s.get('modulation') for s in traffic_samples if s.get('modulation')]
        most_common_modulation = Counter(modulations).most_common(1)[0][0] if modulations else None
        
        # Calculate features
        features = {
            'packet_size_avg': float(np.mean(packet_sizes)) if packet_sizes else 0,
            'packet_size_std': float(np.std(packet_sizes)) if packet_sizes else 0,
            'packet_size_min': float(np.min(packet_sizes)) if packet_sizes else 0,
            'packet_size_max': float(np.max(packet_sizes)) if packet_sizes else 0,
            'packet_rate': len(traffic_samples) / max(1, sum(inter_arrival_times)) if inter_arrival_times else 0,
            'inter_arrival_avg': float(np.mean(inter_arrival_times)) if inter_arrival_times else 0,
            'inter_arrival_std': float(np.std(inter_arrival_times)) if inter_arrival_times else 0,
            'duty_cycle': len([s for s in traffic_samples if s.get('active', True)]) / len(traffic_samples),
            'burst_size': self._estimate_burst_size(traffic_samples),
            'rssi_avg': float(np.mean(rssi_values)) if rssi_values else -100,
            'rssi_std': float(np.std(rssi_values)) if rssi_values else 0,
            'modulation': most_common_modulation,
            'periodicity_score': self._calculate_periodicity(inter_arrival_times) if inter_arrival_times else 0
        }
        
        return features
    
    def _estimate_burst_size(self, traffic_samples: List[Dict]) -> float:
        """Estimate average burst size from traffic samples"""
        # Simple burst detection: packets within 1 second = burst
        if not traffic_samples:
            return 0.0
        
        burst_threshold_s = 1.0
        bursts = []
        current_burst_size = 1
        
        timestamps = [s.get('timestamp') for s in traffic_samples if s.get('timestamp')]
        
        for i in range(1, len(timestamps)):
            if isinstance(timestamps[i], datetime) and isinstance(timestamps[i-1], datetime):
                delta = (timestamps[i] - timestamps[i-1]).total_seconds()
                
                if delta < burst_threshold_s:
                    current_burst_size += 1
                else:
                    bursts.append(current_burst_size)
                    current_burst_size = 1
        
        bursts.append(current_burst_size)
        
        return float(np.mean(bursts)) if bursts else 1.0
    
    def _calculate_periodicity(self, inter_arrival_times: List[float]) -> float:
        """Calculate periodicity score (0=irregular, 1=perfectly periodic)"""
        if not inter_arrival_times or len(inter_arrival_times) < 3:
            return 0.0
        
        # Coefficient of variation: lower = more periodic
        mean_iat = np.mean(inter_arrival_times)
        std_iat = np.std(inter_arrival_times)
        
        if mean_iat == 0:
            return 0.0
        
        cv = std_iat / mean_iat
        
        # Convert to periodicity score: 0.2 CV = 0.8 periodicity
        periodicity = max(0.0, 1.0 - cv)
        
        return float(periodicity)
    
    def _classify_device_ml(self, features: Dict) -> Optional[str]:
        """Classify device using ML model"""
        if not self.classifier or not SKLEARN_AVAILABLE:
            return None
        
        try:
            # Prepare feature vector
            feature_vector = [
                features.get('packet_size_avg', 0),
                features.get('packet_rate', 0),
                features.get('duty_cycle', 0),
                features.get('burst_size', 0),
                features.get('inter_arrival_avg', 0)
            ]
            
            # Scale and predict
            X = self.scaler.transform([feature_vector])
            prediction = self.classifier.predict(X)[0]
            probabilities = self.classifier.predict_proba(X)[0]
            
            # Get confidence
            features['classification_confidence'] = float(np.max(probabilities))
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"ML classification error: {e}")
            return None
    
    def _classify_device_pattern(self, features: Dict) -> str:
        """Classify device using pattern matching"""
        scores = {}
        
        for device_type, signature in self.device_signatures.items():
            score = 0.0
            
            # Check packet size
            packet_size = features.get('packet_size_avg', 0)
            if signature['packet_size_range'][0] <= packet_size <= signature['packet_size_range'][1]:
                score += 1.0
            
            # Check packet rate
            packet_rate = features.get('packet_rate', 0)
            if signature['packet_rate_range'][0] <= packet_rate <= signature['packet_rate_range'][1]:
                score += 1.0
            
            # Check duty cycle
            duty_cycle = features.get('duty_cycle', 0)
            if signature['duty_cycle_range'][0] <= duty_cycle <= signature['duty_cycle_range'][1]:
                score += 1.0
            
            # Check burst size
            burst_size = features.get('burst_size', 0)
            if signature['burst_size_range'][0] <= burst_size <= signature['burst_size_range'][1]:
                score += 1.0
            
            # Check periodicity
            periodicity = features.get('periodicity_score', 0)
            if signature['periodicity'] == 'very_high' and periodicity > 0.8:
                score += 1.0
            elif signature['periodicity'] == 'high' and periodicity > 0.6:
                score += 1.0
            elif signature['periodicity'] == 'medium' and 0.3 < periodicity <= 0.6:
                score += 1.0
            elif signature['periodicity'] == 'low' and periodicity <= 0.3:
                score += 1.0
            
            scores[device_type] = score
        
        # Return device type with highest score
        if scores:
            best_match = max(scores.items(), key=lambda x: x[1])
            features['classification_confidence'] = best_match[1] / 5.0  # Normalize to 0-1
            return best_match[0]
        
        return 'unknown'
    
    def _generate_behavior_signature(self, traffic_samples: List[Dict]) -> Dict[str, Any]:
        """Generate behavioral signature from traffic history"""
        features = self._extract_traffic_features(traffic_samples)
        
        # Active hours (histogram of activity by hour)
        timestamps = [s.get('timestamp') for s in traffic_samples if isinstance(s.get('timestamp'), datetime)]
        hour_distribution = Counter([t.hour for t in timestamps])
        
        signature = {
            'temporal_pattern': {
                'hour_distribution': dict(hour_distribution),
                'most_active_hours': [h for h, c in hour_distribution.most_common(3)],
                'periodicity': features.get('periodicity_score', 0)
            },
            'traffic_characteristics': {
                'avg_packet_size': features.get('packet_size_avg', 0),
                'packet_rate': features.get('packet_rate', 0),
                'burst_pattern': features.get('burst_size', 0),
                'duty_cycle': features.get('duty_cycle', 0)
            },
            'signal_characteristics': {
                'avg_rssi': features.get('rssi_avg', -100),
                'modulation': features.get('modulation', 'unknown')
            },
            'signature_hash': hash(str(features)) % 100000
        }
        
        return signature
    
    def _classify_severity(self, anomaly_score: float) -> str:
        """Classify anomaly severity"""
        if anomaly_score >= 0.8:
            return 'critical'
        elif anomaly_score >= 0.6:
            return 'high'
        elif anomaly_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def get_device_profile(self, device_id: str) -> Optional[Dict]:
        """Get stored device profile"""
        return self.devices.get(device_id)
    
    def list_devices(self) -> List[Dict]:
        """List all profiled devices"""
        return list(self.devices.values())
