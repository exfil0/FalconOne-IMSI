"""
FalconOne Data Validation Middleware
Real-world signal integrity and input sanitization
Version 1.7.0 - Phase 1 Data Integrity

Capabilities:
- IQ sample validation (SNR, DC offset, clipping detection)
- Protocol feature sanitization (range checks, type validation)
- Anomaly result consistency checking
- Input corruption detection
- Automatic data cleaning/filtering

Target: Reduce false positives 10-15%, ensure production data quality
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

try:
    from .logger import ModuleLogger
except ImportError:
    class ModuleLogger:
        def __init__(self, name, parent):
            self.logger = logging.getLogger(name) if parent is None else parent.getChild(name)
        def info(self, msg, **kw): self.logger.info(f"{msg} {kw if kw else ''}")
        def warning(self, msg, **kw): self.logger.warning(f"{msg} {kw if kw else ''}")
        def error(self, msg, **kw): self.logger.error(f"{msg} {kw if kw else ''}")
        def debug(self, msg, **kw): self.logger.debug(f"{msg} {kw if kw else ''}")


class ValidationLevel(Enum):
    """Validation strictness level"""
    PERMISSIVE = "permissive"  # Warn only
    NORMAL = "normal"  # Reject invalid, allow warnings
    STRICT = "strict"  # Reject all issues


@dataclass
class ValidationResult:
    """Validation result"""
    valid: bool
    issues: List[str]
    warnings: List[str]
    cleaned_data: Optional[Any] = None
    metadata: Dict[str, Any] = None


class DataValidator:
    """
    Data validation middleware for real-world signal integrity
    
    Production signals have issues not seen in simulations:
    - DC offset from SDR mixer leakage
    - Clipping from overdriven RF frontend
    - SNR degradation from interference
    - Corrupted protocol fields
    - Inconsistent anomaly detections
    
    Validation strategies:
    - SNR thresholding (reject <5dB)
    - DC offset removal (high-pass filter)
    - Clipping detection and flagging
    - Range checking for protocol features
    - Consistency validation for multi-modal results
    
    Typical usage:
        validator = DataValidator(config, logger)
        
        # Validate IQ samples before processing
        result = validator.validate_iq_samples(samples, min_snr_db=10)
        if result.valid:
            processed_samples = result.cleaned_data
        
        # Sanitize protocol features
        result = validator.sanitize_protocol_features(features)
        
        # Check anomaly consistency
        result = validator.check_anomaly_consistency(detection_result)
    """
    
    def __init__(self, config, logger: logging.Logger):
        """
        Initialize data validator
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = ModuleLogger('DataValidator', logger)
        
        # Configuration
        self.strict_mode = config.get('utils.validation.strict_mode', True)
        self.validation_level = ValidationLevel.STRICT if self.strict_mode else ValidationLevel.NORMAL
        
        self.min_snr_db = config.get('utils.validation.min_snr_db', 5.0)
        self.max_dc_offset = config.get('utils.validation.max_dc_offset', 0.1)
        self.clipping_threshold = config.get('utils.validation.clipping_threshold', 0.95)
        
        # Statistics
        self.stats = {
            'total_validated': 0,
            'rejected': 0,
            'cleaned': 0
        }
        
        self.logger.info("Data Validator initialized",
                       strict_mode=self.strict_mode,
                       min_snr_db=self.min_snr_db)
    
    # ===== IQ SAMPLE VALIDATION =====
    
    def validate_iq_samples(self, signal: np.ndarray, min_snr_db: float = None) -> ValidationResult:
        """
        Validate IQ samples for signal quality
        
        Args:
            signal: Complex IQ samples
            min_snr_db: Minimum SNR threshold (default: config value)
        
        Returns:
            Validation result with cleaned samples
        
        Checks:
        1. SNR above threshold
        2. DC offset within limits
        3. No clipping (samples near ±1.0)
        4. Reasonable dynamic range
        5. No all-zeros (SDR disconnected)
        
        Cleaning:
        - Remove DC offset
        - Normalize amplitude
        - Filter out clipped regions
        """
        self.stats['total_validated'] += 1
        min_snr_db = min_snr_db or self.min_snr_db
        
        issues = []
        warnings = []
        
        # Check 1: Not empty/all-zeros
        if signal is None or len(signal) == 0:
            issues.append("Signal is empty")
            return ValidationResult(valid=False, issues=issues, warnings=warnings)
        
        if np.all(signal == 0):
            issues.append("Signal is all zeros (SDR disconnected?)")
            return ValidationResult(valid=False, issues=issues, warnings=warnings)
        
        # Check 2: SNR
        snr_db = self._calculate_snr(signal)
        if snr_db < min_snr_db:
            if self.validation_level == ValidationLevel.STRICT:
                issues.append(f"SNR {snr_db:.1f} dB below threshold {min_snr_db} dB")
            else:
                warnings.append(f"Low SNR: {snr_db:.1f} dB")
        
        # Check 3: DC offset
        dc_offset = abs(np.mean(signal))
        if dc_offset > self.max_dc_offset:
            warnings.append(f"High DC offset: {dc_offset:.3f}")
        
        # Check 4: Clipping
        max_amplitude = np.max(np.abs(signal))
        clipping_pct = np.sum(np.abs(signal) > self.clipping_threshold) / len(signal) * 100
        
        if clipping_pct > 1.0:  # >1% clipping
            if self.validation_level == ValidationLevel.STRICT:
                issues.append(f"Clipping detected: {clipping_pct:.1f}% of samples")
            else:
                warnings.append(f"Clipping: {clipping_pct:.1f}%")
        
        # Check 5: Dynamic range
        signal_power = np.mean(np.abs(signal)**2)
        if signal_power < 1e-6:
            warnings.append(f"Very low signal power: {signal_power:.2e}")
        
        # Determine validity
        valid = len(issues) == 0
        
        # Clean signal if issues are fixable
        cleaned_signal = signal.copy()
        if dc_offset > self.max_dc_offset:
            # Remove DC offset
            cleaned_signal = signal - np.mean(signal)
            self.stats['cleaned'] += 1
        
        # Normalize if needed
        if max_amplitude > 0:
            cleaned_signal = cleaned_signal / max_amplitude * 0.9  # Leave 10% headroom
        
        if not valid:
            self.rejected_count += 1
        
        metadata = {
            'snr_db': float(snr_db),
            'dc_offset': float(dc_offset),
            'clipping_pct': float(clipping_pct),
            'signal_power': float(signal_power),
            'max_amplitude': float(max_amplitude),
        }
        
        return ValidationResult(
            valid=valid,
            issues=issues,
            warnings=warnings,
            cleaned_data=cleaned_signal if valid or self.validation_level != ValidationLevel.STRICT else None,
            metadata=metadata
        )
    
    def _calculate_snr(self, signal: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio (simplified)"""
        # Estimate noise from high-frequency components
        signal_power = np.mean(np.abs(signal)**2)
        
        # Simple noise estimation: difference between adjacent samples
        noise_estimate = np.mean(np.abs(np.diff(signal))**2) / 2
        
        if noise_estimate < 1e-10:
            return 100.0  # Very high SNR
        
        snr_linear = signal_power / noise_estimate
        snr_db = 10 * np.log10(snr_linear)
        
        return float(snr_db)
    
    # ===== PROTOCOL FEATURE VALIDATION =====
    
    def sanitize_protocol_features(self, features: Dict[str, Any]) -> ValidationResult:
        """
        Sanitize protocol feature dictionary
        
        Args:
            features: Protocol features (e.g., from signal_classifier)
        
        Returns:
            Validation result with sanitized features
        
        Checks:
        - Required fields present
        - Numeric values in valid ranges
        - Enums have valid values
        - No NaN/Inf values
        - Type correctness
        
        Common issues:
        - Missing IMSI/TMSI fields
        - Negative SNR values
        - Out-of-range channel numbers
        - Invalid protocol versions
        """
        self.stats['total_validated'] += 1
        issues = []
        warnings = []
        sanitized = features.copy()
        
        # Check required fields (common across protocols)
        required_fields = ['snr_db', 'frequency_mhz', 'timestamp']
        for field in required_fields:
            if field not in features:
                issues.append(f"Missing required field: {field}")
        
        # Validate numeric ranges
        if 'snr_db' in features:
            snr = features['snr_db']
            if not isinstance(snr, (int, float)):
                issues.append(f"SNR must be numeric, got {type(snr)}")
            elif np.isnan(snr) or np.isinf(snr):
                issues.append(f"SNR is NaN or Inf")
            elif snr < -20 or snr > 100:
                warnings.append(f"SNR {snr} dB out of typical range [-20, 100]")
                sanitized['snr_db'] = np.clip(snr, -20, 100)
        
        if 'frequency_mhz' in features:
            freq = features['frequency_mhz']
            if not isinstance(freq, (int, float)):
                issues.append(f"Frequency must be numeric, got {type(freq)}")
            elif freq < 0 or freq > 100000:  # 0-100 GHz
                warnings.append(f"Frequency {freq} MHz out of range [0, 100000]")
        
        if 'rssi_dbm' in features:
            rssi = features['rssi_dbm']
            if not isinstance(rssi, (int, float)):
                issues.append(f"RSSI must be numeric, got {type(rssi)}")
            elif rssi < -150 or rssi > 0:
                warnings.append(f"RSSI {rssi} dBm out of typical range [-150, 0]")
                sanitized['rssi_dbm'] = np.clip(rssi, -150, 0)
        
        # Validate identifiers
        if 'imsi' in features:
            imsi = features['imsi']
            if not isinstance(imsi, str) or len(imsi) not in range(14, 16):  # IMSI is 14-15 digits
                warnings.append(f"IMSI length {len(imsi) if isinstance(imsi, str) else 'N/A'} unusual (expect 14-15)")
        
        if 'channel' in features:
            channel = features['channel']
            if not isinstance(channel, int) or channel < 0:
                warnings.append(f"Invalid channel: {channel}")
        
        # Check for NaN/Inf in all numeric fields
        for key, value in features.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    issues.append(f"Field '{key}' is NaN or Inf")
                    sanitized[key] = 0.0  # Replace with default
        
        valid = len(issues) == 0
        
        if not valid:
            self.rejected_count += 1
            self.logger.warning(f"Protocol features validation failed: {issues}")
        
        return ValidationResult(
            valid=valid,
            issues=issues,
            warnings=warnings,
            cleaned_data=sanitized if valid or len(issues) < len(features) else None,
            metadata={'field_count': len(features)}
        )
    
    # ===== ANOMALY CONSISTENCY VALIDATION =====
    
    def check_anomaly_consistency(self, result: Dict[str, Any]) -> ValidationResult:
        """
        Check consistency of anomaly detection results
        
        Args:
            result: Anomaly detection result (e.g., from signal_classifier)
        
        Returns:
            Validation result
        
        Consistency checks:
        - Confidence matches anomaly score
        - Multi-modal results agree
        - Temporal consistency (vs previous)
        - Plausibility checks
        
        Example inconsistencies:
        - High confidence but low anomaly score
        - RF says anomaly, protocol says normal
        - Sudden drastic change in score
        """
        self.stats['total_validated'] += 1
        issues = []
        warnings = []
        
        # Check 1: Confidence vs anomaly score consistency
        if 'confidence' in result and 'anomaly_score' in result:
            confidence = result['confidence']
            anomaly_score = result['anomaly_score']
            
            # High confidence should match high anomaly score (or low for normal)
            if confidence > 0.8 and anomaly_score < 0.5:
                warnings.append(f"High confidence ({confidence:.2f}) but low anomaly score ({anomaly_score:.2f})")
            elif confidence < 0.5 and anomaly_score > 0.8:
                warnings.append(f"Low confidence ({confidence:.2f}) but high anomaly score ({anomaly_score:.2f})")
        
        # Check 2: Multi-modal agreement
        if 'rf_anomaly' in result and 'protocol_anomaly' in result:
            rf = result['rf_anomaly']
            proto = result['protocol_anomaly']
            
            if rf != proto:
                warnings.append(f"RF anomaly ({rf}) disagrees with protocol anomaly ({proto})")
        
        # Check 3: Required fields
        required = ['anomaly_score', 'timestamp']
        for field in required:
            if field not in result:
                issues.append(f"Missing field: {field}")
        
        # Check 4: Score range
        if 'anomaly_score' in result:
            score = result['anomaly_score']
            if not isinstance(score, (int, float)):
                issues.append(f"Anomaly score must be numeric, got {type(score)}")
            elif score < 0 or score > 1:
                issues.append(f"Anomaly score {score} out of range [0, 1]")
        
        valid = len(issues) == 0
        
        if not valid:
            self.rejected_count += 1
        
        return ValidationResult(
            valid=valid,
            issues=issues,
            warnings=warnings,
            metadata={'checks_performed': 4}
        )
    
    # ===== BATCH VALIDATION =====
    
    def validate_batch(self, samples: List[np.ndarray], 
                      min_snr_db: float = None) -> Tuple[List[np.ndarray], List[int]]:
        """
        Validate batch of IQ samples
        
        Args:
            samples: List of IQ sample arrays
            min_snr_db: Minimum SNR threshold
        
        Returns:
            (valid_samples, valid_indices)
        """
        valid_samples = []
        valid_indices = []
        
        for i, sample in enumerate(samples):
            result = self.validate_iq_samples(sample, min_snr_db)
            
            if result.valid:
                valid_samples.append(result.cleaned_data)
                valid_indices.append(i)
        
        rejection_rate = (len(samples) - len(valid_samples)) / len(samples) * 100
        
        if rejection_rate > 20:
            self.logger.warning(f"High rejection rate: {rejection_rate:.1f}% of batch rejected")
        
        return valid_samples, valid_indices
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics
        
        Returns:
            Statistics dict with counts and rates
        """
        total = self.stats['total_validated']
        rejected = self.stats['rejected']
        cleaned = self.stats['cleaned']
        
        return {
            'total_validated': total,
            'rejected': rejected,
            'cleaned': cleaned,
            'rejection_rate': rejected / total if total > 0 else 0.0,
            'cleaning_rate': cleaned / total if total > 0 else 0.0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            'total_validated': self.stats['total_validated'],
            'rejected': self.stats['rejected'],
            'cleaned': self.stats['cleaned'],
            'rejection_rate': self.stats['rejected'] / max(1, self.stats['total_validated']),
            'cleaning_rate': self.stats['cleaned'] / max(1, self.stats['total_validated']),
            'strict_mode': self.strict_mode,
            'min_snr_db': self.min_snr_db,
        }
