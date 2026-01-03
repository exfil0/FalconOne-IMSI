"""
FalconOne Concept Drift Detection Module
=========================================

Advanced statistical drift detection algorithms for online ML adaptation.
Implements ADWIN, Page-Hinkley, DDM, EDDM, and KSWIN for robust drift detection.

Author: FalconOne Development Team
Version: 1.9.4
"""

import numpy as np
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Callable, Deque
from collections import deque
from enum import Enum
from scipy import stats

logger = logging.getLogger(__name__)


class DriftSeverity(Enum):
    """Drift severity levels"""
    NONE = "none"
    WARNING = "warning"
    DRIFT = "drift"
    SEVERE = "severe"


@dataclass
class DriftResult:
    """Result of drift detection analysis"""
    detected: bool
    severity: DriftSeverity
    confidence: float
    drift_point: Optional[int]  # Sample index where drift was detected
    statistics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class BaseDriftDetector(ABC):
    """
    Abstract base class for drift detection algorithms.
    """
    
    def __init__(self, name: str = "BaseDetector"):
        self.name = name
        self._sample_count = 0
        self._drift_count = 0
        self._last_drift_point: Optional[int] = None
        self._history: List[DriftResult] = []
    
    @abstractmethod
    def add_element(self, value: float) -> DriftResult:
        """Add new element and check for drift"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset detector state"""
        pass
    
    @property
    def drift_detected(self) -> bool:
        """Check if drift is currently detected"""
        if not self._history:
            return False
        return self._history[-1].detected
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            'name': self.name,
            'samples_processed': self._sample_count,
            'drift_count': self._drift_count,
            'last_drift_point': self._last_drift_point,
        }


class ADWINDetector(BaseDriftDetector):
    """
    ADWIN (ADaptive WINdowing) drift detector.
    
    Maintains a variable-length window of recent items, automatically growing
    when no change is detected and shrinking when change is detected.
    
    Based on: "Learning from Time-Changing Data with Adaptive Windowing"
    by Albert Bifet and Ricard Gavaldà (2007)
    
    Parameters:
        delta: Confidence parameter (default 0.002)
        clock: Number of samples between cut checks (default 32)
        max_buckets: Maximum number of buckets per level (default 5)
        min_window_length: Minimum window length (default 5)
    """
    
    def __init__(
        self,
        delta: float = 0.002,
        clock: int = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
    ):
        super().__init__(name="ADWIN")
        self.delta = delta
        self.clock = clock
        self.max_buckets = max_buckets
        self.min_window_length = min_window_length
        
        # Bucket structures
        self._bucket_totals: List[List[float]] = [[]]
        self._bucket_variances: List[List[float]] = [[]]
        self._bucket_sizes: List[List[int]] = [[]]
        
        # Statistics
        self._total = 0.0
        self._variance = 0.0
        self._width = 0
        self._mint_time = 0
        
    def add_element(self, value: float) -> DriftResult:
        """Add element and check for drift using ADWIN algorithm"""
        self._sample_count += 1
        
        # Insert new element as bucket
        self._insert_element(value)
        
        # Compress buckets
        self._compress_buckets()
        
        # Check for drift
        drift_detected = False
        drift_point = None
        confidence = 0.0
        cut_point = None
        
        if self._width >= self.min_window_length:
            if self._sample_count % self.clock == 0:
                drift_detected, cut_point, confidence = self._detect_change()
                if drift_detected:
                    self._delete_elements(cut_point)
                    self._drift_count += 1
                    self._last_drift_point = self._sample_count
                    drift_point = self._sample_count
        
        result = DriftResult(
            detected=drift_detected,
            severity=DriftSeverity.DRIFT if drift_detected else DriftSeverity.NONE,
            confidence=confidence,
            drift_point=drift_point,
            statistics={
                'window_width': self._width,
                'mean': self.mean,
                'variance': self.variance,
                'total': self._total,
            }
        )
        
        self._history.append(result)
        return result
    
    def _insert_element(self, value: float):
        """Insert new element as a new bucket at level 0"""
        self._width += 1
        self._total += value
        
        # Update running variance using Welford's algorithm
        if self._width > 1:
            delta = value - (self._total - value) / (self._width - 1)
            self._variance += delta * (value - self._total / self._width)
        
        # Add to bucket level 0
        self._bucket_totals[0].append(value)
        self._bucket_variances[0].append(0.0)
        self._bucket_sizes[0].append(1)
    
    def _compress_buckets(self):
        """Compress buckets when there are too many at a level"""
        level = 0
        while level < len(self._bucket_totals):
            if len(self._bucket_totals[level]) > self.max_buckets:
                # Merge two oldest buckets
                n1 = self._bucket_sizes[level][0]
                n2 = self._bucket_sizes[level][1]
                u1 = self._bucket_totals[level][0] / n1
                u2 = self._bucket_totals[level][1] / n2
                
                # Combined statistics
                n_new = n1 + n2
                total_new = self._bucket_totals[level][0] + self._bucket_totals[level][1]
                
                # Combined variance
                var_new = (
                    self._bucket_variances[level][0] +
                    self._bucket_variances[level][1] +
                    n1 * n2 * (u1 - u2) ** 2 / n_new
                )
                
                # Remove old buckets
                del self._bucket_totals[level][:2]
                del self._bucket_variances[level][:2]
                del self._bucket_sizes[level][:2]
                
                # Ensure next level exists
                if level + 1 >= len(self._bucket_totals):
                    self._bucket_totals.append([])
                    self._bucket_variances.append([])
                    self._bucket_sizes.append([])
                
                # Add to next level
                self._bucket_totals[level + 1].append(total_new)
                self._bucket_variances[level + 1].append(var_new)
                self._bucket_sizes[level + 1].append(n_new)
                
            level += 1
    
    def _detect_change(self) -> Tuple[bool, Optional[int], float]:
        """Detect change using statistical test"""
        if self._width < 2 * self.min_window_length:
            return False, None, 0.0
        
        n0 = 0
        n1 = self._width
        u0 = 0.0
        u1 = self._total / self._width
        
        max_cut = None
        max_confidence = 0.0
        
        # Iterate through possible cut points
        for level in range(len(self._bucket_totals)):
            for i in range(len(self._bucket_totals[level])):
                bucket_n = self._bucket_sizes[level][i]
                bucket_total = self._bucket_totals[level][i]
                
                n0 += bucket_n
                n1 -= bucket_n
                u0 = (u0 * (n0 - bucket_n) + bucket_total) / n0 if n0 > 0 else 0
                u1 = (u1 * (n1 + bucket_n) - bucket_total) / n1 if n1 > 0 else 0
                
                if n0 >= self.min_window_length and n1 >= self.min_window_length:
                    # Calculate bound
                    m = 1.0 / (1.0 / n0 + 1.0 / n1)
                    delta_prime = self.delta / np.log(self._width)
                    
                    epsilon = np.sqrt(
                        (0.5 / m) * np.log(4.0 / delta_prime)
                    )
                    
                    difference = abs(u0 - u1)
                    confidence = difference / epsilon if epsilon > 0 else 0
                    
                    if difference >= epsilon:
                        if confidence > max_confidence:
                            max_confidence = confidence
                            max_cut = n0
        
        if max_cut is not None:
            return True, max_cut, min(max_confidence, 1.0)
        
        return False, None, 0.0
    
    def _delete_elements(self, cut_point: int):
        """Delete elements before cut point"""
        deleted = 0
        level = 0
        
        while deleted < cut_point and level < len(self._bucket_totals):
            while (
                deleted < cut_point and
                len(self._bucket_totals[level]) > 0 and
                deleted + self._bucket_sizes[level][0] <= cut_point
            ):
                # Remove bucket
                bucket_n = self._bucket_sizes[level][0]
                bucket_total = self._bucket_totals[level][0]
                bucket_var = self._bucket_variances[level][0]
                
                self._width -= bucket_n
                self._total -= bucket_total
                self._variance -= bucket_var
                
                del self._bucket_totals[level][0]
                del self._bucket_variances[level][0]
                del self._bucket_sizes[level][0]
                
                deleted += bucket_n
            
            level += 1
    
    @property
    def mean(self) -> float:
        """Current window mean"""
        return self._total / self._width if self._width > 0 else 0.0
    
    @property
    def variance(self) -> float:
        """Current window variance"""
        return self._variance / self._width if self._width > 1 else 0.0
    
    @property
    def width(self) -> int:
        """Current window width"""
        return self._width
    
    def reset(self):
        """Reset detector"""
        self._bucket_totals = [[]]
        self._bucket_variances = [[]]
        self._bucket_sizes = [[]]
        self._total = 0.0
        self._variance = 0.0
        self._width = 0
        self._sample_count = 0
        self._history = []


class PageHinkleyDetector(BaseDriftDetector):
    """
    Page-Hinkley Test for drift detection.
    
    Detects change in the average of a Gaussian signal. Useful for 
    detecting gradual concept drift.
    
    Based on: "Continuous inspection schemes" by E.S. Page (1954)
    
    Parameters:
        min_instances: Minimum samples before detection (default 30)
        delta: Magnitude of acceptable changes (default 0.005)
        threshold: Decision threshold for change detection (default 50)
        alpha: Forgetting factor for gradual changes (default 0.9999)
    """
    
    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
    ):
        super().__init__(name="Page-Hinkley")
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        
        # Running statistics
        self._x_mean = 0.0
        self._sum = 0.0
        self._min_sum = float('inf')
        
    def add_element(self, value: float) -> DriftResult:
        """Add element and check for drift using Page-Hinkley test"""
        self._sample_count += 1
        
        # Update running mean with forgetting
        self._x_mean = self._x_mean * self.alpha + value * (1 - self.alpha)
        
        # Update cumulative sum
        self._sum += value - self._x_mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)
        
        # Calculate test statistic
        ph_value = self._sum - self._min_sum
        
        # Check for drift
        drift_detected = False
        drift_point = None
        
        if self._sample_count >= self.min_instances:
            if ph_value > self.threshold:
                drift_detected = True
                drift_point = self._sample_count
                self._drift_count += 1
                self._last_drift_point = self._sample_count
                self.reset()
        
        # Calculate confidence
        confidence = min(ph_value / self.threshold, 1.0) if self.threshold > 0 else 0.0
        
        # Determine severity
        if drift_detected:
            severity = DriftSeverity.DRIFT
        elif confidence > 0.7:
            severity = DriftSeverity.WARNING
        else:
            severity = DriftSeverity.NONE
        
        result = DriftResult(
            detected=drift_detected,
            severity=severity,
            confidence=confidence,
            drift_point=drift_point,
            statistics={
                'ph_value': ph_value,
                'threshold': self.threshold,
                'mean': self._x_mean,
                'cumulative_sum': self._sum,
                'min_sum': self._min_sum,
            }
        )
        
        self._history.append(result)
        return result
    
    def reset(self):
        """Reset detector"""
        self._x_mean = 0.0
        self._sum = 0.0
        self._min_sum = float('inf')
        self._sample_count = 0
        self._history = []


class DDMDetector(BaseDriftDetector):
    """
    DDM (Drift Detection Method) detector.
    
    Uses the error rate of a classifier as indicator. When error rate
    plus standard deviation exceeds thresholds, drift is signaled.
    
    Based on: "Learning with Drift Detection" by Gama et al. (2004)
    
    Parameters:
        min_instances: Minimum samples before detection (default 30)
        warning_level: Warning threshold sigma multiplier (default 2.0)
        out_control_level: Drift threshold sigma multiplier (default 3.0)
    """
    
    def __init__(
        self,
        min_instances: int = 30,
        warning_level: float = 2.0,
        out_control_level: float = 3.0,
    ):
        super().__init__(name="DDM")
        self.min_instances = min_instances
        self.warning_level = warning_level
        self.out_control_level = out_control_level
        
        # Statistics
        self._p_min = float('inf')
        self._s_min = float('inf')
        self._p = 0.0
        self._s = 0.0
        self._warning_level_reached = False
        self._warning_point: Optional[int] = None
        
    def add_element(self, value: float) -> DriftResult:
        """
        Add prediction result (0=correct, 1=error) and check for drift.
        
        Note: Unlike other detectors, DDM expects binary values representing
        classification errors (1) or correct predictions (0).
        """
        self._sample_count += 1
        
        # Update error probability
        self._p += (value - self._p) / self._sample_count
        self._s = np.sqrt(self._p * (1 - self._p) / self._sample_count)
        
        drift_detected = False
        drift_point = None
        severity = DriftSeverity.NONE
        
        if self._sample_count >= self.min_instances:
            # Update minimums
            if self._p + self._s < self._p_min + self._s_min:
                self._p_min = self._p
                self._s_min = self._s
            
            # Calculate levels
            warning_threshold = self._p_min + self.warning_level * self._s_min
            drift_threshold = self._p_min + self.out_control_level * self._s_min
            
            current_level = self._p + self._s
            
            # Check drift
            if current_level >= drift_threshold:
                drift_detected = True
                drift_point = self._sample_count
                severity = DriftSeverity.DRIFT
                self._drift_count += 1
                self._last_drift_point = self._sample_count
                self.reset()
            elif current_level >= warning_threshold:
                severity = DriftSeverity.WARNING
                if not self._warning_level_reached:
                    self._warning_level_reached = True
                    self._warning_point = self._sample_count
            else:
                self._warning_level_reached = False
                self._warning_point = None
        
        # Calculate confidence
        if self._p_min + self._s_min > 0:
            confidence = min(
                (self._p + self._s) / (self._p_min + self.out_control_level * self._s_min),
                1.0
            )
        else:
            confidence = 0.0
        
        result = DriftResult(
            detected=drift_detected,
            severity=severity,
            confidence=confidence,
            drift_point=drift_point,
            statistics={
                'error_rate': self._p,
                'std_error': self._s,
                'min_error_rate': self._p_min,
                'min_std_error': self._s_min,
                'warning_active': self._warning_level_reached,
            }
        )
        
        self._history.append(result)
        return result
    
    def reset(self):
        """Reset detector"""
        self._p_min = float('inf')
        self._s_min = float('inf')
        self._p = 0.0
        self._s = 0.0
        self._warning_level_reached = False
        self._warning_point = None
        self._sample_count = 0
        self._history = []


class KSWINDetector(BaseDriftDetector):
    """
    KSWIN (Kolmogorov-Smirnov WINdowing) detector.
    
    Uses Kolmogorov-Smirnov test to compare distributions between
    recent and reference windows.
    
    Parameters:
        alpha: Significance level for KS test (default 0.005)
        window_size: Size of each window (default 100)
        stat_size: Size of the statistic window (default 30)
    """
    
    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
    ):
        super().__init__(name="KSWIN")
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        
        self._window: Deque[float] = deque(maxlen=window_size)
        
    def add_element(self, value: float) -> DriftResult:
        """Add element and check for drift using KS test"""
        self._sample_count += 1
        self._window.append(value)
        
        drift_detected = False
        drift_point = None
        p_value = 1.0
        ks_statistic = 0.0
        
        if len(self._window) >= self.window_size:
            window_list = list(self._window)
            
            # Split into reference and recent
            reference = window_list[:-self.stat_size]
            recent = window_list[-self.stat_size:]
            
            # Kolmogorov-Smirnov test
            try:
                ks_statistic, p_value = stats.ks_2samp(reference, recent)
                
                if p_value < self.alpha:
                    drift_detected = True
                    drift_point = self._sample_count
                    self._drift_count += 1
                    self._last_drift_point = self._sample_count
                    # Reset window after drift
                    self._window = deque(recent, maxlen=self.window_size)
            except Exception as e:
                logger.warning(f"KS test failed: {e}")
        
        # Calculate confidence
        confidence = 1.0 - p_value
        
        # Determine severity
        if drift_detected:
            severity = DriftSeverity.DRIFT
        elif confidence > 0.9:
            severity = DriftSeverity.WARNING
        else:
            severity = DriftSeverity.NONE
        
        result = DriftResult(
            detected=drift_detected,
            severity=severity,
            confidence=confidence,
            drift_point=drift_point,
            statistics={
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'window_size': len(self._window),
                'alpha': self.alpha,
            }
        )
        
        self._history.append(result)
        return result
    
    def reset(self):
        """Reset detector"""
        self._window = deque(maxlen=self.window_size)
        self._sample_count = 0
        self._history = []


class EnsembleDriftDetector:
    """
    Ensemble of multiple drift detectors for robust detection.
    
    Combines multiple detection algorithms and uses voting or
    confidence aggregation to make final drift decisions.
    
    Parameters:
        detectors: List of drift detectors to use
        voting_threshold: Proportion of detectors that must agree (default 0.5)
        aggregation: Method for combining results ('voting', 'max', 'mean')
    """
    
    def __init__(
        self,
        detectors: Optional[List[BaseDriftDetector]] = None,
        voting_threshold: float = 0.5,
        aggregation: str = 'voting',
    ):
        self.voting_threshold = voting_threshold
        self.aggregation = aggregation
        
        # Default detector ensemble
        self.detectors = detectors or [
            ADWINDetector(delta=0.002),
            PageHinkleyDetector(threshold=50),
            DDMDetector(),
            KSWINDetector(alpha=0.005),
        ]
        
        self._sample_count = 0
        self._drift_count = 0
        self._history: List[Dict[str, Any]] = []
        
    def add_element(self, value: float, is_error: bool = False) -> Dict[str, Any]:
        """
        Add element to all detectors and aggregate results.
        
        Args:
            value: The observation value
            is_error: Whether this is a classification error (for DDM)
            
        Returns:
            Aggregated drift detection result
        """
        self._sample_count += 1
        
        results = {}
        drift_votes = 0
        warning_votes = 0
        confidences = []
        
        for detector in self.detectors:
            # DDM needs error indicator, others use raw value
            if isinstance(detector, DDMDetector):
                result = detector.add_element(float(is_error))
            else:
                result = detector.add_element(value)
            
            results[detector.name] = result
            
            if result.detected:
                drift_votes += 1
            elif result.severity == DriftSeverity.WARNING:
                warning_votes += 1
            
            confidences.append(result.confidence)
        
        # Aggregate
        n_detectors = len(self.detectors)
        
        if self.aggregation == 'voting':
            ensemble_drift = drift_votes / n_detectors >= self.voting_threshold
            ensemble_confidence = drift_votes / n_detectors
        elif self.aggregation == 'max':
            ensemble_drift = max(confidences) > 0.9
            ensemble_confidence = max(confidences)
        else:  # mean
            ensemble_drift = np.mean(confidences) > 0.8
            ensemble_confidence = np.mean(confidences)
        
        # Determine severity
        if ensemble_drift:
            severity = DriftSeverity.DRIFT
            self._drift_count += 1
        elif warning_votes / n_detectors >= 0.5:
            severity = DriftSeverity.WARNING
        else:
            severity = DriftSeverity.NONE
        
        aggregated = {
            'detected': ensemble_drift,
            'severity': severity.value,
            'confidence': ensemble_confidence,
            'sample_count': self._sample_count,
            'individual_results': {
                name: {
                    'detected': r.detected,
                    'severity': r.severity.value,
                    'confidence': r.confidence,
                }
                for name, r in results.items()
            },
            'votes': {
                'drift': drift_votes,
                'warning': warning_votes,
                'total': n_detectors,
            },
        }
        
        self._history.append(aggregated)
        return aggregated
    
    def reset(self):
        """Reset all detectors"""
        for detector in self.detectors:
            detector.reset()
        self._sample_count = 0
        self._history = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ensemble statistics"""
        return {
            'samples_processed': self._sample_count,
            'drift_count': self._drift_count,
            'detectors': [d.get_statistics() for d in self.detectors],
        }


class AdaptiveDriftManager:
    """
    High-level manager for drift detection and adaptation.
    
    Integrates with OnlineAdaptationManager to provide comprehensive
    drift detection and model adaptation capabilities.
    
    Features:
    - Multiple drift detection algorithms
    - Automatic adaptation strategies
    - Performance monitoring
    - Drift history and analytics
    """
    
    def __init__(
        self,
        sensitivity: str = 'medium',
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        Initialize AdaptiveDriftManager.
        
        Args:
            sensitivity: Detection sensitivity ('low', 'medium', 'high')
            callbacks: List of callbacks to invoke on drift
        """
        self.sensitivity = sensitivity
        self.callbacks = callbacks or []
        
        # Configure detectors based on sensitivity
        self._configure_detectors()
        
        # Metrics tracking
        self._performance_window: Deque[float] = deque(maxlen=1000)
        self._drift_events: List[Dict[str, Any]] = []
        self._adaptation_history: List[Dict[str, Any]] = []
        
        # State
        self._current_drift_state = DriftSeverity.NONE
        self._last_adaptation_time = 0.0
        self._consecutive_warnings = 0
        
    def _configure_detectors(self):
        """Configure detectors based on sensitivity level"""
        if self.sensitivity == 'high':
            # More sensitive - faster detection but more false positives
            self.ensemble = EnsembleDriftDetector(
                detectors=[
                    ADWINDetector(delta=0.001),
                    PageHinkleyDetector(threshold=30),
                    DDMDetector(warning_level=1.5, out_control_level=2.5),
                    KSWINDetector(alpha=0.01),
                ],
                voting_threshold=0.25,  # Only 1/4 need to agree
            )
        elif self.sensitivity == 'low':
            # Less sensitive - fewer false positives but slower detection
            self.ensemble = EnsembleDriftDetector(
                detectors=[
                    ADWINDetector(delta=0.005),
                    PageHinkleyDetector(threshold=100),
                    DDMDetector(warning_level=2.5, out_control_level=4.0),
                    KSWINDetector(alpha=0.001),
                ],
                voting_threshold=0.75,  # 3/4 must agree
            )
        else:  # medium (default)
            self.ensemble = EnsembleDriftDetector(
                voting_threshold=0.5,
            )
    
    def observe(
        self,
        prediction_correct: bool,
        accuracy: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Observe a prediction result and check for drift.
        
        Args:
            prediction_correct: Whether prediction was correct
            accuracy: Optional current accuracy metric
            
        Returns:
            Drift status and recommendations
        """
        # Use accuracy if provided, otherwise use error rate
        value = accuracy if accuracy is not None else (1.0 if prediction_correct else 0.0)
        self._performance_window.append(value)
        
        # Check drift
        result = self.ensemble.add_element(value, is_error=not prediction_correct)
        
        # Track state transitions
        new_state = DriftSeverity[result['severity'].upper()]
        old_state = self._current_drift_state
        
        # Handle warnings
        if new_state == DriftSeverity.WARNING:
            self._consecutive_warnings += 1
        else:
            self._consecutive_warnings = 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(result, new_state, old_state)
        
        # Handle drift detection
        if new_state == DriftSeverity.DRIFT or (
            new_state == DriftSeverity.WARNING and self._consecutive_warnings >= 5
        ):
            self._handle_drift(result)
        
        self._current_drift_state = new_state
        
        return {
            **result,
            'recommendations': recommendations,
            'consecutive_warnings': self._consecutive_warnings,
            'current_performance': np.mean(list(self._performance_window)[-100:]) if len(self._performance_window) > 0 else 0.0,
        }
    
    def _generate_recommendations(
        self,
        result: Dict[str, Any],
        new_state: DriftSeverity,
        old_state: DriftSeverity,
    ) -> List[str]:
        """Generate adaptation recommendations"""
        recommendations = []
        
        if new_state == DriftSeverity.DRIFT:
            recommendations.extend([
                "URGENT: Significant concept drift detected",
                "Increase model learning rate temporarily",
                "Consider resetting to previous checkpoint",
                "Enable aggressive experience replay",
                "Review recent data distribution changes",
            ])
        elif new_state == DriftSeverity.WARNING:
            if self._consecutive_warnings >= 3:
                recommendations.extend([
                    "WARNING: Persistent degradation detected",
                    "Monitor closely for full drift",
                    "Increase experience replay frequency",
                ])
            else:
                recommendations.extend([
                    "Minor degradation detected",
                    "Continue monitoring",
                ])
        elif old_state == DriftSeverity.DRIFT and new_state == DriftSeverity.NONE:
            recommendations.extend([
                "Drift recovery detected",
                "Consider reducing learning rate",
                "Save current model as checkpoint",
            ])
        
        return recommendations
    
    def _handle_drift(self, result: Dict[str, Any]):
        """Handle drift event"""
        event = {
            'timestamp': time.time(),
            'sample_count': result['sample_count'],
            'confidence': result['confidence'],
            'individual_votes': result['votes'],
            'performance_at_drift': np.mean(list(self._performance_window)[-50:]) if len(self._performance_window) > 0 else 0.0,
        }
        
        self._drift_events.append(event)
        logger.warning(f"Concept drift detected: {event}")
        
        # Invoke callbacks
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Drift callback error: {e}")
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection"""
        return {
            'current_state': self._current_drift_state.value,
            'total_drifts': len(self._drift_events),
            'consecutive_warnings': self._consecutive_warnings,
            'detector_stats': self.ensemble.get_statistics(),
            'recent_performance': np.mean(list(self._performance_window)[-100:]) if len(self._performance_window) > 0 else 0.0,
            'drift_events': self._drift_events[-10:],  # Last 10 events
        }
    
    def reset(self):
        """Reset manager state"""
        self.ensemble.reset()
        self._performance_window.clear()
        self._current_drift_state = DriftSeverity.NONE
        self._consecutive_warnings = 0


# Prometheus metrics integration
try:
    from prometheus_client import Counter, Gauge, Histogram
    
    DRIFT_EVENTS = Counter(
        'falconone_drift_events_total',
        'Total number of drift events detected',
        ['detector', 'severity']
    )
    
    DRIFT_CONFIDENCE = Gauge(
        'falconone_drift_confidence',
        'Current drift detection confidence',
        ['detector']
    )
    
    PERFORMANCE_METRIC = Gauge(
        'falconone_model_performance',
        'Current model performance metric'
    )
    
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


def create_drift_detector(
    algorithm: str = 'ensemble',
    **kwargs
) -> BaseDriftDetector:
    """
    Factory function to create drift detectors.
    
    Args:
        algorithm: Algorithm name ('adwin', 'page_hinkley', 'ddm', 'kswin', 'ensemble')
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Configured drift detector instance
    """
    algorithms = {
        'adwin': ADWINDetector,
        'page_hinkley': PageHinkleyDetector,
        'ddm': DDMDetector,
        'kswin': KSWINDetector,
    }
    
    if algorithm == 'ensemble':
        return EnsembleDriftDetector(**kwargs)
    elif algorithm in algorithms:
        return algorithms[algorithm](**kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: {list(algorithms.keys()) + ['ensemble']}")
