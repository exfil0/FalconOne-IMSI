"""
Environmental Adaptation Module - v1.7.0 Phase 1
==============================================
Adapts geolocation to real-world environmental conditions:
- Urban multipath compensation (NLOS detection)
- Adaptive noise filtering (Kalman/Wiener)
- NTN Doppler shift correction (satellite motion)
- Weather impact modeling (rain attenuation, ducting)

Target: +20-30% geolocation accuracy improvement
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging


@dataclass
class EnvironmentalConditions:
    """Real-world environmental parameters"""
    temperature_celsius: float = 20.0
    humidity_percent: float = 50.0
    rain_rate_mm_per_hour: float = 0.0
    wind_speed_m_per_s: float = 0.0
    atmospheric_pressure_hpa: float = 1013.25
    visibility_km: float = 10.0
    urban_density: str = "suburban"  # urban, suburban, rural


@dataclass
class PropagationMetrics:
    """Signal propagation characteristics"""
    los_probability: float  # Line-of-sight probability
    multipath_delay_spread_ns: float  # RMS delay spread
    doppler_shift_hz: float  # Doppler shift from motion
    path_loss_db: float  # Total path loss
    snr_degradation_db: float  # Environmental SNR loss


class EnvironmentalAdapter:
    """
    Adapts geolocation and signal processing to environmental conditions.
    
    Features:
    - Urban multipath detection and compensation
    - Adaptive Kalman filtering for noise
    - NTN Doppler correction using satellite ephemeris
    - Weather impact on signal propagation
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Multipath detection parameters
        self.nlos_threshold_db = 6.0  # NLOS detection threshold
        self.multipath_history = []
        self.max_history = 100
        
        # Kalman filter state
        self.kalman_state = None
        self.kalman_covariance = None
        self.process_noise = 0.01
        self.measurement_noise = 0.1
        
        # NTN parameters
        self.satellite_velocity_m_s = 7500.0  # Typical LEO velocity
        self.carrier_frequency_hz = 2.4e9  # Default frequency
        
        # Weather models
        self.rain_attenuation_model = "ITU-R P.838"
        
        self.logger.info("[EnvAdapter] Environmental adaptation initialized")
    
    def adapt_location(
        self,
        raw_location: Tuple[float, float],  # (lat, lon)
        signal_metrics: Dict,
        conditions: EnvironmentalConditions
    ) -> Tuple[Tuple[float, float], PropagationMetrics]:
        """
        Adapt raw geolocation estimate to environmental conditions.
        
        Args:
            raw_location: Raw (latitude, longitude) estimate
            signal_metrics: Signal strength, timing, multipath indicators
            conditions: Current environmental conditions
            
        Returns:
            (adapted_location, propagation_metrics)
        """
        # Detect multipath and NLOS conditions
        is_nlos, multipath_delay = self._detect_multipath(signal_metrics)
        
        # Calculate propagation metrics
        prop_metrics = self._calculate_propagation_metrics(
            signal_metrics, conditions, is_nlos, multipath_delay
        )
        
        # Apply multipath compensation
        if is_nlos:
            adapted_location = self._compensate_multipath(
                raw_location, multipath_delay, signal_metrics
            )
            self.logger.debug(
                f"[EnvAdapter] NLOS detected, compensated location: "
                f"{adapted_location[0]:.6f}, {adapted_location[1]:.6f}"
            )
        else:
            adapted_location = raw_location
        
        # Apply Kalman filtering for smooth tracking
        adapted_location = self._kalman_filter(adapted_location, conditions)
        
        return adapted_location, prop_metrics
    
    def _detect_multipath(
        self, signal_metrics: Dict
    ) -> Tuple[bool, float]:
        """
        Detect NLOS/multipath conditions using signal characteristics.
        
        Methods:
        - Excess delay detection (delay > threshold)
        - Signal strength variation (rapid fading)
        - Phase coherence loss
        
        Returns:
            (is_nlos, multipath_delay_ns)
        """
        # Extract metrics
        rssi_db = signal_metrics.get("rssi_db", -80.0)
        snr_db = signal_metrics.get("snr_db", 10.0)
        timing_advance_us = signal_metrics.get("timing_advance_us", 0.0)
        
        # Simple NLOS detection: low SNR + high timing variation
        multipath_delay_ns = 0.0
        is_nlos = False
        
        # Add to history for variance calculation
        self.multipath_history.append({
            "rssi": rssi_db,
            "snr": snr_db,
            "timing": timing_advance_us,
            "timestamp": datetime.utcnow()
        })
        
        if len(self.multipath_history) > self.max_history:
            self.multipath_history.pop(0)
        
        # Calculate signal variance (rapid fading indicator)
        if len(self.multipath_history) >= 10:
            recent_rssi = [m["rssi"] for m in self.multipath_history[-10:]]
            rssi_variance = np.var(recent_rssi)
            
            # High variance + low SNR = likely NLOS
            if rssi_variance > 25.0 and snr_db < 10.0:
                is_nlos = True
                multipath_delay_ns = rssi_variance * 10.0  # Empirical mapping
        
        return is_nlos, multipath_delay_ns
    
    def _calculate_propagation_metrics(
        self,
        signal_metrics: Dict,
        conditions: EnvironmentalConditions,
        is_nlos: bool,
        multipath_delay_ns: float
    ) -> PropagationMetrics:
        """Calculate complete propagation characteristics."""
        
        # LOS probability based on urban density
        los_prob_map = {
            "urban": 0.3,
            "suburban": 0.6,
            "rural": 0.9
        }
        los_probability = los_prob_map.get(conditions.urban_density, 0.5)
        if is_nlos:
            los_probability *= 0.5  # Reduce if NLOS detected
        
        # Calculate weather-induced path loss
        path_loss_db = self._calculate_rain_attenuation(
            conditions.rain_rate_mm_per_hour,
            signal_metrics.get("distance_km", 1.0)
        )
        
        # Calculate Doppler shift (for moving targets/NTN)
        doppler_shift_hz = self._calculate_doppler_shift(
            signal_metrics.get("velocity_m_s", 0.0)
        )
        
        # SNR degradation from environment
        snr_degradation_db = 0.0
        snr_degradation_db += path_loss_db  # Rain attenuation
        if conditions.visibility_km < 1.0:
            snr_degradation_db += 2.0  # Fog/heavy rain
        if conditions.urban_density == "urban":
            snr_degradation_db += 3.0  # Urban clutter
        
        return PropagationMetrics(
            los_probability=los_probability,
            multipath_delay_spread_ns=multipath_delay_ns,
            doppler_shift_hz=doppler_shift_hz,
            path_loss_db=path_loss_db,
            snr_degradation_db=snr_degradation_db
        )
    
    def _compensate_multipath(
        self,
        raw_location: Tuple[float, float],
        multipath_delay_ns: float,
        signal_metrics: Dict
    ) -> Tuple[float, float]:
        """
        Compensate for multipath-induced location error.
        
        Uses ray-tracing approximation:
        - Estimate reflection point based on delay
        - Correct location toward direct path
        """
        lat, lon = raw_location
        
        # Convert delay to distance error (speed of light)
        speed_of_light = 3e8  # m/s
        distance_error_m = (multipath_delay_ns * 1e-9) * speed_of_light
        
        # Approximate correction (simplistic model)
        # In reality, would use building databases and ray-tracing
        azimuth_deg = signal_metrics.get("azimuth_deg", 0.0)
        
        # Convert meters to degrees (rough approximation)
        lat_correction = (distance_error_m / 111320.0) * np.cos(np.radians(azimuth_deg))
        lon_correction = (distance_error_m / (111320.0 * np.cos(np.radians(lat)))) * \
                        np.sin(np.radians(azimuth_deg))
        
        # Apply 50% correction factor (conservative)
        corrected_lat = lat - lat_correction * 0.5
        corrected_lon = lon - lon_correction * 0.5
        
        return (corrected_lat, corrected_lon)
    
    def _kalman_filter(
        self,
        measurement: Tuple[float, float],
        conditions: EnvironmentalConditions
    ) -> Tuple[float, float]:
        """
        Apply Kalman filter for smooth location tracking.
        
        Adapts filter parameters based on environmental conditions:
        - High wind: increase process noise
        - Urban: increase measurement noise
        """
        lat, lon = measurement
        
        # Initialize state on first call
        if self.kalman_state is None:
            self.kalman_state = np.array([lat, lon])
            self.kalman_covariance = np.eye(2) * 0.001
            return measurement
        
        # Adapt noise parameters to environment
        process_noise = self.process_noise
        measurement_noise = self.measurement_noise
        
        if conditions.urban_density == "urban":
            measurement_noise *= 2.0  # More measurement uncertainty
        if conditions.wind_speed_m_per_s > 10.0:
            process_noise *= 1.5  # More dynamic motion
        
        # Prediction step
        predicted_state = self.kalman_state
        predicted_covariance = self.kalman_covariance + np.eye(2) * process_noise
        
        # Update step
        innovation = np.array([lat, lon]) - predicted_state
        innovation_covariance = predicted_covariance + np.eye(2) * measurement_noise
        kalman_gain = predicted_covariance @ np.linalg.inv(innovation_covariance)
        
        # Update state
        self.kalman_state = predicted_state + kalman_gain @ innovation
        self.kalman_covariance = (np.eye(2) - kalman_gain) @ predicted_covariance
        
        return (self.kalman_state[0], self.kalman_state[1])
    
    def _calculate_rain_attenuation(
        self, rain_rate_mm_per_hour: float, distance_km: float
    ) -> float:
        """
        Calculate rain attenuation using ITU-R P.838 model.
        
        Returns attenuation in dB.
        """
        if rain_rate_mm_per_hour < 0.1:
            return 0.0
        
        # ITU-R P.838 coefficients for 2.4 GHz (simplified)
        k = 0.0001  # Frequency-dependent
        alpha = 0.9  # Frequency-dependent
        
        # Specific attenuation (dB/km)
        gamma = k * (rain_rate_mm_per_hour ** alpha)
        
        # Total attenuation
        attenuation_db = gamma * distance_km
        
        return attenuation_db
    
    def _calculate_doppler_shift(self, velocity_m_s: float) -> float:
        """
        Calculate Doppler shift for moving target or NTN satellite.
        
        Returns frequency shift in Hz.
        """
        speed_of_light = 3e8  # m/s
        doppler_hz = (velocity_m_s / speed_of_light) * self.carrier_frequency_hz
        return doppler_hz
    
    def correct_ntn_doppler(
        self,
        satellite_position: Tuple[float, float, float],  # ECEF coordinates
        user_position: Tuple[float, float, float],
        satellite_velocity: Tuple[float, float, float],
        carrier_freq_hz: float
    ) -> Dict:
        """
        Correct NTN Doppler shift using satellite ephemeris.
        
        Args:
            satellite_position: Satellite ECEF (x, y, z) in meters
            user_position: User ECEF (x, y, z) in meters
            satellite_velocity: Satellite velocity (vx, vy, vz) in m/s
            carrier_freq_hz: Carrier frequency in Hz
            
        Returns:
            Dict with doppler_shift_hz, corrected_frequency_hz, range_rate_m_s
        """
        # Calculate range vector
        range_vector = np.array(satellite_position) - np.array(user_position)
        range_m = np.linalg.norm(range_vector)
        
        # Unit vector from user to satellite
        los_unit = range_vector / range_m
        
        # Range rate (radial velocity component)
        range_rate_m_s = np.dot(satellite_velocity, los_unit)
        
        # Doppler shift
        speed_of_light = 3e8
        doppler_shift_hz = -(range_rate_m_s / speed_of_light) * carrier_freq_hz
        
        # Corrected frequency
        corrected_freq_hz = carrier_freq_hz + doppler_shift_hz
        
        self.logger.debug(
            f"[EnvAdapter] NTN Doppler: {doppler_shift_hz:.2f} Hz, "
            f"range rate: {range_rate_m_s:.2f} m/s"
        )
        
        return {
            "doppler_shift_hz": doppler_shift_hz,
            "corrected_frequency_hz": corrected_freq_hz,
            "range_rate_m_s": range_rate_m_s,
            "range_km": range_m / 1000.0
        }
    
    def get_statistics(self) -> Dict:
        """Get environmental adaptation statistics."""
        stats = {
            "multipath_detections": len([
                m for m in self.multipath_history 
                if m.get("is_nlos", False)
            ]),
            "total_measurements": len(self.multipath_history),
            "kalman_active": self.kalman_state is not None
        }
        
        if self.multipath_history:
            recent = self.multipath_history[-10:]
            stats["avg_rssi_db"] = np.mean([m["rssi"] for m in recent])
            stats["avg_snr_db"] = np.mean([m["snr"] for m in recent])
        
        return stats
