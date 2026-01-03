"""
FalconOne 3D Geolocation Engine (v1.9.1)
Full 3D positioning with altitude support and enhanced MUSIC algorithm

Features:
- Full 3D TDOA/AoA with altitude estimation
- MUSIC algorithm with configurable num_sources
- Enhanced WGS-84 coordinate transformations
- Real dataset integration support
- Target accuracy: <10m horizontal, <15m vertical

References:
- 3GPP TS 38.305 (NR Positioning)
- IEEE 802.11az (Wi-Fi RTT)
- MUSIC algorithm: Schmidt, R.O. (1986)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from scipy.optimize import minimize, least_squares
from scipy.linalg import eigh
from scipy.signal import correlate

try:
    from ..utils.logger import ModuleLogger
except ImportError:
    class ModuleLogger:
        def __init__(self, name, parent):
            self.logger = logging.getLogger(name) if parent else logging.getLogger(__name__)
        def info(self, msg, **kw): self.logger.info(f"{msg} {kw if kw else ''}")
        def warning(self, msg, **kw): self.logger.warning(f"{msg} {kw if kw else ''}")
        def error(self, msg, **kw): self.logger.error(f"{msg} {kw if kw else ''}")
        def debug(self, msg, **kw): self.logger.debug(f"{msg} {kw if kw else ''}")


@dataclass
class Location3D:
    """Full 3D location estimate"""
    latitude: float
    longitude: float
    altitude: float  # meters above WGS-84 ellipsoid
    horizontal_accuracy: float  # meters (95% CI)
    vertical_accuracy: float    # meters (95% CI)
    timestamp: datetime
    method: str
    confidence: float = 1.0
    velocity: Optional[Tuple[float, float, float]] = None  # (vx, vy, vz) m/s
    heading: Optional[float] = None  # degrees from North


@dataclass
class ArrayElement:
    """Antenna array element for DoA estimation"""
    x: float  # meters from array center
    y: float
    z: float = 0.0


class Geolocation3D:
    """
    Full 3D Geolocation Engine with MUSIC Algorithm
    
    Improvements over 2D:
    - Altitude estimation using elevation angles
    - 3D TDOA hyperbolic intersection
    - MUSIC DoA with configurable source count
    - WGS-84 compliant coordinate transforms
    """
    
    # WGS-84 constants
    WGS84_A = 6378137.0           # Semi-major axis (m)
    WGS84_B = 6356752.314245      # Semi-minor axis (m)
    WGS84_E2 = 0.00669437999014   # First eccentricity squared
    WGS84_F = 1 / 298.257223563   # Flattening
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = ModuleLogger('Geolocation3D', logger)
        
        # Sensor network (base stations / SDR receivers)
        self.sensors: Dict[str, Tuple[float, float, float]] = {}  # id -> (lat, lon, alt)
        
        # Antenna array configuration (for MUSIC)
        self.antenna_array: List[ArrayElement] = []
        self.carrier_frequency: float = 2.4e9  # Default 2.4 GHz
        self.wavelength: float = 3e8 / self.carrier_frequency
        
        # MUSIC configuration
        self.music_num_sources = config.get('geolocation.music.num_sources', 3)
        self.music_angular_resolution = config.get('geolocation.music.angular_resolution', 1.0)  # degrees
        
        # 3D Kalman filter state [x, y, z, vx, vy, vz]
        self.kalman_state = np.zeros(6)
        self.kalman_covariance = np.eye(6) * 100.0
        
        # Real dataset integration
        self.propagation_model = config.get('geolocation.propagation_model', 'log_distance')
        self.environment = config.get('geolocation.environment', 'urban')
        
        self.logger.info("3D Geolocation Engine initialized",
                        music_sources=self.music_num_sources,
                        environment=self.environment)
    
    def register_sensor(self, sensor_id: str, lat: float, lon: float, alt: float):
        """Register sensor with full 3D position (WGS-84)"""
        self.sensors[sensor_id] = (lat, lon, alt)
        self.logger.info(f"Sensor registered: {sensor_id} at ({lat:.6f}, {lon:.6f}, {alt:.1f}m)")
    
    def setup_antenna_array(self, array_type: str = 'ula', num_elements: int = 8, 
                           element_spacing: float = 0.5):
        """
        Setup antenna array for DoA estimation
        
        Args:
            array_type: 'ula' (uniform linear), 'uca' (uniform circular), 'planar'
            num_elements: Number of antenna elements
            element_spacing: Element spacing in wavelengths
        """
        spacing_m = element_spacing * self.wavelength
        
        if array_type == 'ula':
            # Uniform Linear Array along x-axis
            self.antenna_array = [
                ArrayElement(x=i * spacing_m - (num_elements - 1) * spacing_m / 2, y=0, z=0)
                for i in range(num_elements)
            ]
        elif array_type == 'uca':
            # Uniform Circular Array in xy-plane
            radius = (num_elements * spacing_m) / (2 * np.pi)
            self.antenna_array = [
                ArrayElement(
                    x=radius * np.cos(2 * np.pi * i / num_elements),
                    y=radius * np.sin(2 * np.pi * i / num_elements),
                    z=0
                )
                for i in range(num_elements)
            ]
        elif array_type == 'planar':
            # 2D planar array for 3D DoA (elevation + azimuth)
            side = int(np.sqrt(num_elements))
            self.antenna_array = []
            for i in range(side):
                for j in range(side):
                    self.antenna_array.append(ArrayElement(
                        x=(i - side / 2) * spacing_m,
                        y=(j - side / 2) * spacing_m,
                        z=0
                    ))
        
        self.logger.info(f"Antenna array configured: {array_type}, {len(self.antenna_array)} elements")
    
    def estimate_doa_music(self, received_signals: np.ndarray, 
                          num_sources: Optional[int] = None) -> List[Tuple[float, float]]:
        """
        MUSIC (Multiple Signal Classification) Direction of Arrival estimation
        
        Args:
            received_signals: (num_snapshots, num_elements) complex array
            num_sources: Number of sources (default: self.music_num_sources)
        
        Returns:
            List of (azimuth, elevation) angles in degrees for each source
        """
        if len(self.antenna_array) == 0:
            self.logger.error("Antenna array not configured")
            return []
        
        num_sources = num_sources or self.music_num_sources
        num_elements = len(self.antenna_array)
        
        if num_sources >= num_elements:
            self.logger.warning(f"num_sources ({num_sources}) must be < num_elements ({num_elements})")
            num_sources = num_elements - 1
        
        try:
            # Compute spatial covariance matrix
            R = (received_signals.conj().T @ received_signals) / received_signals.shape[0]
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = eigh(R)
            
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            # Noise subspace (last num_elements - num_sources eigenvectors)
            noise_subspace = eigenvectors[:, num_sources:]
            
            # MUSIC pseudospectrum scan
            azimuth_range = np.arange(-180, 180, self.music_angular_resolution)
            elevation_range = np.arange(-90, 90, self.music_angular_resolution)
            
            spectrum = np.zeros((len(azimuth_range), len(elevation_range)))
            
            for i, az in enumerate(azimuth_range):
                for j, el in enumerate(elevation_range):
                    # Steering vector for 3D DoA
                    steering = self._steering_vector_3d(np.radians(az), np.radians(el))
                    
                    # MUSIC pseudospectrum
                    projection = noise_subspace.conj().T @ steering
                    spectrum[i, j] = 1.0 / np.real(projection.conj().T @ projection + 1e-12)
            
            # Find peaks (sources)
            peaks = self._find_spectrum_peaks(spectrum, azimuth_range, elevation_range, num_sources)
            
            self.logger.info(f"MUSIC DoA estimation: found {len(peaks)} sources")
            
            return peaks
            
        except Exception as e:
            self.logger.error(f"MUSIC estimation failed: {e}")
            return []
    
    def _steering_vector_3d(self, azimuth: float, elevation: float) -> np.ndarray:
        """
        Compute 3D steering vector for antenna array
        
        Args:
            azimuth: Azimuth angle in radians (0 = North, clockwise)
            elevation: Elevation angle in radians (0 = horizon, +90 = zenith)
        
        Returns:
            Complex steering vector
        """
        # Unit direction vector
        kx = np.cos(elevation) * np.sin(azimuth)
        ky = np.cos(elevation) * np.cos(azimuth)
        kz = np.sin(elevation)
        
        k = 2 * np.pi / self.wavelength  # Wavenumber
        
        # Phase shift for each element
        phases = []
        for elem in self.antenna_array:
            phase = k * (elem.x * kx + elem.y * ky + elem.z * kz)
            phases.append(np.exp(-1j * phase))
        
        return np.array(phases)
    
    def _find_spectrum_peaks(self, spectrum: np.ndarray, 
                            azimuth_range: np.ndarray, 
                            elevation_range: np.ndarray,
                            num_peaks: int) -> List[Tuple[float, float]]:
        """Find top N peaks in MUSIC pseudospectrum"""
        from scipy.ndimage import maximum_filter
        
        # Local maxima detection
        local_max = maximum_filter(spectrum, size=5) == spectrum
        
        # Get peak values and positions
        peak_values = spectrum[local_max]
        peak_indices = np.argwhere(local_max)
        
        # Sort by value
        sorted_idx = np.argsort(peak_values)[::-1]
        
        peaks = []
        for idx in sorted_idx[:num_peaks]:
            i, j = peak_indices[idx]
            az = azimuth_range[i]
            el = elevation_range[j]
            peaks.append((float(az), float(el)))
        
        return peaks
    
    def estimate_3d_tdoa(self, measurements: List[Dict]) -> Optional[Location3D]:
        """
        3D TDOA positioning with altitude estimation
        
        Args:
            measurements: List of {
                'ref_sensor': str,
                'sensor': str,
                'time_diff_ns': float,
                'snr_db': float
            }
        
        Returns:
            Full 3D location estimate
        """
        if len(measurements) < 3:
            self.logger.warning("3D TDOA requires at least 3 measurements")
            return None
        
        c = 299792458.0  # Speed of light
        
        try:
            # Build sensor positions and range differences
            sensor_positions = []
            range_diffs = []
            weights = []
            
            # Reference sensor position
            ref_id = measurements[0]['ref_sensor']
            ref_lat, ref_lon, ref_alt = self.sensors[ref_id]
            ref_x, ref_y, ref_z = self.lla_to_ecef(ref_lat, ref_lon, ref_alt)
            
            for meas in measurements:
                sensor_id = meas['sensor']
                if sensor_id not in self.sensors:
                    continue
                
                lat, lon, alt = self.sensors[sensor_id]
                x, y, z = self.lla_to_ecef(lat, lon, alt)
                
                sensor_positions.append([x, y, z])
                
                # Time difference to range difference
                range_diff = meas['time_diff_ns'] * 1e-9 * c
                range_diffs.append(range_diff)
                
                # Weight by SNR
                snr = meas.get('snr_db', 20)
                weight = 1.0 / (1.0 + np.exp(-(snr - 15) / 5))
                weights.append(weight)
            
            sensor_positions = np.array(sensor_positions)
            range_diffs = np.array(range_diffs)
            weights = np.array(weights)
            
            # Solve 3D hyperbolic positioning
            def residuals(target_pos):
                resid = []
                for i in range(len(sensor_positions)):
                    d_sensor = np.linalg.norm(target_pos - sensor_positions[i])
                    d_ref = np.linalg.norm(target_pos - np.array([ref_x, ref_y, ref_z]))
                    expected_diff = d_sensor - d_ref
                    resid.append((expected_diff - range_diffs[i]) * weights[i])
                return resid
            
            # Initial guess: weighted centroid
            x0 = np.average(sensor_positions, axis=0, weights=weights)
            
            result = least_squares(residuals, x0, method='lm')
            
            if result.success:
                est_x, est_y, est_z = result.x
                est_lat, est_lon, est_alt = self.ecef_to_lla(est_x, est_y, est_z)
                
                # Accuracy from residuals
                horiz_accuracy = np.sqrt(np.mean(np.array(result.fun[:2])**2))
                vert_accuracy = abs(result.fun[2]) if len(result.fun) > 2 else horiz_accuracy * 1.5
                
                return Location3D(
                    latitude=est_lat,
                    longitude=est_lon,
                    altitude=est_alt,
                    horizontal_accuracy=min(horiz_accuracy, 100.0),
                    vertical_accuracy=min(vert_accuracy, 150.0),
                    timestamp=datetime.now(),
                    method='3D-TDOA',
                    confidence=np.mean(weights)
                )
            else:
                self.logger.warning("3D TDOA optimization did not converge")
                return None
            
        except Exception as e:
            self.logger.error(f"3D TDOA estimation failed: {e}")
            return None
    
    def estimate_3d_aoa(self, measurements: List[Dict]) -> Optional[Location3D]:
        """
        3D AoA positioning using azimuth and elevation angles
        
        Args:
            measurements: List of {
                'sensor': str,
                'azimuth_deg': float,
                'elevation_deg': float,
                'snr_db': float
            }
        
        Returns:
            Full 3D location estimate
        """
        if len(measurements) < 2:
            self.logger.warning("3D AoA requires at least 2 measurements")
            return None
        
        try:
            # Build line equations in 3D
            lines = []
            
            for meas in measurements:
                sensor_id = meas['sensor']
                if sensor_id not in self.sensors:
                    continue
                
                lat, lon, alt = self.sensors[sensor_id]
                x, y, z = self.lla_to_ecef(lat, lon, alt)
                
                az_rad = np.radians(meas['azimuth_deg'])
                el_rad = np.radians(meas['elevation_deg'])
                
                # Direction vector
                dx = np.cos(el_rad) * np.sin(az_rad)
                dy = np.cos(el_rad) * np.cos(az_rad)
                dz = np.sin(el_rad)
                
                lines.append({
                    'origin': np.array([x, y, z]),
                    'direction': np.array([dx, dy, dz]),
                    'weight': 1.0 / (1.0 + np.exp(-(meas.get('snr_db', 20) - 15) / 5))
                })
            
            if len(lines) < 2:
                return None
            
            # Find closest point to all lines (weighted)
            def distance_to_lines(point):
                total_dist = 0
                for line in lines:
                    v = point - line['origin']
                    d = line['direction']
                    # Distance from point to line
                    dist = np.linalg.norm(v - np.dot(v, d) * d)
                    total_dist += (dist * line['weight']) ** 2
                return np.sqrt(total_dist)
            
            # Initial guess: midpoint between first two sensor origins
            x0 = (lines[0]['origin'] + lines[1]['origin']) / 2
            
            result = minimize(distance_to_lines, x0, method='L-BFGS-B')
            
            if result.success:
                est_lat, est_lon, est_alt = self.ecef_to_lla(*result.x)
                
                horiz_accuracy = result.fun * 0.7
                vert_accuracy = result.fun * 1.2
                
                return Location3D(
                    latitude=est_lat,
                    longitude=est_lon,
                    altitude=est_alt,
                    horizontal_accuracy=min(horiz_accuracy, 100.0),
                    vertical_accuracy=min(vert_accuracy, 150.0),
                    timestamp=datetime.now(),
                    method='3D-AoA',
                    confidence=np.mean([l['weight'] for l in lines])
                )
            
        except Exception as e:
            self.logger.error(f"3D AoA estimation failed: {e}")
            return None
    
    def apply_3d_kalman(self, measurement: Location3D, dt: float = 1.0) -> Location3D:
        """
        6-state Kalman filter for 3D trajectory smoothing
        State: [x, y, z, vx, vy, vz]
        """
        try:
            # Convert to ECEF
            x_meas, y_meas, z_meas = self.lla_to_ecef(
                measurement.latitude, measurement.longitude, measurement.altitude
            )
            
            # State transition matrix
            F = np.array([
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            
            # Process noise (acceleration uncertainty)
            q = 0.5  # m/s²
            Q = np.eye(6)
            Q[:3, :3] *= (q * dt**2 / 2)**2
            Q[3:, 3:] *= (q * dt)**2
            
            # Prediction
            self.kalman_state = F @ self.kalman_state
            self.kalman_covariance = F @ self.kalman_covariance @ F.T + Q
            
            # Measurement matrix (position only)
            H = np.zeros((3, 6))
            H[:3, :3] = np.eye(3)
            
            z = np.array([x_meas, y_meas, z_meas])
            
            # Measurement noise
            R = np.diag([
                measurement.horizontal_accuracy**2,
                measurement.horizontal_accuracy**2,
                measurement.vertical_accuracy**2
            ])
            
            # Kalman gain
            S = H @ self.kalman_covariance @ H.T + R
            K = self.kalman_covariance @ H.T @ np.linalg.inv(S)
            
            # Update
            innovation = z - H @ self.kalman_state
            self.kalman_state = self.kalman_state + K @ innovation
            self.kalman_covariance = (np.eye(6) - K @ H) @ self.kalman_covariance
            
            # Convert back to LLA
            filtered_lat, filtered_lon, filtered_alt = self.ecef_to_lla(
                self.kalman_state[0], self.kalman_state[1], self.kalman_state[2]
            )
            
            # Velocity in local frame
            velocity = (
                self.kalman_state[3],
                self.kalman_state[4],
                self.kalman_state[5]
            )
            
            # Heading from velocity
            heading = np.degrees(np.arctan2(velocity[0], velocity[1])) % 360
            
            return Location3D(
                latitude=filtered_lat,
                longitude=filtered_lon,
                altitude=filtered_alt,
                horizontal_accuracy=measurement.horizontal_accuracy * 0.7,
                vertical_accuracy=measurement.vertical_accuracy * 0.7,
                timestamp=datetime.now(),
                method=f"{measurement.method}+3DKalman",
                confidence=measurement.confidence,
                velocity=velocity,
                heading=heading
            )
            
        except Exception as e:
            self.logger.error(f"3D Kalman filtering failed: {e}")
            return measurement
    
    # ===== WGS-84 Coordinate Transforms =====
    
    def lla_to_ecef(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        """
        Convert WGS-84 (lat, lon, alt) to ECEF (x, y, z)
        
        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)
            alt: Altitude above ellipsoid (meters)
        
        Returns:
            (x, y, z) in ECEF coordinates (meters)
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Radius of curvature in the prime vertical
        N = self.WGS84_A / np.sqrt(1 - self.WGS84_E2 * np.sin(lat_rad)**2)
        
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - self.WGS84_E2) + alt) * np.sin(lat_rad)
        
        return x, y, z
    
    def ecef_to_lla(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert ECEF (x, y, z) to WGS-84 (lat, lon, alt)
        Uses Bowring's iterative method
        
        Returns:
            (lat, lon, alt) - degrees and meters
        """
        # Longitude is straightforward
        lon = np.degrees(np.arctan2(y, x))
        
        # Iterative latitude and altitude
        p = np.sqrt(x**2 + y**2)
        
        # Initial latitude estimate
        lat = np.arctan2(z, p * (1 - self.WGS84_E2))
        
        # Iterate
        for _ in range(10):
            N = self.WGS84_A / np.sqrt(1 - self.WGS84_E2 * np.sin(lat)**2)
            alt = p / np.cos(lat) - N
            lat_new = np.arctan2(z, p * (1 - self.WGS84_E2 * N / (N + alt)))
            
            if abs(lat_new - lat) < 1e-12:
                break
            lat = lat_new
        
        return np.degrees(lat), lon, alt
    
    def load_real_dataset(self, dataset_path: str) -> bool:
        """
        Load real measurement dataset for AI training/validation
        
        Supports formats:
        - NPZ (numpy compressed)
        - CSV
        - JSON
        
        Expected fields:
        - sensor_id, lat, lon, alt
        - measurement_type, value, snr_db
        - ground_truth_lat, ground_truth_lon, ground_truth_alt (optional)
        """
        try:
            import json
            
            if dataset_path.endswith('.npz'):
                data = np.load(dataset_path, allow_pickle=True)
                self.logger.info(f"Loaded NPZ dataset: {list(data.keys())}")
                return True
            elif dataset_path.endswith('.json'):
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                self.logger.info(f"Loaded JSON dataset: {len(data.get('measurements', []))} measurements")
                return True
            elif dataset_path.endswith('.csv'):
                import csv
                with open(dataset_path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                self.logger.info(f"Loaded CSV dataset: {len(rows)} rows")
                return True
            else:
                self.logger.error(f"Unsupported dataset format: {dataset_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            return False
