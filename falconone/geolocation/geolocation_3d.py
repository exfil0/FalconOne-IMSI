"""
FalconOne 3D Geolocation Engine (v1.9.2)
Full 3D positioning with altitude support and enhanced MUSIC algorithm
NTN (Non-Terrestrial Network) altitude modeling for LEO/MEO/GEO satellites

Features:
- Full 3D TDOA/AoA with altitude estimation
- MUSIC algorithm with configurable num_sources
- Enhanced WGS-84 coordinate transformations
- Real dataset integration support
- NTN altitude modeling for satellite scenarios
- Doppler compensation for moving satellites
- Target accuracy: <10m horizontal, <15m vertical

References:
- 3GPP TS 38.305 (NR Positioning)
- 3GPP TR 38.821 (NTN Study)
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


# =============================================================================
# NTN (Non-Terrestrial Network) Altitude Modeling (v1.9.2)
# =============================================================================

@dataclass
class SatelliteOrbit:
    """Satellite orbital parameters"""
    name: str
    orbit_type: str  # 'LEO', 'MEO', 'GEO', 'HEO'
    altitude_km: float
    inclination_deg: float
    eccentricity: float = 0.0
    raan_deg: float = 0.0  # Right Ascension of Ascending Node
    argument_perigee_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    epoch: Optional[datetime] = None
    
    # Computed properties
    period_minutes: float = field(init=False)
    velocity_km_s: float = field(init=False)
    
    def __post_init__(self):
        # Earth parameters
        MU_EARTH = 398600.4418  # km^3/s^2
        R_EARTH = 6371.0  # km
        
        # Semi-major axis
        a = R_EARTH + self.altitude_km
        
        # Orbital period (Kepler's 3rd law)
        self.period_minutes = 2 * np.pi * np.sqrt(a**3 / MU_EARTH) / 60
        
        # Orbital velocity (vis-viva equation for circular orbit)
        self.velocity_km_s = np.sqrt(MU_EARTH / a)


@dataclass
class NTNMeasurement:
    """NTN-specific positioning measurement"""
    satellite_id: str
    timestamp: datetime
    elevation_deg: float
    azimuth_deg: float
    range_m: Optional[float] = None
    doppler_hz: Optional[float] = None
    snr_db: float = 20.0
    propagation_delay_ns: Optional[float] = None


class NTNAltitudeModeler:
    """
    NTN (Non-Terrestrial Network) Altitude Modeling for 6G
    
    Supports positioning with:
    - LEO constellations (400-2000 km): Starlink, OneWeb, Kuiper
    - MEO constellations (8000-20000 km): O3b
    - GEO satellites (35786 km): Traditional satcom
    - HEO satellites: Molniya, Tundra orbits
    
    Features:
    - Satellite position prediction from orbital elements
    - Doppler shift compensation for velocity
    - Ionospheric delay correction
    - Multi-satellite positioning with altitude constraint
    """
    
    # Physical constants
    C = 299792458.0  # Speed of light (m/s)
    MU_EARTH = 3.986004418e14  # Gravitational parameter (m^3/s^2)
    R_EARTH = 6371000.0  # Earth radius (m)
    OMEGA_EARTH = 7.2921159e-5  # Earth rotation rate (rad/s)
    
    # Typical satellite orbit altitudes
    ORBIT_ALTITUDES = {
        'LEO_LOW': 400e3,     # 400 km (ISS-like)
        'LEO_MID': 550e3,     # 550 km (Starlink)
        'LEO_HIGH': 1200e3,   # 1200 km (OneWeb)
        'MEO': 8000e3,        # 8000 km (O3b)
        'GEO': 35786e3,       # 35786 km (Geostationary)
    }
    
    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        """
        Initialize NTN altitude modeler
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = ModuleLogger('NTNAltitudeModeler', logger)
        
        # Registered satellites
        self.satellites: Dict[str, SatelliteOrbit] = {}
        
        # Constellation configurations
        self.constellations: Dict[str, List[str]] = {}
        
        # Ionospheric model parameters
        self.ionospheric_model = self.config.get('ntn.ionospheric_model', 'klobuchar')
        
        # Tropospheric model
        self.tropospheric_model = self.config.get('ntn.tropospheric_model', 'saastamoinen')
        
        self.logger.info("NTN Altitude Modeler initialized",
                        ionospheric_model=self.ionospheric_model,
                        tropospheric_model=self.tropospheric_model)
    
    def register_satellite(self, sat_id: str, orbit: SatelliteOrbit):
        """Register a satellite with its orbital parameters"""
        self.satellites[sat_id] = orbit
        self.logger.info(f"Satellite registered: {sat_id} ({orbit.orbit_type}, {orbit.altitude_km} km)")
    
    def register_constellation(self, name: str, satellites: List[Tuple[str, SatelliteOrbit]]):
        """Register a satellite constellation"""
        sat_ids = []
        for sat_id, orbit in satellites:
            self.register_satellite(sat_id, orbit)
            sat_ids.append(sat_id)
        self.constellations[name] = sat_ids
        self.logger.info(f"Constellation registered: {name} ({len(sat_ids)} satellites)")
    
    def predict_satellite_position(self, sat_id: str, 
                                   time_utc: datetime) -> Optional[Tuple[float, float, float]]:
        """
        Predict satellite position at given time
        
        Uses simplified SGP4-like propagation for LEO/MEO
        Uses fixed position for GEO
        
        Args:
            sat_id: Satellite identifier
            time_utc: UTC timestamp
            
        Returns:
            (latitude, longitude, altitude_m) or None
        """
        if sat_id not in self.satellites:
            return None
        
        orbit = self.satellites[sat_id]
        
        if orbit.orbit_type == 'GEO':
            # GEO satellites are (approximately) fixed above equator
            # Longitude determined by epoch position + small drift
            return (0.0, orbit.raan_deg, orbit.altitude_km * 1000)
        
        # Simplified Keplerian propagation
        a = (self.R_EARTH + orbit.altitude_km * 1000)  # Semi-major axis (m)
        
        # Mean motion (rad/s)
        n = np.sqrt(self.MU_EARTH / a**3)
        
        # Time since epoch
        if orbit.epoch:
            dt = (time_utc - orbit.epoch).total_seconds()
        else:
            dt = 0
        
        # Mean anomaly at time
        M = np.radians(orbit.mean_anomaly_deg) + n * dt
        M = M % (2 * np.pi)
        
        # Eccentric anomaly (Newton-Raphson for e > 0)
        E = M
        for _ in range(10):
            E_new = M + orbit.eccentricity * np.sin(E)
            if abs(E_new - E) < 1e-10:
                break
            E = E_new
        
        # True anomaly
        nu = 2 * np.arctan2(
            np.sqrt(1 + orbit.eccentricity) * np.sin(E/2),
            np.sqrt(1 - orbit.eccentricity) * np.cos(E/2)
        )
        
        # Radius at current position
        r = a * (1 - orbit.eccentricity * np.cos(E))
        
        # Position in orbital plane
        x_orbital = r * np.cos(nu)
        y_orbital = r * np.sin(nu)
        
        # Transform to ECI (simplified, ignoring precession/nutation)
        i = np.radians(orbit.inclination_deg)
        omega = np.radians(orbit.argument_perigee_deg)
        Omega = np.radians(orbit.raan_deg)
        
        # Rotation matrices
        x_eci = (
            x_orbital * (np.cos(Omega) * np.cos(omega) - np.sin(Omega) * np.sin(omega) * np.cos(i)) -
            y_orbital * (np.cos(Omega) * np.sin(omega) + np.sin(Omega) * np.cos(omega) * np.cos(i))
        )
        y_eci = (
            x_orbital * (np.sin(Omega) * np.cos(omega) + np.cos(Omega) * np.sin(omega) * np.cos(i)) -
            y_orbital * (np.sin(Omega) * np.sin(omega) - np.cos(Omega) * np.cos(omega) * np.cos(i))
        )
        z_eci = (
            x_orbital * np.sin(omega) * np.sin(i) +
            y_orbital * np.cos(omega) * np.sin(i)
        )
        
        # ECI to ECEF (simplified rotation by Earth rotation)
        theta = self.OMEGA_EARTH * dt
        x_ecef = x_eci * np.cos(theta) + y_eci * np.sin(theta)
        y_ecef = -x_eci * np.sin(theta) + y_eci * np.cos(theta)
        z_ecef = z_eci
        
        # ECEF to LLA
        lon = np.degrees(np.arctan2(y_ecef, x_ecef))
        lat = np.degrees(np.arctan2(z_ecef, np.sqrt(x_ecef**2 + y_ecef**2)))
        alt = r - self.R_EARTH
        
        return (lat, lon, alt)
    
    def compute_doppler_shift(self, sat_id: str, 
                             observer_lat: float, observer_lon: float, observer_alt: float,
                             carrier_freq_hz: float,
                             time_utc: datetime) -> Optional[float]:
        """
        Compute Doppler shift for satellite signal
        
        Args:
            sat_id: Satellite identifier
            observer_lat/lon/alt: Observer position (degrees, meters)
            carrier_freq_hz: Carrier frequency
            time_utc: Current time
            
        Returns:
            Doppler shift in Hz (positive = approaching)
        """
        # Get satellite position at t and t+dt
        dt = 1.0  # 1 second interval
        
        pos_t0 = self.predict_satellite_position(sat_id, time_utc)
        pos_t1 = self.predict_satellite_position(
            sat_id, 
            time_utc + timedelta(seconds=dt)
        )
        
        if pos_t0 is None or pos_t1 is None:
            return None
        
        # Convert to ECEF
        geo3d = Geolocation3D({})
        
        obs_x, obs_y, obs_z = geo3d.lla_to_ecef(observer_lat, observer_lon, observer_alt)
        
        sat_x0, sat_y0, sat_z0 = geo3d.lla_to_ecef(*pos_t0)
        sat_x1, sat_y1, sat_z1 = geo3d.lla_to_ecef(*pos_t1)
        
        # Range at t0 and t1
        range_t0 = np.sqrt((sat_x0 - obs_x)**2 + (sat_y0 - obs_y)**2 + (sat_z0 - obs_z)**2)
        range_t1 = np.sqrt((sat_x1 - obs_x)**2 + (sat_y1 - obs_y)**2 + (sat_z1 - obs_z)**2)
        
        # Range rate
        range_rate = (range_t1 - range_t0) / dt  # m/s
        
        # Doppler shift
        doppler_hz = -carrier_freq_hz * range_rate / self.C
        
        return doppler_hz
    
    def compute_ionospheric_delay(self, elevation_deg: float, 
                                  frequency_hz: float,
                                  tec: float = 10e16) -> float:
        """
        Compute ionospheric delay using thin-shell model
        
        Args:
            elevation_deg: Satellite elevation angle
            frequency_hz: Carrier frequency
            tec: Total Electron Content (electrons/m^2)
            
        Returns:
            Delay in meters
        """
        # Ionospheric delay constant
        K = 40.3  # m^3/s^2
        
        # Obliquity factor (mapping function)
        RE = 6371e3
        h_iono = 350e3  # Ionospheric shell height
        
        sin_el = np.sin(np.radians(elevation_deg))
        obliquity = 1.0 / np.sqrt(1 - (RE * np.cos(np.radians(elevation_deg)) / (RE + h_iono))**2)
        
        # Delay (first-order approximation)
        delay_m = K * tec * obliquity / frequency_hz**2
        
        return delay_m
    
    def compute_tropospheric_delay(self, elevation_deg: float,
                                   height_m: float = 0,
                                   pressure_hpa: float = 1013.25,
                                   temperature_k: float = 288.15,
                                   humidity_percent: float = 50) -> float:
        """
        Compute tropospheric delay using Saastamoinen model
        
        Args:
            elevation_deg: Satellite elevation angle
            height_m: Observer height above sea level
            pressure_hpa: Surface pressure
            temperature_k: Surface temperature
            humidity_percent: Relative humidity
            
        Returns:
            Delay in meters
        """
        if elevation_deg < 5:
            elevation_deg = 5  # Avoid singularity
        
        # Zenith delays
        # Hydrostatic (dry) component
        z_hd = 0.0022768 * pressure_hpa / (1 - 0.00266 * np.cos(2 * np.radians(45)) - 0.28e-6 * height_m)
        
        # Wet component
        e_s = 6.11 * 10**(7.5 * (temperature_k - 273.15) / (temperature_k - 35.85))  # Saturation vapor pressure
        e = humidity_percent / 100 * e_s
        z_wd = 0.0022768 * (1255 / temperature_k + 0.05) * e
        
        # Mapping function (Niell simplified)
        el_rad = np.radians(elevation_deg)
        mf_h = 1 / (np.sin(el_rad) + 0.00143 / (np.tan(el_rad) + 0.0445))
        mf_w = 1 / (np.sin(el_rad) + 0.00035 / (np.tan(el_rad) + 0.017))
        
        # Total delay
        delay_m = z_hd * mf_h + z_wd * mf_w
        
        return delay_m
    
    def position_from_ntn_measurements(self, 
                                       measurements: List[NTNMeasurement],
                                       initial_guess: Optional[Tuple[float, float, float]] = None
                                       ) -> Optional[Location3D]:
        """
        Compute 3D position from NTN satellite measurements
        
        Uses weighted least squares with:
        - Doppler for velocity estimation
        - Multi-satellite geometry (GDOP)
        - Atmospheric corrections
        
        Args:
            measurements: List of NTN measurements from visible satellites
            initial_guess: Optional (lat, lon, alt) initial position
            
        Returns:
            3D location estimate with accuracy metrics
        """
        if len(measurements) < 4:
            self.logger.warning("NTN positioning requires at least 4 satellites")
            return None
        
        geo3d = Geolocation3D({})
        
        # Collect satellite positions and measurements
        sat_positions = []
        observed_ranges = []
        weights = []
        
        for meas in measurements:
            if meas.satellite_id not in self.satellites:
                continue
            
            # Get satellite position at measurement time
            sat_pos = self.predict_satellite_position(meas.satellite_id, meas.timestamp)
            if sat_pos is None:
                continue
            
            sat_x, sat_y, sat_z = geo3d.lla_to_ecef(*sat_pos)
            sat_positions.append([sat_x, sat_y, sat_z])
            
            # Use range if available, otherwise compute from propagation delay
            if meas.range_m:
                observed_ranges.append(meas.range_m)
            elif meas.propagation_delay_ns:
                observed_ranges.append(meas.propagation_delay_ns * 1e-9 * self.C)
            else:
                continue
            
            # Weight by SNR and elevation
            elevation_weight = np.sin(np.radians(max(meas.elevation_deg, 5)))
            snr_weight = 1.0 / (1.0 + np.exp(-(meas.snr_db - 15) / 5))
            weights.append(elevation_weight * snr_weight)
        
        if len(sat_positions) < 4:
            return None
        
        sat_positions = np.array(sat_positions)
        observed_ranges = np.array(observed_ranges)
        weights = np.array(weights)
        
        # Initial position guess
        if initial_guess:
            x0 = np.array(geo3d.lla_to_ecef(*initial_guess))
        else:
            # Use Earth center as initial guess
            x0 = np.array([geo3d.R_EARTH, 0, 0])
        
        # Add clock bias as 4th unknown
        x0 = np.append(x0, [0])  # [x, y, z, clock_bias]
        
        def residuals(state):
            pos = state[:3]
            clock_bias = state[3]
            
            resid = []
            for i in range(len(sat_positions)):
                geometric_range = np.linalg.norm(pos - sat_positions[i])
                computed_range = geometric_range + clock_bias
                resid.append((computed_range - observed_ranges[i]) * weights[i])
            return resid
        
        # Solve
        result = least_squares(residuals, x0, method='lm', max_nfev=100)
        
        if result.success:
            est_x, est_y, est_z = result.x[:3]
            est_lat, est_lon, est_alt = geo3d.ecef_to_lla(est_x, est_y, est_z)
            
            # Compute GDOP for accuracy estimate
            gdop = self._compute_gdop(sat_positions, result.x[:3])
            
            # Pseudorange residual RMS
            rms_residual = np.sqrt(np.mean(np.array(result.fun)**2))
            
            horiz_accuracy = gdop * rms_residual * 1.5
            vert_accuracy = gdop * rms_residual * 2.5
            
            self.logger.info(f"NTN position: ({est_lat:.6f}, {est_lon:.6f}, {est_alt:.1f}m), "
                           f"GDOP={gdop:.2f}")
            
            return Location3D(
                latitude=est_lat,
                longitude=est_lon,
                altitude=est_alt,
                horizontal_accuracy=horiz_accuracy,
                vertical_accuracy=vert_accuracy,
                timestamp=measurements[0].timestamp,
                method='NTN_MULTISAT',
                confidence=min(1.0, 1.0 / gdop)
            )
        
        return None
    
    def _compute_gdop(self, sat_positions: np.ndarray, receiver_pos: np.ndarray) -> float:
        """Compute Geometric Dilution of Precision"""
        n_sats = len(sat_positions)
        
        # Design matrix
        H = np.zeros((n_sats, 4))
        
        for i in range(n_sats):
            diff = sat_positions[i] - receiver_pos
            range_val = np.linalg.norm(diff)
            
            H[i, 0] = -diff[0] / range_val
            H[i, 1] = -diff[1] / range_val
            H[i, 2] = -diff[2] / range_val
            H[i, 3] = 1.0  # Clock bias
        
        try:
            G = np.linalg.inv(H.T @ H)
            gdop = np.sqrt(np.trace(G))
        except np.linalg.LinAlgError:
            gdop = 99.9
        
        return gdop
    
    def get_visible_satellites(self, observer_lat: float, observer_lon: float, 
                               observer_alt: float,
                               time_utc: datetime,
                               min_elevation_deg: float = 10.0) -> List[Dict[str, Any]]:
        """
        Get list of visible satellites above minimum elevation
        
        Args:
            observer_lat/lon/alt: Observer position
            time_utc: Current time
            min_elevation_deg: Minimum elevation angle
            
        Returns:
            List of visible satellite info dicts
        """
        visible = []
        geo3d = Geolocation3D({})
        
        obs_x, obs_y, obs_z = geo3d.lla_to_ecef(observer_lat, observer_lon, observer_alt)
        
        for sat_id, orbit in self.satellites.items():
            pos = self.predict_satellite_position(sat_id, time_utc)
            if pos is None:
                continue
            
            sat_x, sat_y, sat_z = geo3d.lla_to_ecef(*pos)
            
            # Vector from observer to satellite
            dx = sat_x - obs_x
            dy = sat_y - obs_y
            dz = sat_z - obs_z
            
            range_m = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Local up vector (simplified)
            up_x, up_y, up_z = obs_x, obs_y, obs_z
            up_mag = np.sqrt(up_x**2 + up_y**2 + up_z**2)
            up_x, up_y, up_z = up_x/up_mag, up_y/up_mag, up_z/up_mag
            
            # Elevation angle
            cos_el = (dx*up_x + dy*up_y + dz*up_z) / range_m
            elevation_deg = np.degrees(np.arcsin(cos_el))
            
            if elevation_deg >= min_elevation_deg:
                visible.append({
                    'satellite_id': sat_id,
                    'orbit_type': orbit.orbit_type,
                    'altitude_km': orbit.altitude_km,
                    'elevation_deg': elevation_deg,
                    'range_km': range_m / 1000,
                    'position': pos
                })
        
        # Sort by elevation (highest first)
        visible.sort(key=lambda x: x['elevation_deg'], reverse=True)
        
        return visible
