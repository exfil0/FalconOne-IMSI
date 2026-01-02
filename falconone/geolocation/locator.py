"""
FalconOne Geolocation Engine
DF/TDOA/AoA-based geolocation for cellular devices
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import logging
from scipy.optimize import least_squares
from scipy.linalg import eig

from ..utils.logger import ModuleLogger


class GeolocatorEngine:
    """Multi-device geolocation using DF/TDOA/AoA"""
    
    def __init__(self, config, logger: logging.Logger, sdr_manager):
        """Initialize geolocation engine"""
        self.config = config
        self.logger = ModuleLogger('Geolocator', logger)
        self.sdr_manager = sdr_manager
        
        self.methods = config.get('geolocation.methods', ['TDOA', 'AoA'])
        self.min_devices = config.get('geolocation.min_devices', 3)
        self.gpsdo_sync = config.get('geolocation.gpsdo_sync', True)
        
        self.logger.info("Geolocation engine initialized", methods=self.methods)
    
    def locate(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Geolocate signal sources
        
        Args:
            signals: List of signal data from monitors
            
        Returns:
            List of location estimates
        """
        results = []
        
        for signal in signals:
            try:
                location = self._estimate_location(signal)
                if location:
                    results.append(location)
            except Exception as e:
                self.logger.error(f"Geolocation error: {e}")
        
        return results
    
    def _estimate_location(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Estimate location using configured methods"""
        try:
            # Get multi-device measurements
            measurements = signal.get('measurements', [])
            
            if len(measurements) < self.min_devices:
                self.logger.warning(f"Insufficient devices: {len(measurements)} < {self.min_devices}")
                return None
            
            # Extract positions and measurements
            positions = [m['position'] for m in measurements if 'position' in m]
            
            lat, lon, method = None, None, None
            accuracy = 0.0
            
            # Try TDOA if time differences available
            if 'TDOA' in self.methods and all('tdoa' in m for m in measurements):
                time_diffs = [m['tdoa'] for m in measurements]
                lat, lon = self._tdoa_triangulation(time_diffs, positions)
                method = 'TDOA'
                accuracy = 50.0  # Typical TDOA accuracy in meters
                
            # Try AoA if angles available
            elif 'AoA' in self.methods and all('angle' in m for m in measurements):
                angles = [m['angle'] for m in measurements]
                lat, lon = self._aoa_triangulation(angles, positions)
                method = 'AoA'
                accuracy = 100.0  # Typical AoA accuracy
            
            if lat is not None and lon is not None:
                return {
                    'signal_id': signal.get('id'),
                    'latitude': lat,
                    'longitude': lon,
                    'accuracy': accuracy,
                    'method': method,
                    'num_sensors': len(measurements)
                }
            
        except Exception as e:
            self.logger.error(f"Location estimation error: {e}")
        
        return None
    
    def _tdoa_triangulation(self, time_diffs: List[float], positions: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        TDOA-based triangulation using Chan-Ho algorithm
        
        Args:
            time_diffs: Time difference of arrivals (in seconds) relative to reference
            positions: Sensor positions as (lat, lon) tuples
            
        Returns:
            Estimated (latitude, longitude)
        """
        try:
            # Speed of light
            c = 299792458  # m/s
            
            # Convert time differences to range differences
            range_diffs = np.array(time_diffs) * c
            
            # Convert lat/lon to cartesian (simplified for small areas)
            # For production, use proper geodetic conversion
            positions_xyz = []
            for lat, lon in positions:
                x = lon * 111320 * np.cos(np.radians(lat))  # meters
                y = lat * 110540  # meters
                positions_xyz.append([x, y, 0])
            
            positions_xyz = np.array(positions_xyz)
            
            # Reference sensor (first one)
            ref_pos = positions_xyz[0]
            
            # Build system of hyperbolic equations
            # Using Fang's algorithm for TDOA positioning
            def residuals(target_pos):
                residuals = []
                for i in range(1, len(positions_xyz)):
                    # Distance from target to sensor i
                    d_i = np.linalg.norm(target_pos - positions_xyz[i])
                    # Distance from target to reference sensor
                    d_0 = np.linalg.norm(target_pos - ref_pos)
                    # Expected range difference
                    expected_diff = d_i - d_0
                    # Residual
                    residuals.append(expected_diff - range_diffs[i-1])
                return residuals
            
            # Initial guess (centroid of sensors)
            x0 = np.mean(positions_xyz, axis=0)
            
            # Solve using least squares
            result = least_squares(residuals, x0[:2])
            
            if result.success:
                # Convert back to lat/lon
                x, y = result.x
                lon = x / (111320 * np.cos(np.radians(positions[0][0])))
                lat = y / 110540
                
                self.logger.debug(f"TDOA solution: ({lat:.6f}, {lon:.6f})")
                return (lat, lon)
            else:
                self.logger.warning("TDOA optimization failed")
                return (0.0, 0.0)
                
        except Exception as e:
            self.logger.error(f"TDOA triangulation error: {e}")
            return (0.0, 0.0)
    
    def _aoa_triangulation(self, angles: List[float], positions: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        AoA-based triangulation using intersection of bearing lines
        
        Args:
            angles: Angles of arrival in degrees (relative to North)
            positions: Sensor positions as (lat, lon) tuples
            
        Returns:
            Estimated (latitude, longitude)
        """
        try:
            # Convert to radians
            angles_rad = np.array(angles) * np.pi / 180
            
            # Convert positions to cartesian
            positions_xyz = []
            for lat, lon in positions:
                x = lon * 111320 * np.cos(np.radians(lat))
                y = lat * 110540
                positions_xyz.append([x, y])
            
            positions_xyz = np.array(positions_xyz)
            
            # Build system of linear equations from bearing lines
            # Each bearing defines a line: y - y_i = tan(theta_i) * (x - x_i)
            # Rearranged: tan(theta_i) * x - y = tan(theta_i) * x_i - y_i
            
            A = []
            b = []
            
            for i in range(len(angles_rad)):
                theta = angles_rad[i]
                x_i, y_i = positions_xyz[i]
                
                # Avoid division by zero for angles close to 90 or 270 degrees
                if abs(np.cos(theta)) > 0.01:
                    tan_theta = np.tan(theta)
                    A.append([tan_theta, -1])
                    b.append(tan_theta * x_i - y_i)
                else:
                    # Vertical line: x = x_i
                    A.append([1, 0])
                    b.append(x_i)
            
            A = np.array(A)
            b = np.array(b)
            
            # Solve overdetermined system using least squares
            solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            if solution.size >= 2:
                x, y = solution
                
                # Convert back to lat/lon
                lon = x / (111320 * np.cos(np.radians(positions[0][0])))
                lat = y / 110540
                
                self.logger.debug(f"AoA solution: ({lat:.6f}, {lon:.6f})")
                return (lat, lon)
            else:
                self.logger.warning("AoA solution invalid")
                return (0.0, 0.0)
                
        except Exception as e:
            self.logger.error(f"AoA triangulation error: {e}")
            return (0.0, 0.0)
    
    def _music_algorithm(self, signal_samples: np.ndarray, num_sources: int = 1) -> List[float]:
        """
        MUSIC (Multiple Signal Classification) algorithm for AoA estimation
        
        Args:
            signal_samples: Received signal samples from antenna array (NxM)
                           N = number of antennas, M = number of samples
            num_sources: Number of signal sources to estimate
            
        Returns:
            List of estimated angles in degrees
        """
        try:
            # Compute covariance matrix
            R = np.cov(signal_samples)
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = eig(R)
            
            # Sort eigenvalues in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Split into signal and noise subspaces
            noise_subspace = eigenvectors[:, num_sources:]
            
            # Compute MUSIC spectrum
            angles = np.linspace(-90, 90, 180)
            spectrum = np.zeros(len(angles))
            
            num_antennas = signal_samples.shape[0]
            antenna_spacing = 0.5  # Half wavelength spacing
            
            for i, theta in enumerate(angles):
                # Steering vector for angle theta
                theta_rad = np.radians(theta)
                a = np.exp(1j * 2 * np.pi * antenna_spacing * 
                          np.arange(num_antennas) * np.sin(theta_rad))
                
                # MUSIC pseudo-spectrum
                a = a.reshape(-1, 1)
                denominator = np.abs(a.conj().T @ noise_subspace @ noise_subspace.conj().T @ a)
                
                if denominator > 1e-10:
                    spectrum[i] = 1.0 / denominator
                else:
                    spectrum[i] = 0.0
            
            # Find peaks (top num_sources)
            peak_indices = np.argsort(spectrum)[-num_sources:]
            estimated_angles = angles[peak_indices]
            
            self.logger.debug(f"MUSIC estimated angles: {estimated_angles}")
            return estimated_angles.tolist()
            
        except Exception as e:
            self.logger.error(f"MUSIC algorithm error: {e}")
            return []
