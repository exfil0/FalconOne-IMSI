"""
NTN (Non-Terrestrial Network) Monitor
Satellite beam tracking and LEO/GEO/MEO satellite support

Version 1.0: Phase 2.6.2 - NTN Beam Tracking
- LEO/MEO/GEO satellite tracking
- Doppler shift compensation
- Ephemeris data integration (TLE/SGP4)
- Beam handover prediction
- Satellite link budget calculation
"""

import threading
import time
from typing import Dict, List, Any, Optional, Tuple
from queue import Queue
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import numpy as np

try:
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..utils.logger import ModuleLogger


@dataclass
class SatelliteInfo:
    """Satellite information"""
    sat_id: int
    name: str
    type: str  # 'LEO', 'MEO', 'GEO'
    tle_line1: str  # TLE Line 1
    tle_line2: str  # TLE Line 2
    frequency_mhz: float
    last_updated: datetime


class NTNMonitor:
    """Non-Terrestrial Network (NTN) monitoring and beam tracking"""
    
    def __init__(self, config, logger: logging.Logger, sdr_manager=None):
        """
        Initialize NTN monitor
        
        Args:
            config: Configuration object
            logger: Logger instance
            sdr_manager: SDR manager instance (optional)
        """
        self.config = config
        self.logger = ModuleLogger('NTN', logger)
        self.sdr_manager = sdr_manager
        
        self.running = False
        self.tracking_thread = None
        self.data_queue = Queue()
        
        # NTN configuration
        self.ntn_enabled = config.get('monitoring.ntn.enabled', True)
        self.tracking_enabled = config.get('monitoring.ntn.tracking', True)
        
        # Satellite database
        self.satellites: Dict[int, SatelliteInfo] = {}
        self._load_satellite_database()
        
        # Tracking state
        self.tracked_satellite = None
        self.doppler_compensation = True
        self.beam_position = {'azimuth_deg': 0.0, 'elevation_deg': 0.0}
        
        # Link budget parameters
        self.ground_station = {
            'latitude_deg': config.get('ntn.ground_station.latitude', 37.7749),
            'longitude_deg': config.get('ntn.ground_station.longitude', -122.4194),
            'altitude_m': config.get('ntn.ground_station.altitude', 10.0),
            'antenna_gain_dbi': config.get('ntn.ground_station.antenna_gain', 25.0)
        }
        
        self.logger.info(f"NTN Monitor initialized: {len(self.satellites)} satellites loaded")
    
    def _load_satellite_database(self):
        """Load satellite TLE database"""
        # Sample satellites (in production: load from CelesTrak or Space-Track)
        sample_satellites = [
            {
                'sat_id': 25544,
                'name': 'ISS (ZARYA)',
                'type': 'LEO',
                'tle_line1': '1 25544U 98067A   23001.00000000  .00016717  00000-0  10270-3 0  9009',
                'tle_line2': '2 25544  51.6400 247.4627 0001532  83.4354  26.5551 15.54225995368503',
                'frequency_mhz': 2200.0
            },
            {
                'sat_id': 40086,
                'name': 'STARLINK-1007',
                'type': 'LEO',
                'tle_line1': '1 40086U 14037E   23001.00000000  .00003456  00000-0  00001-0 0  9992',
                'tle_line2': '2 40086  53.2150 180.4321 0001245  90.1234  45.6789 15.19234567123456',
                'frequency_mhz': 12000.0  # Ku-band downlink
            },
            {
                'sat_id': 28868,
                'name': 'INTELSAT 20',
                'type': 'GEO',
                'tle_line1': '1 28868U 05046A   23001.00000000  .00000010  00000-0  00000-0 0  9998',
                'tle_line2': '2 28868   0.0200 123.4567 0001234 234.5678 125.4321  1.00271234 65432',
                'frequency_mhz': 11500.0  # Ku-band
            }
        ]
        
        for sat_data in sample_satellites:
            sat_info = SatelliteInfo(
                sat_id=sat_data['sat_id'],
                name=sat_data['name'],
                type=sat_data['type'],
                tle_line1=sat_data['tle_line1'],
                tle_line2=sat_data['tle_line2'],
                frequency_mhz=sat_data['frequency_mhz'],
                last_updated=datetime.utcnow()
            )
            self.satellites[sat_info.sat_id] = sat_info
    
    def start_tracking(self, sat_id: int) -> Dict[str, Any]:
        """
        Start tracking specific satellite
        
        Args:
            sat_id: Satellite ID
            
        Returns:
            Tracking initiation result
        """
        try:
            if sat_id not in self.satellites:
                return {'success': False, 'reason': 'satellite_not_found'}
            
            self.tracked_satellite = sat_id
            self.running = True
            
            self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self.tracking_thread.start()
            
            satellite = self.satellites[sat_id]
            self.logger.info(f"Started tracking satellite: {satellite.name} ({satellite.type})")
            
            return {
                'success': True,
                'sat_id': sat_id,
                'sat_name': satellite.name,
                'sat_type': satellite.type,
                'frequency_mhz': satellite.frequency_mhz
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start tracking: {e}")
            return {'success': False, 'error': str(e)}
    
    def stop_tracking(self):
        """Stop satellite tracking"""
        self.logger.info("Stopping NTN tracking...")
        self.running = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=5)
        self.tracked_satellite = None
    
    def _tracking_loop(self):
        """Main tracking loop"""
        while self.running and self.tracked_satellite:
            try:
                # Calculate satellite position
                position = self.calculate_satellite_position(self.tracked_satellite)
                
                if position['success']:
                    # Update beam pointing
                    self.beam_position['azimuth_deg'] = position['azimuth_deg']
                    self.beam_position['elevation_deg'] = position['elevation_deg']
                    
                    # Calculate Doppler shift
                    doppler = self.calculate_doppler_shift(self.tracked_satellite)
                    
                    # Update tracking data
                    self.data_queue.put({
                        'timestamp': datetime.utcnow().isoformat(),
                        'sat_id': self.tracked_satellite,
                        'position': position,
                        'doppler_khz': doppler['doppler_shift_khz'],
                        'beam_position': self.beam_position.copy()
                    })
                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Tracking loop error: {e}")
                time.sleep(5)
    
    def calculate_satellite_position(self, sat_id: int, 
                                    timestamp: datetime = None) -> Dict[str, Any]:
        """
        Calculate satellite position using SGP4 propagator
        Task 2.6.2: Ephemeris data integration
        
        Args:
            sat_id: Satellite ID
            timestamp: Calculation timestamp (default: now)
            
        Returns:
            Satellite position (lat, lon, alt, az, el, range)
        """
        try:
            if sat_id not in self.satellites:
                return {'success': False, 'reason': 'satellite_not_found'}
            
            satellite = self.satellites[sat_id]
            
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            # Parse TLE (simplified - in production: use sgp4 library)
            position = self._propagate_sgp4(satellite, timestamp)
            
            # Calculate look angles (azimuth, elevation, range)
            look_angles = self._calculate_look_angles(
                position,
                self.ground_station['latitude_deg'],
                self.ground_station['longitude_deg'],
                self.ground_station['altitude_m']
            )
            
            result = {
                'success': True,
                'sat_id': sat_id,
                'sat_name': satellite.name,
                'sat_type': satellite.type,
                'timestamp': timestamp.isoformat(),
                'latitude_deg': position['latitude_deg'],
                'longitude_deg': position['longitude_deg'],
                'altitude_km': position['altitude_km'],
                'azimuth_deg': look_angles['azimuth_deg'],
                'elevation_deg': look_angles['elevation_deg'],
                'range_km': look_angles['range_km'],
                'velocity_km_s': position['velocity_km_s'],
                'visible': look_angles['elevation_deg'] > 5.0  # Above 5° horizon
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Position calculation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def calculate_doppler_shift(self, sat_id: int) -> Dict[str, Any]:
        """
        Calculate Doppler shift for satellite signal
        Task 2.6.2: Doppler shift compensation
        
        Args:
            sat_id: Satellite ID
            
        Returns:
            Doppler shift calculation results
        """
        try:
            if sat_id not in self.satellites:
                return {'success': False, 'reason': 'satellite_not_found'}
            
            satellite = self.satellites[sat_id]
            position = self.calculate_satellite_position(sat_id)
            
            if not position['success']:
                return position
            
            # Calculate radial velocity (velocity component toward/away from ground station)
            radial_velocity_km_s = self._calculate_radial_velocity(
                position, self.ground_station
            )
            
            # Doppler shift: f_d = f_0 * (v / c)
            frequency_hz = satellite.frequency_mhz * 1e6
            speed_of_light = 299792.458  # km/s
            doppler_shift_hz = frequency_hz * (radial_velocity_km_s / speed_of_light)
            
            # Doppler rate (rate of change)
            doppler_rate_hz_s = self._estimate_doppler_rate(sat_id, radial_velocity_km_s)
            
            result = {
                'success': True,
                'sat_id': sat_id,
                'sat_name': satellite.name,
                'carrier_freq_mhz': satellite.frequency_mhz,
                'doppler_shift_hz': float(doppler_shift_hz),
                'doppler_shift_khz': float(doppler_shift_hz / 1e3),
                'doppler_rate_hz_s': float(doppler_rate_hz_s),
                'radial_velocity_km_s': float(radial_velocity_km_s),
                'corrected_freq_mhz': float(satellite.frequency_mhz + doppler_shift_hz / 1e6),
                'compensation_enabled': self.doppler_compensation
            }
            
            self.logger.debug(f"Doppler shift: {doppler_shift_hz:.2f} Hz ({doppler_shift_hz/1e3:.2f} kHz)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Doppler calculation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_handover(self, sat_id: int, prediction_duration_min: int = 10) -> Dict[str, Any]:
        """
        Predict satellite handover times
        Task 2.6.2: Beam handover prediction
        
        Args:
            sat_id: Current satellite ID
            prediction_duration_min: Prediction window in minutes
            
        Returns:
            Handover prediction results
        """
        try:
            if sat_id not in self.satellites:
                return {'success': False, 'reason': 'satellite_not_found'}
            
            self.logger.info(f"Predicting handover for sat_id={sat_id}, duration={prediction_duration_min}min")
            
            current_sat = self.satellites[sat_id]
            current_time = datetime.utcnow()
            
            # Calculate current satellite visibility
            visibility_windows = self._calculate_visibility_window(
                sat_id, current_time, prediction_duration_min
            )
            
            # Find candidate satellites for handover
            candidate_sats = []
            for other_sat_id, other_sat in self.satellites.items():
                if other_sat_id != sat_id and other_sat.type == current_sat.type:
                    # Calculate when this satellite becomes visible
                    other_windows = self._calculate_visibility_window(
                        other_sat_id, current_time, prediction_duration_min
                    )
                    
                    if other_windows:
                        candidate_sats.append({
                            'sat_id': other_sat_id,
                            'sat_name': other_sat.name,
                            'visibility_windows': other_windows
                        })
            
            # Determine handover time (when current satellite goes below minimum elevation)
            handover_time = None
            handover_elevation = None
            
            if visibility_windows:
                for window in visibility_windows:
                    if window['end_time']:
                        handover_time = window['end_time']
                        handover_elevation = window['min_elevation_deg']
                        break
            
            result = {
                'success': True,
                'current_sat_id': sat_id,
                'current_sat_name': current_sat.name,
                'current_sat_type': current_sat.type,
                'prediction_time': current_time.isoformat(),
                'prediction_duration_min': prediction_duration_min,
                'handover_predicted': handover_time is not None,
                'handover_time': handover_time.isoformat() if handover_time else None,
                'handover_elevation_deg': handover_elevation,
                'candidate_satellites': candidate_sats,
                'recommended_handover_sat': candidate_sats[0] if candidate_sats else None
            }
            
            if handover_time:
                time_to_handover = (handover_time - current_time).total_seconds() / 60
                self.logger.info(f"Handover predicted in {time_to_handover:.1f} minutes")
            else:
                self.logger.info("No handover needed within prediction window")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Handover prediction error: {e}")
            return {'success': False, 'error': str(e)}
    
    def calculate_link_budget(self, sat_id: int) -> Dict[str, Any]:
        """
        Calculate satellite link budget
        
        Args:
            sat_id: Satellite ID
            
        Returns:
            Link budget calculation
        """
        try:
            if sat_id not in self.satellites:
                return {'success': False, 'reason': 'satellite_not_found'}
            
            satellite = self.satellites[sat_id]
            position = self.calculate_satellite_position(sat_id)
            
            if not position['success']:
                return position
            
            # Link budget parameters
            tx_power_dbw = 10.0  # 10 dBW satellite TX power
            tx_antenna_gain_dbi = 30.0  # Satellite antenna gain
            rx_antenna_gain_dbi = self.ground_station['antenna_gain_dbi']
            
            # Free space path loss: FSPL = 20log10(d) + 20log10(f) + 92.45
            range_km = position['range_km']
            freq_mhz = satellite.frequency_mhz
            fspl_db = 20 * np.log10(range_km) + 20 * np.log10(freq_mhz) + 92.45
            
            # Atmospheric loss (simplified)
            elevation_deg = position['elevation_deg']
            atmospheric_loss_db = self._calculate_atmospheric_loss(elevation_deg)
            
            # Received power: Pr = Pt + Gt + Gr - FSPL - La
            received_power_dbw = (tx_power_dbw + tx_antenna_gain_dbi + 
                                 rx_antenna_gain_dbi - fspl_db - atmospheric_loss_db)
            
            # Link margin (assuming -100 dBW sensitivity)
            rx_sensitivity_dbw = -100.0
            link_margin_db = received_power_dbw - rx_sensitivity_dbw
            
            result = {
                'success': True,
                'sat_id': sat_id,
                'sat_name': satellite.name,
                'frequency_mhz': satellite.frequency_mhz,
                'range_km': range_km,
                'elevation_deg': position['elevation_deg'],
                'tx_power_dbw': tx_power_dbw,
                'tx_antenna_gain_dbi': tx_antenna_gain_dbi,
                'rx_antenna_gain_dbi': rx_antenna_gain_dbi,
                'fspl_db': float(fspl_db),
                'atmospheric_loss_db': float(atmospheric_loss_db),
                'received_power_dbw': float(received_power_dbw),
                'rx_sensitivity_dbw': rx_sensitivity_dbw,
                'link_margin_db': float(link_margin_db),
                'link_available': link_margin_db > 0
            }
            
            self.logger.info(f"Link budget: margin={link_margin_db:.2f} dB "
                           f"({'✓ AVAILABLE' if result['link_available'] else '✗ UNAVAILABLE'})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Link budget calculation error: {e}")
            return {'success': False, 'error': str(e)}
    
    # Helper methods
    
    def _propagate_sgp4(self, satellite: SatelliteInfo, timestamp: datetime) -> Dict[str, Any]:
        """Propagate satellite position using SGP4 (simplified)"""
        # Simplified orbit propagation (in production: use python-sgp4 library)
        # For demonstration: circular orbit approximation
        
        if satellite.type == 'LEO':
            altitude_km = 550.0  # Typical LEO altitude
            orbital_period_min = 96.0  # ~90-100 minutes
        elif satellite.type == 'MEO':
            altitude_km = 20200.0  # GPS altitude
            orbital_period_min = 718.0  # ~12 hours
        else:  # GEO
            altitude_km = 35786.0  # Geostationary altitude
            orbital_period_min = 1436.0  # 24 hours
        
        # Calculate position (simplified)
        earth_radius_km = 6371.0
        orbital_radius_km = earth_radius_km + altitude_km
        
        # Velocity for circular orbit
        mu = 398600.4418  # Earth gravitational parameter (km³/s²)
        velocity_km_s = np.sqrt(mu / orbital_radius_km)
        
        # Random position for simulation (in production: actual SGP4)
        latitude_deg = np.random.uniform(-90, 90) if satellite.type != 'GEO' else 0.0
        longitude_deg = np.random.uniform(-180, 180)
        
        return {
            'latitude_deg': float(latitude_deg),
            'longitude_deg': float(longitude_deg),
            'altitude_km': float(altitude_km),
            'velocity_km_s': float(velocity_km_s)
        }
    
    def _calculate_look_angles(self, sat_position: Dict, observer_lat: float,
                              observer_lon: float, observer_alt: float) -> Dict[str, Any]:
        """Calculate azimuth, elevation, range from observer to satellite"""
        # Simplified look angle calculation (in production: use pyorbital or similar)
        
        # Convert to radians
        sat_lat_rad = np.deg2rad(sat_position['latitude_deg'])
        sat_lon_rad = np.deg2rad(sat_position['longitude_deg'])
        obs_lat_rad = np.deg2rad(observer_lat)
        obs_lon_rad = np.deg2rad(observer_lon)
        
        # Calculate range using haversine formula (simplified)
        dlat = sat_lat_rad - obs_lat_rad
        dlon = sat_lon_rad - obs_lon_rad
        
        a = np.sin(dlat/2)**2 + np.cos(obs_lat_rad) * np.cos(sat_lat_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        earth_radius_km = 6371.0
        ground_distance_km = earth_radius_km * c
        
        # Calculate range considering altitude
        altitude_diff_km = sat_position['altitude_km'] - (observer_alt / 1000.0)
        range_km = np.sqrt(ground_distance_km**2 + altitude_diff_km**2)
        
        # Calculate elevation angle
        elevation_rad = np.arctan2(altitude_diff_km, ground_distance_km)
        elevation_deg = np.rad2deg(elevation_rad)
        
        # Calculate azimuth (simplified)
        y = np.sin(dlon) * np.cos(sat_lat_rad)
        x = np.cos(obs_lat_rad) * np.sin(sat_lat_rad) - np.sin(obs_lat_rad) * np.cos(sat_lat_rad) * np.cos(dlon)
        azimuth_rad = np.arctan2(y, x)
        azimuth_deg = (np.rad2deg(azimuth_rad) + 360) % 360
        
        return {
            'azimuth_deg': float(azimuth_deg),
            'elevation_deg': float(elevation_deg),
            'range_km': float(range_km)
        }
    
    def _calculate_radial_velocity(self, sat_position: Dict, ground_station: Dict) -> float:
        """Calculate radial velocity component"""
        # Simplified calculation (in production: use proper vector math)
        # Positive = moving away, Negative = moving toward
        
        total_velocity = sat_position['velocity_km_s']
        
        # Approximate radial component based on elevation angle
        # When satellite is at zenith: radial velocity ≈ 0
        # When satellite is at horizon: radial velocity ≈ max
        elevation_deg = sat_position.get('elevation_deg', 45.0)
        elevation_factor = np.cos(np.deg2rad(elevation_deg))
        
        # Random direction for simulation
        direction = np.random.choice([-1, 1])
        
        radial_velocity = total_velocity * elevation_factor * direction
        
        return float(radial_velocity)
    
    def _estimate_doppler_rate(self, sat_id: int, radial_velocity: float) -> float:
        """Estimate rate of change of Doppler shift"""
        # Simplified estimation
        # Doppler rate depends on satellite acceleration
        doppler_rate_hz_s = radial_velocity * 100  # Simplified approximation
        
        return doppler_rate_hz_s
    
    def _calculate_visibility_window(self, sat_id: int, start_time: datetime,
                                    duration_min: int) -> List[Dict]:
        """Calculate when satellite is visible from ground station"""
        visibility_windows = []
        
        # Calculate position at intervals
        time_step_min = 1
        num_steps = duration_min // time_step_min
        
        in_window = False
        window_start = None
        
        for i in range(num_steps):
            timestamp = start_time + timedelta(minutes=i * time_step_min)
            position = self.calculate_satellite_position(sat_id, timestamp)
            
            if position['success']:
                visible = position['elevation_deg'] > 5.0  # Above 5° horizon
                
                if visible and not in_window:
                    # Start of visibility window
                    window_start = timestamp
                    in_window = True
                elif not visible and in_window:
                    # End of visibility window
                    visibility_windows.append({
                        'start_time': window_start,
                        'end_time': timestamp,
                        'duration_min': (timestamp - window_start).total_seconds() / 60,
                        'min_elevation_deg': 5.0
                    })
                    in_window = False
        
        # Close any open window
        if in_window:
            visibility_windows.append({
                'start_time': window_start,
                'end_time': None,
                'duration_min': None,
                'min_elevation_deg': 5.0
            })
        
        return visibility_windows
    
    def _calculate_atmospheric_loss(self, elevation_deg: float) -> float:
        """Calculate atmospheric loss based on elevation angle"""
        # Simplified atmospheric loss model
        # Loss increases at lower elevations due to longer path through atmosphere
        
        if elevation_deg >= 90:
            return 0.5  # Minimal loss at zenith
        elif elevation_deg >= 30:
            return 1.0  # Low loss
        elif elevation_deg >= 10:
            return 2.0  # Moderate loss
        else:
            return 5.0  # High loss at low elevations
    
    def get_status(self) -> Dict[str, Any]:
        """Get NTN monitor status"""
        return {
            'running': self.running,
            'ntn_enabled': self.ntn_enabled,
            'tracking_enabled': self.tracking_enabled,
            'num_satellites': len(self.satellites),
            'tracked_satellite': self.tracked_satellite,
            'doppler_compensation': self.doppler_compensation,
            'beam_position': self.beam_position.copy()
        }
    
    def list_satellites(self) -> List[Dict]:
        """List all available satellites"""
        return [
            {
                'sat_id': sat.sat_id,
                'name': sat.name,
                'type': sat.type,
                'frequency_mhz': sat.frequency_mhz,
                'last_updated': sat.last_updated.isoformat()
            }
            for sat in self.satellites.values()
        ]
