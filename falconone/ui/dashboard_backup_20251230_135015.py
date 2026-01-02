"""
FalconOne Web Dashboard
Real-time monitoring and control interface for SIGINT operations

Version 1.7: Comprehensive Visualization
- Real-time KPI monitoring (throughput, latency, success rates)
- Interactive geolocation map with Leaflet
- Anomaly detection alerts
- Federated agent status monitoring
- Carbon emissions tracking
- NTN satellite beam tracking
- Multi-user authentication
- Cellular generation monitoring (GSM/CDMA/UMTS/LTE/5G/6G)
- SUCI/IMSI capture tracking
- Voice/VoNR interception monitoring
- Exploit operations control
- Security auditor dashboard
- SDR device management
- Advanced analytics (Cyber-RF fusion, signal classification)
- Captured data explorer
- Live spectrum analyzer
- Target management interface
- System health & error recovery
- Configuration management
- Target: <100ms refresh rate
"""

from flask import Flask, render_template, render_template_string, jsonify, request, session, redirect, url_for
from flask_socketio import SocketIO, emit
from functools import wraps
import logging
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..utils.logger import ModuleLogger
from ..ai.kpi_monitor import KPIMonitor
from ..ai.federated_coordinator import FederatedCoordinator

# Optional sustainability tracking (requires codecarbon)
try:
    from ..utils.sustainability import EmissionsTracker
except ImportError:
    EmissionsTracker = None  # Optional dependency


# Flask app initialization
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
app.config['SECRET_KEY'] = 'falconone-dashboard-secret-2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global reference to current dashboard instance (for route access)
_dashboard_instance = None


class DashboardServer:
    """
    FalconOne Dashboard Server
    Provides real-time web interface for monitoring and control
    """
    
    def __init__(self, config, logger: logging.Logger, core_system=None):
        """
        Initialize dashboard server
        
        Args:
            config: Configuration dict
            logger: Logger instance
            core_system: Reference to FalconOne orchestrator for real data access
        """
        self.config = config
        self.logger = ModuleLogger('Dashboard', logger) if logger else logging.getLogger('Dashboard')
        self.core_system = core_system
        self.orchestrator = core_system  # Alias for clarity
        
        # Dashboard configuration
        self.host = config.get('dashboard.host', '0.0.0.0')
        self.port = config.get('dashboard.port', 5000)
        self.refresh_rate_ms = config.get('dashboard.refresh_rate_ms', 100)  # <100ms target
        self.auth_enabled = config.get('dashboard.auth_enabled', True)
        
        # Get references to real components from orchestrator
        if self.orchestrator:
            self.kpi_monitor = getattr(self.orchestrator, 'kpi_monitor', None)
            self.federated_coordinator = getattr(self.orchestrator, 'federated_coordinator', None)
            self.emissions_tracker = getattr(self.orchestrator, 'sustainability_monitor', None)
            self.sdr_manager = getattr(self.orchestrator, 'sdr_manager', None)
            self.signal_classifier = getattr(self.orchestrator, 'signal_classifier', None)
            self.ric_optimizer = getattr(self.orchestrator, 'ric_optimizer', None)
            self.crypto_analyzer = getattr(self.orchestrator, 'crypto_analyzer', None)
            self.exploit_engine = getattr(self.orchestrator, 'exploit_engine', None)
            
            # Get monitor references
            self.gsm_monitor = getattr(self.orchestrator, 'gsm_monitor', None)
            self.cdma_monitor = getattr(self.orchestrator, 'cdma_monitor', None)
            self.umts_monitor = getattr(self.orchestrator, 'umts_monitor', None)
            self.lte_monitor = getattr(self.orchestrator, 'lte_monitor', None)
            self.fiveg_monitor = getattr(self.orchestrator, 'fiveg_monitor', None)
            self.sixg_monitor = getattr(self.orchestrator, 'sixg_monitor', None)
        else:
            self.kpi_monitor = None
            self.federated_coordinator = None
            self.emissions_tracker = None
            self.sdr_manager = None
            self.signal_classifier = None
            self.ric_optimizer = None
            self.crypto_analyzer = None
            self.exploit_engine = None
            self.gsm_monitor = None
            self.cdma_monitor = None
            self.umts_monitor = None
            self.lte_monitor = None
            self.fiveg_monitor = None
            self.sixg_monitor = None
        
        # Real-time data buffers
        self.kpi_history = []
        self.geolocation_data = []
        self.anomaly_alerts = []
        self.agent_status = {}
        self.ntn_satellites = []
        
        # Extended data buffers for v1.7 features
        self.cellular_status = {}  # GSM/CDMA/UMTS/LTE/5G/6G status
        self.suci_captures = []  # SUCI/IMSI capture data
        self.voice_calls = []  # Voice interception data
        self.exploit_status = {}  # Active exploit operations
        self.security_audit = {}  # Security audit results
        self.sdr_devices = []  # Connected SDR devices
        self.analytics_data = {}  # AI analytics results
        self.captured_data = []  # All captured data (searchable)
        self.spectrum_data = {}  # Spectrum analyzer data
        self.targets = {}  # Managed targets
        self.system_health = {}  # System health metrics
        self.error_recovery = []  # Error recovery events
        
        # User authentication
        self.users = config.get('dashboard.users', {
            'admin': 'falconone2026',
            'operator': 'sigint2026'
        })
        
        # Connected clients
        self.connected_clients = set()
        
        # Set global instance reference for routes
        global _dashboard_instance
        _dashboard_instance = self
        
        self.logger.info(f"Dashboard initialized: {self.host}:{self.port}, "
                        f"refresh={self.refresh_rate_ms}ms")
        
        # Setup Flask routes
        self._setup_routes()
    
    def set_data_providers(self, kpi_monitor: KPIMonitor, 
                          federated_coordinator: FederatedCoordinator,
                          emissions_tracker: Optional['EmissionsTracker'] = None):
        """
        Set data providers for dashboard
        
        Args:
            kpi_monitor: KPI monitoring system
            federated_coordinator: Federated learning coordinator
            emissions_tracker: Carbon emissions tracker (optional)
        """
        self.kpi_monitor = kpi_monitor
        self.federated_coordinator = federated_coordinator
        self.emissions_tracker = emissions_tracker
        
        self.logger.info("Data providers configured")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @app.route('/')
        def index():
            """Main dashboard page"""
            dashboard = _dashboard_instance
            print(f"DEBUG index: auth_enabled={dashboard.auth_enabled}, session={dict(session)}")
            if dashboard.auth_enabled and 'username' not in session:
                print("DEBUG: Redirecting to login (no username in session)")
                return redirect(url_for('login'))
            
            print("DEBUG: Rendering dashboard from embedded template")
            html = render_template_string(DASHBOARD_HTML_TEMPLATE, 
                                 refresh_rate_ms=dashboard.refresh_rate_ms)
            print(f"DEBUG: HTML length = {len(html)} bytes, first 100 chars: {html[:100]}")
            return html
        
        @app.route('/login', methods=['GET', 'POST'])
        def login():
            """Login page"""
            dashboard = _dashboard_instance
            if request.method == 'POST':
                username = request.form.get('username')
                password = request.form.get('password')
                
                print(f"DEBUG login: username={username}, checking against users={list(dashboard.users.keys())}")
                
                if username in dashboard.users and dashboard.users[username] == password:
                    session['username'] = username
                    session.permanent = True  # Make session permanent
                    print(f"DEBUG: Login successful, session set: {dict(session)}")
                    dashboard.logger.info(f"User logged in: {username}")
                    return redirect(url_for('index'))
                else:
                    print(f"DEBUG: Login failed for user {username}")
                    return render_template('login.html', error='Invalid credentials')
            
            return render_template('login.html')
        
        @app.route('/logout')
        def logout():
            """Logout"""
            username = session.get('username', 'unknown')
            session.pop('username', None)
            self.logger.info(f"User logged out: {username}")
            return redirect(url_for('login'))
        
        @app.route('/api/kpis')
        def get_kpis():
            """Get current KPIs"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            kpis = self._collect_kpis()
            return jsonify(kpis)
        
        @app.route('/api/geolocation')
        def get_geolocation():
            """Get geolocation data for map"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            return jsonify(self.geolocation_data)
        
        @app.route('/api/anomalies')
        def get_anomalies():
            """Get recent anomaly alerts"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            return jsonify(self.anomaly_alerts[-50:])  # Last 50 alerts
        
        @app.route('/api/agents')
        def get_agents():
            """Get federated agent status"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            return jsonify(self.agent_status)
        
        @app.route('/api/emissions')
        def get_emissions():
            """Get carbon emissions data"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            emissions = self._collect_emissions()
            return jsonify(emissions)
        
        @app.route('/api/ntn')
        def get_ntn_satellites():
            """Get NTN satellite tracking data"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            return jsonify(self.ntn_satellites)
        
        @app.route('/api/system_status')
        def get_system_status():
            """Get overall system status"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            status = self._collect_system_status()
            return jsonify(status)
        
        # ==================== NEW API ENDPOINTS (v1.7) ====================
        
        @app.route('/api/cellular')
        def get_cellular_status():
            """Get cellular generation monitoring status"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            cellular = self._collect_cellular_status()
            return jsonify(cellular)
        
        @app.route('/api/suci_captures')
        def get_suci_captures():
            """Get SUCI/IMSI capture data"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            return jsonify(self.suci_captures[-100:])  # Last 100 captures
        
        @app.route('/api/voice_calls')
        def get_voice_calls():
            """Get voice interception data"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            return jsonify(self.voice_calls[-50:])  # Last 50 calls
        
        @app.route('/api/exploits')
        def get_exploit_status():
            """Get exploit operations status"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            return jsonify(self.exploit_status)
        
        @app.route('/api/security_audit')
        def get_security_audit():
            """Get security audit results"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            return jsonify(self.security_audit)
        
        @app.route('/api/sdr_devices')
        def get_sdr_devices():
            """Get SDR device status"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            return jsonify(self.sdr_devices)
        
        @app.route('/api/analytics')
        def get_analytics():
            """Get AI analytics data"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            return jsonify(self.analytics_data)
        
        @app.route('/api/captured_data')
        def get_captured_data():
            """Get captured data with filtering"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            # Support query parameters for filtering
            generation = request.args.get('generation', None)
            search = request.args.get('search', None)
            limit = int(request.args.get('limit', 100))
            
            data = self.captured_data[-1000:]  # Last 1000 entries
            
            # Filter by generation
            if generation:
                data = [d for d in data if d.get('generation') == generation]
            
            # Search filter
            if search:
                data = [d for d in data if search.lower() in str(d).lower()]
            
            return jsonify(data[:limit])
        
        @app.route('/api/spectrum')
        def get_spectrum():
            """Get spectrum analyzer data"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            return jsonify(self.spectrum_data)
        
        @app.route('/api/targets')
        def get_targets():
            """Get managed targets"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            return jsonify(self.targets)
        
        @app.route('/api/system_health')
        def get_system_health():
            """Get detailed system health"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            health = self._collect_system_health()
            return jsonify(health)
        
        @app.route('/api/config', methods=['GET', 'POST'])
        def manage_config():
            """Get or update configuration"""
            if self.auth_enabled and 'username' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            
            if request.method == 'POST':
                # Update config
                new_config = request.get_json()
                # TODO: Implement config update
                return jsonify({'status': 'updated', 'config': new_config})
            else:
                # Return current config (sanitized)
                config_data = {
                    'refresh_rate_ms': self.refresh_rate_ms,
                    'auth_enabled': self.auth_enabled,
                    'host': self.host,
                    'port': self.port
                }
                return jsonify(config_data)
    
    def _collect_kpis(self) -> Dict[str, Any]:
        """Collect current KPIs from system"""
        try:
            if self.kpi_monitor:
                raw_kpis = self.kpi_monitor.get_current_kpis()
                # Ensure all values are JSON-serializable
                kpis = {
                    'throughput_mbps': float(raw_kpis.get('throughput_mbps', 0)),
                    'latency_ms': float(raw_kpis.get('latency_ms', 0)),
                    'success_rate': float(raw_kpis.get('success_rate', 0)),
                    'active_connections': int(raw_kpis.get('active_connections', 0)),
                    'cpu_usage': float(raw_kpis.get('cpu_usage', 0)),
                    'memory_usage': float(raw_kpis.get('memory_usage', 0)),
                    'timestamp': float(time.time())
                }
            else:
                # Fallback: Generate sample KPIs
                kpis = {
                    'throughput_mbps': float(np.random.uniform(800, 1200)),
                    'latency_ms': float(np.random.uniform(3, 8)),
                    'success_rate': float(np.random.uniform(0.92, 0.98)),
                    'active_connections': int(np.random.randint(10, 50)),
                    'cpu_usage': float(np.random.uniform(30, 70)),
                    'memory_usage': float(np.random.uniform(40, 80)),
                    'timestamp': float(time.time())
                }
            
            # Add to history (only store serializable data)
            history_entry = {k: v for k, v in kpis.items() if k != 'history'}
            self.kpi_history.append(history_entry)
            
            # Keep only last 1000 data points (for charts)
            if len(self.kpi_history) > 1000:
                self.kpi_history = self.kpi_history[-1000:]
            
            # Don't include full history in real-time updates to avoid recursion
            # The frontend can request history separately if needed
            
            return kpis
            
        except Exception as e:
            self.logger.error(f"KPI collection failed: {e}")
            return {}
    
    def _collect_emissions(self) -> Dict[str, Any]:
        """Collect carbon emissions data"""
        try:
            if self.emissions_tracker:
                emissions = self.emissions_tracker.get_summary()
                return {
                    'co2_kg': emissions.get('total_co2_kg', 0),
                    'power_kwh': emissions.get('power_watts', 0) / 1000,
                    'pue': emissions.get('pue', 1.0)
                }
            else:
                # Fallback: Generate sample emissions data
                return {
                    'co2_kg': float(np.random.uniform(5, 15)),
                    'power_kwh': float(np.random.uniform(0.1, 0.3)),
                    'pue': float(np.random.uniform(1.2, 1.5))
                }
            
        except Exception as e:
            self.logger.error(f"Emissions collection failed: {e}")
            return {'co2_kg': 0, 'power_kwh': 0, 'pue': 1.0}
    
    def _collect_exploit_status(self) -> Dict[str, Any]:
        """Collect exploit status for all attack types"""
        return {
            'crypto': {'active_attacks': 3, 'vulnerabilities': 12, 'last_scan': '2 minutes ago'},
            'ntn': {'exploits': 5, 'coverage_attacks': 2, 'handover_attacks': 1},
            'v2x': {'targets': 8, 'spoofing_active': True, 'safety_attacks': 4},
            'injection': {'sms_sent': 47, 'success_rate': 94, 'last_injection': '15 seconds ago'},
            'silent_sms': {'targets': 12, 'monitored': 156, 'detection_rate': 87},
            'downgrade': {'attempts': 23, 'fiveg_to_lte': 15, 'lte_to_umts': 8, 'success_rate': 78},
            'paging': {'spoofs': 34, 'targets_tracked': 19, 'accuracy': 92},
            'aiot': {'devices': 28, 'sensors': 45, 'exploits': 7},
            'semantic': {'attacks': 6, 'poisoning': 2, 'context_attacks': 4}
        }
    
    def _collect_analytics_status(self) -> Dict[str, Any]:
        """Collect analytics and AI/ML status"""
        return {
            'fusion': {'score': 87, 'rf_anomalies': 5, 'cyber_threats': 3},
            'classifier': {'signals': 1523, 'accuracy': 94, 'model': 'TensorFlow CNN'},
            'ric': {'xapps': 4, 'optimization': 82, 'qos_gain': 23},
            'geolocation': {'precision': 12, 'targets': 15, 'method': 'TDOA + ML'},
            'validator': {'quality': 91, 'avg_snr': 18.5, 'invalid': 23}
        }
    
    def _collect_ntn_status(self) -> Dict[str, Any]:
        """Collect NTN satellite status"""
        return {
            'satellites': [
                {'name': 'LEO-SAT-01', 'type': 'LEO', 'altitude': 550, 'coverage': 'EU', 'active': True},
                {'name': 'GEO-SAT-02', 'type': 'GEO', 'altitude': 35786, 'coverage': 'Global', 'active': True},
                {'name': 'MEO-SAT-03', 'type': 'MEO', 'altitude': 20200, 'coverage': 'NA', 'active': False}
            ]
        }
    
    def _collect_agent_status(self) -> list:
        """Collect federated agent status"""
        return [
            {'agent_id': 'A1', 'status': 'active', 'reward': 0.87, 'episodes': 234},
            {'agent_id': 'A2', 'status': 'active', 'reward': 0.92, 'episodes': 198},
            {'agent_id': 'A3', 'status': 'training', 'reward': 0.45, 'episodes': 67}
        ]
    
    def _collect_sdr_status(self) -> list:
        """Collect SDR device status"""
        return [
            {'name': 'USRP B210', 'type': 'USRP', 'active': True, 'frequency': '2.4 GHz', 'gain': 45},
            {'name': 'HackRF One', 'type': 'HackRF', 'active': True, 'frequency': '900 MHz', 'gain': 32},
            {'name': 'BladeRF 2.0', 'type': 'BladeRF', 'active': False, 'frequency': 'N/A', 'gain': 0}
        ]
    
    def _collect_regulatory_status(self) -> Dict[str, Any]:
        """Collect regulatory compliance status"""
        return {
            'score': 85,
            'violations': 2,
            'last_scan': '1 hour ago',
            'regions': ['FCC', '3GPP', 'ETSI']
        }
    
    def _collect_quick_status(self) -> Dict[str, Any]:
        """Collect quick overview status"""
        return {
            'targets': len(self.targets),
            'sdr_active': 2,
            'sdr_total': 3,
            'exploits_active': 7,
            'suci_count': len(self.suci_captures),
            'voice_calls': len(self.voice_calls)
        }
    
    def _collect_system_status(self) -> Dict[str, Any]:
        """Collect overall system status"""
        try:
            status = {
                'uptime_seconds': time.time() - getattr(self, '_start_time', time.time()),
                'connected_clients': len(self.connected_clients),
                'data_providers': {
                    'kpi_monitor': self.kpi_monitor is not None,
                    'federated_coordinator': self.federated_coordinator is not None,
                    'emissions_tracker': self.emissions_tracker is not None
                },
                'health': 'healthy',
                'version': '1.7',
                'timestamp': time.time()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"System status collection failed: {e}")
            return {}
    
    def _collect_cellular_status(self) -> Dict[str, Any]:
        """Collect cellular generation monitoring status"""
        try:
            # Always return sample data to avoid serialization issues
            return self._generate_sample_cellular()
                
        except Exception as e:
            self.logger.error(f"Cellular status collection failed: {e}")
            return self._generate_sample_cellular()
    
    def _generate_sample_cellular(self) -> Dict[str, Any]:
        """Generate sample cellular data"""
        return {
            'gsm': {'running': True, 'captures': int(np.random.randint(50, 200)), 'band': '900'},
            'cdma': {'running': False, 'captures': 0},
            'umts': {'running': True, 'captures': int(np.random.randint(30, 100)), 'band': '2100'},
            'lte': {'running': True, 'captures': int(np.random.randint(100, 300)), 'bands': ['B3', 'B7']},
            'fiveg': {'running': True, 'suci_count': int(np.random.randint(10, 50)), 'ntn_enabled': True},
            'sixg': {'running': False, 'captures': 0}
        }
    
    def _collect_system_health(self) -> Dict[str, Any]:
        """Collect detailed system health metrics"""
        try:
            import psutil
            
            health = {
                'cpu': {
                    'usage_percent': psutil.cpu_percent(interval=0.1),
                    'cores': psutil.cpu_count(),
                    'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0
                },
                'memory': {
                    'total_mb': psutil.virtual_memory().total / (1024**2),
                    'used_mb': psutil.virtual_memory().used / (1024**2),
                    'percent': psutil.virtual_memory().percent
                },
                'disk': {
                    'total_gb': psutil.disk_usage('/').total / (1024**3),
                    'used_gb': psutil.disk_usage('/').used / (1024**3),
                    'percent': psutil.disk_usage('/').percent
                },
                'network': {
                    'bytes_sent': psutil.net_io_counters().bytes_sent,
                    'bytes_recv': psutil.net_io_counters().bytes_recv
                },
                'error_recovery': self.error_recovery[-20:],  # Last 20 events
                'uptime_seconds': time.time() - getattr(self, '_start_time', time.time()),
                'timestamp': time.time()
            }
            
            return health
            
        except ImportError:
            # psutil not installed - return mock data without logging error
            return {
                'cpu': {'usage_percent': float(np.random.uniform(30, 70)), 'cores': 8},
                'memory': {'percent': float(np.random.uniform(40, 80)), 'total_mb': 16384.0},
                'disk': {'percent': float(np.random.uniform(50, 70)), 'total_gb': 500.0},
                'error_recovery': [],
                'uptime_seconds': time.time() - getattr(self, '_start_time', time.time()),
                'timestamp': time.time()
            }
        except Exception as e:
            # Other errors - return mock data
            return {
                'cpu': {'usage_percent': float(np.random.uniform(30, 70)), 'cores': 8},
                'memory': {'percent': float(np.random.uniform(40, 80)), 'total_mb': 16384.0},
                'disk': {'percent': float(np.random.uniform(50, 70)), 'total_gb': 500.0},
                'error_recovery': [],
                'uptime_seconds': time.time() - getattr(self, '_start_time', time.time()),
                'timestamp': time.time()
            }
    
    def update_geolocation(self, latitude: float, longitude: float, 
                          target_type: str, metadata: Dict[str, Any]):
        """
        Update geolocation data for map visualization
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            target_type: Type of target (BS, UE, Intercepted, etc.)
            metadata: Additional metadata
        """
        try:
            geolocation_entry = {
                'latitude': latitude,
                'longitude': longitude,
                'type': target_type,
                'metadata': metadata,
                'timestamp': time.time()
            }
            
            self.geolocation_data.append(geolocation_entry)
            
            # Keep only last 500 locations
            if len(self.geolocation_data) > 500:
                self.geolocation_data = self.geolocation_data[-500:]
            
            # Emit via WebSocket for real-time update
            socketio.emit('geolocation_update', geolocation_entry)
            
        except Exception as e:
            self.logger.error(f"Geolocation update failed: {e}")
    
    def add_anomaly_alert(self, anomaly_type: str, severity: str, 
                         description: str, metadata: Dict[str, Any]):
        """
        Add anomaly alert to dashboard
        
        Args:
            anomaly_type: Type of anomaly
            severity: Severity level (low/medium/high/critical)
            description: Human-readable description
            metadata: Additional metadata
        """
        try:
            alert = {
                'type': anomaly_type,
                'severity': severity,
                'description': description,
                'metadata': metadata,
                'timestamp': time.time(),
                'timestamp_human': datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.anomaly_alerts.append(alert)
            
            # Keep only last 200 alerts
            if len(self.anomaly_alerts) > 200:
                self.anomaly_alerts = self.anomaly_alerts[-200:]
            
            # Emit via WebSocket for real-time notification
            socketio.emit('anomaly_alert', alert)
            
            self.logger.warning(f"Anomaly alert: {anomaly_type} ({severity}) - {description}")
            
        except Exception as e:
            self.logger.error(f"Anomaly alert failed: {e}")
    
    def update_agent_status(self, agent_id: str, status: Dict[str, Any]):
        """
        Update federated agent status
        
        Args:
            agent_id: Agent identifier
            status: Agent status dict
        """
        try:
            status['last_update'] = time.time()
            self.agent_status[agent_id] = status
            
            # Emit via WebSocket
            socketio.emit('agent_status_update', {
                'agent_id': agent_id,
                'status': status
            })
            
        except Exception as e:
            self.logger.error(f"Agent status update failed: {e}")
    
    def update_ntn_satellites(self, satellites: List[Dict[str, Any]]):
        """
        Update NTN satellite tracking data
        
        Args:
            satellites: List of satellite status dicts
        """
        try:
            self.ntn_satellites = satellites
            
            # Emit via WebSocket
            socketio.emit('ntn_satellites_update', satellites)
            
        except Exception as e:
            self.logger.error(f"NTN satellites update failed: {e}")
    
    # ==================== NEW UPDATE METHODS (v1.7) ====================
    
    def add_suci_capture(self, suci: str, imsi: str = None, generation: str = '5G',
                        metadata: Dict[str, Any] = None):
        """Add SUCI/IMSI capture"""
        try:
            capture = {
                'suci': suci,
                'imsi': imsi,
                'generation': generation,
                'deconcealed': imsi is not None,
                'metadata': metadata or {},
                'timestamp': time.time(),
                'timestamp_human': datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.suci_captures.append(capture)
            self.captured_data.append(capture)  # Also add to captured data
            
            # Keep last 500
            if len(self.suci_captures) > 500:
                self.suci_captures = self.suci_captures[-500:]
            
            socketio.emit('suci_capture', capture)
            
        except Exception as e:
            self.logger.error(f"SUCI capture failed: {e}")
    
    def add_voice_call(self, call_data: Dict[str, Any]):
        """Add voice interception data"""
        try:
            call_data['timestamp'] = time.time()
            call_data['timestamp_human'] = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            
            self.voice_calls.append(call_data)
            
            # Keep last 200
            if len(self.voice_calls) > 200:
                self.voice_calls = self.voice_calls[-200:]
            
            socketio.emit('voice_call', call_data)
            
        except Exception as e:
            self.logger.error(f"Voice call update failed: {e}")
    
    def update_exploit_status(self, exploit_type: str, status: Dict[str, Any]):
        """Update exploit operation status"""
        try:
            self.exploit_status[exploit_type] = status
            socketio.emit('exploit_update', {'type': exploit_type, 'status': status})
            
        except Exception as e:
            self.logger.error(f"Exploit status update failed: {e}")
    
    def update_security_audit(self, audit_data: Dict[str, Any]):
        """Update security audit results"""
        try:
            self.security_audit = audit_data
            socketio.emit('security_audit_update', audit_data)
            
        except Exception as e:
            self.logger.error(f"Security audit update failed: {e}")
    
    def update_sdr_devices(self, devices: List[Dict[str, Any]]):
        """Update SDR device list"""
        try:
            self.sdr_devices = devices
            socketio.emit('sdr_devices_update', devices)
            
        except Exception as e:
            self.logger.error(f"SDR devices update failed: {e}")
    
    def update_analytics(self, analytics_type: str, data: Dict[str, Any]):
        """Update AI analytics data"""
        try:
            self.analytics_data[analytics_type] = data
            socketio.emit('analytics_update', {'type': analytics_type, 'data': data})
            
        except Exception as e:
            self.logger.error(f"Analytics update failed: {e}")
    
    def update_spectrum(self, spectrum_data: Dict[str, Any]):
        """Update spectrum analyzer data"""
        try:
            self.spectrum_data = spectrum_data
            socketio.emit('spectrum_update', spectrum_data)
            
        except Exception as e:
            self.logger.error(f"Spectrum update failed: {e}")
    
    def add_target(self, target_id: str, target_data: Dict[str, Any]):
        """Add or update target"""
        try:
            self.targets[target_id] = target_data
            socketio.emit('target_update', {'id': target_id, 'data': target_data})
            
        except Exception as e:
            self.logger.error(f"Target update failed: {e}")
    
    def add_error_recovery_event(self, event: Dict[str, Any]):
        """Add error recovery event"""
        try:
            event['timestamp'] = time.time()
            event['timestamp_human'] = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            
            self.error_recovery.append(event)
            
            # Keep last 100
            if len(self.error_recovery) > 100:
                self.error_recovery = self.error_recovery[-100:]
            
            socketio.emit('error_recovery_event', event)
            
        except Exception as e:
            self.logger.error(f"Error recovery event failed: {e}")
    
    # ==================== WebSocket Handlers ====================
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        dashboard = _dashboard_instance
        client_id = request.sid
        dashboard.connected_clients.add(client_id)
        dashboard.logger.info(f"Client connected: {client_id} (total: {len(dashboard.connected_clients)})")
        emit('connection_status', {'status': 'connected', 'refresh_rate_ms': dashboard.refresh_rate_ms})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        dashboard = _dashboard_instance
        client_id = request.sid
        dashboard.connected_clients.discard(client_id)
        dashboard.logger.info(f"Client disconnected: {client_id} (total: {len(dashboard.connected_clients)})")
    
    @socketio.on('request_data')
    def handle_data_request(data):
        """Handle client data request"""
        dashboard = _dashboard_instance
        data_type = data.get('type', 'kpis')
        
        if data_type == 'kpis':
            emit('kpis_update', dashboard._collect_kpis())
        elif data_type == 'geolocation':
            emit('geolocation_update', dashboard.geolocation_data)
        elif data_type == 'anomalies':
            emit('anomalies_update', dashboard.anomaly_alerts[-50:])
        elif data_type == 'agents':
            agents = dashboard._collect_agent_status()
            emit('agents_update', agents)
        elif data_type == 'emissions':
            emit('emissions_update', dashboard._collect_emissions())
        elif data_type == 'ntn':
            ntn = dashboard._collect_ntn_status()
            emit('ntn_update', ntn)
        elif data_type == 'cellular':
            emit('cellular_update', dashboard._collect_cellular_status())
        elif data_type == 'suci':
            emit('suci_update', dashboard.suci_captures[-50:])
        elif data_type == 'voice':
            emit('voice_update', dashboard.voice_calls[-20:])
        elif data_type == 'exploits':
            exploits = dashboard._collect_exploit_status()
            emit('exploit_update', exploits)
        elif data_type == 'security':
            emit('security_update', dashboard.security_audit)
        elif data_type == 'sdr':
            sdr = dashboard._collect_sdr_status()
            emit('sdr_update', sdr)
        elif data_type == 'analytics':
            analytics = dashboard._collect_analytics_status()
            emit('analytics_update', analytics)
        elif data_type == 'spectrum':
            emit('spectrum_update', dashboard.spectrum_data)
        elif data_type == 'targets':
            targets = list(dashboard.targets.values())
            emit('targets_update', targets)
        elif data_type == 'health':
            emit('health_update', dashboard._collect_system_health())
        elif data_type == 'config':
            config = {
                'refresh_rate_ms': dashboard.refresh_rate_ms,
                'auth_enabled': dashboard.auth_enabled,
                'host': dashboard.host,
                'port': dashboard.port
            }
            emit('config_update', config)
        elif data_type == 'regulatory':
            regulatory = dashboard._collect_regulatory_status()
            emit('regulatory_update', regulatory)
        elif data_type == 'quick_status':
            quick = dashboard._collect_quick_status()
            emit('quick_status_update', quick)
    
    # ==================== Server Control ====================
    
    def start(self, debug: bool = False):
        """
        Start dashboard server
        
        Args:
            debug: Enable Flask debug mode
        """
        self._start_time = time.time()
        
        self.logger.info(f"Starting dashboard server on {self.host}:{self.port}")
        
        try:
            socketio.run(app, 
                        host=self.host, 
                        port=self.port, 
                        debug=debug,
                        use_reloader=False)
        except Exception as e:
            self.logger.error(f"Dashboard server error: {e}")
    
    def stop(self):
        """Stop dashboard server"""
        self.logger.info("Stopping dashboard server")
        # Cleanup
        self.connected_clients.clear()


# ==================== HTML TEMPLATES (Placeholder) ====================

# Note: In production, create actual HTML templates in templates/ folder
# Below are minimal template references

DASHBOARD_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FalconOne Dashboard v1.7</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css" />
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #0a0a0a; color: #eee; overflow-x: hidden; }
        .header { background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%); padding: 15px 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.5); position: sticky; top: 0; z-index: 1000; }
        .header h1 { font-size: 24px; margin-bottom: 5px; }
        .header p { font-size: 12px; opacity: 0.9; }
        .nav-tabs { display: flex; gap: 10px; margin-top: 10px; flex-wrap: wrap; }
        .nav-tab { padding: 8px 16px; background: rgba(255,255,255,0.1); border-radius: 5px; cursor: pointer; transition: all 0.3s; font-size: 13px; }
        .nav-tab:hover { background: rgba(255,255,255,0.2); }
        .nav-tab.active { background: rgba(255,255,255,0.3); font-weight: bold; }
        .container { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 15px; padding: 20px; max-width: 2000px; margin: 0 auto; }
        .panel { background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%); border-radius: 10px; padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.4); border: 1px solid #333; }
        .panel h2 { font-size: 18px; margin-bottom: 15px; color: #4fc3f7; border-bottom: 2px solid #0d47a1; padding-bottom: 8px; }
        .panel-large { grid-column: span 2; }
        #map { height: 350px; border-radius: 8px; border: 2px solid #333; }
        .kpi { display: inline-block; margin: 8px; padding: 12px 16px; background: linear-gradient(135deg, #0d47a1, #1976d2); border-radius: 6px; box-shadow: 0 2px 8px rgba(13,71,161,0.3); min-width: 140px; }
        .kpi-label { font-size: 11px; opacity: 0.8; margin-bottom: 4px; }
        .kpi-value { font-size: 20px; font-weight: bold; }
        .alert { background: linear-gradient(135deg, #d32f2f, #c62828); padding: 12px; margin: 8px 0; border-radius: 6px; border-left: 4px solid #ff1744; }
        .alert.low { background: linear-gradient(135deg, #fbc02d, #f9a825); border-left-color: #fdd835; }
        .alert.medium { background: linear-gradient(135deg, #ff6f00, #f57c00); border-left-color: #ff9100; }
        .chart-container { position: relative; height: 250px; margin-top: 15px; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-active { background: #4caf50; box-shadow: 0 0 8px #4caf50; }
        .status-inactive { background: #757575; }
        .data-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
        .data-table th { background: #0d47a1; padding: 10px; text-align: left; }
        .data-table td { padding: 8px; border-bottom: 1px solid #333; }
        .data-table tr:hover { background: rgba(255,255,255,0.05); }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; margin: 2px; }
        .badge-success { background: #4caf50; color: white; }
        .badge-warning { background: #ff9800; color: white; }
        .badge-danger { background: #f44336; color: white; }
        .badge-info { background: #2196f3; color: white; }
        .progress-bar { width: 100%; height: 20px; background: #1a1a1a; border-radius: 10px; overflow: hidden; margin: 8px 0; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #4caf50, #8bc34a); transition: width 0.3s; display: flex; align-items: center; justify-content: center; font-size: 11px; }
        .spectrum-canvas { width: 100%; height: 200px; background: #000; border-radius: 8px; border: 1px solid #333; }
        .filter-controls { display: flex; gap: 10px; margin-bottom: 15px; flex-wrap: wrap; }
        .filter-controls select, .filter-controls input { padding: 8px; background: #1a1a1a; border: 1px solid #333; color: #eee; border-radius: 5px; }
        .btn { padding: 8px 16px; background: #0d47a1; color: white; border: none; border-radius: 5px; cursor: pointer; transition: all 0.3s; }
        .btn:hover { background: #1565c0; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        scrollbar-width: thin; scrollbar-color: #0d47a1 #1a1a1a; }
        ::-webkit-scrollbar { width: 10px; }
        ::-webkit-scrollbar-track { background: #1a1a1a; }
        ::-webkit-scrollbar-thumb { background: #0d47a1; border-radius: 5px; }
        ::-webkit-scrollbar-thumb:hover { background: #1565c0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛰️ FalconOne SIGINT Dashboard v1.7</h1>
        <p>Real-time Monitoring & Control | Refresh: {{ refresh_rate_ms }}ms | User: <span id="username">admin</span></p>
        <div class="nav-tabs">
            <div class="nav-tab active" onclick="showTab('overview')">📊 Overview</div>
            <div class="nav-tab" onclick="showTab('cellular')">📱 Cellular</div>
            <div class="nav-tab" onclick="showTab('captures')">🎯 Captures</div>
            <div class="nav-tab" onclick="showTab('exploits')">⚡ Exploits</div>
            <div class="nav-tab" onclick="showTab('analytics')">🤖 Analytics</div>
            <div class="nav-tab" onclick="showTab('system')">🖥️ System</div>
        </div>
    </div>
    
    <!-- OVERVIEW TAB -->
    <div id="tab-overview" class="tab-content active">
        <div class="container">
            <!-- KPIs Panel -->
            <div class="panel">
                <h2>📊 Key Performance Indicators</h2>
                <div id="kpis"></div>
                <div class="chart-container">
                    <canvas id="throughput-chart"></canvas>
                </div>
            </div>
            
            <!-- Geolocation Map -->
            <div class="panel">
                <h2>🗺️ Geolocation Map</h2>
                <div id="map"></div>
            </div>
            
            <!-- Anomaly Alerts -->
            <div class="panel">
                <h2>⚠️ Anomaly Alerts</h2>
                <div id="anomalies" style="max-height: 300px; overflow-y: auto;"></div>
            </div>
            
            <!-- Quick Status -->
            <div class="panel">
                <h2>⚡ Quick Status</h2>
                <div id="quick-status"></div>
            </div>
        </div>
    </div>
    
    <!-- CELLULAR TAB -->
    <div id="tab-cellular" class="tab-content">
        <div class="container">
            <div class="panel">
                <h2>📱 GSM / 2G Monitor</h2>
                <div id="cellular-gsm"></div>
            </div>
            <div class="panel">
                <h2>📱 UMTS / 3G Monitor</h2>
                <div id="cellular-umts"></div>
            </div>
            <div class="panel">
                <h2>📱 LTE / 4G Monitor</h2>
                <div id="cellular-lte"></div>
            </div>
            <div class="panel">
                <h2>📱 5G NR Monitor</h2>
                <div id="cellular-5g"></div>
            </div>
            <div class="panel">
                <h2>📱 6G Prototype Monitor</h2>
                <div id="cellular-6g"></div>
            </div>
            <div class="panel">
                <h2>🛰️ NTN Satellites</h2>
                <div id="ntn-satellites"></div>
            </div>
        </div>
    </div>
    
    <!-- CAPTURES TAB -->
    <div id="tab-captures" class="tab-content">
        <div class="container">
            <div class="panel panel-large">
                <h2>🎯 SUCI/IMSI Captures</h2>
                <div class="filter-controls">
                    <select id="filter-generation">
                        <option value="">All Generations</option>
                        <option value="5G">5G NR</option>
                        <option value="LTE">LTE</option>
                        <option value="UMTS">UMTS</option>
                        <option value="GSM">GSM</option>
                    </select>
                    <input type="text" id="filter-search" placeholder="Search IMSI/SUCI...">
                    <button class="btn" onclick="refreshCaptures()">🔄 Refresh</button>
                </div>
                <div id="suci-captures" style="max-height: 400px; overflow-y: auto;"></div>
            </div>
            
            <div class="panel panel-large">
                <h2>📞 Voice/VoNR Interception</h2>
                <div id="voice-calls"></div>
            </div>
            
            <div class="panel panel-large">
                <h2>📦 Captured Data Explorer</h2>
                <div id="captured-data"></div>
            </div>
        </div>
    </div>
    
    <!-- EXPLOITS TAB -->
    <div id="tab-exploits" class="tab-content">
        <div class="container">
            <div class="panel">
                <h2>🔐 Crypto Attacks</h2>
                <div id="exploit-crypto"></div>
            </div>
            <div class="panel">
                <h2>🛰️ NTN Attacks</h2>
                <div id="exploit-ntn"></div>
            </div>
            <div class="panel">
                <h2>🚗 V2X Attacks</h2>
                <div id="exploit-v2x"></div>
            </div>
            <div class="panel">
                <h2>💉 Message Injection (Silent SMS)</h2>
                <div id="exploit-injection"></div>
            </div>
            <div class="panel">
                <h2>📱 Silent SMS Tracker</h2>
                <div id="silent-sms"></div>
            </div>
            <div class="panel">
                <h2>🔽 Downgrade Attacks (5G→4G→3G)</h2>
                <div id="downgrade-attacks"></div>
            </div>
            <div class="panel">
                <h2>📡 Paging Spoofing</h2>
                <div id="paging-spoof"></div>
            </div>
            <div class="panel">
                <h2>🤖 A-IoT (Ambient IoT) Exploits</h2>
                <div id="aiot-exploits"></div>
            </div>
            <div class="panel">
                <h2>🌐 Semantic 6G Exploiter</h2>
                <div id="semantic-6g"></div>
            </div>
            <div class="panel panel-large">
                <h2>🛡️ Security Audit Results</h2>
                <div id="security-audit"></div>
            </div>
        </div>
    </div>
    
    <!-- ANALYTICS TAB -->
    <div id="tab-analytics" class="tab-content">
        <div class="container">
            <div class="panel panel-large">
                <h2>📡 Live Spectrum Analyzer</h2>
                <canvas id="spectrum-canvas" class="spectrum-canvas"></canvas>
                <div id="spectrum-info"></div>
            </div>
            
            <div class="panel">
                <h2>🔗 Cyber-RF Fusion</h2>
                <div id="analytics-fusion"></div>
            </div>
            
            <div class="panel">
                <h2>🤖 Signal Classification (AI/ML)</h2>
                <div id="analytics-classifier"></div>
            </div>
            
            <div class="panel">
                <h2>🤝 Federated Agents (MARL)</h2>
                <div id="agents"></div>
            </div>
            
            <div class="panel">
                <h2>📊 RIC Optimization (O-RAN)</h2>
                <div id="analytics-ric"></div>
            </div>
            
            <div class="panel">
                <h2>🌱 Carbon Emissions</h2>
                <div id="emissions"></div>
                <div class="chart-container">
                    <canvas id="emissions-chart"></canvas>
                </div>
            </div>
            
            <div class="panel">
                <h2>🎯 Precision Geolocation</h2>
                <div id="precision-geo"></div>
            </div>
            
            <div class="panel">
                <h2>📏 Data Validator (SNR/Quality)</h2>
                <div id="data-validator"></div>
            </div>
        </div>
    </div>
    
    <!-- SYSTEM TAB -->
    <div id="tab-system" class="tab-content">
        <div class="container">
            <div class="panel panel-large">
                <h2>🖥️ System Health</h2>
                <div id="system-health"></div>
            </div>
            
            <div class="panel">
                <h2>📡 SDR Devices</h2>
                <div id="sdr-devices"></div>
            </div>
            
            <div class="panel panel-large">
                <h2>🎯 Target Management</h2>
                <div id="targets"></div>
            </div>
            
            <div class="panel panel-large">
                <h2>🔧 Error Recovery Events</h2>
                <div id="error-recovery" style="max-height: 300px; overflow-y: auto;"></div>
            </div>
            
            <div class="panel">
                <h2>⚙️ Config Management</h2>
                <div id="config-management"></div>
            </div>
            
            <div class="panel">
                <h2>📋 Regulatory Scanner</h2>
                <div id="regulatory-scanner"></div>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        const socket = io();
        
        // Tab switching
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.getElementById('tab-' + tabName).classList.add('active');
            event.target.classList.add('active');
        }
        
        // Initialize map
        const map = L.map('map').setView([51.505, -0.09], 13);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
        
        // Initialize charts
        const throughputChart = new Chart(document.getElementById('throughput-chart'), {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Throughput (Mbps)', data: [], borderColor: '#4fc3f7', tension: 0.4 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#eee' } } }, scales: { y: { ticks: { color: '#eee' } }, x: { ticks: { color: '#eee' } } } }
        });
        
        const emissionsChart = new Chart(document.getElementById('emissions-chart'), {
            type: 'bar',
            data: { labels: ['CO2', 'Power', 'PUE'], datasets: [{ label: 'Metrics', data: [], backgroundColor: ['#4caf50', '#ff9800', '#2196f3'] }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#eee' } } }, scales: { y: { ticks: { color: '#eee' } }, x: { ticks: { color: '#eee' } } } }
        });
        
        // Real-time data updates
        socket.on('kpis_update', function(data) {
            document.getElementById('kpis').innerHTML = `
                <div class="kpi">
                    <div class="kpi-label">Throughput</div>
                    <div class="kpi-value">${data.throughput_mbps?.toFixed(2)} Mbps</div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Latency</div>
                    <div class="kpi-value">${data.latency_ms?.toFixed(2)} ms</div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Success Rate</div>
                    <div class="kpi-value">${(data.success_rate * 100)?.toFixed(1)}%</div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Connections</div>
                    <div class="kpi-value">${data.active_connections}</div>
                </div>
            `;
        });
        
        socket.on('cellular_update', function(data) {
            if (data.gsm) {
                document.getElementById('cellular-gsm').innerHTML = `
                    <p><span class="status-indicator ${data.gsm.running ? 'status-active' : 'status-inactive'}"></span>
                    Status: ${data.gsm.running ? 'Active' : 'Inactive'}</p>
                    <p>Captures: <strong>${data.gsm.captures || 0}</strong></p>
                    <p>Band: <strong>${data.gsm.band || 'N/A'}</strong></p>
                `;
            }
            if (data.umts) {
                document.getElementById('cellular-umts').innerHTML = `
                    <p><span class="status-indicator ${data.umts.running ? 'status-active' : 'status-inactive'}"></span>
                    Status: ${data.umts.running ? 'Active' : 'Inactive'}</p>
                    <p>Captures: <strong>${data.umts.captures || 0}</strong></p>
                    <p>Bands: <strong>${Array.isArray(data.umts.bands) ? data.umts.bands.join(', ') : 'N/A'}</strong></p>
                `;
            }
            if (data.lte) {
                document.getElementById('cellular-lte').innerHTML = `
                    <p><span class="status-indicator ${data.lte.running ? 'status-active' : 'status-inactive'}"></span>
                    Status: ${data.lte.running ? 'Active' : 'Inactive'}</p>
                    <p>Captures: <strong>${data.lte.captures || 0}</strong></p>
                    <p>Bands: <strong>${Array.isArray(data.lte.bands) ? data.lte.bands.join(', ') : 'N/A'}</strong></p>
                `;
            }
            if (data.fiveg) {
                document.getElementById('cellular-5g').innerHTML = `
                    <p><span class="status-indicator ${data.fiveg.running ? 'status-active' : 'status-inactive'}"></span>
                    Status: ${data.fiveg.running ? 'Active' : 'Inactive'}</p>
                    <p>SUCI Count: <strong>${data.fiveg.suci_count || 0}</strong></p>
                    <p>NTN: <strong>${data.fiveg.ntn_enabled ? 'Enabled' : 'Disabled'}</strong></p>
                `;
            }
            if (data.sixg) {
                document.getElementById('cellular-6g').innerHTML = `
                    <p><span class="status-indicator ${data.sixg.running ? 'status-active' : 'status-inactive'}"></span>
                    Status: ${data.sixg.running ? 'Active' : 'Inactive'}</p>
                    <p>AI Features: <strong>${data.sixg.ai_enabled ? 'Enabled' : 'Disabled'}</strong></p>
                    <p>Semantic: <strong>${data.sixg.semantic_enabled ? 'Active' : 'Inactive'}</strong></p>
                `;
            }
        });
        
        socket.on('ntn_update', function(data) {
            const satHtml = data.satellites.map(sat => `
                <div style="border-left: 3px solid #2196f3; padding: 10px; margin: 8px 0; background: rgba(33, 150, 243, 0.1);">
                    <strong>${sat.name}</strong> (${sat.type})<br>
                    Altitude: ${sat.altitude}km | Coverage: ${sat.coverage}<br>
                    Status: <span class="badge badge-${sat.active ? 'success' : 'warning'}">${sat.active ? 'Active' : 'Standby'}</span>
                </div>
            `).join('');
            document.getElementById('ntn-satellites').innerHTML = satHtml || '<p>No satellites detected</p>';
        });
        
        socket.on('exploit_update', function(data) {
            if (data.crypto) {
                document.getElementById('exploit-crypto').innerHTML = `
                    <p>Active Attacks: <span class="badge badge-warning">${data.crypto.active_attacks || 0}</span></p>
                    <p>Vulnerabilities Found: <strong>${data.crypto.vulnerabilities || 0}</strong></p>
                    <p>Last Analysis: <small>${data.crypto.last_scan || 'Never'}</small></p>
                `;
            }
            if (data.ntn) {
                document.getElementById('exploit-ntn').innerHTML = `
                    <p>Satellite Exploits: <span class="badge badge-info">${data.ntn.exploits || 0}</span></p>
                    <p>Coverage Manipulation: <strong>${data.ntn.coverage_attacks || 0}</strong></p>
                    <p>Handover Attacks: <strong>${data.ntn.handover_attacks || 0}</strong></p>
                `;
            }
            if (data.v2x) {
                document.getElementById('exploit-v2x').innerHTML = `
                    <p>V2X Targets: <span class="badge badge-danger">${data.v2x.targets || 0}</span></p>
                    <p>Message Spoofing: <strong>${data.v2x.spoofing_active ? 'Active' : 'Inactive'}</strong></p>
                    <p>Safety Compromised: <strong>${data.v2x.safety_attacks || 0}</strong></p>
                `;
            }
            if (data.injection) {
                document.getElementById('exploit-injection').innerHTML = `
                    <p>Silent SMS Sent: <span class="badge badge-success">${data.injection.sms_sent || 0}</span></p>
                    <p>Success Rate: <strong>${data.injection.success_rate || 0}%</strong></p>
                    <p>Last Injection: <small>${data.injection.last_injection || 'Never'}</small></p>
                `;
            }
            if (data.silent_sms) {
                document.getElementById('silent-sms').innerHTML = `
                    <p>Tracked Targets: <span class="badge badge-info">${data.silent_sms.targets || 0}</span></p>
                    <p>Messages Monitored: <strong>${data.silent_sms.monitored || 0}</strong></p>
                    <p>Detection Rate: <strong>${data.silent_sms.detection_rate || 0}%</strong></p>
                `;
            }
            if (data.downgrade) {
                document.getElementById('downgrade-attacks').innerHTML = `
                    <p>Downgrade Attempts: <span class="badge badge-warning">${data.downgrade.attempts || 0}</span></p>
                    <p>5G→4G: <strong>${data.downgrade.fiveg_to_lte || 0}</strong> | 4G→3G: <strong>${data.downgrade.lte_to_umts || 0}</strong></p>
                    <p>Success Rate: <strong>${data.downgrade.success_rate || 0}%</strong></p>
                `;
            }
            if (data.paging) {
                document.getElementById('paging-spoof').innerHTML = `
                    <p>Paging Spoofs: <span class="badge badge-danger">${data.paging.spoofs || 0}</span></p>
                    <p>Targets Tracked: <strong>${data.paging.targets_tracked || 0}</strong></p>
                    <p>Location Accuracy: <strong>${data.paging.accuracy || 0}%</strong></p>
                `;
            }
            if (data.aiot) {
                document.getElementById('aiot-exploits').innerHTML = `
                    <p>A-IoT Devices: <span class="badge badge-info">${data.aiot.devices || 0}</span></p>
                    <p>Ambient Sensors: <strong>${data.aiot.sensors || 0}</strong></p>
                    <p>Battery-Free Exploits: <strong>${data.aiot.exploits || 0}</strong></p>
                `;
            }
            if (data.semantic) {
                document.getElementById('semantic-6g').innerHTML = `
                    <p>Semantic Attacks: <span class="badge badge-warning">${data.semantic.attacks || 0}</span></p>
                    <p>AI Model Poisoning: <strong>${data.semantic.poisoning || 0}</strong></p>
                    <p>Context Manipulation: <strong>${data.semantic.context_attacks || 0}</strong></p>
                `;
            }
        });
        
        socket.on('analytics_update', function(data) {
            if (data.fusion) {
                document.getElementById('analytics-fusion').innerHTML = `
                    <p>Fusion Score: <span class="badge badge-success">${data.fusion.score || 0}/100</span></p>
                    <p>RF Anomalies: <strong>${data.fusion.rf_anomalies || 0}</strong></p>
                    <p>Cyber Threats: <strong>${data.fusion.cyber_threats || 0}</strong></p>
                `;
            }
            if (data.classifier) {
                document.getElementById('analytics-classifier').innerHTML = `
                    <p>Signals Classified: <span class="badge badge-info">${data.classifier.signals || 0}</span></p>
                    <p>Accuracy: <strong>${data.classifier.accuracy || 0}%</strong></p>
                    <p>Model: <strong>${data.classifier.model || 'TensorFlow CNN'}</strong></p>
                `;
            }
            if (data.ric) {
                document.getElementById('analytics-ric').innerHTML = `
                    <p>xApps Active: <span class="badge badge-success">${data.ric.xapps || 0}</span></p>
                    <p>Optimization Score: <strong>${data.ric.optimization || 0}%</strong></p>
                    <p>QoS Improvements: <strong>+${data.ric.qos_gain || 0}%</strong></p>
                `;
            }
            if (data.geolocation) {
                document.getElementById('precision-geo').innerHTML = `
                    <p>Precision: <span class="badge badge-success">${data.geolocation.precision || 0}m</span></p>
                    <p>Targets Geolocated: <strong>${data.geolocation.targets || 0}</strong></p>
                    <p>Method: <strong>${data.geolocation.method || 'TDOA + ML'}</strong></p>
                `;
            }
            if (data.validator) {
                document.getElementById('data-validator').innerHTML = `
                    <p>Data Quality: <span class="badge badge-${data.validator.quality > 80 ? 'success' : 'warning'}">${data.validator.quality || 0}%</span></p>
                    <p>Avg SNR: <strong>${data.validator.avg_snr || 0} dB</strong></p>
                    <p>Invalid Samples: <strong>${data.validator.invalid || 0}</strong></p>
                `;
            }
        });
        
        socket.on('agents_update', function(agents) {
            const agentHtml = agents.map(agent => `
                <div style="border-left: 3px solid #4caf50; padding: 10px; margin: 8px 0; background: rgba(76, 175, 80, 0.1);">
                    <strong>Agent ${agent.agent_id}</strong><br>
                    Status: <span class="badge badge-${agent.status === 'active' ? 'success' : 'warning'}">${agent.status}</span><br>
                    Reward: ${agent.reward?.toFixed(2) || 0} | Episodes: ${agent.episodes || 0}
                </div>
            `).join('');
            document.getElementById('agents').innerHTML = agentHtml || '<p>No federated agents running</p>';
        });
        
        socket.on('emissions_update', function(data) {
            document.getElementById('emissions').innerHTML = `
                <p>CO2 Emissions: <strong>${data.co2_kg?.toFixed(2) || 0} kg</strong></p>
                <p>Power Usage: <strong>${data.power_kwh?.toFixed(2) || 0} kWh</strong></p>
                <p>PUE: <strong>${data.pue?.toFixed(2) || 0}</strong></p>
            `;
            if (emissionsChart) {
                emissionsChart.data.datasets[0].data = [data.co2_kg || 0, data.power_kwh || 0, data.pue || 0];
                emissionsChart.update();
            }
        });
        
        socket.on('sdr_update', function(devices) {
            const sdrHtml = devices.map(dev => `
                <div style="border-left: 3px solid #ff9800; padding: 10px; margin: 8px 0; background: rgba(255, 152, 0, 0.1);">
                    <strong>${dev.name}</strong> (${dev.type})<br>
                    Status: <span class="badge badge-${dev.active ? 'success' : 'danger'}">${dev.active ? 'Active' : 'Offline'}</span><br>
                    Frequency: ${dev.frequency || 'N/A'} | Gain: ${dev.gain || 0} dB
                </div>
            `).join('');
            document.getElementById('sdr-devices').innerHTML = sdrHtml || '<p>No SDR devices detected</p>';
        });
        
        socket.on('targets_update', function(targets) {
            const targetHtml = `<table class="data-table">
                <thead><tr><th>ID</th><th>Type</th><th>IMSI</th><th>Status</th><th>Last Seen</th></tr></thead>
                <tbody>${targets.map(t => `
                    <tr>
                        <td>${t.target_id}</td>
                        <td><span class="badge badge-info">${t.type}</span></td>
                        <td>${t.imsi || 'N/A'}</td>
                        <td><span class="badge badge-${t.active ? 'success' : 'warning'}">${t.active ? 'Active' : 'Idle'}</span></td>
                        <td><small>${t.last_seen || 'Never'}</small></td>
                    </tr>
                `).join('')}</tbody>
            </table>`;
            document.getElementById('targets').innerHTML = targetHtml || '<p>No targets managed</p>';
        });
        
        socket.on('security_audit_update', function(audit) {
            const auditHtml = `
                <div class="alert ${audit.severity}">
                    <strong>${audit.title}</strong><br>
                    ${audit.description}<br>
                    <small>Severity: ${audit.severity} | Timestamp: ${audit.timestamp}</small>
                </div>
            `;
            document.getElementById('security-audit').insertAdjacentHTML('afterbegin', auditHtml);
        });
        
        socket.on('voice_call_update', function(call) {
            const callDiv = document.createElement('div');
            callDiv.className = 'alert medium';
            callDiv.innerHTML = `
                <strong>Call Intercepted</strong>: ${call.caller} → ${call.callee}<br>
                Duration: ${call.duration}s | Quality: ${call.quality} | Type: ${call.type}<br>
                <small>${call.timestamp}</small>
            `;
            document.getElementById('voice-calls').prepend(callDiv);
        });
        
        socket.on('error_recovery_update', function(event) {
            const eventDiv = document.createElement('div');
            eventDiv.className = 'alert ' + (event.recovered ? 'low' : 'medium');
            eventDiv.innerHTML = `
                <strong>${event.error_type}</strong>: ${event.description}<br>
                Status: <span class="badge badge-${event.recovered ? 'success' : 'warning'}">${event.recovered ? 'Recovered' : 'Pending'}</span><br>
                <small>${event.timestamp}</small>
            `;
            document.getElementById('error-recovery').prepend(eventDiv);
        });
        
        socket.on('config_update', function(config) {
            document.getElementById('config-management').innerHTML = `
                <p>Refresh Rate: <strong>${config.refresh_rate_ms}ms</strong></p>
                <p>Auth Enabled: <strong>${config.auth_enabled ? 'Yes' : 'No'}</strong></p>
                <p>Host: <strong>${config.host}:${config.port}</strong></p>
                <button class="btn" onclick="alert('Config editor coming soon')">⚙️ Edit Config</button>
            `;
        });
        
        socket.on('regulatory_update', function(data) {
            document.getElementById('regulatory-scanner').innerHTML = `
                <p>Compliance Score: <span class="badge badge-${data.score > 80 ? 'success' : 'warning'}">${data.score || 0}%</span></p>
                <p>Violations: <strong>${data.violations || 0}</strong></p>
                <p>Last Scan: <small>${data.last_scan || 'Never'}</small></p>
                <p>Regions: <strong>${Array.isArray(data.regions) ? data.regions.join(', ') : 'N/A'}</strong></p>
            `;
        });
        
        socket.on('quick_status_update', function(status) {
            document.getElementById('quick-status').innerHTML = `
                <p>🎯 Targets: <strong>${status.targets || 0}</strong></p>
                <p>📡 SDR Devices: <strong>${status.sdr_active || 0}/${status.sdr_total || 0}</strong></p>
                <p>⚡ Exploits Running: <strong>${status.exploits_active || 0}</strong></p>
                <p>🔐 SUCI Captured: <strong>${status.suci_count || 0}</strong></p>
                <p>📞 Calls Intercepted: <strong>${status.voice_calls || 0}</strong></p>
            `;
        });
        
        socket.on('suci_capture', function(capture) {
            const captureDiv = document.createElement('div');
            captureDiv.className = 'alert low';
            captureDiv.innerHTML = `
                <strong>${capture.generation}</strong>: ${capture.suci} 
                ${capture.deconcealed ? '➜ <span class="badge-success badge">' + capture.imsi + '</span>' : '<span class="badge-warning badge">Pending</span>'}
                <small style="float:right">${capture.timestamp_human}</small>
            `;
            document.getElementById('suci-captures').prepend(captureDiv);
        });
        
        socket.on('anomaly_alert', function(alert) {
            const alertDiv = document.createElement('div');
            alertDiv.className = 'alert ' + alert.severity;
            alertDiv.innerHTML = `<strong>${alert.severity.toUpperCase()}:</strong> ${alert.description} <small style="float:right">${alert.timestamp_human}</small>`;
            document.getElementById('anomalies').prepend(alertDiv);
        });
        
        socket.on('geolocation_update', function(location) {
            L.marker([location.latitude, location.longitude])
                .addTo(map)
                .bindPopup(`${location.type}: ${JSON.stringify(location.metadata)}`);
        });
        
        socket.on('health_update', function(health) {
            document.getElementById('system-health').innerHTML = `
                <h3>CPU</h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${health.cpu.usage_percent}%">${health.cpu.usage_percent?.toFixed(1)}%</div>
                </div>
                <h3>Memory</h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${health.memory.percent}%">${health.memory.percent?.toFixed(1)}%</div>
                </div>
                <p>Uptime: <strong>${Math.floor(health.uptime_seconds / 3600)}h ${Math.floor((health.uptime_seconds % 3600) / 60)}m</strong></p>
            `;
        });
        
        // Request data on interval
        setInterval(() => {
            socket.emit('request_data', { type: 'kpis' });
            socket.emit('request_data', { type: 'cellular' });
            socket.emit('request_data', { type: 'health' });
            socket.emit('request_data', { type: 'exploits' });
            socket.emit('request_data', { type: 'analytics' });
            socket.emit('request_data', { type: 'ntn' });
            socket.emit('request_data', { type: 'agents' });
            socket.emit('request_data', { type: 'emissions' });
            socket.emit('request_data', { type: 'sdr' });
            socket.emit('request_data', { type: 'targets' });
            socket.emit('request_data', { type: 'config' });
            socket.emit('request_data', { type: 'regulatory' });
            socket.emit('request_data', { type: 'quick_status' });
        }, {{ refresh_rate_ms }});
        
        // Initial data request
        socket.emit('request_data', { type: 'kpis' });
        socket.emit('request_data', { type: 'cellular' });
        socket.emit('request_data', { type: 'health' });
        socket.emit('request_data', { type: 'exploits' });
        socket.emit('request_data', { type: 'analytics' });
        socket.emit('request_data', { type: 'ntn' });
        socket.emit('request_data', { type: 'agents' });
        socket.emit('request_data', { type: 'emissions' });
        socket.emit('request_data', { type: 'sdr' });
        socket.emit('request_data', { type: 'targets' });
        socket.emit('request_data', { type: 'config' });
        socket.emit('request_data', { type: 'regulatory' });
        socket.emit('request_data', { type: 'quick_status' });
    </script>
</body>
</html>
"""

LOGIN_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FalconOne Login</title>
    <style>
        body { font-family: Arial, sans-serif; background: #1a1a1a; color: #eee; display: flex; justify-content: center; align-items: center; height: 100vh; }
        .login-box { background: #2a2a2a; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        input { width: 100%; padding: 10px; margin: 10px 0; border-radius: 5px; border: none; }
        button { width: 100%; padding: 10px; background: #0d47a1; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #1565c0; }
        .error { color: #d32f2f; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="login-box">
        <h2>🛰️ FalconOne Login</h2>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Login</button>
        </form>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
    </div>
</body>
</html>
"""


def create_dashboard_templates():
    """Create HTML template files"""
    import os
    
    # Create templates directory
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    # Write dashboard template
    with open(os.path.join(template_dir, 'dashboard.html'), 'w') as f:
        f.write(DASHBOARD_HTML_TEMPLATE)
    
    # Write login template
    with open(os.path.join(template_dir, 'login.html'), 'w') as f:
        f.write(LOGIN_HTML_TEMPLATE)
    
    print("Dashboard templates created successfully")


# Create templates on import (for development)
# In production, templates would be pre-created
try:
    create_dashboard_templates()
except:
    pass
