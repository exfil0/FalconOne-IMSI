"""
FalconOne Configuration Management
YAML-based configuration with validation and nested key access
v1.8.0: Added hot-reload capability with file watching
"""

import yaml
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import os

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


class Config:
    """Configuration manager with nested key access"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration with hot-reload support (v1.8.0)
        
        Args:
            config_path: Path to config YAML (default: config/config.yaml)
        """
        if config_path is None:
            # Try multiple default locations
            possible_paths = [
                Path(__file__).parent.parent.parent / 'config' / 'config.yaml',
                Path(__file__).parent.parent.parent / 'config' / 'falconone.yaml',
                Path.cwd() / 'config' / 'config.yaml',
                Path.cwd() / 'config.yaml'
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
            
            if config_path is None:
                # Use default config
                config_path = str(possible_paths[0])
        
        self.config_path = Path(config_path)
        self.config_file = str(self.config_path)
        self._config = self._load_config()
        
        # Hot-reload support (v1.8.0)
        self.enable_hot_reload = True
        self.reload_callbacks: List[Callable] = []
        self.config_observer = None
        self._reload_lock = threading.Lock()
        
        if WATCHDOG_AVAILABLE:
            self._setup_hot_reload()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            print(f"[INFO] Config file not found: {self.config_path}. Using defaults.")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config or {}
        except Exception as e:
            print(f"[WARNING] Failed to load config: {e}. Using defaults.")
            return self._get_default_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with nested key support
        
        Args:
            key: Dot-separated key (e.g., 'exploitation.scapy_integration')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            key: Dot-separated key
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def _setup_hot_reload(self):
        """Setup file watcher for configuration changes (v1.8.0)"""
        if not self.enable_hot_reload or not WATCHDOG_AVAILABLE:
            return
        
        try:
            class ConfigChangeHandler(FileSystemEventHandler):
                def __init__(self, config_instance):
                    self.config = config_instance
                    self.last_modified = 0
                
                def on_modified(self, event):
                    # Debounce: ignore rapid successive events
                    current_time = time.time()
                    if current_time - self.last_modified < 1.0:
                        return
                    
                    if event.src_path.endswith(('.yaml', '.yml')):
                        self.last_modified = current_time
                        self.config._reload_configuration()
            
            self.config_observer = Observer()
            event_handler = ConfigChangeHandler(self)
            
            watch_path = os.path.dirname(self.config_file)
            if watch_path and os.path.exists(watch_path):
                self.config_observer.schedule(event_handler, watch_path, recursive=False)
                self.config_observer.start()
                print(f"[INFO] Configuration hot-reload enabled for {self.config_file}")
            
        except Exception as e:
            print(f"[WARNING] Hot-reload setup failed: {e}")
            self.enable_hot_reload = False
    
    def _reload_configuration(self):
        """Reload configuration and notify subscribers (v1.8.0)"""
        with self._reload_lock:
            try:
                # Small delay to ensure file write is complete
                time.sleep(0.5)
                
                # Reload YAML file
                with open(self.config_file, 'r') as f:
                    new_config = yaml.safe_load(f)
                
                if not new_config:
                    print(f"[ERROR] Reloaded config is empty - reload aborted")
                    return
                
                # Detect changes
                old_config = self._config.copy()
                changes = self._detect_changes(old_config, new_config)
                
                # Update configuration
                self._config = new_config
                
                # Notify callbacks
                self._notify_reload_callbacks(changes)
                
                print(f"[INFO] Configuration reloaded successfully ({len(changes)} changes)")
                
            except Exception as e:
                print(f"[ERROR] Configuration reload error: {e}")
    
    def _detect_changes(self, old_config: Dict, new_config: Dict) -> List[Dict]:
        """Detect what changed between configurations"""
        changes = []
        
        def compare_dicts(old, new, path=''):
            all_keys = set(list(old.keys()) + list(new.keys()))
            
            for key in all_keys:
                current_path = f"{path}.{key}" if path else key
                
                if key not in old:
                    changes.append({'type': 'added', 'path': current_path, 'value': new[key]})
                elif key not in new:
                    changes.append({'type': 'removed', 'path': current_path, 'value': old[key]})
                elif old[key] != new[key]:
                    if isinstance(old[key], dict) and isinstance(new[key], dict):
                        compare_dicts(old[key], new[key], current_path)
                    else:
                        changes.append({
                            'type': 'modified',
                            'path': current_path,
                            'old_value': old[key],
                            'new_value': new[key]
                        })
        
        compare_dicts(old_config, new_config)
        return changes
    
    def register_reload_callback(self, callback: Callable):
        """Register callback to be notified of configuration changes (v1.8.0)"""
        self.reload_callbacks.append(callback)
    
    def _notify_reload_callbacks(self, changes: List[Dict]):
        """Notify all registered callbacks of configuration changes"""
        for callback in self.reload_callbacks:
            try:
                callback(changes)
            except Exception as e:
                print(f"[ERROR] Reload callback error: {e}")
    
    def shutdown_hot_reload(self):
        """Shutdown hot-reload file watcher"""
        if self.config_observer:
            try:
                self.config_observer.stop()
                self.config_observer.join(timeout=2)
            except Exception as e:
                print(f"[WARNING] Error stopping config observer: {e}")
    
    def save(self):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    def validate(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if valid, False otherwise
        """
        required_keys = [
            'logging.level',
            'safety.audit_logging'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                print(f"[ERROR] Required config key missing: {key}")
                return False
        
        return True
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'system': {
                'name': 'FalconOne',
                'version': '1.4',
                'mode': 'research'
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/falconone.log',
                'audit_dir': 'logs/audit'
            },
            'exploitation': {
                'scapy_integration': True,
                'evasion_mode': True,
                'ml_evasion': True,
                'gan_enabled': True
            },
            'monitoring': {
                'gsm': {'enabled': True},
                'umts': {'enabled': True},
                'lte': {'enabled': True},
                '5g': {'enabled': True},
                '6g': {'enabled': False, 'prototype': True}
            },
            'sdr': {
                'devices': ['USRP', 'BladeRF', 'LimeSDR'],
                'priority': 'USRP',
                'sample_rate': 10e6,
                'center_freq': 2140e6,
                'gain': 40,
                'bandwidth': 20e6,
                'failover_enabled': True,
                'failover_threshold_ms': 10000
            },
            'ai': {
                'enable_marl': True,
                'enable_gan': True,
                'enable_federated': True,
                'model_cache_dir': 'cache/models'
            },
            'safety': {
                'require_faraday_cage': False,  # Disabled for dev/testing - ENABLE IN PRODUCTION
                'audit_logging': True,
                'max_power_dbm': 20,
                'ethical_mode': True
            },
            'dashboard': {
                'enabled': True,
                'host': '0.0.0.0',
                'port': 5000,
                'refresh_rate_ms': 100,
                'auth_enabled': True,
                'users': {
                    'admin': 'falconone2026',
                    'operator': 'sigint2026'
                }
            },
            'geolocation': {
                'method': 'tdoa',
                'min_stations': 3,
                'algorithm': 'chan_ho'
            },
            'crypto': {
                'quantum_enabled': True,
                'target_algorithms': ['Kyber', 'NTRU', 'SABER']
            },
            'sustainability': {
                'enabled': True,
                'emissions_tracking': True,
                'power_optimization': True,
                'target_reduction_percent': 20
            }
        }


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get configuration with environment overrides
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Config instance
    """
    # Check for environment variable override
    env_config_path = os.getenv('FALCONONE_CONFIG')
    if env_config_path and Path(env_config_path).exists():
        config_path = env_config_path
    
    return Config(config_path)
