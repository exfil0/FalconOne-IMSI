"""
FalconOne Configuration Management
Handles all system settings, device configurations, and runtime parameters
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Central configuration manager for FalconOne system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        return str(Path(__file__).parent.parent / "config" / "config.yaml")
    
    def load_config(self) -> None:
        """Load configuration from file"""
        if not os.path.exists(self.config_path):
            self._create_default_config()
        
        with open(self.config_path, 'r') as f:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                self.config = yaml.safe_load(f)
            else:
                self.config = json.load(f)
    
    def _create_default_config(self) -> None:
        """Create default configuration file"""
        default_config = {
            'system': {
                'name': 'FalconOne',
                'version': '1.1.0',
                'environment': 'research',
                'log_level': 'INFO',
                'log_dir': '/var/log/falconone',
                'data_dir': '/var/lib/falconone'
            },
            'sdr': {
                'devices': ['USRP', 'BladeRF', 'RTL-SDR', 'HackRF'],
                'priority': 'USRP',
                'sample_rate': 23.04e6,
                'center_freq': 2.14e9,
                'gain': 40,
                'bandwidth': 20e6
            },
            'monitoring': {
                'gsm': {
                    'enabled': True,
                    'bands': ['GSM900', 'GSM1800'],
                    'arfcn_scan': True,
                    'tools': ['gr-gsm', 'kalibrate-rtl', 'OsmocomBB']
                },
                'umts': {
                    'enabled': True,
                    'bands': ['UMTS2100', 'UMTS1900'],
                    'tools': ['gr-umts']
                },
                'cdma2000': {
                    'enabled': True,
                    'bands': ['CDMA800', 'CDMA1900'],
                    'tools': ['gr-cdma']
                },
                'lte': {
                    'enabled': True,
                    'bands': [1, 3, 7, 20, 28],
                    'tools': ['LTESniffer', 'srsRAN']
                },
                '5g': {
                    'enabled': True,
                    'mode': 'SA',  # SA or NSA
                    'bands': ['n1', 'n78', 'n79'],
                    'tools': ['srsRAN Project', 'Sni5Gect']
                },
                '6g': {
                    'enabled': False,
                    'prototype': True,
                    'tools': ['OAI']
                }
            },
            'core_network': {
                'open5gs': {
                    'enabled': True,
                    'mcc': '001',
                    'mnc': '01',
                    'amf_addr': '127.0.0.5',
                    'upf_addr': '127.0.0.7'
                }
            },
            'ai_ml': {
                'signal_classification': {
                    'enabled': True,
                    'model': 'CNN',
                    'accuracy_threshold': 0.90
                },
                'suci_deconcealment': {
                    'enabled': True,
                    'model': 'RoBERTa',
                    'quantization': True
                },
                'kpi_monitoring': {
                    'enabled': True,
                    'model': 'LSTM'
                },
                'payload_generation': {
                    'enabled': False,
                    'model': 'GAN'
                }
            },
            'cryptanalysis': {
                'enabled': False,
                'sca': {
                    'tool': 'Riscure Inspector',
                    'traces_required': 10000
                },
                'dfa': {
                    'tool': 'Riscure Huracan',
                    'fault_count': 100
                }
            },
            'geolocation': {
                'enabled': True,
                'methods': ['TDOA', 'AoA', 'DF'],
                'min_devices': 3,
                'gpsdo_sync': True,
                'accuracy_target': 50  # meters
            },
            'voice_interception': {
                'enabled': True,
                'protocols': ['VoLTE', 'VoNR'],
                'codecs': ['AMR', 'EVS']
            },
            'exploitation': {
                'enabled': False,
                'scapy_integration': True,
                'dos_testing': False,
                'downgrade_attacks': False
            },
            'compliance': {
                'faraday_cage': True,
                'cvd_enabled': True,
                'rica_compliance': True,
                'icasa_license': False,
                'popia_compliance': True
            },
            'performance': {
                'cpu_cores': 8,
                'gpu_enabled': True,
                'memory_limit_gb': 32,
                'thermal_monitoring': True
            }
        }
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Write default config
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        self.config = default_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key
        
        Args:
            key: Configuration key (e.g., 'sdr.sample_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-notation key
        
        Args:
            key: Configuration key (e.g., 'sdr.sample_rate')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save current configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def validate(self) -> bool:
        """
        Validate configuration for required fields and legal compliance
        
        Returns:
            True if configuration is valid
        """
        # Check Faraday cage requirement
        if not self.get('compliance.faraday_cage'):
            raise ValueError("Faraday cage must be enabled for legal compliance")
        
        # Check CVD is enabled
        if not self.get('compliance.cvd_enabled'):
            raise ValueError("Coordinated Vulnerability Disclosure must be enabled")
        
        # Validate SDR devices
        devices = self.get('sdr.devices', [])
        if not devices:
            raise ValueError("At least one SDR device must be configured")
        
        return True
    
    def __repr__(self) -> str:
        return f"<Config: {self.config_path}>"
