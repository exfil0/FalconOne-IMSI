#!/usr/bin/env python3
"""
FalconOne Production Environment Validator
Validates all required environment variables and configuration for production deployment

Usage:
    python validate_production_env.py
    
Exit codes:
    0 - All checks passed
    1 - Critical failures found
    2 - Warnings found (non-critical)
"""

import os
import sys
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class EnvironmentValidator:
    """Validates production environment configuration"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []
        
    def check_env_var(self, var_name: str, required: bool = True, min_length: int = 0) -> bool:
        """Check if environment variable is set and valid"""
        value = os.getenv(var_name)
        
        if value is None:
            if required:
                self.errors.append(f"❌ {var_name}: NOT SET (REQUIRED)")
                return False
            else:
                self.warnings.append(f"⚠️  {var_name}: NOT SET (optional)")
                return False
        
        if min_length > 0 and len(value) < min_length:
            if required:
                self.errors.append(f"❌ {var_name}: TOO SHORT (minimum {min_length} characters)")
                return False
            else:
                self.warnings.append(f"⚠️  {var_name}: TOO SHORT (recommended {min_length}+ characters)")
                return False
        
        # Mask sensitive values in output
        display_value = value[:8] + "..." if len(value) > 8 else value
        self.passed.append(f"✅ {var_name}: SET ({len(value)} chars)")
        return True
    
    def check_file_exists(self, file_path: str, description: str, required: bool = True) -> bool:
        """Check if required file exists"""
        if os.path.exists(file_path):
            self.passed.append(f"✅ {description}: FOUND ({file_path})")
            return True
        else:
            if required:
                self.errors.append(f"❌ {description}: NOT FOUND ({file_path})")
                return False
            else:
                self.warnings.append(f"⚠️  {description}: NOT FOUND ({file_path})")
                return False
    
    def check_config_value(self, config_path: str, key_path: str, expected_value: any) -> bool:
        """Check config.yaml for specific value"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Navigate nested keys
            keys = key_path.split('.')
            value = config
            for key in keys:
                value = value.get(key, None)
                if value is None:
                    self.warnings.append(f"⚠️  Config {key_path}: NOT SET")
                    return False
            
            if value == expected_value:
                self.passed.append(f"✅ Config {key_path}: {value}")
                return True
            else:
                self.warnings.append(f"⚠️  Config {key_path}: {value} (expected {expected_value})")
                return False
                
        except Exception as e:
            self.errors.append(f"❌ Config check failed: {e}")
            return False
    
    def check_python_package(self, package_name: str, min_version: str = None) -> bool:
        """Check if Python package is installed"""
        try:
            __import__(package_name)
            self.passed.append(f"✅ Package {package_name}: INSTALLED")
            return True
        except ImportError:
            self.errors.append(f"❌ Package {package_name}: NOT INSTALLED")
            return False
    
    def validate_all(self) -> int:
        """Run all validation checks"""
        logger.info("="*70)
        logger.info("FalconOne v1.9.0 - Production Environment Validation")
        logger.info("="*70 + "\n")
        
        # === CRITICAL SECURITY VARIABLES ===
        logger.info("🔐 Checking Critical Security Variables...")
        self.check_env_var("FALCONONE_SECRET_KEY", required=True, min_length=32)
        self.check_env_var("FALCONONE_DB_KEY", required=True, min_length=16)
        self.check_env_var("SIGNAL_BUS_KEY", required=False, min_length=16)
        logger.info("")
        
        # === O-RAN INTEGRATION ===
        logger.info("🌐 Checking O-RAN Integration...")
        self.check_env_var("ORAN_RIC_ENDPOINT", required=False)
        self.check_env_var("ORAN_RIC_ENDPOINT_NTN", required=False)
        logger.info("")
        
        # === EXTERNAL APIS ===
        logger.info("🔌 Checking External APIs...")
        self.check_env_var("OPENCELLID_API_KEY", required=False, min_length=10)
        self.check_env_var("SPACETRACK_USERNAME", required=False)
        self.check_env_var("SPACETRACK_PASSWORD", required=False)
        logger.info("")
        
        # === PRODUCTION SETTINGS ===
        logger.info("⚙️  Checking Production Settings...")
        env = os.getenv("FALCONONE_ENV", "development")
        if env == "production":
            self.passed.append(f"✅ FALCONONE_ENV: {env}")
        else:
            self.warnings.append(f"⚠️  FALCONONE_ENV: {env} (should be 'production')")
        
        log_level = os.getenv("FALCONONE_LOG_LEVEL", "INFO")
        if log_level in ["WARNING", "ERROR", "CRITICAL"]:
            self.passed.append(f"✅ FALCONONE_LOG_LEVEL: {log_level}")
        else:
            self.warnings.append(f"⚠️  FALCONONE_LOG_LEVEL: {log_level} (production should use WARNING/ERROR/CRITICAL)")
        logger.info("")
        
        # === FILE EXISTENCE ===
        logger.info("📁 Checking Required Files...")
        self.check_file_exists("config/config.yaml", "Configuration file", required=True)
        self.check_file_exists("requirements.txt", "Requirements file", required=True)
        self.check_file_exists(".env", "Environment file", required=False)
        logger.info("")
        
        # === CONFIG.YAML VALIDATION ===
        logger.info("📝 Checking config.yaml Settings...")
        if os.path.exists("config/config.yaml"):
            self.check_config_value("config/config.yaml", "system.environment", "production")
            self.check_config_value("config/config.yaml", "signal_bus.enable_encryption", True)
            self.check_config_value("config/config.yaml", "database.encrypt", True)
        logger.info("")
        
        # === PYTHON PACKAGES ===
        logger.info("🐍 Checking Critical Python Packages...")
        self.check_python_package("numpy")
        self.check_python_package("scipy")
        self.check_python_package("astropy")
        self.check_python_package("flask")
        self.check_python_package("flask_socketio")
        self.check_python_package("flask_limiter")
        self.check_python_package("bcrypt")
        logger.info("")
        
        # === FALCONONE MODULES ===
        logger.info("🦅 Checking FalconOne Modules...")
        try:
            from falconone.monitoring.isac_monitor import ISACMonitor
            self.passed.append("✅ ISAC Monitor: LOADABLE")
        except Exception as e:
            self.errors.append(f"❌ ISAC Monitor: LOAD FAILED ({e})")
        
        try:
            from falconone.exploit.isac_exploiter import ISACExploiter
            self.passed.append("✅ ISAC Exploiter: LOADABLE")
        except Exception as e:
            self.errors.append(f"❌ ISAC Exploiter: LOAD FAILED ({e})")
        
        try:
            from falconone.monitoring.ntn_6g_monitor import NTN6GMonitor
            self.passed.append("✅ NTN Monitor: LOADABLE")
        except Exception as e:
            self.errors.append(f"❌ NTN Monitor: LOAD FAILED ({e})")
        
        try:
            from falconone.le.intercept_enhancer import InterceptEnhancer
            self.passed.append("✅ LE Mode: LOADABLE")
        except Exception as e:
            self.errors.append(f"❌ LE Mode: LOAD FAILED ({e})")
        logger.info("")
        
        # === PRINT SUMMARY ===
        logger.info("="*70)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*70)
        
        if self.passed:
            logger.info(f"\n✅ PASSED ({len(self.passed)} checks):")
            for msg in self.passed[:5]:  # Show first 5
                logger.info(f"  {msg}")
            if len(self.passed) > 5:
                logger.info(f"  ... and {len(self.passed) - 5} more")
        
        if self.warnings:
            logger.info(f"\n⚠️  WARNINGS ({len(self.warnings)} checks):")
            for msg in self.warnings:
                logger.info(f"  {msg}")
        
        if self.errors:
            logger.info(f"\n❌ ERRORS ({len(self.errors)} checks):")
            for msg in self.errors:
                logger.info(f"  {msg}")
        
        logger.info("\n" + "="*70)
        
        # Determine exit code
        if self.errors:
            logger.error("❌ VALIDATION FAILED - Critical errors found")
            logger.error("Fix errors above before deploying to production")
            logger.error("See PRODUCTION_DEPLOYMENT.md for detailed setup instructions")
            return 1
        elif self.warnings:
            logger.warning("⚠️  VALIDATION PASSED WITH WARNINGS")
            logger.warning("Review warnings above for optimal production configuration")
            return 2
        else:
            logger.info("✅ VALIDATION PASSED - Ready for production deployment")
            return 0


def main():
    """Main entry point"""
    validator = EnvironmentValidator()
    exit_code = validator.validate_all()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
