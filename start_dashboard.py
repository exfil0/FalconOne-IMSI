#!/usr/bin/env python3
"""
FalconOne Dashboard Launcher
Start the web dashboard for FalconOne SIGINT platform
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def start_dashboard():
    """Start the FalconOne dashboard"""
    print("="*70)
    print("FalconOne SIGINT Platform Dashboard")
    print("Version 1.7.0 Phase 1 + Setup Wizard")
    print("="*70)
    print()
    
    try:
        # Import dashboard components
        print("[*] Loading dashboard components...")
        from falconone.ui.dashboard import DashboardServer, app, socketio
        import logging
        
        # Setup basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger('FalconOne')
        
        # Create dashboard instance (without orchestrator for standalone mode)
        print("[*] Initializing dashboard server...")
        config = {
            'dashboard': {
                'host': '127.0.0.1',
                'port': 5000,
                'refresh_rate_ms': 100,
                'debug': True
            }
        }
        
        dashboard = DashboardServer(config=config, logger=logger, core_system=None)
        
        print()
        print("="*70)
        print("Dashboard Ready!")
        print("="*70)
        print()
        print("Dashboard URL: http://127.0.0.1:5000")
        print()
        print("Features Available:")
        print("   - Overview Tab: System status and KPIs")
        print("   - Cellular Tab: GSM/CDMA/UMTS/LTE/5G/6G monitoring")
        print("   - Captures Tab: IMSI/SUCI/Voice captures")
        print("   - Exploits Tab: Exploitation operations")
        print("   - Analytics Tab: AI/ML analytics")
        print("   - Setup Wizard Tab: SDR device installation (NEW)")
        print("   - v1.7.0 Tab: Phase 1 features")
        print("   - System Tab: Health and configuration")
        print()
        print("Default Credentials:")
        print("   Username: admin")
        print("   Password: admin")
        print("   WARNING: Change password after first login!")
        print()
        print("LEGAL NOTICE: For research and authorized use only.")
        print("              Must operate within Faraday cage environment.")
        print()
        print("="*70)
        print()
        print("[*] Starting Flask server...")
        print("[*] Press Ctrl+C to stop the server")
        print()
        
        # Start the dashboard server
        socketio.run(
            app,
            host=config['dashboard']['host'],
            port=config['dashboard']['port'],
            debug=config['dashboard']['debug'],
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        print("\n\n[*] Dashboard server stopped by user")
        sys.exit(0)
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("\nMissing dependencies. Please install required packages:")
        print("   py -m pip install flask flask-socketio python-socketio pyyaml numpy")
        sys.exit(1)
    except Exception as e:
        print(f"\nError starting dashboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    start_dashboard()
