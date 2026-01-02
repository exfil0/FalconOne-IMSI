#!/usr/bin/env python3
"""
FalconOne System Launcher
Quick start script for running the FalconOne system
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from falconone.core.main import FalconOne
from falconone.core.config import Config


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='FalconOne IMSI/TMSI and SMS Catcher v1.1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python run.py                    # Run with default config
  python run.py -c custom.yaml     # Run with custom config
  python run.py --status           # Show system status
  python run.py --validate         # Validate configuration

⚠️  LEGAL NOTICE: For research and authorized use only.
    Must operate within Faraday cage environment.
        '''
    )
    
    parser.add_argument(
        '-c', '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status and exit'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate configuration and exit'
    )
    
    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version and exit'
    )
    
    args = parser.parse_args()
    
    # Show version
    if args.version:
        print("FalconOne v1.1.0")
        print("Multi-Generation Cellular Monitoring Platform")
        return 0
    
    # Validate config
    if args.validate:
        try:
            config = Config(args.config)
            config.validate()
            print("✓ Configuration is valid")
            return 0
        except Exception as e:
            print(f"✗ Configuration error: {e}")
            return 1
    
    # Show status
    if args.status:
        try:
            falcon = FalconOne(args.config)
            status = falcon.get_status()
            print("\nFalconOne System Status")
            print("=" * 50)
            print(f"Running: {status['running']}")
            print(f"Modules: {len(status['modules'])}")
            for module_name, module_status in status['modules'].items():
                print(f"  - {module_name}: {module_status}")
            return 0
        except Exception as e:
            print(f"Error getting status: {e}")
            return 1
    
    # Run system
    try:
        print("""
╔═══════════════════════════════════════════════════════════╗
║         FalconOne IMSI/TMSI and SMS Catcher v1.1         ║
║                 Multi-Generation SIGINT Platform          ║
║                    TOP CONFIDENTIAL                       ║
╚═══════════════════════════════════════════════════════════╝

⚠️  LEGAL NOTICE:
   - Research and authorized use only
   - Must operate within Faraday cage
   - Comply with RICA, ICASA, POPIA
   - Follow CVD protocols

Initializing...
        """)
        
        falcon = FalconOne(args.config)
        falcon.start()
        
    except KeyboardInterrupt:
        print("\n\n[!] Shutdown requested by user")
        return 0
    except Exception as e:
        print(f"\n\n[!] Critical error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
