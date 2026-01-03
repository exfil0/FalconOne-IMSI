#!/usr/bin/env python3
"""
FalconOne Application Entry Point
Version 1.9.4

Usage:
    python main.py                 # Interactive menu (future)
    python -m falconone.cli.main   # CLI commands
"""

if __name__ == '__main__':
    from falconone.cli.main import main
    main()
