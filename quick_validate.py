#!/usr/bin/env python3
"""
Quick Validation - Dashboard & Wizard Features
Tests core functionality without requiring all optional dependencies
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_dashboard_import():
    """Test dashboard module can be imported"""
    print("="*70)
    print("TEST 1: Dashboard Import")
    print("="*70)
    
    try:
        from falconone.ui.dashboard import DashboardServer
        print("PASS: Dashboard module imported successfully")
        return True
    except SyntaxError as e:
        print(f"FAIL: Syntax error in dashboard - {e}")
        return False
    except Exception as e:
        print(f"FAIL: Import error - {e}")
        return False

def test_wizard_methods():
    """Test wizard methods exist"""
    print("\n" + "="*70)
    print("TEST 2: Wizard Methods")
    print("="*70)
    
    try:
        from falconone.ui.dashboard import DashboardServer
        
        required_methods = [
            '_check_system_dependencies',
            '_install_dependency',
            '_get_device_wizard_status',
            '_check_python_package',
        ]
        
        all_pass = True
        for method in required_methods:
            if hasattr(DashboardServer, method):
                print(f"PASS: {method} exists")
            else:
                print(f"FAIL: {method} missing")
                all_pass = False
        
        return all_pass
    except Exception as e:
        print(f"FAIL: Could not check methods - {e}")
        return False

def test_api_endpoints():
    """Test API endpoints are defined"""
    print("\n" + "="*70)
    print("TEST 3: API Endpoints")
    print("="*70)
    
    try:
        dashboard_path = Path(__file__).parent / 'falconone' / 'ui' / 'dashboard.py'
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        endpoints = [
            '/api/check_dependencies',
            '/api/install_dependency',
            '/api/device_wizard_status',
        ]
        
        all_pass = True
        for endpoint in endpoints:
            if endpoint in content:
                print(f"PASS: {endpoint} found")
            else:
                print(f"FAIL: {endpoint} missing")
                all_pass = False
        
        return all_pass
    except Exception as e:
        print(f"FAIL: Could not check endpoints - {e}")
        return False

def test_ui_components():
    """Test UI components are present"""
    print("\n" + "="*70)
    print("TEST 4: UI Components")
    print("="*70)
    
    try:
        dashboard_path = Path(__file__).parent / 'falconone' / 'ui' / 'dashboard.py'
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        components = [
            ('tab-setup', 'Setup Wizard Tab'),
            ('checkDependencies()', 'Check Dependencies Function'),
            ('refreshDeviceStatus()', 'Refresh Device Status Function'),
            ('connected-devices-overview', 'Connected Devices Overview'),
            ('dependencies-panel', 'Dependencies Panel'),
            ('action-sections', 'Dynamic Action Sections'),
            ("'uhd'", 'USRP Device Support'),
            ("'hackrf'", 'HackRF Device Support'),
            ("'bladerf'", 'bladeRF Device Support'),
            ("'limesdr'", 'LimeSDR Device Support'),
        ]
        
        all_pass = True
        for component, description in components:
            if component in content:
                print(f"PASS: {description}")
            else:
                print(f"FAIL: {description} missing")
                all_pass = False
        
        return all_pass
    except Exception as e:
        print(f"FAIL: Could not check UI components - {e}")
        return False

def test_syntax():
    """Test Python syntax is valid"""
    print("\n" + "="*70)
    print("TEST 5: Python Syntax Validation")
    print("="*70)
    
    import py_compile
    import tempfile
    
    try:
        dashboard_path = Path(__file__).parent / 'falconone' / 'ui' / 'dashboard.py'
        
        # Compile to check syntax
        with tempfile.TemporaryDirectory() as tmpdir:
            py_compile.compile(str(dashboard_path), doraise=True)
        
        print("PASS: No syntax errors found")
        return True
    except SyntaxError as e:
        print(f"FAIL: Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"FAIL: Compilation error - {e}")
        return False

def test_required_packages():
    """Test required packages are installed"""
    print("\n" + "="*70)
    print("TEST 6: Required Packages")
    print("="*70)
    
    packages = [
        'flask',
        'flask_socketio',
        'yaml',
        'numpy',
    ]
    
    all_pass = True
    for package in packages:
        try:
            __import__(package)
            print(f"PASS: {package} installed")
        except ImportError:
            print(f"FAIL: {package} not installed")
            all_pass = False
    
    return all_pass

def main():
    """Run all quick validation tests"""
    print("\n" + "="*70)
    print("FALCONONE QUICK VALIDATION")
    print("Dashboard & Setup Wizard Feature Tests")
    print("="*70 + "\n")
    
    results = []
    
    # Run tests
    results.append(('Syntax Validation', test_syntax()))
    results.append(('Required Packages', test_required_packages()))
    results.append(('Dashboard Import', test_dashboard_import()))
    results.append(('Wizard Methods', test_wizard_methods()))
    results.append(('API Endpoints', test_api_endpoints()))
    results.append(('UI Components', test_ui_components()))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "OK" if result else "XX"
        print(f"[{symbol}] {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSUCCESS: All tests passed! Dashboard and wizard are ready.")
        return 0
    else:
        print(f"\nWARNING: {total - passed} test(s) failed. Review errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
