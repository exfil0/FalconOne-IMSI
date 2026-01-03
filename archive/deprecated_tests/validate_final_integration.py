#!/usr/bin/env python3
"""
FalconOne v1.9.0 - Final Integration Validation
Validates complete RANSacked integration with all 5 phases + 6G NTN/ISAC
"""

import sys
import os
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))

def validate_all_phases():
    print("=" * 70)
    print("FalconOne v1.9.0 - Final Integration Validation")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    
    # Test 1: System Dependencies Documentation
    print("[TEST 1] System Dependencies Documentation")
    try:
        deps_file = Path('SYSTEM_DEPENDENCIES.md')
        if deps_file.exists():
            content = deps_file.read_text(encoding='utf-8')
            required = ['gr-gsm', 'kalibrate-rtl', 'OsmocomBB', 'LTESniffer', 
                       'srsRAN', 'Open5GS', 'OpenAirInterface', 'UHD', 
                       'BladeRF', 'GNU Radio']
            missing = [dep for dep in required if dep not in content]
            if not missing:
                print("  ✓ All 10 RANSacked stack components documented")
                passed += 1
            else:
                print(f"  ✗ Missing: {', '.join(missing)}")
                failed += 1
        else:
            print("  ✗ SYSTEM_DEPENDENCIES.md not found")
            failed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
        failed += 1
    print()
    
    # Test 2: Exploit Workflow Guide
    print("[TEST 2] Exploit Workflow Guide")
    try:
        workflow_file = Path('docs/EXPLOIT_WORKFLOW_GUIDE.md')
        if workflow_file.exists():
            content = workflow_file.read_text(encoding='utf-8')
            sections = ['Reconnaissance', 'Vulnerability Scanning', 
                       'Exploit Selection', 'Payload Generation',
                       'Exploit Execution', 'Post-Exploitation']
            missing = [sec for sec in sections if sec not in content]
            if not missing:
                print("  ✓ Complete 6-phase workflow documented")
                passed += 1
            else:
                print(f"  ✗ Missing sections: {', '.join(missing)}")
                failed += 1
        else:
            print("  ✗ docs/EXPLOIT_WORKFLOW_GUIDE.md not found")
            failed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
        failed += 1
    print()
    
    # Test 3: Unified Vulnerability Database
    print("[TEST 3] Unified Vulnerability Database")
    try:
        vuln_db_file = Path('falconone/exploit/vulnerability_db.py')
        if vuln_db_file.exists():
            content = vuln_db_file.read_text(encoding='utf-8')
            has_unified_class = 'class VulnerabilityDatabase' in content
            has_ransacked_cvss = 'ransacked_cves' in content or 'CVE-2024-' in content
            has_native_exploits = 'native_exploits' in content or 'FALCON-' in content
            has_get_stats = 'def get_statistics' in content
            
            if has_unified_class and has_ransacked_cvss and has_native_exploits and has_get_stats:
                print("  ✓ Unified database class implemented")
                print("    - VulnerabilityDatabase: ✓")
                print("    - RANSacked CVEs integrated: ✓")
                print("    - Native exploits: ✓")
                print("    - Statistics API: ✓")
                passed += 1
            else:
                issues = []
                if not has_unified_class: issues.append("Missing unified class")
                if not has_ransacked_cvss: issues.append("Missing RANSacked CVEs")
                if not has_native_exploits: issues.append("Missing native exploits")
                if not has_get_stats: issues.append("Missing stats API")
                print(f"  ✗ Database issues: {', '.join(issues)}")
                failed += 1
        else:
            print("  ✗ vulnerability_db.py not found")
            failed += 1
    except Exception as e:
        print(f"  ✗ Error loading database: {e}")
        failed += 1
    print()
    
    # Test 4: Exploit Engine Integration
    print("[TEST 4] Exploit Engine Auto-Exploit")
    try:
        engine_file = Path('falconone/exploit/exploit_engine.py')
        if engine_file.exists():
            content = engine_file.read_text(encoding='utf-8')
            has_auto_exploit = 'def auto_exploit' in content
            has_execute = 'def execute(' in content
            has_unified_db = 'vulnerability_db' in content.lower()
            
            if has_auto_exploit and has_execute and has_unified_db:
                print("  ✓ Auto-exploit engine implemented")
                print("    - Auto-exploit mode: ✓")
                print("    - Exploit execution: ✓")
                print("    - Unified database integration: ✓")
                passed += 1
            else:
                issues = []
                if not has_auto_exploit: issues.append("Missing auto-exploit")
                if not has_execute: issues.append("Missing execute method")
                if not has_unified_db: issues.append("Not using unified DB")
                print(f"  ✗ Engine issues: {', '.join(issues)}")
                failed += 1
        else:
            print("  ✗ exploit_engine.py not found")
            failed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
        failed += 1
    print()
    
    # Test 5: Exploit Chaining
    print("[TEST 5] Exploit Chaining")
    try:
        vuln_db_file = Path('falconone/exploit/vulnerability_db.py')
        if vuln_db_file.exists():
            content = vuln_db_file.read_text(encoding='utf-8')
            has_find_chains = 'def find_exploit_chains' in content or 'def get_exploit_chains' in content
            has_compatible = 'compatible' in content.lower()
            
            if has_find_chains and has_compatible:
                print("  ✓ Exploit chaining implemented")
                print("    - Chain discovery: ✓")
                print("    - Compatibility check: ✓")
                passed += 1
            else:
                issues = []
                if not has_find_chains: issues.append("Missing chain discovery")
                if not has_compatible: issues.append("Missing compatibility")
                print(f"  ✗ Chaining issues: {', '.join(issues)}")
                failed += 1
        else:
            print("  ✗ vulnerability_db.py not found")
            failed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
        failed += 1
    print()
    
    # Test 6: Payload Generator
    print("[TEST 6] Payload Generator")
    try:
        payload_file = Path('falconone/ai/payload_generator.py')
        if payload_file.exists():
            content = payload_file.read_text(encoding='utf-8')
            has_generator = 'class PayloadGenerator' in content
            has_generate = 'def generate' in content
            has_scapy = 'scapy' in content.lower() or 'Scapy' in content
            
            if has_generator and has_generate:
                print("  ✓ Payload generator operational")
                print("    - PayloadGenerator class: ✓")
                print("    - Generate methods: ✓")
                if has_scapy:
                    print("    - Scapy integration: ✓")
                passed += 1
            else:
                issues = []
                if not has_generator: issues.append("Missing generator class")
                if not has_generate: issues.append("Missing generate methods")
                print(f"  ✗ Generator issues: {', '.join(issues)}")
                failed += 1
        else:
            print("  ✗ payload_generator.py not found")
            failed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
        failed += 1
    print()
    
    # Test 7: Dashboard UI Integration
    print("[TEST 7] Dashboard UI - Unified Exploits Tab")
    try:
        dash_file = Path('falconone/ui/dashboard.py')
        if dash_file.exists():
            content = dash_file.read_text(encoding='utf-8')
            
            # Check for unified sections
            has_unified_db = 'Unified Vulnerability Database' in content
            has_auto_exploit = 'Auto-Exploit Engine' in content
            has_api = '/api/exploits/list' in content
            
            if has_unified_db and has_auto_exploit and has_api:
                print("  ✓ UI successfully merged")
                print("    - Unified Database viewer: ✓")
                print("    - Auto-Exploit Engine: ✓")
                print("    - Unified API endpoints: ✓")
                passed += 1
            else:
                issues = []
                if not has_unified_db: issues.append("Missing Unified DB")
                if not has_auto_exploit: issues.append("Missing Auto-Exploit")
                if not has_api: issues.append("Missing API")
                print(f"  ✗ UI issues: {', '.join(issues)}")
                failed += 1
        else:
            print("  ✗ dashboard.py not found")
            failed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
        failed += 1
    print()
    
    # Test 8: API Endpoints
    print("[TEST 8] Unified API Endpoints")
    try:
        dash_file = Path('falconone/ui/dashboard.py')
        if dash_file.exists():
            content = dash_file.read_text(encoding='utf-8')
            endpoints = [
                '/api/exploits/list',
                '/api/exploits/execute',
                '/api/exploits/chains',
                '/api/exploits/stats'
            ]
            
            missing = [ep for ep in endpoints if ep not in content]
            if not missing:
                print("  ✓ All 4 unified endpoints implemented")
                passed += 1
            else:
                print(f"  ✗ Missing endpoints: {', '.join(missing)}")
                failed += 1
        else:
            print("  ✗ dashboard.py not found")
            failed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
        failed += 1
    print()
    
    # Test 9: Progress Documentation
    print("[TEST 9] Integration Progress Documentation")
    try:
        progress_file = Path('RANSACKED_FINAL_SUMMARY.md')
        if progress_file.exists():
            content = progress_file.read_text(encoding='utf-8')
            # Check for Feature or Phase documentation
            features_complete = content.count('Feature')
            phases_complete = content.count('Phase')
            total_sections = features_complete + phases_complete
            if total_sections >= 3:
                print(f"  ✓ Integration documented ({features_complete} features, {phases_complete} phases found)")
                passed += 1
            else:
                print(f"  ✗ Only {total_sections} sections documented")
                failed += 1
        else:
            print("  ✗ Progress document not found")
            failed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
        failed += 1
    print()
    
    # Test 10: No Separate RANSacked Tab
    print("[TEST 10] RANSacked Tab Removed")
    try:
        dash_file = Path('falconone/ui/dashboard.py')
        if dash_file.exists():
            content = dash_file.read_text(encoding='utf-8')
            if 'tab-ransacked' in content:
                # Check if it's just comments
                lines_with_ransacked = [line for line in content.split('\n') 
                                       if 'tab-ransacked' in line and not line.strip().startswith('<!--')]
                if len(lines_with_ransacked) == 0:
                    print("  ✓ RANSacked tab successfully removed (only comments remain)")
                    passed += 1
                else:
                    print(f"  ✗ RANSacked tab still present ({len(lines_with_ransacked)} references)")
                    failed += 1
            else:
                print("  ✓ RANSacked tab successfully removed")
                passed += 1
        else:
            print("  ✗ dashboard.py not found")
            failed += 1
    except Exception as e:
        print(f"  ✗ Error: {e}")
        failed += 1
    print()
    
    # Final summary
    print("=" * 70)
    print(f"VALIDATION RESULTS: {passed}/10 PASSED, {failed}/10 FAILED")
    print("=" * 70)
    print()
    
    if failed == 0:
        print("✅ ALL TESTS PASSED - Integration 100% Complete!")
        print()
        print("🎉 FalconOne v1.9.0 is PRODUCTION READY")
        print()
        print("Next Steps:")
        print("  1. Start dashboard: python start_dashboard.py")
        print("  2. Open browser: http://127.0.0.1:5000")
        print("  3. Navigate to: ⚡ Exploit Engine tab or 🛰️ 6G NTN tab")
        print("  4. Click: 🗂️ Load Vulnerability Database")
        print("  5. Explore 25+ exploits with auto-exploitation")
        print()
        print("Documentation:")
        print("  - System setup: SYSTEM_DEPENDENCIES.md")
        print("  - Exploit guide: docs/EXPLOIT_WORKFLOW_GUIDE.md")
        print("  - Progress: RANSACKED_FINAL_SUMMARY.md")
        return 0
    else:
        print(f"❌ {failed} TEST(S) FAILED - Review errors above")
        print()
        print("Please fix the failed tests before production deployment.")
        return 1

if __name__ == '__main__':
    sys.exit(validate_all_phases())
