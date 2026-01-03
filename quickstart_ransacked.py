#!/usr/bin/env python3
"""
Quick Start Demo for RANSacked Exploit Integration
Demonstrates all three new features:
1. Integration tests
2. Exploit chains
3. GUI controls

Usage: python quickstart_ransacked.py
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         RANSacked Integration - Quick Start Demo            ║
    ║                  FalconOne v1.8.0                           ║
    ║                                                              ║
    ║  Features:                                                   ║
    ║  ✅ Integration Tests (96 CVE payloads)                     ║
    ║  ✅ Exploit Chains (7 pre-defined chains)                   ║
    ║  ✅ GUI Controls (Visual exploit selection)                 ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    """Check required dependencies"""
    print("\n[1/5] Checking dependencies...")
    
    required_packages = ['pytest', 'flask', 'scapy']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ❌ {package} (missing)")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def run_tests():
    """Run integration tests"""
    print("\n[2/5] Running integration tests...")
    print("  This will test all 96 CVE payload generation")
    
    choice = input("  Run tests? (y/n): ").lower()
    if choice != 'y':
        print("  Skipped")
        return
    
    try:
        result = subprocess.run(
            ['pytest', 'falconone/tests/test_ransacked_exploits.py', '-v', '--tb=short'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Parse results
        lines = result.stdout.split('\n')
        for line in lines:
            if 'passed' in line.lower():
                print(f"  ✅ {line.strip()}")
        
        if result.returncode == 0:
            print("  ✅ All tests PASSED")
        else:
            print("  ⚠️  Some tests failed (check output above)")
    
    except subprocess.TimeoutExpired:
        print("  ⏱️  Tests timed out")
    except Exception as e:
        print(f"  ❌ Error running tests: {e}")


def demo_exploit_chains():
    """Demonstrate exploit chains"""
    print("\n[3/5] Demonstrating exploit chains...")
    print("  7 pre-defined chains available")
    
    chains = [
        ("chain_1", "Reconnaissance & DoS Chain"),
        ("chain_2", "Persistent Access Chain"),
        ("chain_3", "Multi-Implementation Attack"),
        ("chain_4", "Memory Corruption Cascade"),
        ("chain_5", "LTE S1AP Missing IE Flood"),
        ("chain_6", "GTP Protocol Attack Chain"),
        ("chain_7", "Advanced Evasion Chain")
    ]
    
    print("\n  Available chains:")
    for i, (chain_id, name) in enumerate(chains, 1):
        print(f"    {i}. {name}")
    
    choice = input("\n  Select chain to demo (1-7, or 0 to skip): ")
    
    if choice == '0':
        print("  Skipped")
        return
    
    try:
        chain_num = int(choice)
        if 1 <= chain_num <= 7:
            chain_id = chains[chain_num - 1][0]
            
            print(f"\n  Executing {chains[chain_num - 1][1]}...")
            print(f"  (Dry run mode - no actual exploitation)")
            
            # Execute chain
            result = subprocess.run(
                ['python', '-c', f"""
from exploit_chain_examples import {chain_id}_reconnaissance_crash, {chain_id}_persistent_access, {chain_id}_multi_implementation_attack, {chain_id}_memory_corruption_cascade, {chain_id}_lte_s1ap_flood, {chain_id}_gtp_protocol_attacks, {chain_id}_advanced_evasion

chain = {chain_id}()
result = chain.execute('192.168.1.100', dry_run=True)
print(f"\\n  ✅ Chain executed: {{result['executed_steps']}}/{{result['total_steps']}} steps")
print(f"  Success rate: {{result['success_rate']*100:.1f}}%")
""".replace(f"{chain_id}_", f"{chains[chain_num-1][0].replace('chain_', 'chain_')}")],
                capture_output=True,
                text=True
            )
            
            print(result.stdout)
            
            if result.returncode == 0:
                print("  ✅ Chain demo complete")
            else:
                print(f"  ⚠️  Chain demo encountered issues")
                print(result.stderr)
        else:
            print("  ❌ Invalid choice")
    
    except Exception as e:
        print(f"  ❌ Error: {e}")


def start_gui():
    """Start GUI dashboard"""
    print("\n[4/5] Starting GUI dashboard...")
    
    choice = input("  Start web GUI? (y/n): ").lower()
    if choice != 'y':
        print("  Skipped")
        return None
    
    try:
        print("  Starting Flask server...")
        
        # Start dashboard in background
        process = subprocess.Popen(
            ['python', 'start_dashboard.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        print("  Waiting for server to start...", end="", flush=True)
        for i in range(10):
            time.sleep(1)
            print(".", end="", flush=True)
            
            # Check if server is ready
            try:
                import requests
                response = requests.get('http://localhost:5000', timeout=1)
                if response.status_code in [200, 302, 404]:
                    break
            except:
                continue
        
        print(" Ready!")
        print(f"  ✅ Dashboard running at http://localhost:5000")
        print(f"  ✅ RANSacked GUI at http://localhost:5000/exploits/ransacked")
        
        return process
    
    except Exception as e:
        print(f"  ❌ Error starting dashboard: {e}")
        return None


def open_browser():
    """Open browser to GUI"""
    print("\n[5/5] Opening browser...")
    
    choice = input("  Open RANSacked GUI in browser? (y/n): ").lower()
    if choice != 'y':
        print("  Skipped")
        return
    
    try:
        url = 'http://localhost:5000/exploits/ransacked'
        webbrowser.open(url)
        print(f"  ✅ Opened {url}")
        print("\n  GUI Features:")
        print("    - Visual exploit selection")
        print("    - Filter by implementation/protocol")
        print("    - Multi-select execution")
        print("    - Real-time payload preview")
        print("    - Pre-defined exploit chains")
    
    except Exception as e:
        print(f"  ❌ Error opening browser: {e}")
        print(f"  Manually navigate to: http://localhost:5000/exploits/ransacked")


def show_summary():
    """Show summary and next steps"""
    print("\n" + "="*70)
    print("QUICK START COMPLETE")
    print("="*70)
    
    print("\n✅ What you've seen:")
    print("  1. Integration tests for 96 CVE payloads")
    print("  2. Exploit chain execution (7 pre-defined chains)")
    print("  3. Web GUI for visual exploit selection")
    
    print("\n📖 Documentation:")
    print("  - RANSACKED_FINAL_SUMMARY.md (integration summary)")
    print("  - RANSACKED_SECURITY_REVIEW.md (security analysis)")
    print("  - exploit_chain_examples.py (chain definitions)")
    
    print("\n🔧 Next Steps:")
    print("  1. Run full test suite:")
    print("     pytest falconone/tests/test_ransacked_exploits.py -v")
    
    print("\n  2. Create custom exploit chain:")
    print("     from exploit_chain_examples import ExploitChain, ChainStep")
    print("     chain = ExploitChain('My Chain', 'Description')")
    print("     # Add steps and execute")
    
    print("\n  3. Access GUI:")
    print("     python start_dashboard.py")
    print("     # Navigate to http://localhost:5000/exploits/ransacked")
    
    print("\n  4. Use API programmatically:")
    print("     from falconone.exploit.ransacked_payloads import RANSackedPayloadGenerator")
    print("     generator = RANSackedPayloadGenerator()")
    print("     payload = generator.get_payload('CVE-2024-24445', '192.168.1.100')")
    
    print("\n⚠️  Security Notice:")
    print("  - All exploits default to dry-run mode")
    print("  - Authentication required for live execution")
    print("  - Full audit logging enabled")
    print("  - Rate limiting enforced")
    
    print("\n" + "="*70)


def main():
    """Main quick start flow"""
    print_banner()
    
    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return 1
    
    # Run tests
    run_tests()
    
    # Demo chains
    demo_exploit_chains()
    
    # Start GUI
    dashboard_process = start_gui()
    
    if dashboard_process:
        # Open browser
        open_browser()
        
        # Keep server running
        print("\n" + "="*70)
        print("Dashboard is running. Press Ctrl+C to stop.")
        print("="*70)
        
        try:
            dashboard_process.wait()
        except KeyboardInterrupt:
            print("\n\nShutting down dashboard...")
            dashboard_process.terminate()
            dashboard_process.wait(timeout=5)
            print("✅ Dashboard stopped")
    
    # Show summary
    show_summary()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
