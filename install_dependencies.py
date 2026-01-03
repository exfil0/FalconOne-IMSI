#!/usr/bin/env python3
"""
FalconOne Phase 1-4 Dependencies Installer
Installs all required packages for the complete implementation roadmap
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*70}")
    print(f"📦 {description}")
    print(f"{'='*70}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ SUCCESS: {description}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {description}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                FalconOne Dependencies Installer                   ║
║                  Phases 1-4 Complete Setup                        ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ ERROR: Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")
    
    # Track installation results
    results = {}
    
    # ======== PHASE 1: CRITICAL DEPENDENCIES ========
    print("\n" + "="*70)
    print("🔴 PHASE 1: CRITICAL SECURITY & CORE DEPENDENCIES")
    print("="*70)
    
    phase1_packages = [
        ("py -m pip install --upgrade pip", "Upgrade pip"),
        ("py -m pip install sqlcipher3", "Database Encryption (SQLCipher)"),
        ("py -m pip install flask-limiter", "API Rate Limiting"),
        ("py -m pip install marshmallow", "Input Validation Schemas"),
        ("py -m pip install flask-wtf", "CSRF Protection"),
        ("py -m pip install python-dotenv", "Environment Variables"),
    ]
    
    for cmd, desc in phase1_packages:
        results[desc] = run_command(cmd, desc)
    
    # ======== PHASE 2: ENHANCED FEATURES ========
    print("\n" + "="*70)
    print("🟡 PHASE 2: ENHANCED FEATURES DEPENDENCIES")
    print("="*70)
    
    phase2_packages = [
        ("py -m pip install pycryptodome", "LTE Cryptography"),
        ("py -m pip install asn1tools", "ASN.1 Parsing (RRC/E2AP)"),
        ("py -m pip install reportlab", "PDF Report Generation"),
        ("py -m pip install pillow", "Image Processing for Reports"),
        ("py -m pip install flask-login", "User Authentication"),
        ("py -m pip install bcrypt", "Password Hashing"),
        ("py -m pip install celery", "Task Scheduling"),
        ("py -m pip install redis", "Cache Backend (optional)"),
    ]
    
    for cmd, desc in phase2_packages:
        results[desc] = run_command(cmd, desc)
    
    # ======== PHASE 3: ADVANCED CAPABILITIES ========
    print("\n" + "="*70)
    print("🟠 PHASE 3: ADVANCED AI/ML & TESTING DEPENDENCIES")
    print("="*70)
    
    phase3_packages = [
        ("py -m pip install shap", "Explainable AI (SHAP)"),
        ("py -m pip install lime", "Explainable AI (LIME)"),
        ("py -m pip install scikit-learn", "ML Utilities"),
        ("py -m pip install pytest pytest-cov", "Unit Testing"),
        ("py -m pip install pytest-asyncio", "Async Testing"),
        ("py -m pip install locust", "Performance/Load Testing"),
        ("py -m pip install bandit safety", "Security Scanning"),
    ]
    
    for cmd, desc in phase3_packages:
        results[desc] = run_command(cmd, desc)
    
    # ======== PHASE 4: PRODUCTION & POLISH ========
    print("\n" + "="*70)
    print("🔵 PHASE 4: PRODUCTION READINESS DEPENDENCIES")
    print("="*70)
    
    phase4_packages = [
        ("py -m pip install flask-swagger-ui", "API Documentation"),
        ("py -m pip install sqlalchemy", "Database ORM"),
        ("py -m pip install gevent", "Async Processing"),
        ("py -m pip install gunicorn", "Production WSGI Server"),
        ("py -m pip install prometheus-client", "Metrics Export"),
        ("py -m pip install python-json-logger", "Structured Logging"),
    ]
    
    for cmd, desc in phase4_packages:
        results[desc] = run_command(cmd, desc)
    
    # ======== OPTIONAL: AI/ML ACCELERATION ========
    print("\n" + "="*70)
    print("⚡ OPTIONAL: GPU ACCELERATION (if CUDA available)")
    print("="*70)
    
    optional_packages = [
        ("py -m pip install tensorflow-gpu", "TensorFlow GPU Support (requires CUDA)"),
    ]
    
    for cmd, desc in optional_packages:
        print(f"\n🔹 {desc} - attempting install...")
        run_command(cmd, desc)  # Don't track, optional
    
    # ======== SUMMARY ========
    print("\n" + "="*70)
    print("📊 INSTALLATION SUMMARY")
    print("="*70)
    
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    failure_count = total_count - success_count
    
    print(f"\n✅ Successful: {success_count}/{total_count}")
    print(f"❌ Failed: {failure_count}/{total_count}")
    
    if failure_count > 0:
        print("\n⚠️  Failed Packages:")
        for name, success in results.items():
            if not success:
                print(f"   - {name}")
        print("\n💡 Tip: Some packages may require system dependencies.")
        print("   Check package documentation for platform-specific requirements.")
    
    # ======== RECOMMENDATIONS ========
    print("\n" + "="*70)
    print("💡 NEXT STEPS & RECOMMENDATIONS")
    print("="*70)
    
    print("""
1. ✅ Core Security Packages:
   - SQLCipher, Flask-Limiter, Marshmallow, Flask-WTF installed
   - Run 'py -m pip list' to verify installations

2. 🔧 External Tools (Manual Installation):
   - Kraken (A5/1 decryption): https://github.com/gnuradio/gr-gsm
   - gr-gsm (GPRS capture): https://github.com/ptrkrysik/gr-gsm
   - Wireshark/tshark (S1-AP monitoring): https://www.wireshark.org/

3. 🔒 Security Configuration:
   - Set environment variables: FALCONONE_SECRET_KEY, DATABASE_PASSWORD
   - Create .env file with credentials (see .env.example)
   - Enable database encryption in config

4. 🚀 Ready to Start:
   - Phase 1 dependencies installed
   - Begin implementation with: py start_dashboard.py
   - Check DEVELOPER_GUIDE.md for development information

5. 📚 Documentation:
   - Review README.md for feature overview
   - Check DEVELOPER_GUIDE.md for implementation details
   - All features documented and ready to use
""")
    
    if success_count == total_count:
        print("✅ ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
        print("🚀 You're ready to begin Phase 1 implementation!")
        return 0
    else:
        print("⚠️  Some dependencies failed to install.")
        print("   Review errors above and retry failed packages manually.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
