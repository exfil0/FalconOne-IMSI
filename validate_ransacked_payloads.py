#!/usr/bin/env python3
"""
RANSacked Payload Validation Script
Validates all 97 CVE exploit payloads
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from falconone.exploit.ransacked_payloads import RANSackedPayloadGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_payloads():
    """Validate all RANSacked payloads"""
    print("\n" + "="*80)
    print("RANSacked Payload Validation")
    print("="*80 + "\n")
    
    generator = RANSackedPayloadGenerator()
    
    # Get statistics
    stats = generator.get_stats()
    print(f"📊 Total CVEs: {stats['total_cves']}")
    print(f"\n📦 By Implementation:")
    for impl, count in stats['by_implementation'].items():
        print(f"  - {impl}: {count} CVEs")
    
    print(f"\n🔌 By Protocol:")
    for protocol, count in stats['by_protocol'].items():
        print(f"  - {protocol}: {count} CVEs")
    
    # Validate all CVEs
    print("\n" + "="*80)
    print("Validating Individual Payloads")
    print("="*80 + "\n")
    
    all_cves = generator.list_cves()
    passed = 0
    failed = 0
    failed_cves = []
    
    for cve_id in all_cves:
        try:
            payload = generator.get_payload(cve_id, target_ip="192.168.1.100")
            
            if not payload:
                print(f"❌ {cve_id}: Payload is None")
                failed += 1
                failed_cves.append(cve_id)
                continue
            
            # Validate payload structure
            assert payload.packet is not None, "packet is None"
            assert isinstance(payload.packet, bytes), "packet is not bytes"
            assert len(payload.packet) > 0, "packet is empty"
            assert payload.protocol in ['NGAP', 'S1AP', 'NAS', 'GTP', 'GTP-U'], f"invalid protocol: {payload.protocol}"
            assert payload.description, "description is empty"
            assert payload.success_indicators, "success_indicators is empty"
            assert len(payload.success_indicators) > 0, "no success indicators"
            
            print(f"✅ {cve_id}: OK ({len(payload.packet)} bytes, {payload.protocol})")
            passed += 1
            
        except Exception as e:
            print(f"❌ {cve_id}: {str(e)}")
            failed += 1
            failed_cves.append(cve_id)
    
    # Summary
    print("\n" + "="*80)
    print("Validation Summary")
    print("="*80 + "\n")
    
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {failed}/{total}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if failed_cves:
        print(f"\n⚠️  Failed CVEs:")
        for cve in failed_cves:
            print(f"  - {cve}")
    
    # Validate specific implementations
    print("\n" + "="*80)
    print("Implementation Tests")
    print("="*80 + "\n")
    
    implementations = generator.list_implementations()
    for impl in implementations:
        payloads = generator.get_implementation_payloads(impl)
        print(f"✅ {impl}: {len(payloads)} payloads")
    
    # Test specific CVE details
    print("\n" + "="*80)
    print("Sample CVE Details")
    print("="*80 + "\n")
    
    sample_cves = ['CVE-2024-24445', 'CVE-2023-37024', 'CVE-2023-37002', 'VULN-J01']
    for cve in sample_cves:
        info = generator.get_cve_info(cve)
        if info:
            print(f"\n{cve}:")
            print(f"  Implementation: {info['implementation']}")
            print(f"  Description: {info['description'][:100]}...")
    
    print("\n" + "="*80)
    
    if failed == 0:
        print("✅ ALL PAYLOADS VALIDATED SUCCESSFULLY!")
        print("="*80 + "\n")
        return 0
    else:
        print(f"⚠️  {failed} PAYLOAD(S) FAILED VALIDATION")
        print("="*80 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(validate_payloads())
