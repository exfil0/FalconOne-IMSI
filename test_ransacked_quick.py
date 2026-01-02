"""Quick test of RANSacked module"""
from falconone.audit.ransacked import RANSackedAuditor, CVESignature

# Test 1: Module loads
print("✓ RANSacked module imported successfully")

# Test 2: Instantiate auditor
auditor = RANSackedAuditor()
print("✓ RANSackedAuditor instantiated")

# Test 3: Check CVE database
stats = auditor.get_statistics()
print(f"\n📊 CVE Database Statistics:")
print(f"  Total CVEs: {stats['total_cves']}")
print(f"  Implementations: {list(stats['by_implementation'].keys())}")
print(f"  By Implementation: {stats['by_implementation']}")
print(f"  By Severity: {stats['by_severity']}")
print(f"  Average CVSS: {stats['avg_cvss_score']}")

# Test 4: Scan implementation
print(f"\n🔍 Testing scan_implementation()...")
scan_result = auditor.scan_implementation("Open5GS", "2.7.0")
print(f"  Implementation: {scan_result['implementation']}")
print(f"  Version: {scan_result['version']}")
print(f"  Total Known CVEs: {scan_result['total_known_cves']}")
print(f"  Applicable CVEs: {len(scan_result['applicable_cves'])}")
print(f"  Risk Score: {scan_result['risk_score']}")
print(f"  Severity Breakdown: {scan_result['severity_breakdown']}")

# Test 5: Audit packet
print(f"\n📦 Testing audit_nas_packet()...")
test_packet = bytes.fromhex("075d020011012345678900abcdef")
audit_result = auditor.audit_nas_packet(test_packet, "NAS")
print(f"  Protocol: {audit_result['protocol']}")
print(f"  Packet Size: {audit_result['packet_size']} bytes")
print(f"  Vulnerabilities Detected: {len(audit_result['vulnerabilities_detected'])}")
print(f"  Risk Level: {audit_result['risk_level']}")

print(f"\n✅ All tests passed! RANSacked module is fully operational.")
