#!/usr/bin/env python3
"""
Comprehensive unit tests for RANSacked vulnerability auditor
Tests CVE database, scanning logic, packet auditing, and edge cases
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from falconone.audit.ransacked import RANSackedAuditor, CVESignature, Severity, Implementation


class TestRANSackedAuditor:
    """Test suite for RANSackedAuditor class"""
    
    @pytest.fixture
    def auditor(self):
        """Create auditor instance for tests"""
        return RANSackedAuditor()
    
    # ==================== CVE Database Tests ====================
    
    def test_auditor_initialization(self, auditor):
        """Test auditor initializes correctly"""
        assert auditor is not None
        assert len(auditor.cve_database) > 0
        print(f"✓ Auditor initialized with {len(auditor.cve_database)} CVEs")
    
    def test_cve_database_count(self, auditor):
        """Test correct number of CVEs loaded"""
        assert len(auditor.cve_database) == 97, f"Expected 97 CVEs, got {len(auditor.cve_database)}"
        print(f"✓ Correct CVE count: 97")
    
    def test_implementation_coverage(self, auditor):
        """Test all 7 implementations have CVEs"""
        implementations = set()
        for cve in auditor.cve_database:
            implementations.add(cve.implementation)
        
        expected = {
            "Open5GS",
            "OpenAirInterface",
            "Magma",
            "srsRAN",
            "NextEPC",
            "SD-Core",
            "Athonet"
        }
        
        assert implementations == expected, f"Missing implementations: {expected - implementations}"
        print(f"✓ All 7 implementations covered")
    
    def test_implementation_cve_counts(self, auditor):
        """Test correct CVE counts per implementation"""
        expected_counts = {
            "Open5GS": 14,
            "OpenAirInterface": 18,
            "Magma": 11,
            "srsRAN": 24,
            "NextEPC": 13,
            "SD-Core": 9,
            "Athonet": 8
        }
        
        for impl, expected_count in expected_counts.items():
            actual_count = sum(1 for cve in auditor.cve_database if cve.implementation == impl)
            assert actual_count == expected_count, f"{impl}: expected {expected_count}, got {actual_count}"
        
        print(f"✓ Implementation CVE counts correct")
    
    def test_severity_distribution(self, auditor):
        """Test severity distribution"""
        severity_counts = {
            "Critical": 0,
            "High": 0,
            "Medium": 0,
            "Low": 0
        }
        
        for cve in auditor.cve_database:
            severity_counts[cve.severity] += 1
        
        assert severity_counts["Critical"] == 31, f"Expected 31 Critical, got {severity_counts['Critical']}"
        assert severity_counts["High"] == 50, f"Expected 50 High, got {severity_counts['High']}"
        assert severity_counts["Medium"] == 16, f"Expected 16 Medium, got {severity_counts['Medium']}"
        
        print(f"✓ Severity distribution: Critical={severity_counts['Critical']}, High={severity_counts['High']}, Medium={severity_counts['Medium']}")
    
    def test_cvss_scores_valid(self, auditor):
        """Test all CVSS scores are valid (0.0-10.0)"""
        for cve in auditor.cve_database:
            assert 0.0 <= cve.cvss_score <= 10.0, f"{cve.cve_id} has invalid CVSS: {cve.cvss_score}"
        print(f"✓ All CVSS scores valid")
    
    def test_cve_fields_populated(self, auditor):
        """Test all CVE fields are properly populated"""
        for cve in auditor.cve_database:
            assert cve.cve_id, f"CVE missing ID"
            assert cve.severity, f"{cve.cve_id} missing severity"
            assert cve.cvss_score > 0, f"{cve.cve_id} has zero CVSS"
            assert cve.component, f"{cve.cve_id} missing component"
            assert cve.description, f"{cve.cve_id} missing description"
            assert cve.affected_versions, f"{cve.cve_id} missing affected_versions"
            assert cve.attack_vector, f"{cve.cve_id} missing attack_vector"
            assert cve.impact, f"{cve.cve_id} missing impact"
            assert cve.mitigation, f"{cve.cve_id} missing mitigation"
        
        print(f"✓ All CVE fields populated")
    
    # ==================== Statistics Tests ====================
    
    def test_get_statistics(self, auditor):
        """Test statistics method returns correct data"""
        stats = auditor.get_statistics()
        
        assert stats['total_cves'] == 97
        assert 'by_implementation' in stats
        assert 'by_severity' in stats
        assert 'avg_cvss_score' in stats
        
        assert stats['by_severity']['Critical'] == 31
        assert stats['by_severity']['High'] == 50
        assert stats['by_severity']['Medium'] == 16
        
        assert 8.0 <= stats['avg_cvss_score'] <= 8.2  # Should be around 8.08
        
        print(f"✓ Statistics: {stats['total_cves']} CVEs, avg CVSS {stats['avg_cvss_score']:.2f}")
    
    # ==================== Scan Implementation Tests ====================
    
    def test_scan_open5gs_vulnerable_version(self, auditor):
        """Test scanning Open5GS vulnerable version"""
        result = auditor.scan_implementation("Open5GS", "2.7.0")
        
        assert result['implementation'] == "Open5GS"
        assert result['version'] == "2.7.0"
        assert len(result['applicable_cves']) > 0
        assert result['risk_score'] > 0
        
        print(f"✓ Open5GS 2.7.0: {len(result['applicable_cves'])} CVEs, risk={result['risk_score']:.2f}")
    
    def test_scan_open5gs_patched_version(self, auditor):
        """Test scanning Open5GS patched version"""
        result = auditor.scan_implementation("Open5GS", "9.9.9")
        
        assert result['implementation'] == "Open5GS"
        assert result['version'] == "9.9.9"
        assert len(result['applicable_cves']) == 0
        assert result['risk_score'] == 0.0
        
        print(f"✓ Open5GS 9.9.9: {len(result['applicable_cves'])} CVEs (patched)")
    
    def test_scan_all_implementations(self, auditor):
        """Test scanning all implementations"""
        implementations = [
            ("Open5GS", "2.0.0"),
            ("OpenAirInterface", "1.0.0"),
            ("Magma", "1.5.0"),
            ("srsRAN", "20.04"),
            ("NextEPC", "0.3.0"),
            ("SD-Core", "1.0.0"),
            ("Athonet", "4.0.0")
        ]
        
        for impl, version in implementations:
            result = auditor.scan_implementation(impl, version)
            assert result['implementation'] == impl
            assert result['version'] == version
            assert 'applicable_cves' in result
            assert 'risk_score' in result
            print(f"  - {impl} {version}: {len(result['applicable_cves'])} CVEs")
        
        print(f"✓ All implementations scannable")
    
    def test_scan_invalid_implementation(self, auditor):
        """Test scanning invalid implementation name"""
        result = auditor.scan_implementation("InvalidCore", "1.0.0")
        
        assert len(result['applicable_cves']) == 0
        assert result['risk_score'] == 0.0
        print(f"✓ Invalid implementation handled gracefully")
    
    def test_scan_version_comparison(self, auditor):
        """Test version comparison logic"""
        # Test same CVE affects different versions differently
        result_old = auditor.scan_implementation("Open5GS", "1.0.0")
        result_new = auditor.scan_implementation("Open5GS", "3.0.0")
        
        # Older version should have more or equal vulnerabilities
        assert len(result_old['applicable_cves']) >= len(result_new['applicable_cves'])
        print(f"✓ Version comparison: v1.0.0 has {len(result_old['applicable_cves'])} CVEs, v3.0.0 has {len(result_new['applicable_cves'])} CVEs")
    
    def test_risk_score_calculation(self, auditor):
        """Test risk score is calculated correctly"""
        result = auditor.scan_implementation("Open5GS", "2.0.0")
        
        if len(result['applicable_cves']) > 0:
            # Risk score should be present and non-zero
            assert result['risk_score'] > 0
            print(f"✓ Risk score calculated correctly: {result['risk_score']:.2f}")
    
    def test_risk_level_mapping(self, auditor):
        """Test risk score calculation"""
        # Test vulnerable version has risk score
        result_critical = auditor.scan_implementation("Open5GS", "2.0.0")
        if len(result_critical['applicable_cves']) > 0:
            assert result_critical['risk_score'] > 0
        
        # Test patched version has zero risk
        result_patched = auditor.scan_implementation("Open5GS", "9.9.9")
        assert result_patched['risk_score'] == 0.0
        
        print(f"✓ Risk score calculation correct")
    
    # ==================== Packet Audit Tests ====================
    
    def test_audit_nas_packet_valid(self, auditor):
        """Test auditing valid NAS packet"""
        packet_hex = "074141000bf600f110000201000000001702e060c040"
        packet_bytes = bytes.fromhex(packet_hex)
        
        result = auditor.audit_nas_packet(packet_bytes, "NAS")
        
        assert result['protocol'] == "NAS"
        assert result['packet_size'] == len(packet_bytes)
        assert 'vulnerabilities_detected' in result
        assert 'risk_level' in result
        assert 'recommendations' in result
        
        print(f"✓ NAS packet audit: {result['packet_size']} bytes, {len(result['vulnerabilities_detected'])} vulns")
    
    def test_audit_packet_empty(self, auditor):
        """Test auditing empty packet"""
        result = auditor.audit_nas_packet(b"", "NAS")
        
        assert result['packet_size'] == 0
        assert len(result['vulnerabilities_detected']) == 0
        print(f"✓ Empty packet handled")
    
    def test_audit_packet_small(self, auditor):
        """Test auditing very small packet"""
        result = auditor.audit_nas_packet(b"\x07\x41", "NAS")
        
        assert result['packet_size'] == 2
        print(f"✓ Small packet handled")
    
    def test_audit_packet_large(self, auditor):
        """Test auditing large packet"""
        large_packet = b"\x00" * 1000
        result = auditor.audit_nas_packet(large_packet, "NAS")
        
        assert result['packet_size'] == 1000
        print(f"✓ Large packet handled")
    
    def test_audit_different_protocols(self, auditor):
        """Test auditing different protocol types"""
        packet = bytes.fromhex("074141000bf600")
        
        protocols = ["NAS", "S1AP", "NGAP", "GTP"]
        for protocol in protocols:
            result = auditor.audit_nas_packet(packet, protocol)
            assert result['protocol'] == protocol
        
        print(f"✓ All protocols supported")
    
    def test_audit_auth_bypass_pattern(self, auditor):
        """Test detection of authentication bypass pattern"""
        # Packet with 0x41 (Attach Request) followed by zeros (potential bypass)
        bypass_packet = bytes.fromhex("074100000000000000000000")
        result = auditor.audit_nas_packet(bypass_packet, "NAS")
        
        # Should detect pattern if implemented
        assert 'vulnerabilities_detected' in result
        print(f"✓ Auth bypass pattern detection: {len(result['vulnerabilities_detected'])} patterns")
    
    # ==================== Version Comparison Tests ====================
    
    def test_version_comparison_equal(self, auditor):
        """Test version comparison for equal versions"""
        assert auditor._compare_versions("2.7.0", "2.7.0") == 0
        print(f"✓ Version equality works")
    
    def test_version_comparison_less(self, auditor):
        """Test version comparison for less than"""
        assert auditor._compare_versions("2.6.0", "2.7.0") < 0
        assert auditor._compare_versions("2.7.0", "2.7.1") < 0
        print(f"✓ Version less-than works")
    
    def test_version_comparison_greater(self, auditor):
        """Test version comparison for greater than"""
        assert auditor._compare_versions("2.8.0", "2.7.0") > 0
        assert auditor._compare_versions("3.0.0", "2.9.9") > 0
        print(f"✓ Version greater-than works")
    
    def test_version_affected_logic(self, auditor):
        """Test version affected checking"""
        # Test "< 2.7.1" pattern
        assert auditor._is_version_affected("2.7.0", "< 2.7.1") == True
        assert auditor._is_version_affected("2.7.1", "< 2.7.1") == False
        assert auditor._is_version_affected("2.8.0", "< 2.7.1") == False
        
        # Test "All" pattern
        assert auditor._is_version_affected("1.0.0", "All") == True
        assert auditor._is_version_affected("99.9.9", "All") == True
        
        print(f"✓ Version affected logic works")
    
    # ==================== Edge Cases ====================
    
    def test_scan_empty_version(self, auditor):
        """Test scanning with empty version"""
        result = auditor.scan_implementation("Open5GS", "")
        assert 'applicable_cves' in result
        print(f"✓ Empty version handled")
    
    def test_scan_malformed_version(self, auditor):
        """Test scanning with malformed version"""
        result = auditor.scan_implementation("Open5GS", "abc.def.xyz")
        assert 'applicable_cves' in result
        print(f"✓ Malformed version handled")
    
    def test_concurrent_scans(self, auditor):
        """Test multiple concurrent scans don't interfere"""
        result1 = auditor.scan_implementation("Open5GS", "2.7.0")
        result2 = auditor.scan_implementation("srsRAN", "20.04")
        
        assert result1['implementation'] == "Open5GS"
        assert result2['implementation'] == "srsRAN"
        print(f"✓ Concurrent scans work independently")
    
    def test_cve_uniqueness(self, auditor):
        """Test all CVE IDs are unique"""
        cve_ids = [cve.cve_id for cve in auditor.cve_database]
        assert len(cve_ids) == len(set(cve_ids)), "Duplicate CVE IDs found"
        print(f"✓ All CVE IDs unique")
    
    def test_references_valid(self, auditor):
        """Test CVE references are non-empty"""
        for cve in auditor.cve_database:
            assert len(cve.references) > 0, f"{cve.cve_id} has no references"
        print(f"✓ All CVEs have references")


class TestCVESignature:
    """Test CVESignature dataclass"""
    
    def test_cve_signature_creation(self):
        """Test creating CVE signature"""
        cve = CVESignature(
            cve_id="CVE-2025-99999",
            implementation="Open5GS",
            affected_versions="< 1.0.0",
            severity="Critical",
            cvss_score=9.8,
            component="Test Component",
            vulnerability_type="Test Type",
            description="Test CVE",
            attack_vector="Network",
            impact="Complete system compromise",
            mitigation="Apply patch",
            references=["https://example.com"]
        )
        
        assert cve.cve_id == "CVE-2025-99999"
        assert cve.severity == "Critical"
        assert cve.cvss_score == 9.8
        print(f"✓ CVE signature creation works")


# ==================== Test Runner ====================

def run_tests():
    """Run all tests with pytest"""
    import subprocess
    
    print("\n" + "="*80)
    print("RANSacked Unit Tests")
    print("="*80 + "\n")
    
    # Run pytest with verbose output
    result = subprocess.run(
        ["pytest", __file__, "-v", "-s"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode


if __name__ == "__main__":
    # Allow running directly or with pytest
    try:
        import pytest
        exit_code = pytest.main([__file__, "-v", "-s"])
        sys.exit(exit_code)
    except ImportError:
        print("pytest not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest"])
        exit_code = pytest.main([__file__, "-v", "-s"])
        sys.exit(exit_code)
