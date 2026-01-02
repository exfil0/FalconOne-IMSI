"""
API Integration Tests for RANSacked Endpoints
Tests all 3 API endpoints with various scenarios
"""

import requests
import json
import time
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:5000"

def print_header(text):
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)

def test_statistics_api():
    """Test GET /api/audit/ransacked/stats"""
    print_header("TEST 1: Statistics API")
    
    try:
        response = requests.get(f"{BASE_URL}/api/audit/ransacked/stats", timeout=5)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {response.elapsed.total_seconds():.3f}s")
        
        if response.status_code == 200:
            data = response.json()
            # API returns direct JSON, not wrapped
            print(f"\n✓ SUCCESS")
            print(f"Total CVEs: {data['total_cves']}")
            print(f"Average CVSS: {data['avg_cvss_score']}")
            print(f"\nSeverity Breakdown:")
            for sev, count in data['by_severity'].items():
                print(f"  {sev:12s}: {count:3d}")
            print(f"\nImplementation Counts:")
            for impl, count in data['by_implementation'].items():
                print(f"  {impl:20s}: {count:3d}")
            return True
        else:
            print(f"✗ FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ FAILED: Dashboard not running. Start with: python main.py dashboard")
        return False
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        return False


def test_scan_api():
    """Test POST /api/audit/ransacked/scan"""
    print_header("TEST 2: Implementation Scan API")
    
    test_cases = [
        ("Open5GS", "2.7.0", True),  # Should find vulnerabilities
        ("Open5GS", "9.9.9", False),  # Should find none (patched)
        ("srsRAN", "20.04", True),   # Should find vulnerabilities
        ("InvalidCore", "1.0.0", False),  # Invalid implementation
    ]
    
    results = []
    for impl, version, expect_vulns in test_cases:
        print(f"\n Testing: {impl} v{version}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/audit/ransacked/scan",
                json={"implementation": impl, "version": version},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                # API returns direct JSON with applicable_cves array
                vuln_count = len(data.get('applicable_cves', []))
                risk_score = data.get('risk_score', 0.0)
                
                print(f"  ✓ Vulnerabilities: {vuln_count}")
                print(f"    Risk Score: {risk_score:.2f}")
                print(f"    Implementation: {data.get('implementation')} v{data.get('version')}")
                
                if expect_vulns and vuln_count > 0:
                    print(f"    Result: PASS (found {vuln_count} CVEs as expected)")
                    results.append(True)
                elif not expect_vulns and vuln_count == 0:
                    print(f"    Result: PASS (no CVEs as expected)")
                    results.append(True)
                else:
                    print(f"    Result: FAIL (unexpected vulnerability count)")
                    results.append(False)
            elif response.status_code == 400:
                # Expected for invalid implementation
                print(f"  ✓ Validation error (expected): {response.json().get('error', '')[:60]}")
                results.append(True)
            else:
                print(f"  ✗ HTTP {response.status_code}: {response.text[:100]}")
                results.append(False)
                
        except Exception as e:
            print(f"  ✗ Exception: {type(e).__name__}: {e}")
            results.append(False)
    
    return all(results)


def test_packet_audit_api():
    """Test POST /api/audit/ransacked/packet"""
    print_header("TEST 3: Packet Audit API")
    
    test_cases = [
        ("074141000bf600f110000201000000001702e060c040", "NAS", "Valid NAS packet"),
        ("", "NAS", "Empty packet"),
        ("07", "NAS", "Very small packet"),
        ("XYZ", "NAS", "Invalid hex"),
    ]
    
    results = []
    for packet_hex, protocol, description in test_cases:
        print(f"\n Testing: {description}")
        print(f"  Packet: {packet_hex[:40]}{'...' if len(packet_hex) > 40 else ''}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/audit/ransacked/packet",
                json={"packet_hex": packet_hex, "protocol": protocol},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code in [200, 400]:  # 400 is expected for invalid hex
                data = response.json()
                
                if response.status_code == 200:
                    # API returns direct JSON with vulnerabilities_detected array
                    print(f"  ✓ Packet Size: {data['packet_size']} bytes")
                    print(f"    Vulnerabilities: {len(data['vulnerabilities_detected'])}") 
                    print(f"    Risk Level: {data['risk_level']}")
                    if data['vulnerabilities_detected']:
                        print(f"    Detected: {len(data['vulnerabilities_detected'])} patterns")
                        for vuln in data['vulnerabilities_detected'][:2]:
                            print(f"      - {vuln['cve_id']}: {vuln['description'][:60]}")
                    results.append(True)
                elif response.status_code == 400:
                    print(f"  ✓ Validation error (expected): {data.get('error', '')[:60]}")
                    results.append(True)
            else:
                print(f"  ✗ HTTP {response.status_code}")
                results.append(False)
                
        except Exception as e:
            print(f"  ✗ Exception: {type(e).__name__}: {e}")
            results.append(False)
    
    return all(results)


def test_rate_limiting():
    """Test rate limiting on scan endpoint"""
    print_header("TEST 4: Rate Limiting")
    
    print("\n Testing: Rapid requests to /api/audit/ransacked/scan (limit: 10/min)")
    
    # Send 12 requests quickly
    responses = []
    for i in range(12):
        try:
            response = requests.post(
                f"{BASE_URL}/api/audit/ransacked/scan",
                json={"implementation": "Open5GS", "version": "2.7.0"},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            responses.append(response.status_code)
            print(f"  Request {i+1:2d}: {response.status_code}", end="")
            if response.status_code == 429:
                print(" (Rate limited ✓)")
            else:
                print()
        except Exception as e:
            print(f"  Request {i+1}: Error - {e}")
            return False
    
    # Count rate limited responses
    rate_limited_count = responses.count(429)
    if rate_limited_count >= 2:
        print(f"\n✓ Rate limiting working: {rate_limited_count} requests blocked")
        return True
    else:
        print(f"\n✗ Rate limiting may not be working: only {rate_limited_count} requests blocked")
        return False


def test_error_handling():
    """Test error handling"""
    print_header("TEST 5: Error Handling")
    
    print("\\n Waiting 65 seconds for rate limit cooldown...")
    time.sleep(65)  # Wait for rate limit to reset
    
    test_cases = [
        ("Invalid JSON", None, "Invalid request body"),
        ("Missing fields", {"implementation": "Open5GS"}, "Missing version field"),
        ("Invalid types", {"implementation": 123, "version": "2.7.0"}, "Invalid data types"),
    ]
    
    results = []
    for description, payload, expected_error in test_cases:
        print(f"\n Testing: {description}")
        
        try:
            if payload is None:
                response = requests.post(
                    f"{BASE_URL}/api/audit/ransacked/scan",
                    data="invalid json",
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )
            else:
                response = requests.post(
                    f"{BASE_URL}/api/audit/ransacked/scan",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )
            
            if response.status_code in [400, 422]:
                data = response.json()
                print(f"  ✓ Error handled correctly: HTTP {response.status_code}")
                print(f"    Message: {data.get('error', '')[:80]}")
                results.append(True)
            else:
                print(f"  ✗ Expected error response, got: HTTP {response.status_code}")
                results.append(False)
                
        except Exception as e:
            print(f"  ✗ Exception: {type(e).__name__}: {e}")
            results.append(False)
    
    return all(results)


def main():
    print("\n" + "=" * 80)
    print(" RANSacked API Integration Tests")
    print("=" * 80)
    print("\nEnsure dashboard is running: python main.py dashboard")
    print(f"Testing endpoint: {BASE_URL}")
    
    tests = [
        ("Statistics API", test_statistics_api),
        ("Scan API", test_scan_api),
        ("Packet Audit API", test_packet_audit_api),
        ("Rate Limiting", test_rate_limiting),
        ("Error Handling", test_error_handling),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            time.sleep(1)  # Brief delay between test suites
        except Exception as e:
            print(f"\n✗ Test Suite Failed: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 80)
    print(" TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:25s}: {status}")
    
    print(f"\nTotal: {passed}/{total} test suites passed")
    print("=" * 80 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
