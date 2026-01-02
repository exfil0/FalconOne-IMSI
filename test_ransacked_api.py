#!/usr/bin/env python3
"""
Test RANSacked API endpoints
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_stats_endpoint():
    """Test GET /api/audit/ransacked/stats"""
    print("\n" + "="*80)
    print("TEST 1: Statistics Endpoint")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/api/audit/ransacked/stats")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data.get('success')}")
        if data.get('success'):
            stats = data.get('data', {})
            print(f"\nTotal CVEs: {stats.get('total_cves')}")
            print(f"Implementations: {stats.get('implementation_count')}")
            print(f"Average CVSS: {stats.get('avg_cvss')}")
            print(f"\nSeverity Breakdown:")
            for severity, count in stats.get('severity_counts', {}).items():
                print(f"  {severity}: {count}")
            print(f"\nImplementation CVE Counts:")
            for impl, count in stats.get('implementation_cve_counts', {}).items():
                print(f"  {impl}: {count}")
            return True
    else:
        print(f"ERROR: {response.text}")
        return False

def test_scan_endpoint():
    """Test POST /api/audit/ransacked/scan"""
    print("\n" + "="*80)
    print("TEST 2: Implementation Scan Endpoint")
    print("="*80)
    
    test_data = {
        "implementation": "Open5GS",
        "version": "2.7.0"
    }
    
    print(f"Scanning: {test_data['implementation']} v{test_data['version']}")
    
    response = requests.post(
        f"{BASE_URL}/api/audit/ransacked/scan",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data.get('success')}")
        if data.get('success'):
            result = data.get('data', {})
            print(f"\nImplementation: {result.get('implementation')}")
            print(f"Version: {result.get('version')}")
            print(f"Vulnerabilities Found: {result.get('vulnerability_count')}")
            print(f"Risk Level: {result.get('risk_level')}")
            print(f"Risk Score: {result.get('risk_score')}")
            
            print(f"\nDetected CVEs:")
            for cve in result.get('vulnerabilities', [])[:5]:  # Show first 5
                print(f"  - {cve['cve_id']}: {cve['severity']} (CVSS {cve['cvss_score']})")
                print(f"    {cve['description'][:80]}...")
            return True
    else:
        print(f"ERROR: {response.text}")
        return False

def test_packet_audit_endpoint():
    """Test POST /api/audit/ransacked/packet"""
    print("\n" + "="*80)
    print("TEST 3: Packet Audit Endpoint")
    print("="*80)
    
    # Sample NAS attach request packet
    test_data = {
        "packet_hex": "074141000bf600f110000201000000001702e060c040",
        "protocol": "NAS"
    }
    
    print(f"Auditing {test_data['protocol']} packet: {test_data['packet_hex']}")
    
    response = requests.post(
        f"{BASE_URL}/api/audit/ransacked/packet",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data.get('success')}")
        if data.get('success'):
            result = data.get('data', {})
            print(f"\nProtocol: {result.get('protocol')}")
            print(f"Packet Size: {result.get('packet_size')} bytes")
            print(f"Vulnerabilities Detected: {result.get('vulnerability_count')}")
            print(f"Risk Level: {result.get('risk_level')}")
            
            detected_vulns = result.get('detected_vulnerabilities', [])
            if detected_vulns:
                print(f"\nDetected Patterns:")
                for vuln in detected_vulns:
                    print(f"  - {vuln['cve_id']}: {vuln['pattern']}")
            else:
                print(f"\n✓ No vulnerability patterns detected")
                
            print(f"\nRecommendations:")
            for rec in result.get('recommendations', [])[:3]:
                print(f"  - {rec}")
            return True
    else:
        print(f"ERROR: {response.text}")
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("RANSacked API Endpoint Tests")
    print("="*80)
    
    tests = [
        ("Statistics", test_stats_endpoint),
        ("Implementation Scan", test_scan_endpoint),
        ("Packet Audit", test_packet_audit_endpoint)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nEXCEPTION in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*80 + "\n")
