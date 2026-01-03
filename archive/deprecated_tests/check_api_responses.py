import requests
import json

BASE_URL = "http://127.0.0.1:5000"

print("=" * 80)
print("Checking API Response Formats")
print("=" * 80)

# Test 1: Stats endpoint
print("\n1. GET /api/audit/ransacked/stats")
print("-" * 80)
response = requests.get(f"{BASE_URL}/api/audit/ransacked/stats")
print(f"Status: {response.status_code}")
print(f"Response (first 500 chars):")
print(response.text[:500])
print()

# Test 2: Scan endpoint
print("\n2. POST /api/audit/ransacked/scan")
print("-" * 80)
response = requests.post(
    f"{BASE_URL}/api/audit/ransacked/scan",
    json={"implementation": "Open5GS", "version": "2.7.0"},
    headers={"Content-Type": "application/json"}
)
print(f"Status: {response.status_code}")
print(f"Response (first 500 chars):")
print(response.text[:500])
print()

# Test 3: Packet endpoint
print("\n3. POST /api/audit/ransacked/packet")
print("-" * 80)
response = requests.post(
    f"{BASE_URL}/api/audit/ransacked/packet",
    json={"packet_hex": "074141000bf600f110000201000000001702e060c040", "protocol": "NAS"},
    headers={"Content-Type": "application/json"}
)
print(f"Status: {response.status_code}")
print(f"Response (first 500 chars):")
print(response.text[:500])
print()

print("=" * 80)
