# FalconOne API Documentation

**Version:** 3.4.0  
**Last Updated:** January 3, 2026  
**Base URL:** `http://localhost:5000/api` (Development) | `https://api.falconone.example.com` (Production)  
**Authentication:** JWT Bearer Token  
**Content-Type:** `application/json`

**Changelog v3.4.0:**
- Added 6G NTN API endpoints (5 new endpoints for satellite monitoring and exploitation)
- Added ISAC API endpoints (4 new endpoints for Integrated Sensing & Communications)
- Added LE Mode API endpoints (6 new endpoints for Law Enforcement warrant validation)
- Updated documentation to match v1.9.0 implementation

**Changelog v3.3.0:**
- Added RANSacked Integration Testing Suite (485 lines, 8 test classes, 100+ tests)
- Added RANSacked Exploit Chain Framework (7 pre-defined chains with 80-95% success rates)
- Added RANSacked GUI Controls (10 new REST API endpoints for visual exploit selection)
- New endpoints: `/api/ransacked/payloads`, `/api/ransacked/payload/<cve>`, `/api/ransacked/generate`, `/api/ransacked/execute`, `/api/ransacked/chains/available`, `/api/ransacked/chains/execute`, `/api/ransacked/stats`
- Enhanced rate limiting: 60/30/5/3 requests per minute for RANSacked operations

**Changelog v3.2.0:**
- Added RANSacked Vulnerability Auditor API (96 CVEs across 7 cellular core implementations)
- New endpoints: `/api/audit/ransacked/scan`, `/api/audit/ransacked/packet`, `/api/audit/ransacked/stats`

---

## Table of Contents
- [Authentication](#authentication)
- [Targets API](#targets-api)
- [Scanning API](#scanning-api)
- [Exploits API](#exploits-api)
- [RANSacked Exploit Integration API](#ransacked-exploit-integration-api) **(NEW v1.8.0)**
- [6G NTN API](#6g-ntn-api) **(NEW v1.9.0)**
- [ISAC API](#isac-api) **(NEW v1.9.0)**
- [LE Mode API](#le-mode-api) **(NEW v1.9.0)**
- [Monitoring API](#monitoring-api)
- [AI/ML API](#aiml-api)
- [O-RAN API](#o-ran-api)
- [System Tools API](#system-tools-api)
- [RANSacked Vulnerability Auditor API](#ransacked-vulnerability-auditor-api)
- [Users API](#users-api)
- [Multi-Tenant API](#multi-tenant-api)
- [Audit API](#audit-api)
- [Error Codes](#error-codes)

---

## Authentication

### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "secure_password"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": 1,
    "username": "admin",
    "email": "admin@example.com",
    "role": "admin"
  },
  "expires_in": 3600
}
```

### Refresh Token
```http
POST /api/auth/refresh
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600
}
```

### Logout
```http
POST /api/auth/logout
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Logged out successfully"
}
```

---

## Targets API

### List Targets
```http
GET /api/targets?page=1&limit=50&status=active
Authorization: Bearer {token}
```

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `limit` (optional): Results per page (default: 50, max: 100)
- `status` (optional): Filter by status (`active`, `inactive`)
- `network_type` (optional): Filter by network type (`5G`, `LTE`, `GSM`)

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "imsi": "310150123456789",
      "imei": "352099001761481",
      "msisdn": "+1234567890",
      "network_type": "5G",
      "last_seen": "2025-12-31T10:30:00Z",
      "latitude": 37.7749,
      "longitude": -122.4194,
      "status": "active"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 150,
    "pages": 3
  }
}
```

### Create Target
```http
POST /api/targets
Authorization: Bearer {token}
Content-Type: application/json

{
  "imsi": "310150123456789",
  "imei": "352099001761481",
  "msisdn": "+1234567890",
  "network_type": "5G"
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "imsi": "310150123456789",
    "imei": "352099001761481",
    "msisdn": "+1234567890",
    "network_type": "5G",
    "created_at": "2025-12-31T10:30:00Z",
    "status": "active"
  }
}
```

### Get Target Details
```http
GET /api/targets/{id}
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "imsi": "310150123456789",
    "imei": "352099001761481",
    "msisdn": "+1234567890",
    "network_type": "5G",
    "last_seen": "2025-12-31T10:30:00Z",
    "latitude": 37.7749,
    "longitude": -122.4194,
    "status": "active",
    "captures": 15,
    "exploits": 3
  }
}
```

### Update Target
```http
PUT /api/targets/{id}
Authorization: Bearer {token}
Content-Type: application/json

{
  "latitude": 37.7750,
  "longitude": -122.4195,
  "status": "inactive"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "latitude": 37.7750,
    "longitude": -122.4195,
    "status": "inactive",
    "updated_at": "2025-12-31T11:00:00Z"
  }
}
```

### Delete Target
```http
DELETE /api/targets/{id}
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Target deleted successfully"
}
```

---

## Scanning API

### Start Scan
```http
POST /api/scan/start
Authorization: Bearer {token}
Content-Type: application/json

{
  "target_id": 1,
  "scan_type": "full",
  "frequency_range": {
    "start": 2400,
    "end": 2500
  },
  "duration": 60
}
```

**Response (202 Accepted):**
```json
{
  "success": true,
  "scan_id": "scan_12345",
  "status": "running",
  "estimated_completion": "2025-12-31T11:01:00Z"
}
```

### Get Scan Status
```http
GET /api/scan/{scan_id}
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "scan_id": "scan_12345",
    "status": "completed",
    "progress": 100,
    "started_at": "2025-12-31T10:30:00Z",
    "completed_at": "2025-12-31T10:31:00Z",
    "results": {
      "signals_detected": 25,
      "targets_found": 3,
      "captures": 15
    }
  }
}
```

### Stop Scan
```http
POST /api/scan/{scan_id}/stop
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Scan stopped successfully"
}
```

---

## Exploits API

### List Available Exploits
```http
GET /api/exploits
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": "dos_attack",
      "name": "Denial of Service",
      "description": "Disrupt target's network connectivity",
      "risk_level": "high",
      "parameters": ["target_id", "attack_type", "duration"]
    },
    {
      "id": "downgrade_attack",
      "name": "Network Downgrade",
      "description": "Force target to lower network generation",
      "risk_level": "medium",
      "parameters": ["target_id", "target_network"]
    }
  ]
}
```

### Execute Exploit
```http
POST /api/exploits/execute
Authorization: Bearer {token}
Content-Type: application/json

{
  "exploit_id": "dos_attack",
  "target_id": 1,
  "parameters": {
    "attack_type": "frequency_jamming",
    "duration": 30,
    "power_level": 0.8
  }
}
```

**Response (202 Accepted):**
```json
{
  "success": true,
  "operation_id": "op_67890",
  "status": "executing",
  "estimated_completion": "2025-12-31T11:01:00Z"
}
```

### Get Exploit Status
```http
GET /api/exploits/operations/{operation_id}
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "operation_id": "op_67890",
    "exploit_id": "dos_attack",
    "target_id": 1,
    "status": "completed",
    "result": "success",
    "started_at": "2025-12-31T10:30:00Z",
    "completed_at": "2025-12-31T10:31:00Z",
    "metrics": {
      "packets_sent": 10000,
      "success_rate": 0.98
    }
  }
}
```

---

## RANSacked Exploit Integration API

**New in v1.8.0**: Complete RANSacked exploit integration with 96 CVE payloads, 7 exploit chains, and visual GUI controls.

### List All RANSacked Payloads
```http
GET /api/ransacked/payloads?implementation=oai_5g&protocol=NGAP&search=CVE-2024
Authorization: Bearer {token}
```

**Rate Limit:** 60 requests/minute

**Query Parameters:**
- `implementation` (optional): Filter by implementation (`oai_5g`, `open5gs_5g`, `magma_lte`, `open5gs_lte`, `misc`)
- `protocol` (optional): Filter by protocol (`NGAP`, `S1AP`, `NAS`, `GTP`, `GTP-U`)
- `search` (optional): Search CVE ID or description

**Response (200 OK):**
```json
{
  "success": true,
  "total": 96,
  "filtered": 11,
  "payloads": [
    {
      "cve_id": "CVE-2024-24445",
      "implementation": "oai_5g",
      "protocol": "NGAP",
      "severity": "critical",
      "description": "Registration request with NULL pointer",
      "success_indicators": ["AMF crash", "Core dump", "Connection reset"],
      "payload_size": 48
    }
  ],
  "implementations": ["oai_5g", "open5gs_5g", "magma_lte", "open5gs_lte", "misc"],
  "protocols": ["NGAP", "S1AP", "NAS", "GTP", "GTP-U"]
}
```

### Get Payload Details
```http
GET /api/ransacked/payload/CVE-2024-24445
Authorization: Bearer {token}
```

**Rate Limit:** 30 requests/minute

**Response (200 OK):**
```json
{
  "success": true,
  "cve_id": "CVE-2024-24445",
  "info": {
    "implementation": "oai_5g",
    "protocol": "NGAP",
    "severity": "critical",
    "description": "Registration request with NULL pointer dereference",
    "impact": "AMF crash, denial of service",
    "cvss_score": 9.8
  },
  "protocol": "NGAP",
  "description": "Registration request with NULL pointer",
  "success_indicators": ["AMF crash", "Core dump", "Connection reset"],
  "payload_size": 48,
  "payload_preview": "7e 00 41 2d e0 00 00 03 00 55 00 05 04 00 c0 a8 01 64 00 79 40 0b 0a 00 00...",
  "execution_template": {
    "target_ip": "192.168.1.100",
    "dry_run": true,
    "capture_traffic": false
  }
}
```

### Generate Payload
```http
POST /api/ransacked/generate
Authorization: Bearer {token}
Content-Type: application/json

{
  "cve_id": "CVE-2024-24445",
  "target_ip": "192.168.1.100"
}
```

**Rate Limit:** 30 requests/minute

**Response (200 OK):**
```json
{
  "success": true,
  "cve_id": "CVE-2024-24445",
  "target_ip": "192.168.1.100",
  "payload_base64": "fgBBLe...",
  "payload_hex": "7e 00 41 2d e0 00 00 03 00 55 00 05 04 00 c0 a8 01 64...",
  "protocol": "NGAP",
  "size": 48,
  "description": "Registration request with NULL pointer",
  "generated_at": "2026-01-02T10:30:00Z"
}
```

### Execute Exploit
```http
POST /api/ransacked/execute
Authorization: Bearer {token}
Content-Type: application/json

{
  "cve_id": "CVE-2024-24445",
  "target_ip": "192.168.1.100",
  "options": {
    "dry_run": true,
    "capture_traffic": false
  }
}
```

**Rate Limit:** 5 requests/minute

**Response (200 OK):**
```json
{
  "success": true,
  "execution_id": "exec_abc123",
  "cve_id": "CVE-2024-24445",
  "target_ip": "192.168.1.100",
  "dry_run": true,
  "result": "DRY RUN: Would execute CVE-2024-24445 against 192.168.1.100 (NGAP, 48 bytes)",
  "timestamp": "2026-01-02T10:30:00Z"
}
```

**Audit Log Entry:**
```
[2026-01-02 10:30:00] [CRITICAL] [AUDIT] RANSacked EXPLOIT EXECUTION - User: admin, IP: 127.0.0.1, CVE: CVE-2024-24445, Target: 192.168.1.100, Dry Run: true
```

### List Available Exploit Chains
```http
GET /api/ransacked/chains/available
Authorization: Bearer {token}
```

**Rate Limit:** 30 requests/minute

**Response (200 OK):**
```json
{
  "success": true,
  "total": 7,
  "chains": [
    {
      "id": "chain_1",
      "name": "Reconnaissance & DoS Chain",
      "description": "3-step chain targeting Open5GS 5G with progressive DoS escalation",
      "steps": 3,
      "implementations": ["open5gs_5g"],
      "cves": ["CVE-2024-24425", "CVE-2024-24428", "CVE-2024-24427"],
      "estimated_time_ms": 700,
      "success_rate": 0.95
    },
    {
      "id": "chain_5",
      "name": "LTE S1AP Missing IE Flood",
      "description": "10-step massive flood targeting LTE MME with missing IEs",
      "steps": 10,
      "implementations": ["magma_lte", "open5gs_lte"],
      "cves": ["CVE-2023-37025", "CVE-2023-37026", "...", "CVE-2024-24432"],
      "estimated_time_ms": 500,
      "success_rate": 0.95
    }
  ]
}
```

### Execute Exploit Chain
```http
POST /api/ransacked/chains/execute
Authorization: Bearer {token}
Content-Type: application/json

{
  "chain_id": "chain_1",
  "target_ip": "192.168.1.100",
  "options": {
    "dry_run": true
  }
}
```

**Rate Limit:** 3 requests/minute

**Response (200 OK):**
```json
{
  "success": true,
  "chain_id": "chain_1",
  "chain_name": "Reconnaissance & DoS Chain",
  "target_ip": "192.168.1.100",
  "dry_run": true,
  "total_steps": 3,
  "executed_steps": 3,
  "success_rate": 1.0,
  "results": [
    {
      "step": 1,
      "cve_id": "CVE-2024-24425",
      "stage": "DISCOVERY",
      "success": true,
      "message": "DRY RUN: Would execute payload"
    },
    {
      "step": 2,
      "cve_id": "CVE-2024-24428",
      "stage": "IMPACT",
      "success": true,
      "message": "DRY RUN: Would execute payload"
    },
    {
      "step": 3,
      "cve_id": "CVE-2024-24427",
      "stage": "IMPACT",
      "success": true,
      "message": "DRY RUN: Would execute payload"
    }
  ],
  "execution_time_ms": 715,
  "timestamp": "2026-01-02T10:30:00Z"
}
```

### Get RANSacked Statistics
```http
GET /api/ransacked/stats
Authorization: Bearer {token}
```

**Rate Limit:** 60 requests/minute

**Response (200 OK):**
```json
{
  "success": true,
  "total_cves": 96,
  "by_implementation": {
    "oai_5g": 11,
    "open5gs_5g": 6,
    "magma_lte": 25,
    "open5gs_lte": 26,
    "misc": 28
  },
  "by_protocol": {
    "NGAP": 12,
    "S1AP": 62,
    "NAS": 15,
    "GTP": 3,
    "GTP-U": 2,
    "Unknown": 2
  },
  "by_severity": {
    "critical": 75,
    "high": 21
  },
  "available_chains": 7
}
```

---

## 6G NTN API

**New in v1.9.0**: Complete 6G Non-Terrestrial Networks (NTN) monitoring and exploitation API with support for LEO, MEO, GEO, HAPS, and UAV satellite types.

### Start NTN Monitoring
```http
POST /api/ntn_6g/monitor
Authorization: Bearer {token}
Content-Type: application/json

{
  "sat_type": "LEO",
  "duration_sec": 60,
  "use_isac": true,
  "frequency_ghz": 150,
  "le_mode": false,
  "warrant_id": "optional-warrant-id"
}
```

**Rate Limit:** 10 requests/minute

**Parameters:**
- `sat_type` (string, required): Satellite type - `LEO`, `MEO`, `GEO`, `HAPS`, `UAV`
- `duration_sec` (int, optional): Monitoring duration 1-300 seconds (default: 60)
- `use_isac` (bool, optional): Enable ISAC sensing (default: true)
- `frequency_ghz` (float, optional): Center frequency in GHz (default: 150)
- `le_mode` (bool, optional): Enable Law Enforcement mode (default: false)
- `warrant_id` (string, optional): Required if le_mode is true

**Response (200 OK):**
```json
{
  "success": true,
  "timestamp": "2026-01-02T10:00:00Z",
  "satellite_type": "LEO",
  "technology": "6G_NTN",
  "signal_detected": true,
  "signal_strength_dbm": -95.5,
  "doppler_shift_hz": 15234.5,
  "isac_data": {
    "range_m": 550000,
    "velocity_mps": 7500,
    "angle_deg": 45.0,
    "snr_db": 18.5
  },
  "evidence_hash": "abc123..."
}
```

### Execute NTN Exploit
```http
POST /api/ntn_6g/exploit
Authorization: Bearer {token}
Content-Type: application/json

{
  "exploit_type": "beam_hijacking",
  "sat_type": "LEO",
  "target_beam_id": "beam_001",
  "le_mode": false,
  "warrant_id": "optional-warrant-id"
}
```

**Rate Limit:** 5 requests/minute

**Parameters:**
- `exploit_type` (string, required): One of:
  - `beam_hijacking` - Hijack satellite beam via RIS manipulation (75% success)
  - `handover_poisoning` - AI orchestration attack on handover (65% success)
  - `downlink_spoofing` - Inject fake downlink signals (70% success)
  - `feeder_link_attack` - Attack ground-satellite link (60% success)
  - `nb_iot_ntn` - NB-IoT satellite exploitation (68% success)
  - `timing_advance` - Timing advance manipulation (72% success)
  - `quantum_attack` - Quantum key distribution attack (35% success)

**Response (200 OK):**
```json
{
  "success": true,
  "exploit_type": "beam_hijacking",
  "satellite_type": "LEO",
  "execution_time_ms": 1250,
  "result": {
    "status": "success",
    "beam_captured": true,
    "old_beam_id": "beam_original",
    "new_beam_id": "beam_001"
  },
  "evidence_hash": "def456..."
}
```

### List Tracked Satellites
```http
GET /api/ntn_6g/satellites
Authorization: Bearer {token}
```

**Rate Limit:** 20 requests/minute

**Response (200 OK):**
```json
{
  "satellites": [
    {
      "sat_id": "STARLINK-1234",
      "sat_type": "LEO",
      "altitude_km": 550,
      "inclination_deg": 53.0,
      "longitude_deg": -122.5,
      "latitude_deg": 37.8,
      "velocity_kms": 7.5,
      "doppler_hz": 15234.5,
      "visible": true,
      "signal_strength_dbm": -95.5
    }
  ],
  "count": 15,
  "timestamp": "2026-01-02T10:00:00Z"
}
```

### Get Satellite Ephemeris
```http
GET /api/ntn_6g/ephemeris/{sat_id}
Authorization: Bearer {token}
```

**Rate Limit:** 10 requests/minute

**Response (200 OK):**
```json
{
  "sat_id": "STARLINK-1234",
  "ephemeris": {
    "epoch": "2026-01-02T10:00:00Z",
    "semi_major_axis_km": 6928,
    "eccentricity": 0.0001,
    "inclination_deg": 53.0,
    "raan_deg": 120.5,
    "arg_perigee_deg": 45.0,
    "mean_anomaly_deg": 180.0
  },
  "predictions": [
    {
      "time": "2026-01-02T10:10:00Z",
      "latitude_deg": 38.5,
      "longitude_deg": -121.0,
      "altitude_km": 550.1,
      "doppler_hz": 14500.0
    }
  ]
}
```

### Get NTN Statistics
```http
GET /api/ntn_6g/statistics
Authorization: Bearer {token}
```

**Rate Limit:** 20 requests/minute

**Response (200 OK):**
```json
{
  "total_sessions": 10,
  "satellites_tracked": 5,
  "doppler_measurements": 100,
  "isac_measurements": 100,
  "doppler_stats": {
    "mean_hz": 12000.5,
    "max_hz": 35000.0
  },
  "isac_stats": {
    "mean_range_km": 550.0,
    "mean_snr_db": 18.5
  }
}
```

---

## ISAC API

**New in v1.9.0**: Integrated Sensing and Communications (ISAC) monitoring and exploitation API supporting monostatic, bistatic, and cooperative sensing modes.

### Start ISAC Monitoring
```http
POST /api/isac/monitor
Authorization: Bearer {token}
Content-Type: application/json

{
  "mode": "monostatic",
  "duration_sec": 10,
  "frequency_ghz": 150.0,
  "waveform_type": "OFDM",
  "le_mode": false,
  "warrant_id": "WARRANT-12345"
}
```

**Rate Limit:** 10 requests/minute

**Parameters:**
- `mode` (string, required): `monostatic`, `bistatic`, `cooperative`
- `duration_sec` (int, optional): Duration 1-60 seconds (default: 10)
- `frequency_ghz` (float, optional): Frequency in GHz (default: 150)
- `waveform_type` (string, optional): `OFDM`, `DFT-s-OFDM`, `FMCW`
- `le_mode` (bool, optional): Enable LE mode (default: false)
- `warrant_id` (string, optional): Required if le_mode is true

**Response (200 OK):**
```json
{
  "success": true,
  "timestamp": "2026-01-02T10:00:00Z",
  "mode": "monostatic",
  "waveform": "OFDM",
  "targets_detected": 3,
  "sensing_data": [
    {
      "range_m": 250.5,
      "velocity_mps": 15.2,
      "angle_deg": 45.0,
      "snr_db": 22.5
    }
  ],
  "privacy_analysis": {
    "breach_detected": false,
    "risk_level": "low"
  }
}
```

### Execute ISAC Exploit
```http
POST /api/isac/exploit
Authorization: Bearer {token}
Content-Type: application/json

{
  "exploit_type": "waveform_manipulation",
  "target_freq_ghz": 150.0,
  "le_mode": false,
  "warrant_id": "WARRANT-12345"
}
```

**Rate Limit:** 5 requests/minute

**Parameters:**
- `exploit_type` (string, required): One of:
  - `waveform_manipulation` - Inject malformed waveforms (80% success)
  - `ai_poisoning` - ML model poisoning attack (65% success)
  - `privacy_breach` - Exploit sensing for tracking (60% success)
  - `e2sm_hijack` - E2SM control plane attack (70% success)
  - `quantum_attack` - Quantum radar exploitation (35% success)

**Response (200 OK):**
```json
{
  "success": true,
  "exploit_type": "waveform_manipulation",
  "execution_time_ms": 450,
  "result": {
    "status": "success",
    "targets_affected": 2,
    "dos_achieved": true,
    "data_leaked": false
  }
}
```

### Get Sensing Data
```http
GET /api/isac/sensing_data?limit=10&mode=monostatic
Authorization: Bearer {token}
```

**Rate Limit:** 20 requests/minute

**Query Parameters:**
- `limit` (int, optional): Number of entries (default: 10, max: 100)
- `mode` (string, optional): Filter by mode

**Response (200 OK):**
```json
{
  "data": [
    {
      "mode": "monostatic",
      "range_m": 250.5,
      "velocity_mps": 15.2,
      "angle_deg": 45.0,
      "snr_db": 22.5,
      "timestamp": 1704240000.0
    }
  ],
  "count": 10
}
```

### Get ISAC Statistics
```http
GET /api/isac/statistics
Authorization: Bearer {token}
```

**Rate Limit:** 20 requests/minute

**Response (200 OK):**
```json
{
  "monitoring": {
    "total_sessions": 100,
    "monostatic_count": 50,
    "bistatic_count": 30,
    "cooperative_count": 20,
    "avg_range_m": 350.5,
    "avg_velocity_mps": 12.3,
    "privacy_breaches_detected": 5
  },
  "exploitation": {
    "total_exploits": 50,
    "waveform_attacks": 20,
    "ai_poisoning_attacks": 10,
    "success_rate": 0.70
  }
}
```

---

## LE Mode API

**New in v1.9.0**: Law Enforcement Mode API for warrant validation, evidence chain management, and legally-compliant interception operations.

### Validate Warrant
```http
POST /api/le/warrant/validate
Authorization: Bearer {token}
Content-Type: application/json

{
  "warrant_id": "WRT-2026-00123",
  "warrant_image": "base64_encoded_image",
  "metadata": {
    "jurisdiction": "Southern District NY",
    "case_number": "2026-CR-00123",
    "authorized_by": "Judge Smith",
    "valid_until": "2026-06-30T23:59:59Z",
    "target_identifiers": ["001010123456789"],
    "operator": "officer_jones"
  }
}
```

**Rate Limit:** 10 requests/minute

**Response (200 OK):**
```json
{
  "success": true,
  "warrant_id": "WRT-2026-00123",
  "status": "validated",
  "valid_until": "2026-06-30T23:59:59Z",
  "message": "LE Mode activated with warrant WRT-2026-00123"
}
```

### Enhance Exploit for LE
```http
POST /api/le/enhance_exploit
Authorization: Bearer {token}
Content-Type: application/json

{
  "exploit_type": "imsi_catch",
  "warrant_id": "WRT-2026-00123",
  "parameters": {
    "target_imsi": "001010123456789"
  }
}
```

**Rate Limit:** 5 requests/minute

**Response (200 OK):**
```json
{
  "success": true,
  "enhanced": true,
  "evidence_hash": "sha256:abc123...",
  "timestamp": "2026-01-02T10:00:00Z",
  "audit_entry_id": "aud_12345"
}
```

### Get Evidence Details
```http
GET /api/le/evidence/{evidence_id}
Authorization: Bearer {token}
```

**Rate Limit:** 20 requests/minute

**Response (200 OK):**
```json
{
  "evidence_id": "EVD-2026-00001",
  "warrant_id": "WRT-2026-00123",
  "created_at": "2026-01-02T10:00:00Z",
  "data_type": "intercept_capture",
  "hash": "sha256:abc123...",
  "chain_valid": true,
  "metadata": {
    "target_imsi": "001010123456789",
    "operator": "officer_jones"
  }
}
```

### Verify Evidence Chain
```http
GET /api/le/chain/verify
Authorization: Bearer {token}
```

**Rate Limit:** 10 requests/minute

**Response (200 OK):**
```json
{
  "chain_valid": true,
  "total_entries": 150,
  "verified_entries": 150,
  "last_verified": "2026-01-02T10:00:00Z"
}
```

### Get LE Statistics
```http
GET /api/le/statistics
Authorization: Bearer {token}
```

**Rate Limit:** 20 requests/minute

**Response (200 OK):**
```json
{
  "active_warrants": 3,
  "total_evidence_items": 150,
  "operations_today": 12,
  "chain_integrity": true
}
```

### Export Evidence Package
```http
POST /api/le/evidence/export
Authorization: Bearer {token}
Content-Type: application/json

{
  "warrant_id": "WRT-2026-00123",
  "format": "court_package",
  "include_chain": true
}
```

**Rate Limit:** 5 requests/minute

**Response (200 OK):**
```json
{
  "success": true,
  "export_id": "EXP-2026-00001",
  "format": "court_package",
  "file_path": "/exports/WRT-2026-00123_court_package.zip",
  "hash": "sha256:def456...",
  "entries_included": 25
}
```

---

## Monitoring API

### Get Network Metrics
```http
GET /api/monitoring/metrics?network_type=5G&duration=3600
Authorization: Bearer {token}
```

**Query Parameters:**
- `network_type` (optional): `5G`, `LTE`, `GSM`
- `duration` (optional): Time range in seconds (default: 3600)

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "network_type": "5G",
    "metrics": {
      "rssi": -75.5,
      "rsrp": -95.2,
      "rsrq": -12.8,
      "sinr": 18.5,
      "throughput_mbps": 150.3
    },
    "timestamp": "2025-12-31T10:30:00Z"
  }
}
```

### Get Anomalies
```http
GET /api/monitoring/anomalies?severity=high&limit=20
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "type": "signal_anomaly",
      "severity": "high",
      "description": "Unexpected signal power drop",
      "target_id": 1,
      "detected_at": "2025-12-31T10:25:00Z",
      "resolved": false
    }
  ]
}
```

---

## AI/ML API

### Classify Signal
```http
POST /api/ml/classify
Authorization: Bearer {token}
Content-Type: application/json

{
  "signal_data": [0.1, 0.2, 0.3, ...],
  "model_version": "latest"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "prediction": "5G_NR",
    "confidence": 0.95,
    "probabilities": {
      "5G_NR": 0.95,
      "LTE": 0.04,
      "GSM": 0.01
    },
    "model_version": "v2.5.0"
  }
}
```

### Get Explanation
```http
POST /api/ml/explain
Authorization: Bearer {token}
Content-Type: application/json

{
  "signal_data": [0.1, 0.2, 0.3, ...],
  "prediction": "5G_NR",
  "method": "shap"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "method": "shap",
    "feature_importance": {
      "frequency": 0.35,
      "signal_strength": 0.28,
      "bandwidth": 0.22,
      "modulation": 0.15
    },
    "confidence": 0.95,
    "visualization_url": "/api/ml/visualizations/exp_123.png"
  }
}
```

### Train Model (Online Learning)
```http
POST /api/ml/train
Authorization: Bearer {token}
Content-Type: application/json

{
  "training_data": [
    {"features": [0.1, 0.2, ...], "label": "5G_NR"},
    {"features": [0.3, 0.4, ...], "label": "LTE"}
  ],
  "model_id": "online_classifier"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "model_id": "online_classifier",
    "samples_processed": 100,
    "accuracy": 0.94,
    "drift_detected": false,
    "model_version": "v2.5.1"
  }
}
```

---

## O-RAN API

### List xApps
```http
GET /api/oran/xapps
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": "xapp_001",
      "name": "TrafficSteeringXApp",
      "state": "RUNNING",
      "version": "1.0.0",
      "deployed_at": "2025-12-31T09:00:00Z"
    }
  ]
}
```

### Deploy xApp
```http
POST /api/oran/xapps/deploy
Authorization: Bearer {token}
Content-Type: application/json

{
  "xapp_class": "AnomalyDetectionXApp",
  "xapp_id": "xapp_002",
  "config": {
    "threshold": 3.0,
    "window_size": 100
  }
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "xapp_id": "xapp_002",
    "state": "STARTING",
    "message": "xApp deployment initiated"
  }
}
```

### Get E2 Nodes
```http
GET /api/oran/e2/nodes
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "node_id": "gnb_001",
      "plmn_id": "310150",
      "connected": true,
      "functions": ["E2SM-KPM", "E2SM-RC"]
    }
  ]
}
```

---

## System Tools API

### Get System Tools Status
```http
GET /api/system_tools/status
```

**Response (200 OK):**
```json
{
  "tools": {
    "gr-gsm": {
      "id": "gr-gsm",
      "name": "gr-gsm",
      "description": "GSM signal monitoring with GNU Radio",
      "category": "GSM",
      "icon": "📱",
      "installed": true,
      "status": "ready",
      "version": "1.0.0",
      "install_cmd": "sudo apt-get install gr-gsm"
    }
  },
  "total": 13,
  "installed": 5,
  "missing": 8,
  "completion_percent": 38.5,
  "timestamp": 1735660800.123
}
```

### Install System Tool
```http
POST /api/system_tools/install
Content-Type: application/json

{
  "tool": "gr-gsm"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "tool": "gr-gsm",
  "command": "sudo apt-get install gr-gsm",
  "message": "Installation command prepared for gr-gsm",
  "instructions": "Please run the following command in your terminal with sudo privileges:\nsudo apt-get install gr-gsm",
  "requires_sudo": true
}
```

### Uninstall System Tool
```http
POST /api/system_tools/uninstall
Content-Type: application/json

{
  "tool": "gr-gsm"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "tool": "gr-gsm",
  "command": "sudo apt-get remove --purge gr-gsm",
  "message": "Uninstallation command prepared for gr-gsm",
  "instructions": "Please run the following command in your terminal with sudo privileges:\nsudo apt-get remove --purge gr-gsm",
  "requires_sudo": true
}
```

### Test System Tool
```http
POST /api/system_tools/test
Content-Type: application/json

{
  "tool": "gr-gsm"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "tool": "gr-gsm",
  "working": true,
  "message": "gr-gsm is working correctly",
  "output": "grgsm_livemon version 1.0.0\n..."
}
```

---

## Users API

### List Users (Admin Only)
```http
GET /api/users?role=operator&status=active
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "username": "admin",
      "email": "admin@example.com",
      "role": "admin",
      "status": "active",
      "last_login": "2025-12-31T10:00:00Z"
    }
  ]
}
```

### Create User (Admin Only)
```http
POST /api/users
Authorization: Bearer {token}
Content-Type: application/json

{
  "username": "operator1",
  "email": "operator1@example.com",
  "password": "SecurePass123!",
  "role": "operator"
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "id": 2,
    "username": "operator1",
    "email": "operator1@example.com",
    "role": "operator",
    "created_at": "2025-12-31T10:30:00Z"
  }
}
```

---

## Multi-Tenant API

### Get Tenant Info
```http
GET /api/tenants/current
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "tenant_id": "tenant_abc123",
    "tenant_name": "Acme Corp",
    "subscription_tier": "professional",
    "status": "active",
    "quota": {
      "max_targets": 500,
      "max_scans_per_hour": 50,
      "max_storage_mb": 10000
    },
    "usage": {
      "targets_count": 120,
      "scans_last_hour": 8,
      "storage_used_mb": 2500.5
    }
  }
}
```

### Get Usage Statistics
```http
GET /api/tenants/current/usage
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "targets_count": 120,
    "scans_last_hour": 8,
    "storage_used_mb": 2500.5,
    "api_calls_last_minute": 15,
    "users_count": 5,
    "quota_warnings": []
  }
}
```

---

## Audit API

### Get Audit Logs
```http
GET /api/audit/logs?event_type=AUTH&user_id=1&limit=50
Authorization: Bearer {token}
```

**Query Parameters:**
- `event_type` (optional): `AUTH`, `ACCESS`, `EXPLOIT`, `SYSTEM`
- `user_id` (optional): Filter by user ID
- `severity` (optional): `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- `start_time` (optional): ISO 8601 timestamp
- `end_time` (optional): ISO 8601 timestamp

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "event_id": "evt_123",
      "event_type": "AUTH",
      "action": "LOGIN",
      "user_id": "1",
      "resource": "dashboard",
      "severity": "INFO",
      "timestamp": "2025-12-31T10:00:00Z",
      "ip_address": "192.168.1.100",
      "details": {
        "success": true
      }
    }
  ]
}
```

### Verify Blockchain Integrity
```http
GET /api/audit/blockchain/verify
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "chain_valid": true,
    "total_blocks": 150,
    "total_events": 5000,
    "genesis_timestamp": "2025-01-01T00:00:00Z",
    "last_block_hash": "0000a1b2c3d4e5f6..."
  }
}
```

---

## RANSacked Vulnerability Auditor API

The RANSacked API provides vulnerability scanning and packet auditing capabilities based on the RANSacked project's research, which identified 119 vulnerabilities (96 CVEs) across 7 cellular core implementations.

### Scan Implementation

Scan a cellular core implementation for known vulnerabilities.

```http
POST /api/audit/ransacked/scan
Content-Type: application/json
Authorization: Bearer <token>

{
  "implementation": "Open5GS",
  "version": "2.7.0"
}
```

**Parameters:**
- `implementation` (string, required): Name of the implementation to scan  
  Valid values: `Open5GS`, `OpenAirInterface`, `Magma`, `srsRAN`, `NextEPC`, `SD-Core`, `Athonet`
- `version` (string, optional): Specific version to check (e.g., "2.7.0", "24.04")  
  If omitted, returns all known CVEs for the implementation

**Response (200 OK):**
```json
{
  "implementation": "Open5GS",
  "version": "2.7.0",
  "scan_time": "2025-12-31T15:30:00.000Z",
  "total_known_cves": 14,
  "applicable_cves": [
    {
      "cve_id": "CVE-2023-45917",
      "implementation": "Open5GS",
      "affected_versions": "2.5.0 - 2.6.4",
      "severity": "High",
      "cvss_score": 8.1,
      "component": "AMF - NAS Security",
      "vulnerability_type": "Replay Attack",
      "description": "NAS Security Mode Command can be replayed to downgrade security",
      "attack_vector": "Captured SMC replayed to force weaker encryption algorithm",
      "impact": "Security downgrade, plaintext NAS messages, eavesdropping",
      "mitigation": "Upgrade to >= 2.6.5, implement strict nonce validation and replay protection",
      "references": [
        "https://nvd.nist.gov/vuln/detail/CVE-2023-45917"
      ]
    }
  ],
  "severity_breakdown": {
    "Critical": 3,
    "High": 5,
    "Medium": 2,
    "Low": 0
  },
  "risk_score": 8.45
}
```

**Error Responses:**

- **400 Bad Request**: Missing or invalid implementation name
  ```json
  {
    "error": "Missing required field: implementation"
  }
  ```
  
- **400 Bad Request**: Invalid implementation
  ```json
  {
    "error": "Invalid implementation",
    "valid_implementations": ["Open5GS", "OpenAirInterface", "Magma", "srsRAN", "NextEPC", "SD-Core", "Athonet"]
  }
  ```

- **429 Too Many Requests**: Rate limit exceeded (10 scans/minute)
  ```json
  {
    "error": "Rate limit exceeded"
  }
  ```

**Rate Limiting:**  
- 10 requests per minute per user

---

### Audit Packet

Audit a captured cellular packet for vulnerability signatures.

```http
POST /api/audit/ransacked/packet
Content-Type: application/json
Authorization: Bearer <token>

{
  "packet_hex": "075d02001101234567890abcdef",
  "protocol": "NAS"
}
```

**Parameters:**
- `packet_hex` (string, required): Hexadecimal representation of the packet bytes
- `protocol` (string, optional, default: "NAS"): Protocol type  
  Valid values: `NAS`, `S1AP`, `NGAP`, `GTP`

**Response (200 OK):**
```json
{
  "protocol": "NAS",
  "packet_size": 28,
  "audit_time": "2025-12-31T15:31:00.000Z",
  "vulnerabilities_detected": [
    {
      "cve_id": "CVE-2023-45917",
      "description": "Possible NAS Security Mode Command replay",
      "confidence": "High"
    }
  ],
  "risk_level": "High",
  "recommendations": [
    "Investigate packet source and validate authentication",
    "Check core network version for known CVE patches",
    "Enable enhanced security monitoring",
    "Review recent security alerts for similar patterns"
  ]
}
```

**Response (200 OK - No Vulnerabilities):**
```json
{
  "protocol": "NAS",
  "packet_size": 28,
  "audit_time": "2025-12-31T15:32:00.000Z",
  "vulnerabilities_detected": [],
  "risk_level": "Low",
  "recommendations": []
}
```

**Error Responses:**

- **400 Bad Request**: Missing packet data
  ```json
  {
    "error": "Missing required field: packet_hex"
  }
  ```

- **400 Bad Request**: Invalid hex format
  ```json
  {
    "error": "Invalid packet_hex format (must be hexadecimal)"
  }
  ```

- **400 Bad Request**: Invalid protocol
  ```json
  {
    "error": "Invalid protocol",
    "valid_protocols": ["NAS", "S1AP", "NGAP", "GTP"]
  }
  ```

**Rate Limiting:**  
- 20 requests per minute per user

---

### Get CVE Statistics

Retrieve overall statistics about the RANSacked CVE database.

```http
GET /api/audit/ransacked/stats
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "total_cves": 97,
  "by_implementation": {
    "Open5GS": 14,
    "OpenAirInterface": 18,
    "Magma": 11,
    "srsRAN": 24,
    "NextEPC": 13,
    "SD-Core": 9,
    "Athonet": 8
  },
  "by_severity": {
    "Critical": 27,
    "High": 45,
    "Medium": 19,
    "Low": 6
  },
  "avg_cvss_score": 7.85
}
```

**cURL Examples:**

```bash
# Scan Open5GS for vulnerabilities
curl -X POST http://localhost:5000/api/audit/ransacked/scan \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"implementation": "Open5GS", "version": "2.7.0"}'

# Audit a NAS packet
curl -X POST http://localhost:5000/api/audit/ransacked/packet \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"packet_hex": "075d02001101234567890abcdef", "protocol": "NAS"}'

# Get CVE database statistics
curl http://localhost:5000/api/audit/ransacked/stats \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Security Notes:**
- All RANSacked audit activities are logged with user information and timestamps
- Scan results may contain sensitive vulnerability information - use responsibly
- Packet auditing processes packet data in-memory without persistent storage
- Rate limiting prevents abuse and resource exhaustion

**Use Cases:**
1. **Pre-Deployment Security Assessment**: Scan planned cellular core implementation before deployment
2. **Vulnerability Management**: Track known CVEs affecting current infrastructure
3. **Penetration Testing**: Verify vulnerability presence during authorized security assessments
4. **Incident Response**: Audit captured packets during security investigations
5. **Compliance Auditing**: Generate vulnerability reports for regulatory compliance

---

## Error Codes

### HTTP Status Codes
- **200 OK**: Successful request
- **201 Created**: Resource created successfully
- **202 Accepted**: Request accepted for processing
- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Missing or invalid authentication
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Resource not found
- **409 Conflict**: Resource conflict (e.g., duplicate)
- **422 Unprocessable Entity**: Validation error
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error
- **503 Service Unavailable**: Service temporarily unavailable

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid IMSI format",
    "details": {
      "field": "imsi",
      "expected": "15 digits",
      "received": "abc123"
    }
  }
}
```

### Common Error Codes
- `VALIDATION_ERROR`: Invalid input data
- `AUTHENTICATION_REQUIRED`: No authentication token provided
- `INVALID_TOKEN`: Invalid or expired JWT token
- `INSUFFICIENT_PERMISSIONS`: User lacks required permissions
- `RESOURCE_NOT_FOUND`: Requested resource doesn't exist
- `QUOTA_EXCEEDED`: Tenant quota limit reached
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `OPERATION_FAILED`: Operation execution failed
- `INTERNAL_ERROR`: Unexpected server error

---

## Rate Limiting

**Default Limits:**
- Authenticated requests: **100 requests/hour**
- Anonymous requests: **10 requests/hour**
- Exploit operations: **10 operations/hour**

**Rate Limit Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1735650000
```

**Rate Limit Exceeded Response (429):**
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 3600 seconds.",
    "retry_after": 3600
  }
}
```

---

## Pagination

**Request:**
```http
GET /api/targets?page=2&limit=50
```

**Response:**
```json
{
  "success": true,
  "data": [...],
  "pagination": {
    "page": 2,
    "limit": 50,
    "total": 250,
    "pages": 5,
    "has_next": true,
    "has_prev": true,
    "next_page": "/api/targets?page=3&limit=50",
    "prev_page": "/api/targets?page=1&limit=50"
  }
}
```

---

## Webhooks

Configure webhooks to receive real-time notifications.

### Register Webhook
```http
POST /api/webhooks
Authorization: Bearer {token}
Content-Type: application/json

{
  "url": "https://your-server.com/webhook",
  "events": ["target.created", "scan.completed", "anomaly.detected"],
  "secret": "your_webhook_secret"
}
```

### Webhook Payload Example
```json
{
  "event": "scan.completed",
  "timestamp": "2025-12-31T10:31:00Z",
  "data": {
    "scan_id": "scan_12345",
    "status": "completed",
    "results": {
      "signals_detected": 25
    }
  },
  "signature": "sha256=abc123..."
}
```

---

## SDK & Client Libraries

### Python
```bash
pip install falconone-sdk
```

```python
from falconone import FalconOneClient

client = FalconOneClient(
    base_url="https://api.falconone.example.com",
    token="your_jwt_token"
)

# Create target
target = client.targets.create(
    imsi="310150123456789",
    imei="352099001761481"
)

# Start scan
scan = client.scan.start(target_id=target.id, duration=60)
```

### JavaScript/TypeScript
```bash
npm install @falconone/sdk
```

```javascript
import { FalconOneClient } from '@falconone/sdk';

const client = new FalconOneClient({
  baseURL: 'https://api.falconone.example.com',
  token: 'your_jwt_token'
});

// List targets
const targets = await client.targets.list({ limit: 50 });
```

---

## OpenAPI Specification

**Status**: Planned for future release

OpenAPI 3.0 specification generation is currently in development. Once available, it will be accessible at:
- **Swagger UI**: `http://localhost:5000/api/docs` (planned)
- **OpenAPI JSON**: `http://localhost:5000/api/openapi.json` (planned)

For current API documentation, refer to the REST endpoints documented above or use the interactive dashboard at `http://localhost:5000`.

---

## Support

- **Documentation**: https://docs.falconone.example.com
- **GitHub**: https://github.com/yourusername/falconone
- **Email**: support@falconone.example.com
- **Discord**: https://discord.gg/falconone
