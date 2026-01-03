# FalconOne User Manual

**Version:** 3.5.0  
**Last Updated:** January 3, 2026  
**Audience:** Security Researchers, Network Engineers, Operators

---

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Dashboard Overview](#dashboard-overview)
4. [Target Management](#target-management)
5. [Network Scanning](#network-scanning)
6. [Signal Monitoring](#signal-monitoring)
7. [Exploit Operations](#exploit-operations)
8. [RANSacked Exploit Operations](#ransacked-exploit-operations) **(NEW v1.8.0)**
9. [6G NTN Operations](#6g-ntn-operations) **(NEW v1.9.0)**
10. [ISAC Framework](#isac-framework) **(NEW v1.9.0)**
11. [Law Enforcement Mode](#law-enforcement-mode) **(NEW v1.9.0)**
12. [System Tools Management](#system-tools-management)
13. [AI/ML Features](#aiml-features)
14. [O-RAN Integration](#o-ran-integration)
15. [Reports & Exports](#reports--exports)
16. [Security & Compliance](#security--compliance)
17. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is FalconOne?

FalconOne is a comprehensive 5G/LTE/GSM security testing and analysis platform designed for:
- **Security Researchers**: Test cellular network vulnerabilities
- **Network Engineers**: Monitor and analyze network performance
- **Operators**: Manage security operations across multiple networks

### Key Features

✅ **Multi-Network Support**: 5G NR, LTE, GSM, CDMA  
✅ **Real-Time Monitoring**: Signal analysis, KPI tracking, anomaly detection  
✅ **Advanced Exploits**: DOS attacks, downgrade attacks, MITM  
✅ **RANSacked Integration**: 96 CVE payloads, 7 exploit chains, visual GUI controls **(NEW v1.8.0)**  
✅ **Comprehensive Testing**: 100+ integration tests, performance benchmarks **(NEW v1.8.0)**  
✅ **AI/ML Integration**: Signal classification, explainable AI, online learning  
✅ **O-RAN Support**: Near-RT RIC, E2 interface, xApp deployment  
✅ **Enterprise Features**: Multi-tenancy, RBAC, blockchain audit trail  
✅ **Cloud-Native**: AWS/Azure/GCP deployment with auto-scaling  

### System Requirements

**Hardware:**
- CPU: 4+ cores (8+ recommended)
- RAM: 8GB minimum (16GB recommended)
- Storage: 50GB+ available space
- SDR: USRP B200/B210, HackRF One, or BladeRF

**Software:**
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker (optional, recommended)

---

## Getting Started

### Installation

#### Option 1: Quick Start (Docker)
```bash
# Clone repository
git clone https://github.com/yourusername/falconone.git
cd falconone

# Start with Docker Compose
docker-compose up -d

# Access dashboard
open http://localhost:5000
```

#### Option 2: Manual Installation
```bash
# Install dependencies
python install_dependencies.py

# Configure environment
cp .env.example .env
nano .env  # Edit configuration

# Initialize database
python -c "from falconone.utils.database import FalconOneDatabase; db = FalconOneDatabase(); db.create_tables()"

# Start application
python main.py
```

### First Login

1. Navigate to `http://localhost:5000`
2. **Default Credentials** (⚠️ Change immediately!):
   - Username: `admin`
   - Password: `admin`
3. Click "Login"
4. You'll be prompted to change your password

### Initial Configuration

1. **Change Admin Password**:
   - Go to Settings → Account
   - Enter new password (min 8 characters)
   - Click "Update Password"

2. **Configure SDR**:
   - Go to Settings → SDR Configuration
   - Select your SDR device (USRP B210, HackRF, etc.)
   - Set sample rate, frequency range
   - Click "Test Connection"

3. **Set Up Multi-Tenancy** (Enterprise):
   - Go to Settings → Tenants
   - Create your organization
   - Configure resource quotas

---

## Dashboard Overview

### Main Dashboard

The main dashboard provides a real-time overview of system status:

**Widgets:**
- **Active Targets**: Number of currently tracked targets
- **Running Scans**: Active scanning operations
- **Recent Alerts**: Latest security alerts and anomalies
- **Network Health**: Signal quality metrics (RSSI, RSRP, RSRQ)
- **System Status**: CPU, memory, disk usage

**Navigation Menu:**
- 🎯 **Targets**: Manage tracked devices
- 📡 **Scanning**: Start/monitor scans
- 🔍 **Monitoring**: Real-time network monitoring
- 💥 **Exploits**: Execute security tests
- 🤖 **AI/ML**: Machine learning features
- 📊 **Reports**: Generate and export reports
- 👥 **Users**: User management (admin only)
- ⚙️ **Settings**: System configuration

### User Roles

**Admin** (Full Access):
- All features enabled
- User management
- System configuration
- Audit log access

**Operator** (Execute Operations):
- Target management
- Scanning and monitoring
- Exploit execution
- Report generation

**Viewer** (Read-Only):
- View targets and scans
- View monitoring data
- View reports
- No execution permissions

---

## Target Management

### Creating a Target

1. Click **Targets** in the navigation menu
2. Click **"+ New Target"** button
3. Fill in target information:
   - **IMSI**: International Mobile Subscriber Identity (15 digits)
   - **IMEI**: International Mobile Equipment Identity (15 digits)
   - **MSISDN**: Phone number (optional)
   - **Network Type**: 5G, LTE, or GSM
4. Click **"Create Target"**

**Example:**
```
IMSI: 310150123456789
IMEI: 352099001761481
MSISDN: +1234567890
Network Type: 5G
```

### Target List View

The target list displays:
- Target ID
- IMSI (masked for privacy)
- Network type
- Last seen timestamp
- Location (if available)
- Status (Active/Inactive)

**Actions:**
- 👁️ **View**: See detailed target information
- ✏️ **Edit**: Update target details
- 📍 **Track**: View location history
- 🗑️ **Delete**: Remove target (requires confirmation)

### Target Details Page

Click on any target to view detailed information:

**Overview Tab:**
- Basic information (IMSI, IMEI, MSISDN)
- Network information (MCC, MNC, Cell ID)
- Location (map view with history)
- Last activity timestamp

**Captures Tab:**
- List of SUCI captures
- Decrypted IMSI values
- Capture timestamps
- Signal strength at capture

**Exploits Tab:**
- History of exploit operations
- Operation status and results
- Timestamps and durations

**Activity Timeline:**
- Chronological activity log
- Network attachments
- Location updates
- Exploit operations

---

## Network Scanning

### Starting a Scan

1. Click **Scanning** → **"New Scan"**
2. Configure scan parameters:
   - **Target**: Select target or "Scan All"
   - **Network Type**: 5G, LTE, GSM, or All
   - **Frequency Range**: 
     - 5G: 600 MHz - 6 GHz
     - LTE: 700 MHz - 2.6 GHz
     - GSM: 850/900/1800/1900 MHz
   - **Duration**: 10s to 3600s (1 hour)
   - **Scan Type**: Quick, Standard, Deep
3. Click **"Start Scan"**

**Scan Types:**

**Quick Scan** (10-30 seconds):
- Basic signal detection
- Identify active networks
- Fast overview

**Standard Scan** (1-5 minutes):
- Comprehensive signal analysis
- Target identification
- SUCI capture attempts

**Deep Scan** (5-30 minutes):
- Exhaustive frequency sweep
- Detailed signal analysis
- Multiple capture attempts
- Network topology mapping

### Monitoring Active Scans

The scanning page displays:
- **Scan ID**: Unique identifier
- **Status**: Running, Completed, Failed
- **Progress**: Percentage complete
- **Frequency**: Current scanning frequency
- **Signals Detected**: Count of detected signals
- **Targets Found**: Number of identified targets

**Actions:**
- ⏸️ **Pause**: Temporarily pause scanning
- ⏹️ **Stop**: Terminate scan early
- 📊 **View Results**: See detailed results

### Scan Results

After completion, view results:

**Summary:**
- Total scan duration
- Frequencies scanned
- Signals detected
- Targets identified
- SUCI captures

**Detailed Results:**
Each detected signal shows:
- Frequency (MHz)
- Signal strength (RSSI)
- Network type (5G/LTE/GSM)
- Cell ID
- PLMN (MCC + MNC)

**Export Options:**
- CSV: Raw data export
- JSON: API-compatible format
- PDF: Professional report

---

## Signal Monitoring

### Real-Time Monitoring

1. Click **Monitoring** → **"Start Monitoring"**
2. Select network type (5G, LTE, or GSM)
3. Choose monitoring duration
4. Click **"Start"**

**Monitored Metrics:**

**5G NR:**
- RSRP (Reference Signal Received Power)
- RSRQ (Reference Signal Received Quality)
- SINR (Signal-to-Interference-plus-Noise Ratio)
- SS-RSRP (Synchronization Signal RSRP)
- Beam ID
- Serving cell metrics

**LTE:**
- RSRP
- RSRQ
- RSSI (Received Signal Strength Indicator)
- CQI (Channel Quality Indicator)
- Cell-specific information

**GSM:**
- RSSI
- RXLEV (Received Signal Level)
- RXQUAL (Received Signal Quality)
- ARFCN (Absolute Radio Frequency Channel Number)

### Anomaly Detection

FalconOne automatically detects anomalies:

**Types of Anomalies:**
- Sudden signal drops
- Unexpected cell changes
- Abnormal handover patterns
- Interference detection
- Fake base station indicators

**Anomaly Alerts:**
When detected, you'll see:
- 🔴 **Critical**: Immediate action required
- 🟠 **Warning**: Investigate soon
- 🟡 **Info**: Monitor situation

**Investigating Anomalies:**
1. Click on anomaly alert
2. View detailed metrics at time of detection
3. Check nearby cell information
4. Review historical data
5. Mark as "Investigated" or "False Positive"

---

## Exploit Operations

⚠️ **Warning**: Exploit operations may be illegal in your jurisdiction. Use only on authorized networks with proper permissions.

### Available Exploits

**1. Denial of Service (DOS) Attack**
- **Purpose**: Disrupt target's connectivity
- **Methods**: Frequency jamming, RACH exhaustion, paging saturation
- **Risk Level**: High
- **Legal**: Requires authorization

**2. Downgrade Attack**
- **Purpose**: Force target to lower network generation
- **Target**: 5G → LTE → 3G
- **Risk Level**: Medium
- **Legal**: Requires authorization

**3. Man-in-the-Middle (MITM)**
- **Purpose**: Intercept communications
- **Method**: Fake base station
- **Risk Level**: High
- **Legal**: Strictly controlled

### Executing an Exploit

1. Click **Exploits** → **"Execute"**
2. Select exploit type
3. Choose target
4. Configure parameters:
   - Duration (seconds)
   - Power level (0.0 - 1.0)
   - Attack type (for DOS)
5. Review warning and legal disclaimer
6. Check "I have authorization"
7. Click **"Execute"**

**Example: DOS Attack**
```
Target: IMSI 310150123456789
Exploit: DOS Attack
Attack Type: Frequency Jamming
Duration: 30 seconds
Power Level: 0.8
```

### Operation Monitoring

Monitor active operations:
- **Operation ID**: Unique identifier
- **Status**: Executing, Completed, Failed
- **Progress**: Percentage
- **Elapsed Time**: Duration so far
- **Metrics**: Packets sent, success rate

**Actions:**
- ⏹️ **Stop**: Terminate operation immediately
- 📊 **View Details**: See detailed metrics
- 📝 **Export Log**: Download operation log

### Operation Results

After completion:

**Success Metrics:**
- Total duration
- Packets sent/received
- Success rate (%)
- Target response
- Network impact

**Evidence Collection:**
- Operation log
- Network captures
- Before/after comparisons
- Timestamps

---

## RANSacked Exploit Operations

**New in v1.8.0**: Complete RANSacked vulnerability exploit integration with 96 CVE payloads, 7 pre-defined exploit chains, and visual GUI controls.

⚠️ **Critical Warning**: RANSacked exploits target real vulnerabilities in production cellular core implementations. Use ONLY in authorized lab environments with proper permissions.

### Overview

The RANSacked module provides:
- **96 CVE Exploits**: Payloads for vulnerabilities in 7 cellular implementations
- **7 Exploit Chains**: Pre-configured multi-step attack scenarios
- **Visual GUI**: Interactive exploit selection and execution
- **Comprehensive Testing**: 100+ integration tests with performance benchmarks

**Supported Implementations:**
- **OAI 5G**: 11 CVEs (OpenAirInterface 5G Core)
- **Open5GS 5G**: 6 CVEs (Open5GS 5G Core)
- **Magma LTE**: 25 CVEs (Magma LTE Core)
- **Open5GS LTE**: 26 CVEs (Open5GS LTE Core)
- **Miscellaneous**: 28 CVEs (srsRAN, NextEPC, SD-Core, Athonet, OAI GTP)

### Accessing RANSacked GUI

1. Navigate to **Exploits** → **"RANSacked Exploits"**
2. Or visit: `http://localhost:5000/exploits/ransacked`

**GUI Features:**
- 📊 **Statistics Bar**: View total CVEs, selected count, implementations, protocols
- 🔍 **Advanced Filtering**: Filter by implementation, protocol, or search
- 🎯 **Multi-Select**: Click cards to select multiple exploits
- 👁️ **Payload Preview**: View hex payload before execution
- 🚀 **Bulk Execution**: Execute multiple exploits at once
- ⛓️ **Chain Execution**: Run pre-defined exploit chains

### Filtering Exploits

**By Implementation:**
```
[Implementation: Open5GS 5G ▼]
Results: 6 CVEs
```

**By Protocol:**
```
[Protocol: NGAP ▼]
Results: 12 CVEs (5G control plane)
```

**By Search:**
```
[Search: CVE-2024-24445]
Results: 1 CVE
```

### Viewing Exploit Details

1. **Click any exploit card** to select
2. **Click "View Details"** button
3. **Review information**:
   - CVE ID and severity
   - Implementation and protocol
   - Success indicators
   - Payload preview (hex dump)
   - Payload size
4. **Close modal** to return to grid

**Example Details:**
```
CVE-2024-24445
Implementation: OAI 5G
Protocol: NGAP
Severity: Critical
Success Indicators:
  ✓ AMF crash
  ✓ Core dump
  ✓ Connection reset
Payload Size: 48 bytes
Payload Preview:
7e 00 41 2d e0 00 00 03 00 55 00 05 04 00 c0 a8
01 64 00 79 40 0b 0a 00 00 00 00 00 00 00 00 00
```

### Executing Individual Exploits

1. **Select one or more exploits** (click cards)
2. **Click "Execute Selected"** button
3. **Configure execution**:
   - Target IP: `192.168.1.100`
   - ☑️ Dry Run Mode (recommended)
   - ☐ Capture Traffic
4. **Review warning message**
5. **Click "Confirm Execution"**

**Dry Run Mode** (Recommended):
- Generates payload but does NOT send
- Safe for testing and validation
- Shows what WOULD be executed
- No impact on target

**Live Execution** (Requires Authorization):
- Actually sends exploit payload
- May cause target crash/DOS
- Logged in audit trail
- Rate limited: 5 executions/minute

**Example Execution:**
```
Selected: 3 CVEs
Target: 192.168.1.100
Dry Run: ✓ Enabled

Results:
✓ CVE-2024-24445 - DRY RUN: Would execute NGAP payload (48 bytes)
✓ CVE-2024-24428 - DRY RUN: Would execute NGAP payload (52 bytes)
✓ CVE-2024-24427 - DRY RUN: Would execute NGAP payload (48 bytes)

Total: 3/3 successful (100%)
```

### Using Exploit Chains

Exploit chains combine multiple CVEs in multi-step attacks:

**Available Chains:**

| Chain ID | Name | Steps | Target | Success Rate |
|----------|------|-------|--------|--------------|
| chain_1 | Reconnaissance & DoS | 3 | Open5GS 5G | 95% |
| chain_2 | Persistent Access | 3 | OAI 5G | 90% |
| chain_3 | Multi-Implementation Attack | 5 | Multiple | 85% |
| chain_4 | Memory Corruption Cascade | 5 | Magma/OAI | 80% |
| chain_5 | LTE S1AP Flood | 10 | Magma/Open5GS | 95% |
| chain_6 | GTP Protocol Attacks | 5 | Open5GS/OAI | 88% |
| chain_7 | Advanced Evasion | 4 | Open5GS/OAI | 92% |

**Executing a Chain:**

1. Navigate to **Exploits** → **"RANSacked Exploits"**
2. Click **"Execute Chain"** dropdown
3. Select chain (e.g., "chain_1")
4. Configure target IP
5. Enable dry run (recommended)
6. Review chain steps
7. Click **"Execute Chain"**

**Example: Chain 1 (Reconnaissance & DoS)**
```
Chain: Reconnaissance & DoS Chain
Target: 192.168.1.100
Steps: 3
Estimated Time: 700ms
Dry Run: ✓ Enabled

Execution:
[Step 1/3] DISCOVERY - CVE-2024-24425
  ✓ DRY RUN: Would execute payload (45 bytes, NGAP)

[Step 2/3] IMPACT - CVE-2024-24428
  ✓ DRY RUN: Would execute payload (52 bytes, NGAP)

[Step 3/3] IMPACT - CVE-2024-24427
  ✓ DRY RUN: Would execute payload (48 bytes, NGAP)

Result: 3/3 steps successful (100%)
Execution Time: 715ms
Success Rate: 95%
```

### Using API Endpoints

**List All Payloads:**
```bash
curl -X GET http://localhost:5000/api/ransacked/payloads \
  -H "Authorization: Bearer <token>"
```

**Execute Single Exploit:**
```bash
curl -X POST http://localhost:5000/api/ransacked/execute \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "cve_id": "CVE-2024-24445",
    "target_ip": "192.168.1.100",
    "options": {
      "dry_run": true,
      "capture_traffic": false
    }
  }'
```

**Execute Exploit Chain:**
```bash
curl -X POST http://localhost:5000/api/ransacked/chains/execute \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "chain_id": "chain_1",
    "target_ip": "192.168.1.100",
    "options": {
      "dry_run": true
    }
  }'
```

### Running Integration Tests

Validate RANSacked functionality:

```bash
# Run all tests
pytest falconone/tests/test_ransacked_exploits.py -v

# Run specific test class
pytest falconone/tests/test_ransacked_exploits.py::TestPayloadGeneration -v

# Run performance benchmarks
pytest falconone/tests/test_ransacked_exploits.py::TestPerformance -v
```

**Expected Results:**
```
TestPayloadGeneration::test_oai_5g_payloads ✓ PASSED (11/11 CVEs)
TestPayloadGeneration::test_all_implementations ✓ PASSED (96/96 CVEs)
TestPerformance::test_bulk_generation_speed ✓ PASSED (<2 seconds)

================== 108 tests passed ==================
```

### Quick Start Demo

Run interactive quick-start guide:

```bash
python quickstart_ransacked.py
```

**Demo Features:**
1. Dependency check (pytest, flask, scapy)
2. Run integration tests
3. Demo exploit chains
4. Start GUI dashboard
5. Open browser automatically

### Security & Best Practices

**✓ Always Use Dry Run First**
- Test payloads without sending
- Validate target configuration
- Review execution logs

**✓ Rate Limiting Enforced**
- List payloads: 60/minute
- Generate payload: 30/minute
- Execute exploit: 5/minute
- Execute chain: 3/minute

**✓ Comprehensive Audit Logging**
```
[2026-01-02 10:30:00] [CRITICAL] [AUDIT] RANSacked EXPLOIT EXECUTION
User: admin, IP: 127.0.0.1
CVE: CVE-2024-24445
Target: 192.168.1.100
Dry Run: true
```

**✓ Authentication Required**
- Session-based authentication
- RBAC enforcement
- Auto-logout on timeout

**✗ Never Use on Production Networks**
- Exploits cause real damage
- May violate laws
- Only authorized lab environments

### Troubleshooting

**GUI Not Loading:**
```bash
# Check dashboard running
ps aux | grep python | grep dashboard

# Restart dashboard
python start_dashboard.py
```

**Tests Failing:**
```bash
# Check pytest installed
pip install pytest

# Run with verbose output
pytest falconone/tests/test_ransacked_exploits.py -v -s
```

**Chain Execution Fails:**
```bash
# Always use dry_run first
python -c "
from exploit_chain_examples import chain_1_reconnaissance_crash
chain = chain_1_reconnaissance_crash()
result = chain.execute('127.0.0.1', dry_run=True)
print(result)
"
```

---

## 6G NTN Operations

**New in v1.9.0**: Complete 6G Non-Terrestrial Networks (NTN) monitoring and exploitation capabilities.

### Supported Satellite Types

| Type | Altitude | Typical Doppler | Use Case |
|------|----------|-----------------|----------|
| **LEO** | 300-2000 km | ±40 kHz | Starlink, OneWeb - Low latency broadband |
| **MEO** | 2000-35786 km | ±5 kHz | O3b, SES - Moderate latency |
| **GEO** | 35,786 km | ±500 Hz | Traditional satellite - High latency |
| **HAPS** | 20-50 km | ±1 kHz | High-Altitude Platforms - Regional coverage |
| **UAV** | 0-20 km | Variable | Unmanned Aerial Vehicles - Tactical |

### NTN Monitoring

1. Navigate to **6G NTN** tab in the dashboard
2. Select satellite type from dropdown (LEO/MEO/GEO/HAPS/UAV)
3. Configure monitoring parameters:
   - **Duration**: 1-300 seconds
   - **Frequency**: Center frequency in GHz (default: 150 GHz for sub-THz)
   - **ISAC Enable**: Toggle integrated sensing
4. Click **Start Monitoring**

### NTN Exploitation

Available exploit types:
- **Beam Hijacking** (75% success) - RIS-assisted beam manipulation
- **Handover Poisoning** (65% success) - AI orchestration attack
- **Downlink Spoofing** (70% success) - Inject fake downlink signals
- **Feeder Link Attack** (60% success) - Ground-satellite link attack
- **NB-IoT NTN** (68% success) - NB-IoT satellite exploitation
- **Timing Advance** (72% success) - Timing advance manipulation
- **Quantum Attack** (35% success) - QKD exploitation

---

## ISAC Framework

**New in v1.9.0**: Integrated Sensing and Communications framework for joint radar-communication operations.

### Sensing Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Monostatic** | Same TX/RX antenna | Close-range, single device |
| **Bistatic** | Separate TX/RX | Long-range, stealth detection |
| **Cooperative** | Multiple sensors | Wide area, high accuracy |

### Waveform Types

- **OFDM**: Standard 5G waveform, good sensing resolution
- **DFT-s-OFDM**: Lower PAPR, better for uplink sensing
- **FMCW**: Traditional radar, highest range resolution

### Starting ISAC Monitoring

1. Navigate to **ISAC** tab
2. Select sensing mode
3. Configure parameters:
   - **Frequency**: 100-300 GHz for sub-THz
   - **Waveform**: OFDM/DFT-s-OFDM/FMCW
   - **Duration**: 1-60 seconds
4. Click **Start Sensing**

### ISAC Exploitation

Available exploit types:
- **Waveform Manipulation** (80% success) - Inject malformed waveforms
- **AI Poisoning** (65% success) - ML model poisoning
- **Privacy Breach** (60% success) - Sensing-based tracking
- **E2SM Hijack** (70% success) - Control plane attack
- **Quantum Attack** (35% success) - Quantum radar exploitation

---

## Law Enforcement Mode

**New in v1.9.0**: Legally-compliant interception operations with warrant validation and evidence chain management.

### ⚠️ Important Legal Notice

LE Mode is designed for **authorized law enforcement operations only**. Use requires:
- Valid court-issued warrant
- Proper jurisdiction
- Authorized personnel credentials

### Activating LE Mode

1. Navigate to **LE Mode** tab
2. Enter warrant details:
   - **Warrant ID**: Court-issued warrant number
   - **Jurisdiction**: Issuing court
   - **Case Number**: Associated case
   - **Authorized By**: Issuing judge
   - **Valid Until**: Expiration date
   - **Target Identifiers**: IMSIs/IMEIs
3. Upload warrant image (optional)
4. Click **Validate Warrant**

### Evidence Chain Management

All operations in LE Mode are:
- **Cryptographically hashed**: SHA-256 evidence integrity
- **Chain-of-custody tracked**: Complete audit trail
- **Tamper-evident**: Any modification detected
- **Export-ready**: Court-compatible packages

### Exporting Evidence

1. Navigate to **LE Mode > Evidence**
2. Select warrant
3. Choose export format:
   - **Court Package**: Complete legal bundle
   - **Raw Data**: Technical analysis format
4. Click **Export**

---

## System Tools Management

**Version 1.7.0 introduces a comprehensive interface for managing external system tools required for cellular monitoring operations.**

### Overview

The System Tools Management tab provides unified management for:
- **gr-gsm** - GSM signal monitoring
- **kalibrate-rtl** - GSM base station scanner
- **OsmocomBB** - Mobile communications baseband
- **LTESniffer** - LTE downlink/uplink sniffer
- **srsRAN** - 4G/5G software radio
- **srsRAN Project** - 5G RAN implementation
- **Open5GS** - 5G Core Network
- **OpenAirInterface** - 5G software
- **UHD** - USRP Hardware Driver
- **BladeRF** - bladeRF driver
- **GNU Radio** - Software radio toolkit
- **SoapySDR** - SDR API
- **gr-osmosdr** - GNU Radio OsmoSDR

### Accessing System Tools

1. **Open Dashboard**: Navigate to http://127.0.0.1:5000
2. **Click "System Tools"** tab in the left sidebar
3. **View Status**: See which tools are installed/missing

### Managing Tools

#### View Tool Status
- **Green Badge (✅)**: Tool is installed and working
- **Red Badge (❌)**: Tool is not installed
- **Completion %**: Shows overall installation progress

#### Install a Tool
1. Click **"Install"** button on a missing tool
2. Copy the installation command from the modal
3. Open your system terminal
4. Run the command with `sudo` privileges
5. Return to dashboard and click **"Refresh Status"**

#### Test a Tool
1. Click **"Test"** button on an installed tool
2. View test results in the notification
3. Check diagnostic output in the details modal

#### Uninstall a Tool
1. Click **"Uninstall"** button
2. Confirm the action
3. Copy the removal command
4. Run in terminal with `sudo`
5. Refresh status to verify

### Tool Categories

**GSM Tools**
- Essential for 2G network monitoring
- Required for IMSI catching operations

**LTE Tools**
- LTE/4G network analysis
- SUCI capture and decoding

**5G Tools**
- 5G NR monitoring
- Network slicing detection
- Beam management analysis

**SDR Drivers**
- Hardware device drivers
- Low-level RF access

**Frameworks**
- Software radio toolkits
- Signal processing libraries

### Best Practices

1. **Install in Order**: Start with SDR drivers, then frameworks, then specific tools
2. **Test After Installation**: Always test tools to verify functionality
3. **Keep Updated**: Regularly check for tool updates
4. **System Requirements**: Ensure adequate disk space and permissions

### Troubleshooting

**Tool Not Detected**:
- Verify PATH environment variable includes tool location
- Restart terminal after installation
- Check sudo privileges

**Installation Failed**:
- Review system package manager (apt, yum, pacman)
- Check internet connectivity
- Verify system compatibility

**Test Failed**:
- Ensure tool dependencies are installed
- Check tool-specific documentation
- Review system logs for errors

---

## AI/ML Features

### Signal Classification

Automatically classify network signals using machine learning.

**Using the Classifier:**
1. Click **AI/ML** → **"Classify Signal"**
2. Upload signal file (CSV, WAV, or IQ format)
3. Click **"Classify"**

**Results:**
- **Prediction**: Network type (5G, LTE, GSM)
- **Confidence**: 0-100%
- **Probabilities**: Per-class probabilities

**Example Output:**
```
Prediction: 5G_NR
Confidence: 95%
Probabilities:
  - 5G_NR: 95%
  - LTE: 4%
  - GSM: 1%
```

### Explainable AI

Understand why the model made a prediction.

1. After classification, click **"Explain Prediction"**
2. Choose explanation method:
   - **SHAP**: Feature importance with Shapley values
   - **LIME**: Local interpretable model
   - **Hybrid**: Combination of both
3. View feature importance:
   - Frequency: 35%
   - Signal Strength: 28%
   - Bandwidth: 22%
   - Modulation: 15%

**Visualizations:**
- Bar chart of feature importance
- Waterfall plot (SHAP)
- Individual conditional expectation (ICE) plots

### Online Learning

Train models incrementally with new data.

1. Click **AI/ML** → **"Online Learning"**
2. Upload training data (labeled samples)
3. Configure:
   - Batch size: 50-200 samples
   - Learning rate: Auto or manual
   - Drift detection: Enabled/Disabled
4. Click **"Train"**

**Model Versioning:**
- Each training creates a new version
- Automatic rollback if performance degrades
- Compare versions side-by-side

---

## O-RAN Integration

### Near-RT RIC

The Near-Real-Time RAN Intelligent Controller manages xApps.

**Starting the RIC:**
1. Click **O-RAN** → **"RIC Dashboard"**
2. Click **"Start RIC"**
3. Monitor RIC status (Starting → Running)

**RIC Metrics:**
- Active xApps count
- Connected E2 nodes
- SDL (Shared Data Layer) status
- A1 policies loaded

### xApp Management

**Deploying an xApp:**
1. Click **O-RAN** → **"xApps"** → **"Deploy"**
2. Select xApp type:
   - **Traffic Steering**: Load balancing
   - **Anomaly Detection**: KPI monitoring
   - **Resource Optimization**: PRB allocation
3. Configure parameters
4. Click **"Deploy"**

**Available xApps:**

**Traffic Steering xApp:**
- Monitors UE metrics (throughput, latency)
- Makes handover decisions
- Balances load across cells

**Anomaly Detection xApp:**
- Monitors KPIs (RSRP, throughput, latency)
- Detects anomalies (Z-score, IQR methods)
- Generates alerts

**Resource Optimization xApp:**
- Analyzes PRB utilization
- Optimizes resource allocation
- Improves spectral efficiency

### E2 Interface

Connect external E2 nodes (gNBs, eNBs).

**Registering an E2 Node:**
1. Click **O-RAN** → **"E2 Nodes"** → **"Register"**
2. Enter node details:
   - Node ID: gnb_001
   - PLMN ID: 310150
   - E2 Address: 192.168.1.100:36422
3. Click **"Register"**

**E2 Subscriptions:**
Create subscriptions to receive indications:
1. Select E2 node
2. Click **"Create Subscription"**
3. Choose service model (E2SM-KPM, E2SM-RC)
4. Set report period (1000ms - 60000ms)
5. Click **"Subscribe"**

---

## Reports & Exports

### Generating Reports

1. Click **Reports** → **"Generate Report"**
2. Select report type:
   - **Summary Report**: Overview of all activities
   - **Target Report**: Specific target analysis
   - **Scan Report**: Detailed scan results
   - **Exploit Report**: Operation summaries
   - **Compliance Report**: Audit trail
3. Configure:
   - Date range
   - Targets to include
   - Detail level (Summary, Standard, Detailed)
4. Click **"Generate"**

**Report Formats:**
- **PDF**: Professional presentation
- **CSV**: Raw data for analysis
- **JSON**: API-compatible format
- **HTML**: Web-viewable report

### Exporting Data

**Bulk Export:**
1. Click **Reports** → **"Export Data"**
2. Select tables:
   - ☑️ Targets
   - ☑️ SUCI Captures
   - ☑️ Exploit Operations
   - ☑️ Anomaly Alerts
3. Choose format (CSV, JSON, SQL)
4. Click **"Export All"**

**Scheduled Exports:**
Set up automatic exports:
1. Go to Settings → Scheduled Tasks
2. Click **"Add Schedule"**
3. Configure:
   - Frequency: Daily, Weekly, Monthly
   - Export type: Full or Incremental
   - Destination: Local, S3, FTP
4. Click **"Save"**

---

## Security & Compliance

### Blockchain Audit Trail

All security-sensitive operations are logged on an immutable blockchain.

**Viewing Audit Logs:**
1. Click **Security** → **"Audit Trail"**
2. Filter by:
   - Event type (AUTH, ACCESS, EXPLOIT)
   - User
   - Date range
   - Severity
3. Click **"Search"**

**Blockchain Verification:**
1. Click **"Verify Chain Integrity"**
2. System checks:
   - Hash integrity
   - Previous block links
   - Proof of Work validity
3. View results (Valid/Invalid)

### Multi-Tenant Security

**Tenant Isolation:**
- Data encrypted per-tenant
- Separate database schemas
- Network-level isolation

**Resource Quotas:**
View your quota usage:
1. Click **Settings** → **"Quota Usage"**
2. See limits:
   - Max Targets: 500
   - Max Scans/Hour: 50
   - Storage: 10 GB
3. Current usage displayed

**Compliance Features:**
- GDPR-compliant data handling
- Automatic data retention policies
- Audit logs for compliance reporting
- Right to deletion support

### Zero-Knowledge Proofs

Authenticate without revealing credentials:
1. Enable ZKP: Settings → Security → Zero-Knowledge Auth
2. On next login, you'll prove knowledge of password
3. Server never sees actual password
4. Quantum-resistant authentication

---

## Troubleshooting

### Common Issues

**Problem: SDR Not Detected**
```
Solution:
1. Check USB connection
2. Install SDR drivers: uhd_images_downloader (USRP) or hackrf_info (HackRF)
3. Verify permissions: sudo usermod -a -G plugdev $USER
4. Restart application
```

**Problem: Scan Returns No Results**
```
Solution:
1. Verify frequency range matches your location's bands
2. Check antenna connection
3. Increase scan duration
4. Try different gain settings (default: 40)
5. Ensure SDR has clear line of sight
```

**Problem: Cannot Login**
```
Solution:
1. Check username/password (case-sensitive)
2. Clear browser cache
3. Verify database connection: psql -h localhost -U falconone
4. Reset password: python reset_password.py admin
```

**Problem: High CPU Usage**
```
Solution:
1. Reduce number of concurrent scans
2. Lower SDR sample rate
3. Disable real-time visualization
4. Check for runaway processes: ps aux | grep falconone
```

**Problem: Database Errors**
```
Solution:
1. Check database health: Settings → System → Database Health
2. Run integrity check
3. Restore from backup if needed
4. Check disk space: df -h
```

### Log Files

**Application Logs:**
```
Location: logs/falconone.log
Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

Viewing recent errors:
tail -f logs/falconone.log | grep ERROR
```

**Audit Logs:**
```
Location: Blockchain (immutable)
Access: Dashboard → Security → Audit Trail
```

**SDR Logs:**
```
Location: logs/sdr/
Files: sdr_YYYYMMDD.log
```

### Getting Help

**Documentation:**
- User Manual: This document
- API Docs: API_DOCUMENTATION.md
- Developer Guide: DEVELOPER_GUIDE.md

**Support Channels:**
- Email: support@falconone.example.com
- Discord: https://discord.gg/falconone
- GitHub Issues: https://github.com/yourusername/falconone/issues

**Emergency Support:**
- Critical issues: emergency@falconone.example.com
- Response time: 4 hours (business days)

---

## Appendix

### Keyboard Shortcuts

- `Ctrl+K`: Quick search
- `Ctrl+N`: New target
- `Ctrl+S`: Start scan
- `Ctrl+R`: Refresh dashboard
- `Esc`: Close modal
- `?`: Show help overlay

### Glossary

- **IMSI**: International Mobile Subscriber Identity
- **IMEI**: International Mobile Equipment Identity
- **SUCI**: Subscription Concealed Identifier
- **RSRP**: Reference Signal Received Power
- **RSRQ**: Reference Signal Received Quality
- **E2**: Interface between RIC and RAN nodes
- **xApp**: Application running on Near-RT RIC
- **ZKP**: Zero-Knowledge Proof

### Legal Disclaimer

FalconOne is intended for authorized security research and testing only. Users are solely responsible for ensuring compliance with all applicable laws and regulations. Unauthorized use of this software may result in civil and criminal penalties.

---

**Manual Version:** 3.0.0  
**Last Updated:** December 31, 2025  
**Next Review:** March 31, 2026
