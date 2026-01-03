# FalconOne Installation Guide

**Version:** 1.9.0  
**Last Updated:** January 2026  
**Status:** Production Ready (All Features Validated ✅)

---

## What's New in v1.9.0

- ✅ **6G NTN Integration**: LEO/MEO/GEO/HAPS/UAV satellite support with sub-THz monitoring
- ✅ **ISAC Framework**: Integrated Sensing & Communications with 8 exploitation CVEs
- ✅ **O-RAN Integration**: E2SM-RC/KPM interfaces with xApp deployment
- ✅ **Production Ready**: Real data flows, no hardcoded values, comprehensive validation
- ✅ **Enhanced Security**: LE warrant enforcement, rate limiting, evidence chain logging

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Installation Methods](#installation-methods)
   - [Method 1: Ubuntu Installation (Recommended)](#method-1-ubuntu-installation-recommended)
   - [Method 2: Docker Deployment](#method-2-docker-deployment)
   - [Method 3: Kubernetes Deployment](#method-3-kubernetes-deployment)
   - [Method 4: Manual Installation](#method-4-manual-installation)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [Post-Installation](#post-installation)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Requirements
- **CPU:** Multi-core processor (4+ cores recommended)
- **RAM:** 8GB minimum, 16GB+ recommended
- **Storage:** 10GB free space minimum
- **SDR Hardware:** (Optional) Compatible SDR device for RF operations
  - USRP B200/B210
  - HackRF One
  - LimeSDR
  - BladeRF

### Software Requirements
- **Operating System:** Ubuntu 20.04 LTS or later (recommended)
- **Python:** 3.10 or higher (3.13 tested)
- **pip:** Latest version
- **Git:** For cloning repository

### Network Requirements
- Internet connection for dependency installation
- Open ports for dashboard (default: 5000)
- SDR device access permissions

---

## Prerequisites

### 1. Install Python 3.10+

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

**Verify Installation:**
```bash
python3 --version  # Should be 3.10 or higher
```

### 2. Install System Dependencies

```bash
sudo apt install -y \
    build-essential \
    cmake \
    git \
    libusb-1.0-0-dev \
    pkg-config \
    libfftw3-dev \
    libtool \
    automake
```

### 3. Install SDR Libraries (Optional)

**For USRP:**
```bash
sudo apt install -y uhd-host libuhd-dev
```

**For HackRF:**
```bash
sudo apt install -y hackrf libhackrf-dev
```

**For LimeSDR:**
```bash
sudo add-apt-repository -y ppa:myriadrf/drivers
sudo apt update
sudo apt install -y limesuite liblimesuite-dev
```

---

## Installation Methods

### Method 1: Ubuntu Installation (Recommended)

This is the **recommended production installation method** for Ubuntu systems.

#### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/falconone.git
cd falconone
```

#### Step 2: Run Automated Installer

```bash
chmod +x install_ubuntu.sh
sudo ./install_ubuntu.sh
```

The installer will:
- ✅ Install system dependencies
- ✅ Create Python virtual environment
- ✅ Install Python packages from requirements.txt
- ✅ Configure SDR access permissions
- ✅ Set up configuration files

#### Step 3: Activate Environment

```bash
source .venv/bin/activate
```

#### Step 4: Verify Installation

```bash
python validate_features.py
```

**Expected Output:**
```
======================================================================
TOTAL: 8/8 PASSED, 0/8 FAILED
======================================================================
🎉 ALL FEATURES VALIDATED SUCCESSFULLY! 🎉
```

---

### Method 2: Docker Deployment

Ideal for **containerized deployments** and **development environments**.

#### Step 1: Build Docker Image

```bash
docker build -t falconone:1.7.0 .
```

#### Step 2: Run Container

**Basic Run:**
```bash
docker run -d \
  --name falconone \
  --privileged \
  -p 5000:5000 \
  -v $(pwd)/config:/app/config \
  falconone:1.7.0
```

**With SDR Device (USB passthrough):**
```bash
docker run -d \
  --name falconone \
  --privileged \
  --device=/dev/bus/usb \
  -p 5000:5000 \
  -v $(pwd)/config:/app/config \
  falconone:1.7.0
```

#### Step 3: Verify Container

```bash
docker exec falconone python validate_features.py
```

---

### Method 3: Kubernetes Deployment

For **production-scale** and **cloud deployments**.

#### Step 1: Create Namespace

```bash
kubectl create namespace falconone
```

#### Step 2: Apply Deployment

```bash
kubectl apply -f k8s-deployment.yaml -n falconone
```

#### Step 3: Verify Deployment

```bash
kubectl get pods -n falconone
kubectl logs -f deployment/falconone -n falconone
```

#### Step 4: Expose Service (Optional)

```bash
kubectl port-forward -n falconone service/falconone 5000:5000
```

---

### Method 4: Manual Installation

For **custom setups** or **development**.

#### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/falconone.git
cd falconone
```

#### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Core Dependencies Installed:**
- `scapy` - Packet manipulation and network analysis
- `scipy` - Scientific computing (TDOA, AoA algorithms)
- `scikit-learn` - Machine learning for classification
- `numpy` - Numerical processing
- `requests` - HTTP client for API interactions
- `pyyaml` - Configuration file parsing
- `click` - CLI framework
- `colorama` - Terminal color output

#### Step 4: Configure System

```bash
cp config/config.yaml.example config/config.yaml
nano config/config.yaml  # Edit configuration
```

#### Step 5: Verify Installation

```bash
python validate_features.py
```

---

## Configuration

### 1. Main Configuration File

Edit `config/config.yaml`:

```yaml
# FalconOne Configuration
version: 1.7.0

# Monitoring Configuration
monitoring:
  gsm: true
  umts: true
  lte: true
  5g: true
  6g: false  # Experimental
  cdma: false

# SDR Configuration
sdr:
  device: "usrp"  # Options: usrp, hackrf, limesdr, bladerf
  sample_rate: 20000000  # 20 MHz
  gain: 40
  frequency: 2450000000  # 2.45 GHz

# AI/ML Configuration
ai:
  enable_federated: false
  model_path: "./models"
  
# Security Configuration
security:
  jurisdiction: "usa"  # Options: usa, eu, jp
  audit_interval: 3600  # seconds
  block_non_compliant: true
  
# Data Validation
validation:
  strict_mode: false
  min_snr_db: 5.0
  
# Error Recovery
recovery:
  max_retries: 5
  backoff_base: 2
  checkpoint_enabled: true
```

### 2. FalconOne-Specific Configuration

Edit `config/falconone.yaml`:

```yaml
# Advanced FalconOne Configuration
features:
  error_recovery: true
  data_validation: true
  security_auditor: true
  aiot_analyzer: true
  semantic_exploiter: true
  cyber_rf_fusion: true
  regulatory_scanner: true
  precision_geolocation: true

# Geolocation Configuration
geolocation:
  nlos_detection: true
  v2x_fusion: true
  kalman_smoothing: true
  
# Exploit Configuration (Research Use Only)
exploit:
  enabled: false
  require_authorization: true
  log_all_operations: true
```

### 3. Set Environment Variables

```bash
export FALCONONE_CONFIG=./config/config.yaml
export FALCONONE_LOG_LEVEL=INFO
export FALCONONE_DATA_DIR=/var/lib/falconone
```

---

## Verification

### 1. Run Feature Validation

```bash
python validate_features.py
```

**All 8 Features Should Pass:**

**v1.7.0 Phase 1 Features:**
- ✅ Error Recovery Framework
- ✅ Data Validation Middleware
- ✅ Security Auditor

**v1.6.2 Features:**
- ✅ Rel-20 A-IoT Analyzer
- ✅ Semantic 6G Exploiter
- ✅ Cyber-RF Fusion
- ✅ Regulatory Scanner
- ✅ Precision Geolocation

### 2. Run Integration Tests

```bash
pytest tests/test_integration.py -v
```

### 3. Run System Tests

```bash
pytest tests/test_system.py -v
```

### 4. Check SDR Connectivity (If Using SDR)

**USRP:**
```bash
uhd_find_devices
uhd_usrp_probe
```

**HackRF:**
```bash
hackrf_info
```

**LimeSDR:**
```bash
LimeUtil --find
```

---

## Post-Installation

### 1. Create Data Directories

```bash
sudo mkdir -p /var/lib/falconone/logs
sudo mkdir -p /var/lib/falconone/captures
sudo mkdir -p /var/lib/falconone/checkpoints
sudo chown -R $USER:$USER /var/lib/falconone
```

### 2. Set Up Systemd Service (Optional)

Create `/etc/systemd/system/falconone.service`:

```ini
[Unit]
Description=FalconOne Cellular Intelligence Platform
After=network.target

[Service]
Type=simple
User=falconone
WorkingDirectory=/opt/falconone
Environment="PATH=/opt/falconone/.venv/bin"
ExecStart=/opt/falconone/.venv/bin/python run.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and Start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable falconone
sudo systemctl start falconone
sudo systemctl status falconone
```

### 3. Configure Logging

```bash
# Rotate logs
sudo nano /etc/logrotate.d/falconone
```

Add:
```
/var/lib/falconone/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0640 falconone falconone
    sharedscripts
}
```

### 4. Set Up Dashboard Access

The web dashboard runs on port 5000 by default.

**Access locally:**
```
http://localhost:5000
```

**Access remotely (with firewall configuration):**
```bash
sudo ufw allow 5000/tcp
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'scipy'`

**Solution:**
```bash
source .venv/bin/activate  # Ensure virtual environment is active
pip install scipy scikit-learn
```

#### 2. SDR Device Not Found

**Problem:** SDR device not detected

**Solution:**
```bash
# Check USB permissions
sudo usermod -a -G plugdev $USER
sudo udevadm control --reload-rules
sudo udevadm trigger

# Re-plug SDR device and check
lsusb
```

#### 3. Permission Denied Errors

**Problem:** Cannot access `/dev/bus/usb` or similar

**Solution:**
```bash
# Add user to required groups
sudo usermod -a -G dialout,plugdev $USER
# Log out and log back in for changes to take effect
```

#### 4. Feature Validation Failures

**Problem:** Some features fail validation

**Solution:**
```bash
# Re-run installation
pip install --upgrade -r requirements.txt

# Check Python version
python3 --version  # Must be 3.10+

# Verify all dependencies
pip list | grep -E "scipy|scikit-learn|scapy|numpy"
```

#### 5. Configuration File Errors

**Problem:** YAML parsing errors

**Solution:**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"

# Reset to defaults
cp config/config.yaml.example config/config.yaml
```

#### 6. Out of Memory Errors

**Problem:** Python crashes with memory errors

**Solution:**
```bash
# Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Debug Mode

Enable verbose logging:

```bash
export FALCONONE_LOG_LEVEL=DEBUG
python run.py --debug
```

### Getting Help

**Check logs:**
```bash
tail -f /var/lib/falconone/logs/falconone.log
```

**Run diagnostics:**
```bash
python -m falconone.cli diagnostics
```

**Community Support:**
- GitHub Issues: https://github.com/your-org/falconone/issues
- Documentation: https://docs.falconone.io

---

## Security Considerations

### Production Deployment

1. **Change Default Passwords** - Update all default credentials
2. **Enable HTTPS** - Use TLS certificates for web dashboard
3. **Restrict Network Access** - Use firewall rules to limit access
4. **Regular Updates** - Keep dependencies up to date
5. **Audit Logging** - Enable comprehensive audit logging
6. **Compliance** - Ensure regulatory compliance for your jurisdiction

### Legal Notice

⚠️ **IMPORTANT:** FalconOne is designed for authorized security research and testing only. Unauthorized interception of cellular communications is illegal in most jurisdictions. Users are responsible for ensuring compliance with all applicable laws and regulations.

---

## Next Steps

After successful installation:

1. ✅ **Read the [README.md](README.md)** for usage instructions
2. ✅ **Review [config/config.yaml](config/config.yaml)** and customize settings
3. ✅ **Run validation:** `python validate_features.py`
4. ✅ **Start the system:** `python run.py`
5. ✅ **Access dashboard:** http://localhost:5000
6. ✅ **Explore CLI:** `python -m falconone.cli --help`

---

## Version History

### v1.7.0 Phase 1 (December 30, 2025)
- ✅ Error Recovery Framework with circuit breakers
- ✅ Data Validation Middleware with SNR filtering
- ✅ Security Auditor with FCC/ETSI compliance
- ✅ Production cleanup and optimization
- ✅ All 8 features validated (100% passing)

### v1.6.2
- Rel-20 A-IoT encryption analysis
- Semantic 6G communication exploitation
- Cyber-RF event fusion
- Multi-jurisdiction regulatory scanning
- Precision geolocation with NLOS/V2X

---

**Installation Complete! 🎉**

For additional support, consult the [README.md](README.md) or open an issue on GitHub.
