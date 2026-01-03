# FalconOne v1.9.0 - Production Deployment Guide

**Last Updated:** January 3, 2026  
**Version:** 1.9.0 (ISAC/NTN Integration)

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Environment Variables](#environment-variables)
3. [Configuration](#configuration)
4. [Dependencies](#dependencies)
5. [Database Setup](#database-setup)
6. [SDR Hardware Configuration](#sdr-hardware-configuration)
7. [Security Hardening](#security-hardening)
8. [Performance Tuning](#performance-tuning)
9. [Monitoring & Logging](#monitoring--logging)
10. [Troubleshooting](#troubleshooting)

---

## Pre-Deployment Checklist

Before deploying FalconOne v1.9.0 to production, ensure:

- [ ] All environment variables are set (see below)
- [ ] SDR hardware is connected and calibrated
- [ ] Database is initialized and encrypted
- [ ] SSL/TLS certificates are configured
- [ ] Firewall rules are configured
- [ ] LE mode warrants are uploaded (if applicable)
- [ ] O-RAN RIC endpoints are configured
- [ ] All dependencies are installed
- [ ] Security scan passed (run `python -m falconone.tests.security_scan`)
- [ ] Test suite passed (run `pytest falconone/tests/`)

---

## Environment Variables

### **Critical Security Variables** (REQUIRED)

```bash
# Flask Secret Key (REQUIRED - generate with: python -c "import secrets; print(secrets.token_hex(32))")
export FALCONONE_SECRET_KEY="your-64-character-hex-secret-key-here"

# Database Encryption Key (REQUIRED for SQLCipher)
export FALCONONE_DB_KEY="your-database-encryption-key-here"

# Signal Bus Encryption Key (REQUIRED if signal_bus.enable_encryption=true)
export SIGNAL_BUS_KEY="your-signal-bus-encryption-key-here"
```

### **O-RAN Integration** (REQUIRED for ISAC/NTN exploits)

```bash
# O-RAN RIC Endpoints
export ORAN_RIC_ENDPOINT="http://ric.production.example.com:8090/e2"  # ISAC E2 endpoint
export ORAN_RIC_ENDPOINT_NTN="http://ric-ntn.production.example.com:8080"  # NTN RIC endpoint
```

### **External APIs** (Optional but recommended)

```bash
# OpenCellID API for tower discovery (get key from https://opencellid.org/)
export OPENCELLID_API_KEY="your-opencellid-api-key"

# Space-Track.org for satellite TLE data (alternative to Celestrak)
export SPACETRACK_USERNAME="your-spacetrack-username"
export SPACETRACK_PASSWORD="your-spacetrack-password"
```

### **Production Settings**

```bash
# Environment
export FALCONONE_ENV="production"  # Options: development, research, production
export FALCONONE_LOG_LEVEL="WARNING"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Dashboard
export FALCONONE_DASHBOARD_PORT="5000"
export FALCONONE_DASHBOARD_HOST="0.0.0.0"  # WARNING: Only bind to 0.0.0.0 behind firewall/proxy
```

### **Law Enforcement Mode** (If using LE features)

```bash
# LE Mode Configuration
export LE_MODE_ENABLED="true"
export WARRANT_STORAGE_PATH="/secure/warrants/"  # Must be encrypted filesystem
export OCR_API_KEY="your-ocr-api-key"  # For warrant OCR validation
```

---

## Configuration

### **config.yaml Production Settings**

Edit `config/config.yaml` and set:

```yaml
system:
  environment: production  # IMPORTANT: Set to production
  log_level: WARNING  # Reduce verbosity in production
  
# Enable security features
signal_bus:
  enable_encryption: true  # REQUIRED in production
  
# Enable ISAC/NTN if needed
monitoring:
  isac:
    enabled: true  # Set to true if using ISAC features
  ntn_6g:
    enabled: true  # Set to true if using NTN features
    
# SDR configuration
sdr:
  devices:
    - USRP  # Production: Use USRP X410 or N310
  priority: USRP
  
# Database
database:
  encrypt: true  # REQUIRED in production (uses SQLCipher)
```

---

## Dependencies

### **System Dependencies**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3-pip \
    libusb-1.0-0-dev \
    libsoapysdr-dev \
    soapysdr-module-uhd \
    uhd-host \
    build-essential

# Download UHD FPGA images (for USRP)
sudo uhd_images_downloader
```

### **Python Dependencies**

```bash
# Production installation
pip install -r requirements.txt --no-cache-dir

# Verify critical packages
python -c "import numpy, scipy, astropy, qutip; print('Core packages OK')"
python -c "import flask, flask_socketio, flask_limiter; print('Web packages OK')"
python -c "from falconone.monitoring.isac_monitor import ISACMonitor; print('ISAC OK')"
python -c "from falconone.exploit.isac_exploiter import ISACExploiter; print('ISAC Exploit OK')"
```

---

## Database Setup

### **Initialize Database**

```bash
# Create database directory
mkdir -p /var/lib/falconone/data

# Set permissions (production: use dedicated user)
chmod 700 /var/lib/falconone/data

# Initialize database with encryption
export FALCONONE_DB_KEY="your-encryption-key"
python -c "from falconone.utils.database import FalconOneDatabase; db = FalconOneDatabase('/var/lib/falconone/data/falconone.db'); print('Database initialized')"
```

### **Backup Strategy**

```bash
# Daily encrypted backups
0 2 * * * /usr/local/bin/backup_falconone.sh

# backup_falconone.sh:
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
sqlite3 /var/lib/falconone/data/falconone.db ".backup /backup/falconone_$DATE.db"
gpg --encrypt --recipient admin@example.com /backup/falconone_$DATE.db
```

---

## SDR Hardware Configuration

### **USRP Setup** (Recommended for production)

```bash
# Test USRP connectivity
uhd_find_devices

# Expected output:
# [INFO] Device: USRP X410
# [INFO] Serial: 12345678

# Test SDR streaming
uhd_rx_samples --freq 2.14e9 --rate 23.04e6 --duration 5 --file test.dat

# Verify in FalconOne
python -c "from falconone.sdr.sdr_layer import get_sdr_manager; sdr = get_sdr_manager(); print(f'SDR: {sdr.device_type}, Status: {sdr.is_active()}')"
```

### **Calibration**

```python
from falconone.sdr.sdr_layer import get_sdr_manager

sdr = get_sdr_manager()
sdr.calibrate_device()  # Performs DC offset, IQ imbalance correction
```

---

## Security Hardening

### **1. Firewall Configuration**

```bash
# Allow only dashboard port from internal network
sudo ufw allow from 10.0.0.0/8 to any port 5000

# Block all other incoming
sudo ufw default deny incoming
sudo ufw enable
```

### **2. SSL/TLS (Production Dashboard)**

```nginx
# nginx reverse proxy configuration
server {
    listen 443 ssl http2;
    server_name falcon.example.com;
    
    ssl_certificate /etc/ssl/certs/falcon.crt;
    ssl_certificate_key /etc/ssl/private/falcon.key;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### **3. Run as Non-Root**

```bash
# Create dedicated user
sudo useradd -r -s /bin/false falconone

# Set ownership
sudo chown -R falconone:falconone /var/lib/falconone

# Run with systemd
sudo systemctl start falconone
```

### **4. Security Scan**

```bash
# Run before each deployment
python -m falconone.tests.security_scan

# Expected output:
# ✓ Bandit scan complete: 0 issues
# ✓ Safety scan complete: 0 vulnerabilities
# Overall Status: ✅ PASS
```

---

## Performance Tuning

### **1. Orchestrator Scaling**

```yaml
# config.yaml
orchestrator:
  dynamic_scaling: true
  scaling_thresholds:
    cpu_high: 0.85
    memory_high: 0.80
```

### **2. Database Optimization**

```sql
-- Create indexes for frequent queries
CREATE INDEX idx_targets_imsi ON targets(imsi);
CREATE INDEX idx_captures_timestamp ON captures(timestamp);

-- Vacuum regularly
VACUUM;
ANALYZE;
```

### **3. Rate Limiting**

```python
# dashboard.py already configured
# ISAC endpoints: 5-20 rpm
# NTN endpoints: 5-20 rpm
```

---

## Monitoring & Logging

### **1. Structured Logging**

```yaml
# config.yaml
system:
  log_level: WARNING
  log_dir: /var/log/falconone
```

### **2. Log Rotation**

```bash
# /etc/logrotate.d/falconone
/var/log/falconone/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
}
```

### **3. Prometheus Metrics** (Optional)

```python
# Already configured in monitoring/prometheus.yml
# Endpoints: http://localhost:9090
```

---

## Troubleshooting

### **Common Issues**

#### **1. "FALCONONE_SECRET_KEY not set"**
```bash
# Generate and export
python -c "import secrets; print(secrets.token_hex(32))" > .secret_key
export FALCONONE_SECRET_KEY=$(cat .secret_key)
```

#### **2. "No SDR available for frequency scanning"**
```bash
# Check SDR connection
uhd_find_devices
lsusb | grep "Ettus"

# Restart UHD service
sudo systemctl restart uhd-usrp
```

#### **3. "Orchestrator required for IMSI capture - chain aborted"**
```python
# Ensure orchestrator is initialized
from falconone.core.orchestrator import FalconOneOrchestrator
orchestrator = FalconOneOrchestrator(config_path='config/config.yaml')
orchestrator.initialize()
```

#### **4. "E2 injection failed: Connection refused"**
```bash
# Verify O-RAN RIC endpoint
curl -v http://ric.example.com:8090/e2/health

# Check environment variable
echo $ORAN_RIC_ENDPOINT
```

#### **5. "OpenCellID API key required"**
```bash
# Get free API key from https://opencellid.org/
export OPENCELLID_API_KEY="your-key-here"
```

### **Logs to Check**

```bash
# Application logs
tail -f /var/log/falconone/falconone.log

# ISAC monitoring
tail -f /var/log/falconone/isac_monitor.log

# Exploit operations
tail -f /var/log/falconone/exploit_engine.log

# LE mode audit trail
tail -f logs/audit/le_mode_audit.log
```

---

## Production Checklist Summary

### **Pre-Launch**
- [ ] All environment variables set and validated
- [ ] Config file reviewed and set to `environment: production`
- [ ] Database encrypted and backed up
- [ ] SDR calibrated and tested
- [ ] Security scan passed
- [ ] Test suite passed (87%+ coverage)
- [ ] Rate limiting configured
- [ ] Firewall rules applied
- [ ] SSL/TLS configured
- [ ] Logs rotated and monitored

### **Post-Launch**
- [ ] Monitor logs for errors
- [ ] Verify SDR connectivity
- [ ] Check O-RAN API connectivity
- [ ] Test ISAC/NTN endpoints
- [ ] Verify LE mode warrant validation
- [ ] Monitor system resources (CPU/memory)
- [ ] Backup verification

---

## Support

For production deployment support:
- **Documentation:** See README.md, USER_MANUAL.md, API_DOCUMENTATION.md
- **Issues:** GitHub Issues (for non-sensitive bugs)
- **Security:** Report via encrypted channel (see SECURITY.md)

---

**Version:** 1.9.0  
**Status:** Production-Ready  
**Last Validated:** January 3, 2026
