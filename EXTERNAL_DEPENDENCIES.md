# FalconOne External Dependencies Guide

## Version 1.9.2

This document specifies all external dependencies, installation requirements, and graceful degradation behavior for FalconOne.

---

## Table of Contents

1. [Core Dependencies](#core-dependencies)
2. [Optional Dependencies](#optional-dependencies)
3. [System Tools](#system-tools)
4. [Graceful Degradation](#graceful-degradation)
5. [Installation Guides](#installation-guides)
6. [Troubleshooting](#troubleshooting)

---

## Core Dependencies

These packages are **required** for FalconOne to function:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.24.0 | Array operations, IQ signal processing |
| PyYAML | >=6.0.1 | Configuration file parsing |
| requests | >=2.32.0 | HTTP communications |
| cryptography | >=42.0.0 | Secure crypto operations |
| asyncio | (stdlib) | Async I/O operations |

### Installation

```bash
pip install numpy>=1.24.0 PyYAML>=6.0.1 requests>=2.32.0 cryptography>=42.0.0
```

---

## Optional Dependencies

FalconOne supports graceful degradation when these packages are unavailable.

### TensorFlow (AI/ML Features)

**Package:** `tensorflow>=2.14.0`

**Features Enabled:**
- CNN-based signal classification
- LSTM temporal pattern detection
- Neural network anomaly detection
- Deep learning model training

**When Unavailable:**
- Falls back to heuristic frequency-based classification
- Uses rule-based pattern matching
- Reduced accuracy (~70% vs ~95%)

**Installation:**
```bash
# CPU-only (smaller, works everywhere)
pip install tensorflow>=2.14.0

# GPU support (NVIDIA CUDA required)
pip install tensorflow[and-cuda]>=2.14.0
```

**Verification:**
```python
from falconone.ai.signal_classifier import TF_AVAILABLE
print(f"TensorFlow available: {TF_AVAILABLE}")
```

---

### Ray / RLlib (Multi-Agent Reinforcement Learning)

**Packages:** 
- `ray>=2.7.0`
- `ray[rllib]>=2.7.0`

**Features Enabled:**
- Multi-agent RIC optimization
- Distributed RL training
- Advanced policy learning
- O-RAN xApp optimization

**When Unavailable:**
- Falls back to DQN with experience replay
- Uses epsilon-greedy action selection
- Single-agent optimization only

**Installation:**
```bash
pip install ray[rllib]>=2.7.0
```

**Verification:**
```python
from falconone.ai.ric_optimizer import RAY_AVAILABLE
print(f"Ray/RLlib available: {RAY_AVAILABLE}")
```

---

### OpenAI Gym (RL Environments)

**Package:** `gym>=0.26.0` or `gymnasium>=0.28.0`

**Features Enabled:**
- Custom RL environments
- Standard API for RL training
- Environment wrappers

**When Unavailable:**
- Uses simplified internal environment
- Reduced state space

**Installation:**
```bash
pip install gymnasium>=0.28.0
```

---

### Pyshark (Packet Parsing)

**Package:** `pyshark>=0.5.0`

**Features Enabled:**
- PCAP file parsing
- Live packet capture analysis
- Protocol dissection

**When Unavailable:**
- Uses subprocess-based tshark parsing
- Reduced parsing capabilities

**Requirements:**
- Wireshark/tshark must be installed system-wide

**Installation:**
```bash
pip install pyshark>=0.5.0
```

---

### Kraken (Crypto Analysis)

**Package:** External system tool

**Features Enabled:**
- A5/1 cryptographic analysis
- GSM encryption key recovery
- Known-plaintext attacks

**When Unavailable:**
- Crypto analysis disabled
- Returns warning messages

**Installation:**
```bash
# Ubuntu/Debian
git clone https://github.com/joswr1ght/Kraken
cd Kraken && make install
```

---

## System Tools

These external tools enhance FalconOne capabilities:

### SDR Tools

| Tool | Purpose | Required |
|------|---------|----------|
| **gr-gsm** | GSM signal processing | For GSM monitoring |
| **kalibrate-rtl** | Frequency calibration | Optional |
| **srsRAN** | 4G/5G signal processing | For LTE/5G |
| **OpenBTS** | GSM base station | For active testing |

### Network Tools

| Tool | Purpose | Required |
|------|---------|----------|
| **tshark** | Packet analysis | For pyshark |
| **nmap** | Network scanning | Optional |
| **tcpdump** | Raw packet capture | Optional |

---

## Graceful Degradation

FalconOne implements graceful degradation patterns throughout:

### Pattern 1: Import-Time Flags

```python
# At module level
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# In methods
if TF_AVAILABLE:
    # Use neural network
    return self._classify_with_model(signal)
else:
    # Use heuristic fallback
    return self._classify_heuristic(signal)
```

### Pattern 2: Lazy Loading

```python
def _ensure_models_loaded(self):
    """Load models only when first needed"""
    if self._models_loaded:
        return
    
    if TF_AVAILABLE:
        self._load_tf_models()
    else:
        self.logger.warning("TensorFlow unavailable, using heuristics")
    
    self._models_loaded = True
```

### Pattern 3: Feature Flags

```python
class FeatureFlags:
    AI_CLASSIFICATION = TF_AVAILABLE
    MULTI_AGENT_RL = RAY_AVAILABLE
    CRYPTO_ANALYSIS = KRAKEN_AVAILABLE
    ADVANCED_PARSING = PYSHARK_AVAILABLE
```

---

## Installation Guides

### Minimal Installation (Core Only)

```bash
pip install -r requirements.txt
```

**Capabilities:**
- Basic signal monitoring
- Heuristic classification
- Simple RL optimization
- CLI interface

### Full Installation (All Features)

```bash
# Core
pip install -r requirements.txt

# AI/ML
pip install tensorflow>=2.14.0
pip install ray[rllib]>=2.7.0
pip install gymnasium>=0.28.0

# Analysis
pip install pyshark>=0.5.0

# Install system tools
sudo apt install -y wireshark-common gr-gsm kalibrate-rtl
```

### Docker Installation

```bash
docker build -t falconone:1.9.2 .
docker run -it --privileged falconone:1.9.2
```

The Docker image includes all dependencies.

---

## Troubleshooting

### TensorFlow Issues

**Problem:** `ModuleNotFoundError: No module named 'tensorflow'`

**Solution:**
```bash
pip install tensorflow>=2.14.0
# Or for M1/M2 Macs:
pip install tensorflow-macos
```

**Problem:** CUDA/cuDNN version mismatch

**Solution:**
```bash
# Use CPU-only version
pip uninstall tensorflow
pip install tensorflow-cpu>=2.14.0
```

---

### Ray/RLlib Issues

**Problem:** Ray cluster initialization fails

**Solution:**
```bash
# Start local cluster
ray start --head --port=6379

# Set environment
export RAY_ADDRESS="auto"
```

---

### Pyshark Issues

**Problem:** `TSharkNotFoundException`

**Solution:**
```bash
# Install Wireshark
sudo apt install wireshark-common tshark

# Or on macOS
brew install wireshark
```

---

### SDR Issues

**Problem:** No SDR device detected

**Solution:**
```bash
# Check device
lsusb | grep RTL

# Install udev rules
sudo cp rtl-sdr.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

---

## Dependency Status Check

Run this to check all dependencies:

```python
from falconone.utils.dependency_check import check_all_dependencies

status = check_all_dependencies()
for dep, available in status.items():
    print(f"{dep}: {'✓' if available else '✗'}")
```

Expected output:
```
numpy: ✓
tensorflow: ✓ (or ✗ with fallback)
ray: ✓ (or ✗ with fallback)
pyshark: ✓ (or ✗ with fallback)
kraken: ✓ (or ✗ - disabled)
```

---

## Version Compatibility Matrix

| FalconOne | Python | TensorFlow | Ray | NumPy |
|-----------|--------|------------|-----|-------|
| 1.9.2 | 3.10+ | 2.14+ | 2.7+ | 1.24+ |
| 1.9.1 | 3.10+ | 2.13+ | 2.6+ | 1.23+ |
| 1.9.0 | 3.9+ | 2.12+ | 2.5+ | 1.22+ |

---

## Security Considerations

All dependencies are regularly audited for vulnerabilities. Version 1.9.2 includes security updates for:

- **Jinja2 >=3.1.6** - CVE-2024-22195 (XSS)
- **Werkzeug >=3.0.6** - CVE-2024-34069 (Path traversal)
- **cryptography >=42.0.0** - Multiple CVEs
- **Pillow >=10.3.0** - Buffer overflow fixes
- **requests >=2.32.0** - CVE-2024-35195 (SSRF)
- **pycryptodome >=3.19.1** - Timing attack fixes

Run security audit:
```bash
pip-audit --strict
```

---

*Document Version: 1.9.2 | Last Updated: 2025-01-14*
