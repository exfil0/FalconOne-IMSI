# Installation Guide

Quick setup guide for FalconOne Intelligence Platform v1.9.8.

## Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- **Git** for version control
- **SDR Hardware** (optional): HackRF, USRP, RTL-SDR, BladeRF, or LimeSDR

## Quick Install

### 1. Clone Repository

```bash
git clone https://github.com/exfil0/FalconOne-IMSI.git
cd FalconOne-IMSI
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure

```bash
cp config/config.yaml.example config/config.yaml
# Edit config/config.yaml with your settings
```

### 5. Run

```bash
python main.py
# Or use the CLI
python -m falconone.cli
```

---

## Dependency Groups

### Core (Required)
```bash
pip install numpy scipy pyyaml aiohttp cryptography
```

### AI/ML
```bash
pip install tensorflow scikit-learn torch
```

### Voice Processing
```bash
pip install opuslib pyannote.audio resemblyzer webrtcvad
```

### Post-Quantum Crypto
```bash
pip install liboqs-python pqcrypto
```

### SDR Support
```bash
pip install pyrtlsdr SoapySDR
```

### Visualization
```bash
pip install matplotlib plotly dash
```

---

## Platform-Specific Notes

### Windows

1. Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for compiled dependencies
2. For SDR: Install [Zadig](https://zadig.akeo.ie/) for USB driver replacement

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3-dev libffi-dev libsndfile1 portaudio19-dev
# For SDR support
sudo apt install librtlsdr-dev libhackrf-dev
```

### macOS

```bash
brew install portaudio libsndfile
# For SDR support
brew install hackrf librtlsdr
```

---

## Docker Installation

```bash
docker build -t falconone:latest .
docker run -it --rm falconone:latest
```

Or use Docker Compose:

```bash
docker-compose up -d
```

---

## Kubernetes Deployment

```bash
kubectl apply -f k8s-deployment.yaml
```

See [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md) for cloud-specific instructions.

---

## Verify Installation

```bash
# Run validation script
python validate_system.py

# Run tests
pytest falconone/tests/ -v --tb=short
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| SDR not detected | Check USB drivers, run with `sudo` on Linux |
| CUDA errors | Install matching CUDA version for TensorFlow |
| Import errors | Ensure virtual environment is activated |

For detailed troubleshooting, see [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md).

---

## Next Steps

- [USAGE.md](USAGE.md) – Common workflows and examples
- [CONTRIBUTING.md](CONTRIBUTING.md) – Development setup
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) – API reference
