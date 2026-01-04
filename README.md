# FalconOne IMSI/TMSI and SMS Catcher

**Version 1.9.6** | Multi-Generation Cellular Monitoring Platform  
**Status:** Production-Ready | **Classification:** TOP CONFIDENTIAL

---

## Overview

FalconOne is a comprehensive cellular signal intelligence platform supporting GSM through 6G NTN (Non-Terrestrial Networks). It provides real-time signal monitoring, AI-powered classification, voice interception, and post-quantum secure communications.

### Key Capabilities

| Category | Features |
|----------|----------|
| **Signal Monitoring** | GSM, LTE, 5G NR, 6G NTN (LEO/MEO/GEO), ISAC sensing |
| **AI/ML** | Transformer classifiers, DQN RIC optimization, online learning |
| **Voice Processing** | VoLTE/VoNR/VoWiFi, Opus/AMR/EVS codecs, speaker diarization |
| **Security** | Post-quantum crypto (Kyber/Dilithium hybrids), audit logging |
| **Exploitation** | 96+ CVE database, RANSacked framework, LE mode with warrants |

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/exfil0/FalconOne-IMSI.git
cd FalconOne-IMSI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure and run
cp config/config.yaml.example config/config.yaml
python main.py
```

📖 **Detailed setup:** [INSTALL.md](INSTALL.md)

---

## Documentation

| Guide | Description |
|-------|-------------|
| [INSTALL.md](INSTALL.md) | Installation & dependencies |
| [USAGE.md](USAGE.md) | Common workflows & examples |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup & guidelines |
| [API_DOCUMENTATION.md](API_DOCUMENTATION.md) | REST API reference |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

### Additional Documentation

- [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md) – Kubernetes & cloud deployment
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) – Architecture deep-dive
- [LE_MODE_QUICKSTART.md](LE_MODE_QUICKSTART.md) – Law enforcement mode
- [QUICKSTART.md](QUICKSTART.md) – Extended quick start guide

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FalconOne Platform                       │
├─────────────┬─────────────┬─────────────┬─────────────────┤
│   SDR       │     AI      │   Voice     │    Security     │
│  Capture    │  Analysis   │  Processing │    & Crypto     │
├─────────────┼─────────────┼─────────────┼─────────────────┤
│ HackRF      │ Signal      │ VoLTE/VoNR  │ Post-Quantum    │
│ USRP        │ Classifier  │ Opus/AMR    │ Kyber+X25519    │
│ RTL-SDR     │ Anomaly     │ Diarization │ Dilithium+Ed25519│
│ BladeRF     │ Detection   │ VAD         │ OQS Integration │
│ LimeSDR     │ RIC DQN     │ Transcribe  │ Audit Logging   │
└─────────────┴─────────────┴─────────────┴─────────────────┘
```

---

## Version History

### v1.9.6 (January 2026) – Bug Fixes
- Fixed `DataValidator.rejected_count` AttributeError
- Fixed `SignalClassifier.get_anomaly_report` incomplete implementation
- Fixed `RICOptimizer` gym import ordering bug
- Added comprehensive test suites (PQC, voice, MARL)

### v1.9.5 (January 2026) – Voice & PQC
- Opus codec support with speaker diarization
- Post-quantum hybrid schemes (X25519+Kyber, Ed25519+Dilithium)
- OQS library integration
- Quantum attack simulation

### v1.9.4 (January 2026) – Gap Analysis
- SDR device failover management
- Voice codec support (AMR, EVS, OPUS, SILK)
- Concept drift detection (ADWIN, DDM, KSWIN)
- Security scanning CI/CD pipeline

📖 **Full history:** [CHANGELOG.md](CHANGELOG.md)

---

## Testing

```bash
# Run all tests
pytest falconone/tests/ -v

# Run specific test suites
pytest falconone/tests/test_post_quantum.py -v      # PQC tests
pytest falconone/tests/test_voice_interceptor.py -v # Voice tests
pytest falconone/tests/test_marl.py -v              # Multi-agent RL tests

# With coverage
pytest falconone/tests/ --cov=falconone --cov-report=html
```

---

## Project Structure

```
FalconOne-IMSI/
├── falconone/              # Main package
│   ├── ai/                 # AI/ML modules
│   ├── core/               # Orchestration
│   ├── crypto/             # Cryptography (PQC)
│   ├── sdr/                # SDR handling
│   ├── voice/              # Voice processing
│   └── tests/              # Unit tests
├── config/                 # Configuration
├── docs/                   # Documentation
├── INSTALL.md              # Installation guide
├── USAGE.md                # Usage guide
├── CONTRIBUTING.md         # Development guide
└── requirements.txt        # Dependencies
```

---

## Requirements

- **Python:** 3.10+ (3.11/3.12 recommended)
- **OS:** Ubuntu 24.04, Windows 11, macOS
- **SDR:** HackRF, USRP, RTL-SDR, BladeRF, LimeSDR (optional)
- **GPU:** CUDA-compatible (optional, for AI acceleration)

---

## License

Research and Development Use Only. See LICENSE for details.

---

## Support

- **Issues:** [GitHub Issues](https://github.com/exfil0/FalconOne-IMSI/issues)
- **Security:** security@falconone-project.example

---

*For the complete feature list and detailed documentation, see [CHANGELOG.md](CHANGELOG.md) and [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md).*
