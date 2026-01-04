# Usage Guide

Common workflows and examples for FalconOne v1.9.8.

## Quick Start

```bash
# Activate environment
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Start with default config
python main.py

# Start with custom config
python main.py --config config/custom.yaml

# Start dashboard only
python start_dashboard.py
# Access at: http://127.0.0.1:5000
```

---

## Dashboard Navigation (v1.9.8)

The dashboard UI features a reorganized navigation structure:

### 5 Collapsible Categories

| Category | Tabs |
|----------|------|
| **📊 MONITORING** | Dashboard, System Health, Carbon Emissions, SDR Devices, SDR Failover |
| **🎯 OPERATIONS** | Captures & IMSI, Cellular Monitor, Voice/VoNR, Target Management, Terminal |
| **⚡ EXPLOITATION** | Exploit Engine, Post-Quantum Crypto, 6G NTN Satellite, ISAC/V2X/Semantic |
| **🤖 ANALYTICS** | AI Classification, Data Validator |
| **⚙️ ADMINISTRATION** | Setup Wizard, Vulnerability Audit, Documentation |

### UI Features
- **Persistent Status Bar**: Real-time KPIs always visible (throughput, latency, success rate)
- **Role Selector**: Switch between Operator/Analyst/Admin views
- **Theme Toggle**: Dark/Light mode in sidebar footer
- **Collapsible Navigation**: Click category headers to expand/collapse

---

## CLI Commands

### System Status
```bash
python -m falconone.cli status
python -m falconone.cli health
```

### Signal Monitoring
```bash
# Start GSM monitoring
python -m falconone.cli monitor --band gsm900

# Start LTE monitoring
python -m falconone.cli monitor --band lte --arfcn 3350

# Start 5G NR monitoring
python -m falconone.cli monitor --band nr --frequency 3.5e9
```

### Data Export
```bash
python -m falconone.cli export --format json --output data.json
python -m falconone.cli export --format csv --output data.csv
```

---

## Python API Examples

### Basic Signal Analysis

```python
from falconone.core.orchestrator import FalconOneOrchestrator
from falconone.utils.config import Config

# Initialize
config = Config('config/config.yaml')
orchestrator = FalconOneOrchestrator(config)

# Start monitoring
orchestrator.start()

# Get current status
status = orchestrator.get_status()
print(f"Active devices: {status['active_devices']}")
```

### AI Signal Classification

```python
from falconone.ai.signal_classifier import SignalClassifier
import numpy as np

# Initialize classifier
classifier = SignalClassifier(config, logger)

# Classify IQ samples
iq_samples = np.random.randn(1024) + 1j * np.random.randn(1024)
result = classifier.classify_signal(iq_samples)

print(f"Signal type: {result['classification']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Voice Interception & Analysis

```python
from falconone.voice.interceptor import VoiceInterceptor

interceptor = VoiceInterceptor(config, logger)

# Decode Opus stream
decoded = interceptor.decode_opus_stream(rtp_packets, ssrc=0x12345678)

# Analyze call (diarization + VAD)
analysis = interceptor.analyze_call(audio_bytes, sample_rate=16000)

print(f"Speakers detected: {len(analysis['speakers'])}")
for speaker in analysis['speakers']:
    print(f"  {speaker['id']}: {speaker['speaking_time']:.1f}s")
```

### Post-Quantum Cryptography

```python
from falconone.crypto.post_quantum import HybridKEMScheme, HybridSignatureScheme

# Hybrid Key Exchange (X25519 + Kyber768)
kem = HybridKEMScheme(classical='x25519', pq='kyber768')
keypair = kem.keygen()

# Sender encapsulates
ciphertext, sender_secret = kem.encapsulate(keypair.combined_public_key)

# Receiver decapsulates
receiver_secret = kem.decapsulate(ciphertext, keypair)

assert sender_secret == receiver_secret  # Shared secret established

# Hybrid Signatures (Ed25519 + Dilithium3)
sig_scheme = HybridSignatureScheme(classical='ed25519', pq='dilithium3')
sig_keypair = sig_scheme.keygen()

message = b"Authenticated message"
signature = sig_scheme.sign(message, sig_keypair)

is_valid = sig_scheme.verify(message, signature, sig_keypair.combined_public_key)
print(f"Signature valid: {is_valid}")
```

### O-RAN RIC Optimization

```python
from falconone.ai.ric_optimizer import RICOptimizer

optimizer = RICOptimizer(config, logger)

# Get current state
state = optimizer.get_current_state()

# Get recommended action
action = optimizer.get_action(state)

# Apply action and get reward
reward = optimizer.apply_action(action)

print(f"Action: {action}, Reward: {reward:.3f}")
```

---

## Configuration Examples

### Basic Configuration (`config/config.yaml`)

```yaml
# SDR Settings
sdr:
  device: hackrf
  sample_rate: 2.4e6
  frequency: 935e6
  gain: 40

# AI Settings
ai:
  signal_classifier:
    model: transformer
    confidence_threshold: 0.85
  
  ric:
    learning_rate: 0.001
    epsilon: 0.1

# Voice Processing
voice:
  sample_rate: 16000
  diarization:
    min_speakers: 2
    max_speakers: 5

# Security
security:
  audit_enabled: true
  pqc_enabled: true
```

### LE Mode Configuration

```yaml
le_mode:
  enabled: true
  warrant_required: true
  evidence_chain: true
  
  intercept:
    target_imsi: "310260*"
    log_all_comms: true
```

---

## Dashboard

### Start Web Dashboard

```bash
python start_dashboard.py
# Access at http://localhost:8050
```

### Dashboard Features

- Real-time signal visualization
- Device status monitoring
- AI classification results
- KPI metrics and alerts

---

## Common Workflows

### 1. IMSI Catching

```bash
# Start IMSI catcher mode
python -m falconone.cli catch --band gsm900 --output imsi_log.csv
```

### 2. VoLTE Interception

```bash
# Start VoLTE monitoring
python -m falconone.cli volte --target-imsi 310260123456789
```

### 3. Anomaly Detection

```python
from falconone.ai.signal_classifier import SignalClassifier

classifier = SignalClassifier(config, logger)

# Check for anomalies
result = classifier.detect_anomaly(iq_samples)

if result['anomaly_detected']:
    print(f"Anomaly score: {result['anomaly_score']:.2f}")
    print(f"Type: {result['anomaly_type']}")
```

### 4. SDR Failover

```python
from falconone.sdr.sdr_failover import SDRFailoverManager

failover = SDRFailoverManager(config, logger)

# Add devices
failover.add_device('hackrf_primary', 'hackrf', priority=1)
failover.add_device('rtlsdr_backup', 'rtlsdr', priority=2)

# Start monitoring
failover.start_monitoring()
```

---

## Output Formats

### JSON Export
```json
{
  "imsi": "310260123456789",
  "timestamp": "2026-01-04T12:00:00Z",
  "signal_strength": -75,
  "cell_id": 12345,
  "classification": "legitimate"
}
```

### CSV Export
```csv
imsi,timestamp,signal_strength,cell_id,classification
310260123456789,2026-01-04T12:00:00Z,-75,12345,legitimate
```

---

## Performance Tips

1. **Use GPU acceleration**: Set `TF_FORCE_GPU_ALLOW_GROWTH=true`
2. **Batch processing**: Process signals in batches of 1000+ samples
3. **Async operations**: Use `asyncio` for concurrent SDR reads
4. **Memory management**: Clear buffers regularly with `orchestrator.clear_buffers()`

---

## Next Steps

- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) – Full API reference
- [CONTRIBUTING.md](CONTRIBUTING.md) – Development guide
- [CHANGELOG.md](CHANGELOG.md) – Version history
