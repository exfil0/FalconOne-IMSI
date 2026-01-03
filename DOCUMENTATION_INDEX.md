# FalconOne Documentation Index

**Version:** 1.9.3  
**Last Updated:** January 4, 2026  
**Status:** Production Ready ✅

---

## 📚 Core Documentation (8 Essential Files)

| Document | Purpose | Audience |
|----------|---------|----------|
| [README.md](README.md) | Project overview, features, architecture | Everyone |
| [QUICKSTART.md](QUICKSTART.md) | 5-minute setup guide | New Users |
| [INSTALLATION.md](INSTALLATION.md) | Detailed installation | Administrators |
| [USER_MANUAL.md](USER_MANUAL.md) | Complete user guide | Operators |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Architecture, APIs, contribution | Developers |
| [API_DOCUMENTATION.md](API_DOCUMENTATION.md) | REST API reference (v3.4.0) | Developers |
| [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) | Production setup | DevOps |
| [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md) | Docker, K8s deployment | DevOps |

---

## 🎯 Quick Start Paths

### New Users
1. **[QUICKSTART.md](QUICKSTART.md)** - Get running in 5 minutes
2. **[USER_MANUAL.md](USER_MANUAL.md)** - Learn all features
3. **[DASHBOARD_MANAGEMENT_GUIDE.md](DASHBOARD_MANAGEMENT_GUIDE.md)** - Master the UI

### Operators & Researchers  
1. **[EXPLOIT_QUICK_REFERENCE.md](EXPLOIT_QUICK_REFERENCE.md)** - Exploit operations cheat sheet
2. **[LE_MODE_QUICKSTART.md](LE_MODE_QUICKSTART.md)** - Law Enforcement mode guide
3. **[docs/EXPLOIT_WORKFLOW_GUIDE.md](docs/EXPLOIT_WORKFLOW_GUIDE.md)** - Detailed exploit workflows

### Developers
1. **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - Architecture and APIs
2. **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - REST endpoint reference
3. **[PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md)** - Performance tuning

### System Administrators
1. **[INSTALLATION.md](INSTALLATION.md)** - Full installation guide
2. **[PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md)** - Production setup
3. **[CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md)** - Container deployment
4. **[SYSTEM_DEPENDENCIES.md](SYSTEM_DEPENDENCIES.md)** - External tools setup

---

## 📁 Documentation Structure

```
FalconOne/
├── README.md                      # Main project overview
├── QUICKSTART.md                  # Quick setup guide
├── INSTALLATION.md                # Detailed installation
├── USER_MANUAL.md                 # User guide
├── DEVELOPER_GUIDE.md             # Developer reference
├── API_DOCUMENTATION.md           # REST API docs
├── PRODUCTION_DEPLOYMENT.md       # Production setup
├── CLOUD_DEPLOYMENT.md            # Docker/K8s deployment
├── CHANGELOG.md                   # Version history
│
├── Operational Guides/
│   ├── DASHBOARD_MANAGEMENT_GUIDE.md   # Dashboard UI guide
│   ├── EXPLOIT_QUICK_REFERENCE.md      # Exploit cheat sheet
│   ├── LE_MODE_QUICKSTART.md           # LE mode guide
│   ├── PERFORMANCE_OPTIMIZATION.md     # Performance tuning
│   ├── SYSTEM_DEPENDENCIES.md          # External tools
│   └── SYSTEM_TOOLS_MANAGEMENT.md      # Tools management
│
├── Feature Documentation/
│   ├── 6G_NTN_INTEGRATION_COMPLETE.md  # 6G satellite features
│   └── SYSTEM_DOCUMENTATION.md         # Full system reference
│
├── docs/
│   └── EXPLOIT_WORKFLOW_GUIDE.md       # Detailed exploit workflows
│
└── archive/deprecated_docs/            # Archived documentation
```

---

## 🔧 Testing & Validation

### Quick Validation
```bash
# Run quick validation (recommended)
python quick_validate.py

# Run full test suite
pytest falconone/tests/ -v
```

### Test Structure
```
falconone/tests/
├── conftest.py              # Pytest fixtures
├── test_authentication.py   # Auth tests
├── test_database.py         # Database tests
├── test_e2e.py              # End-to-end tests
├── test_exploitation.py     # Exploit tests
├── test_integration.py      # Integration tests
├── test_isac.py             # ISAC tests
├── test_le_mode.py          # LE mode tests
├── test_ntn_6g.py           # 6G NTN tests
├── test_sdr_failover.py     # SDR tests
├── integration/             # Integration test suites
├── locustfile.py            # Load testing
└── security_scan.py         # Security tests
```

---

## 📋 Document Descriptions

### Core Documents

| Document | Description |
|----------|-------------|
| **README.md** | Complete project overview with features from v1.2 to v1.9.0, architecture diagrams, and status tracking |
| **QUICKSTART.md** | Minimal setup: clone → install → configure → run dashboard in under 5 minutes |
| **INSTALLATION.md** | Full installation covering all dependencies, SDR hardware, optional packages |
| **USER_MANUAL.md** | Comprehensive guide to all features: monitoring, exploitation, analysis |
| **DEVELOPER_GUIDE.md** | Code architecture, module structure, API design, contribution guidelines |
| **API_DOCUMENTATION.md** | Full REST API reference with 50+ endpoints, examples, schemas |
| **PRODUCTION_DEPLOYMENT.md** | Security hardening, scaling, monitoring for production |
| **CLOUD_DEPLOYMENT.md** | Docker, docker-compose, Kubernetes manifests and configs |

### Operational Guides

| Document | Description |
|----------|-------------|
| **DASHBOARD_MANAGEMENT_GUIDE.md** | Complete UI guide with tab-by-tab walkthrough |
| **EXPLOIT_QUICK_REFERENCE.md** | Quick reference cards for all exploit operations |
| **LE_MODE_QUICKSTART.md** | Law Enforcement mode: warrants, evidence chain, compliance |
| **PERFORMANCE_OPTIMIZATION.md** | Tuning guides, caching, profiling, bottleneck fixes |
| **SYSTEM_DEPENDENCIES.md** | External tools: SoapySDR, gr-gsm, kalibrate, etc. |
| **SYSTEM_TOOLS_MANAGEMENT.md** | Tool installation, configuration, troubleshooting |

### Feature Documentation

| Document | Description |
|----------|-------------|
| **6G_NTN_INTEGRATION_COMPLETE.md** | 6G NTN satellite integration: LEO/MEO/GEO, Doppler, beam tracking |
| **SYSTEM_DOCUMENTATION.md** | Comprehensive system reference (all modules) |
| **CHANGELOG.md** | Version history with all changes from v1.0 to v1.9.0 |

---

## 🗄️ Archived Documentation

Historical documents have been moved to `archive/deprecated_docs/`:

- LE_MODE_COMPLETION_REPORT.md - LE mode development completion
- LE_MODE_IMPLEMENTATION_SUMMARY.md - Implementation details
- LE_MODE_VERIFICATION.md - Verification results
- RANSACKED_*.md - RANSacked audit reports
- DOCUMENTATION_*.md - Documentation cleanup logs
- Various audit and status reports

These are preserved for reference but are no longer actively maintained.

---

## 📊 Documentation Statistics

| Metric | Value |
|--------|-------|
| Core Documents | 8 |
| Operational Guides | 6 |
| Feature Docs | 3 |
| Total Active Docs | 17 |
| Archived Docs | 17 |
| Test Files | 12 |

---

## 🔄 Last Updated

- **Documentation consolidation:** January 3, 2026
- **Archived redundant docs:** 17 files
- **Consolidated test files:** 3 files
- **Active documentation:** 17 files
