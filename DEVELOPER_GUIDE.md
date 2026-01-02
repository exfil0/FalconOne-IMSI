# FalconOne Developer Guide

**Version:** 3.1.0  
**Last Updated:** January 2, 2026  
**Audience:** Software Developers, Contributors

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Development Setup](#development-setup)
3. [Codebase Structure](#codebase-structure)
4. [Core Components](#core-components)
5. [Testing](#testing)
6. [RANSacked Integration Testing](#ransacked-integration-testing) **(NEW v1.8.0)**
7. [Contributing](#contributing)
8. [Code Standards](#code-standards)
9. [API Development](#api-development)
10. [Database Schema](#database-schema)
11. [Deployment](#deployment)

---

## Architecture Overview

### System Architecture

FalconOne follows a **modular, event-driven architecture**:

```
┌─────────────────────────────────────────────────┐
│                  Web Dashboard                   │
│              (Flask + JavaScript)                │
└──────────────────┬──────────────────────────────┘
                   │ HTTP/WebSocket
┌──────────────────▼──────────────────────────────┐
│              REST API Layer                      │
│        (Flask-RESTX, JWT Authentication)         │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│             Core Orchestrator                    │
│      (Signal Bus, Task Queue, Events)            │
└─┬─────────┬─────────┬────────────┬──────────┬──┘
  │         │         │            │          │
  │   ┌─────▼────┐ ┌─▼──────┐  ┌──▼─────┐ ┌─▼───────┐
  │   │   SDR    │ │   AI   │  │ Crypto │ │  O-RAN  │
  │   │  Layer   │ │Engine  │  │ Engine │ │   RIC   │
  │   └──────────┘ └────────┘  └────────┘ └─────────┘
  │
┌─▼──────────────────────────────────────────────┐
│           Data Persistence Layer                │
│   PostgreSQL | Redis | Blockchain | S3          │
└─────────────────────────────────────────────────┘
```

### Design Patterns

**1. Event-Driven Architecture**
- Components communicate via `SignalBus`
- Loose coupling, high scalability
- Example: `signal_bus.emit('target_detected', target_data)`

**2. Strategy Pattern**
- SDR drivers (USRP, HackRF, BladeRF)
- Exploit engines (DOS, MITM, Downgrade)
- AI models (SHAP, LIME, Online Learning)

**3. Observer Pattern**
- Real-time monitoring subscriptions
- WebSocket notifications
- xApp event callbacks

**4. Factory Pattern**
- Signal classifier factory
- Exploit engine factory
- xApp factory

**5. Repository Pattern**
- Database abstractions
- `TargetRepository`, `ScanRepository`
- Testable, swappable backends

### Technology Stack

**Backend:**
- **Python 3.11+**: Core language
- **Flask 3.0**: Web framework
- **Flask-RESTX**: REST API with Swagger
- **SQLAlchemy**: ORM
- **Celery**: Asynchronous task queue
- **Redis**: Cache, message broker
- **PostgreSQL**: Primary database

**Frontend:**
- **JavaScript ES6+**: Client-side logic
- **Chart.js**: Data visualization
- **Leaflet**: Map rendering
- **Socket.IO**: WebSocket communication

**AI/ML:**
- **scikit-learn**: Machine learning
- **SHAP**: Model explainability
- **LIME**: Local interpretable explanations
- **River**: Online learning

**SDR:**
- **UHD (USRP)**: Ettus USRP devices
- **HackRF**: HackRF One support
- **BladeRF**: Nuand BladeRF support

**Security:**
- **pqcrypto**: Post-quantum cryptography
- **pycryptodome**: AES-GCM encryption
- **PyJWT**: JWT authentication

---

## Development Setup

### Prerequisites

```bash
# Python 3.11+
python --version

# PostgreSQL 15+
psql --version

# Redis 7+
redis-cli --version

# Git
git --version
```

### Local Development Environment

**1. Clone Repository**
```bash
git clone https://github.com/yourusername/falconone.git
cd falconone
```

**2. Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools
```

**4. Configure Environment**
```bash
cp .env.example .env
nano .env  # Edit configuration
```

**Required Environment Variables:**
```env
# Database
DATABASE_URL=postgresql://falconone:password@localhost:5432/falconone_dev
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here-change-in-production
JWT_SECRET_KEY=your-jwt-secret-key-here

# SDR
SDR_TYPE=usrp  # or hackrf, bladerf
SDR_SAMPLE_RATE=10000000
SDR_GAIN=40

# AI/ML
MODEL_PATH=models/
ENABLE_EXPLAINABILITY=true

# O-RAN
RIC_ENABLED=true
RIC_PORT=36421

# Logging
LOG_LEVEL=DEBUG
LOG_FILE=logs/falconone.log
```

**5. Initialize Database**
```bash
# Create database
createdb falconone_dev

# Run migrations
python -c "from falconone.utils.database import FalconOneDatabase; db = FalconOneDatabase(); db.create_tables()"
```

**6. Start Development Server**
```bash
# Terminal 1: Main application
python main.py

# Terminal 2: Celery worker
celery -A falconone.tasks worker --loglevel=info

# Terminal 3: Redis
redis-server
```

**7. Run Tests**
```bash
pytest
```

### IDE Setup

**VS Code Recommended Extensions:**
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.black-formatter",
    "ms-toolsai.jupyter",
    "redhat.vscode-yaml",
    "ms-azuretools.vscode-docker"
  ]
}
```

**PyCharm Configuration:**
1. Open project in PyCharm
2. Configure Python interpreter → Virtual environment
3. Enable Django support (Settings → Languages & Frameworks → Django)
4. Set pytest as default test runner

---

## Codebase Structure

### Complete Project Structure

```
FalconOne App/
├── main.py                          # Main application entry point  
├── run.py                           # Alternative run script
├── start_dashboard.py               # Dashboard launcher
├── setup.py                         # Package setup
├── install_dependencies.py          # Dependency installer
├── simple_test.py                   # Simple validation (5 tests)
├── quick_validate.py                # Comprehensive validation (6 tests)
├── requirements.txt                 # Python dependencies
├── pytest.ini                       # Pytest configuration
├── docker-compose.yml               # Docker orchestration
├── Dockerfile                       # Container image
├── deploy.sh                        # Deployment script
├── k8s-production.yaml              # Production K8s (550+ lines)
│
├── config/                          # Configuration
│   ├── config.yaml                  # Main config
│   └── falconone.yaml               # System config
│
├── logging/                         # ELK Stack
│   ├── elasticsearch-template.json
│   ├── kibana.yml
│   └── logstash.conf
│
├── monitoring/                      # Prometheus/Grafana
│   ├── prometheus.yml               # 6 scrape targets
│   ├── alerts.yml                   # 23 alert rules
│   ├── grafana-datasource.yml
│   └── grafana-dashboard-system.json
│
├── terraform/aws/                   # Infrastructure as Code
│
└── falconone/                       # Core application package
    ├── __init__.py
    │
    ├── ai/                          # AI/ML (11 modules)
    │   ├── device_profiler.py       # Device profiling & telemetry
    │   ├── explainable_ai.py        # SHAP/LIME interpretability
    │   ├── federated_coordinator.py # Federated learning coordinator
    │   ├── graph_topology.py        # GNN topology inference
    │   ├── kpi_monitor.py           # KPI tracking & anomaly detection
    │   ├── model_zoo.py             # ML model registry (5 models)
    │   ├── online_learning.py       # Incremental learning
    │   ├── payload_generator.py     # GAN-based payload generation
    │   ├── ric_optimizer.py         # DQN/MARL RIC optimizer
    │   ├── signal_classifier.py     # CNN-LSTM signal classification
    │   └── suci_deconcealment.py    # Transformer SUCI analysis
    │
    ├── analysis/                    # Analysis (1 module)
    │   └── cyber_rf_fuser.py        # Cyber-RF intelligence fusion
    │
    ├── cli/                         # CLI (1 module)
    │   └── main.py                  # Command-line interface
    │
    ├── cloud/                       # Cloud (1 module)
    │   └── storage.py               # AWS S3/Azure/GCP integration
    │
    ├── core/                        # Core (6 modules)
    │   ├── config.py                # Configuration manager
    │   ├── detector_scanner.py      # Rogue BS detection
    │   ├── main.py                  # Core orchestrator
    │   ├── multi_tenant.py          # Multi-tenancy support
    │   ├── orchestrator.py          # Task orchestration
    │   └── signal_bus.py            # Event bus
    │
    ├── crypto/                      # Cryptography (3 modules)
    │   ├── analyzer.py              # CPA/DPA side-channel analysis
    │   ├── quantum_resistant.py     # PQC (Kyber/Dilithium)
    │   └── zkp.py                   # Zero-knowledge proofs
    │
    ├── exploit/                     # Exploitation (6 modules)
    │   ├── crypto_attacks.py        # Cryptographic attacks
    │   ├── exploit_engine.py        # DOS/Downgrade/MITM engine
    │   ├── message_injector.py      # SMS/Paging injection
    │   ├── ntn_attacks.py           # Satellite (NTN) attacks
    │   ├── semantic_exploiter.py    # 6G semantic exploits
    │   └── v2x_attacks.py           # V2X (PC5) attacks
    │
    ├── geolocation/                 # Geolocation (3 modules)
    │   ├── environmental_adapter.py # NLOS/multipath compensation
    │   ├── locator.py               # TDOA/AoA/MUSIC algorithms
    │   └── precision_geolocation.py # Enhanced geolocation
    │
    ├── monitoring/                  # Monitoring (13 modules)
    │   ├── aiot_monitor.py          # AIoT traffic monitoring
    │   ├── aiot_rel20_analyzer.py   # 3GPP Rel-20 AIoT
    │   ├── cdma_monitor.py          # CDMA/IS-95
    │   ├── fiveg_monitor.py         # 5G NR
    │   ├── gsm_monitor.py           # GSM 2G
    │   ├── lte_monitor.py           # LTE 4G
    │   ├── ntn_monitor.py           # NTN satellite
    │   ├── pdcch_tracker.py         # PDCCH/DCI (TS 38.212)
    │   ├── profiler.py              # Prometheus profiler
    │   ├── sixg_monitor.py          # 6G ISAC/JCAS
    │   ├── suci_fingerprinter.py    # SUCI fingerprinting
    │   ├── umts_monitor.py          # UMTS 3G
    │   └── vonr_interceptor.py      # VoNR/AMR-WB/EVS
    │
    ├── notifications/               # Alerting (2 modules)
    │   ├── alert_rules.py           # Alert rule engine
    │   └── email_alerts.py          # Email notifications
    │
    ├── oran/                        # O-RAN (3 modules)
    │   ├── e2_interface.py          # E2 protocol handler
    │   ├── near_rt_ric.py           # Near-RT RIC
    │   └── ric_xapp.py              # xApp framework
    │
    ├── sdr/                         # SDR (1 module)
    │   └── sdr_layer.py             # USRP/HackRF/BladeRF/LimeSDR
    │
    ├── security/                    # Security (2 modules)
    │   ├── auditor.py               # FCC/ETSI compliance auditor
    │   └── blockchain_audit.py      # Blockchain audit trail
    │
    ├── sim/                         # SIM (1 module)
    │   └── sim_manager.py           # pySim integration
    │
    ├── simulator/                   # Simulation (1 module)
    │   └── sim_engine.py            # 5G core simulation
    │
    ├── tasks/                       # Celery (5 modules)
    │   ├── celery_app.py            # Celery application
    │   ├── exploit_tasks.py         # Background exploits
    │   ├── monitoring_tasks.py      # Background monitoring
    │   ├── scan_tasks.py            # Background scanning
    │   └── schedules.py             # Scheduled tasks
    │
    ├── tests/                       # Test Suite (17 files)
    │   ├── conftest.py              # Pytest fixtures
    │   ├── e2e_validation.py        # E2E validation
    │   ├── locustfile.py            # Load testing
    │   ├── security_scan.py         # Security scanning
    │   ├── test_authentication.py   # Auth tests
    │   ├── test_database.py         # Database tests
    │   ├── test_e2e.py              # E2E tests
    │   ├── test_e2_interface.py     # O-RAN E2 tests
    │   ├── test_explainable_ai.py   # XAI tests
    │   ├── test_exploitation.py     # Exploit tests
    │   ├── test_integration.py      # Integration tests
    │   ├── test_online_learning.py  # Online learning tests
    │   ├── test_sdr_failover.py     # SDR failover tests
    │   ├── validation_suite.py      # Validation framework
    │   └── integration/             # Integration tests
    │       ├── test_e2e_exploit.py
    │       └── test_e2e_monitoring.py
    │
    ├── ui/                          # UI (2 modules + assets)
    │   ├── dashboard.py             # Flask dashboard (7900+ lines)
    │   ├── i18n.py                  # i18n (8 languages)
    │   ├── static/
    │   │   ├── css/
    │   │   │   ├── accessibility.css    # WCAG 2.1 AA
    │   │   │   ├── dark-mode.css        # Dark theme
    │   │   │   └── responsive.css       # Mobile-responsive
    │   │   ├── i18n/                    # 8 languages
    │   │   │   ├── en.json  # English
    │   │   │   ├── es.json  # Spanish
    │   │   │   ├── fr.json  # French
    │   │   │   ├── de.json  # German
    │   │   │   ├── zh.json  # Chinese
    │   │   │   ├── ja.json  # Japanese
    │   │   │   ├── ar.json  # Arabic
    │   │   │   └── ru.json  # Russian
    │   │   └── js/
    │   │       ├── i18n.js              # i18n loader
    │   │       └── theme-toggle.js      # Dark mode toggle
    │   └── templates/
    │       ├── dashboard.html           # (Embedded)
    │       └── login.html
    │
    ├── utils/                       # Utilities (9 modules)
    │   ├── config.py                # Config loader
    │   ├── database.py              # Database abstraction
    │   ├── data_validator.py        # SNR/DC offset validation
    │   ├── error_recoverer.py       # Circuit breakers & recovery
    │   ├── exceptions.py            # Custom exceptions
    │   ├── logger.py                # Structured logging
    │   ├── performance.py           # Performance utilities
    │   ├── regulatory_scanner.py    # FCC/ETSI compliance
    │   └── sustainability.py        # CodeCarbon tracking
    │
    └── voice/                       # Voice (2 modules)
        ├── amr_decoder.py           # AMR-NB/WB/EVS decoder
        └── sip_parser.py            # SIP/RTP parser
```

### Project Statistics

- **Total Python Modules**: 100+ modules
- **Total Lines of Code**: ~50,000+
- **Test Coverage**: >80%
- **Documentation**: 10 markdown files (~10,000 lines)
- **Supported SDRs**: 4 (USRP, HackRF, BladeRF, LimeSDR)
- **Cellular Generations**: 7 (GSM, CDMA, UMTS, LTE, 5G NR, 6G, NTN)
- **UI Languages**: 8 (EN, ES, FR, DE, ZH, JA, AR, RU)
- **Prometheus Metrics**: 6 scrape targets
- **Alert Rules**: 23 configured alerts
- **Grafana Dashboards**: 8 panels

### Module Descriptions

#### ai/ - Artificial Intelligence & Machine Learning
- **device_profiler.py**: Device fingerprinting and telemetry collection
- **explainable_ai.py**: SHAP/LIME model interpretability
- **federated_coordinator.py**: Federated learning coordination
- **graph_topology.py**: GNN-based network topology inference
- **kpi_monitor.py**: KPI tracking and anomaly detection
- **model_zoo.py**: Centralized ML model registry
- **online_learning.py**: Incremental/online learning
- **payload_generator.py**: GAN-based payload generation
- **ric_optimizer.py**: DQN/MARL RIC optimization
- **signal_classifier.py**: CNN-LSTM signal classification
- **suci_deconcealment.py**: Transformer-based SUCI analysis

#### monitoring/ - Protocol Monitoring
- **aiot_monitor.py**: AIoT traffic monitoring
- **aiot_rel20_analyzer.py**: 3GPP Release 20 AIoT analysis
- **cdma_monitor.py**: CDMA/IS-95 monitoring
- **fiveg_monitor.py**: 5G NR monitoring
- **gsm_monitor.py**: GSM 2G monitoring
- **lte_monitor.py**: LTE 4G monitoring
- **ntn_monitor.py**: NTN satellite monitoring
- **pdcch_tracker.py**: PDCCH/DCI parsing (3GPP TS 38.212)
- **profiler.py**: Prometheus metrics exporter
- **sixg_monitor.py**: 6G ISAC/JCAS monitoring
- **suci_fingerprinter.py**: SUCI fingerprinting
- **umts_monitor.py**: UMTS 3G monitoring
- **vonr_interceptor.py**: VoNR/AMR-WB/EVS interception

#### exploit/ - Exploitation Engines
- **crypto_attacks.py**: Cryptographic attacks
- **exploit_engine.py**: DOS/Downgrade/MITM attacks
- **message_injector.py**: SMS/paging injection
- **ntn_attacks.py**: Satellite (NTN) attacks
- **semantic_exploiter.py**: 6G semantic exploits
- **v2x_attacks.py**: V2X (PC5) attacks

#### utils/ - Utility Modules
- **data_validator.py**: SNR thresholding, DC offset removal
- **error_recoverer.py**: Circuit breakers, SDR reconnection
- **regulatory_scanner.py**: FCC/ETSI/ARIB compliance checking
- **performance.py**: Signal caching, resource pooling

---

## Core Components

### Signal Bus

The `SignalBus` is the central event distribution system.

**Usage Example:**
```python
from falconone.core.signal_bus import SignalBus

signal_bus = SignalBus()

# Subscribe to events
@signal_bus.subscribe('target_detected')
def handle_target_detected(data):
    print(f"New target: {data['imsi']}")

# Emit events
signal_bus.emit('target_detected', {
    'imsi': '310150123456789',
    'network_type': '5G'
})
```

**Built-in Events:**
- `target_detected`: New target discovered
- `scan_completed`: Scan finished
- `exploit_started`: Exploit operation started
- `anomaly_detected`: Anomaly in network
- `xapp_deployed`: xApp deployed to RIC

### Orchestrator

The `Orchestrator` manages system-wide operations.

**Example:**
```python
from falconone.core.orchestrator import Orchestrator

orchestrator = Orchestrator()

# Start scan
scan_id = orchestrator.start_scan(
    target_id=1,
    network_type='5G',
    duration=60
)

# Check status
status = orchestrator.get_scan_status(scan_id)
print(f"Progress: {status['progress']}%")
```

### SDR Driver Interface

All SDR drivers implement the `SDRDriver` interface.

**Base Class:**
```python
from abc import ABC, abstractmethod

class SDRDriver(ABC):
    @abstractmethod
    def initialize(self, config: dict) -> bool:
        """Initialize SDR device"""
        pass
    
    @abstractmethod
    def start_capture(self, frequency: float, sample_rate: float):
        """Start capturing samples"""
        pass
    
    @abstractmethod
    def stop_capture(self):
        """Stop capturing"""
        pass
    
    @abstractmethod
    def get_samples(self, count: int) -> np.ndarray:
        """Get IQ samples"""
        pass
```

**Implementing a New Driver:**
```python
from falconone.sdr.sdr_driver import SDRDriver
import numpy as np

class MySDRDriver(SDRDriver):
    def initialize(self, config: dict) -> bool:
        # Initialize your SDR
        self.device = MySDR.open()
        return True
    
    def start_capture(self, frequency: float, sample_rate: float):
        self.device.set_frequency(frequency)
        self.device.set_sample_rate(sample_rate)
        self.device.start()
    
    def stop_capture(self):
        self.device.stop()
    
    def get_samples(self, count: int) -> np.ndarray:
        return self.device.read_samples(count)
```

### AI Model Integration

**Adding a New Model:**
```python
from falconone.ai.model_zoo import ModelZoo
from sklearn.ensemble import RandomForestClassifier

# Register model
model_zoo = ModelZoo()
model = RandomForestClassifier(n_estimators=100)
model_zoo.register_model(
    model_id='rf_classifier',
    model=model,
    version='1.0.0',
    metadata={'type': 'classification', 'input_dim': 128}
)

# Use model
predictions = model_zoo.predict('rf_classifier', features)
```

### Exploit Engine

**Creating a New Exploit:**
```python
from falconone.exploit.exploit_engine import ExploitEngine

class MyExploit(ExploitEngine):
    def __init__(self):
        super().__init__(
            exploit_id='my_exploit',
            name='My Custom Exploit',
            description='Does something interesting',
            risk_level='medium'
        )
    
    def validate_parameters(self, params: dict) -> bool:
        required = ['target_id', 'parameter1']
        return all(k in params for k in required)
    
    def execute(self, params: dict) -> dict:
        target_id = params['target_id']
        
        # Your exploit logic here
        result = self._perform_exploit(target_id)
        
        return {
            'success': True,
            'metrics': result
        }
    
    def _perform_exploit(self, target_id):
        # Implementation
        pass
```

---

## Testing

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests
│   ├── test_signal_bus.py
│   ├── test_sdr_driver.py
│   └── test_crypto.py
├── integration/             # Integration tests
│   ├── test_scan_workflow.py
│   ├── test_exploit_execution.py
│   └── test_ai_pipeline.py
├── e2e/                     # End-to-end tests
│   └── test_complete_workflow.py
├── performance/             # Performance tests
│   └── locustfile.py
└── security/                # Security tests
    └── security_scan.py
```

### Running Tests

**All Tests:**
```bash
pytest
```

**Specific Test:**
```bash
pytest tests/unit/test_signal_bus.py
```

**With Coverage:**
```bash
pytest --cov=falconone --cov-report=html
```

**Markers:**
```bash
pytest -m unit        # Unit tests only
pytest -m integration # Integration tests only
pytest -m slow        # Slow tests
pytest -m "not slow"  # Skip slow tests
```

### Writing Tests

**Unit Test Example:**
```python
import pytest
from falconone.core.signal_bus import SignalBus

class TestSignalBus:
    def test_subscribe_and_emit(self):
        """Test basic subscription and emission"""
        bus = SignalBus()
        received = []
        
        @bus.subscribe('test_event')
        def handler(data):
            received.append(data)
        
        bus.emit('test_event', {'value': 42})
        
        assert len(received) == 1
        assert received[0]['value'] == 42
    
    def test_unsubscribe(self):
        """Test unsubscribing from events"""
        bus = SignalBus()
        received = []
        
        @bus.subscribe('test_event')
        def handler(data):
            received.append(data)
        
        bus.unsubscribe('test_event', handler)
        bus.emit('test_event', {'value': 42})
        
        assert len(received) == 0
```

**Integration Test Example:**
```python
import pytest
from falconone.core.orchestrator import Orchestrator

@pytest.mark.integration
class TestScanWorkflow:
    def test_complete_scan(self, orchestrator, test_target):
        """Test complete scan workflow"""
        # Start scan
        scan_id = orchestrator.start_scan(
            target_id=test_target.id,
            network_type='5G',
            duration=5  # Short for testing
        )
        
        assert scan_id is not None
        
        # Wait for completion
        import time
        time.sleep(6)
        
        # Check results
        status = orchestrator.get_scan_status(scan_id)
        assert status['status'] == 'completed'
        assert status['signals_detected'] > 0
```

### Test Fixtures

**conftest.py:**
```python
import pytest
from falconone.utils.database import FalconOneDatabase
from falconone.core.orchestrator import Orchestrator

@pytest.fixture(scope='session')
def test_db():
    """Create test database"""
    db = FalconOneDatabase('postgresql://localhost/falconone_test')
    db.create_tables()
    yield db
    db.drop_tables()

@pytest.fixture
def orchestrator(test_db):
    """Orchestrator instance"""
    return Orchestrator(database=test_db)

@pytest.fixture
def test_target(test_db):
    """Test target"""
    from falconone.models import Target
    target = Target(
        imsi='310150123456789',
        imei='352099001761481',
        network_type='5G'
    )
    test_db.add(target)
    return target
```

---

## RANSacked Integration Testing

**New in v1.8.0**: Comprehensive testing framework for RANSacked exploit integration covering all 96 CVE payloads.

### Test Suite Overview

**File:** `falconone/tests/test_ransacked_exploits.py` (700+ lines)

**Test Classes:**
- `TestPayloadGeneration` - Individual CVE payload generation
- `TestImplementationFiltering` - Filter by implementation
- `TestCVEInformation` - CVE metadata retrieval
- `TestPayloadValidation` - Payload structure validation
- `TestErrorHandling` - Edge cases and error conditions
- `TestProtocolDistribution` - Protocol distribution validation
- `TestPerformance` - Performance benchmarks
- `TestIntegration` - Compatibility with ExploitationEngine

**Coverage:** 100+ tests covering all 96 CVEs, 5 implementations, 5 protocols

### Running RANSacked Tests

**All RANSacked Tests:**
```bash
pytest falconone/tests/test_ransacked_exploits.py -v
```

**Specific Test Class:**
```bash
pytest falconone/tests/test_ransacked_exploits.py::TestPayloadGeneration -v
```

**Performance Benchmarks:**
```bash
pytest falconone/tests/test_ransacked_exploits.py::TestPerformance -v
```

**With Coverage:**
```bash
pytest falconone/tests/test_ransacked_exploits.py --cov=falconone.exploit --cov-report=html
```

### Test Structure

**Fixtures:**
```python
@pytest.fixture
def generator():
    """RANSacked payload generator instance"""
    from falconone.exploit.ransacked_payloads import RANSackedPayloadGenerator
    return RANSackedPayloadGenerator()

@pytest.fixture
def target_ip():
    """Default target IP for testing"""
    return "192.168.1.100"
```

**Parametrized Test Example:**
```python
@pytest.mark.parametrize("cve_id", [
    'CVE-2024-24445', 'CVE-2024-24450', 'CVE-2024-24444',
    'CVE-2024-24451', 'CVE-2024-24442', 'CVE-2024-24447',
    'CVE-2024-24443', 'CVE-2024-24446', 'CVE-2024-24449',
    'CVE-2024-24448', 'CVE-2024-24452'
])
def test_oai_5g_payloads(self, generator, target_ip, cve_id):
    """Test OAI 5G payload generation for all 11 CVEs"""
    payload = generator.get_payload(cve_id, target_ip)
    assert payload is not None
    assert isinstance(payload, bytes)
    assert len(payload) > 0
```

**Performance Test Example:**
```python
def test_bulk_payload_generation_speed(self, generator, target_ip):
    """Test bulk generation of all 96 payloads (should be <2 seconds)"""
    import time
    
    all_cves = generator.list_cves()
    start_time = time.time()
    
    payloads = []
    for cve_id in all_cves:
        payload = generator.get_payload(cve_id, target_ip)
        payloads.append(payload)
    
    elapsed_time = time.time() - start_time
    
    assert len(payloads) == 96
    assert elapsed_time < 2.0  # Should complete in under 2 seconds
    print(f"\nBulk generation time: {elapsed_time:.3f}s ({elapsed_time/96*1000:.2f}ms per payload)")
```

**Validation Test Example:**
```python
def test_all_payloads_valid_structure(self, generator, target_ip):
    """Validate all 96 payloads have correct structure"""
    all_cves = generator.list_cves()
    
    for cve_id in all_cves:
        payload = generator.get_payload(cve_id, target_ip)
        info = generator.get_cve_info(cve_id)
        
        # Validate payload
        assert payload is not None
        assert isinstance(payload, bytes)
        assert len(payload) > 0
        
        # Validate reasonable size (10 bytes to 10 KB)
        assert 10 <= len(payload) <= 10240
        
        # Validate info
        assert info is not None
        assert 'implementation' in info
        assert 'protocol' in info
        assert 'severity' in info
```

### Expected Test Results

```bash
$ pytest falconone/tests/test_ransacked_exploits.py -v

collected 108 items

test_ransacked_exploits.py::TestPayloadGeneration::test_generator_initialization PASSED [1%]
test_ransacked_exploits.py::TestPayloadGeneration::test_cve_mapping_complete PASSED [2%]
test_ransacked_exploits.py::TestPayloadGeneration::test_implementations_list PASSED [3%]
test_ransacked_exploits.py::TestPayloadGeneration::test_oai_5g_payloads[CVE-2024-24445] PASSED [4%]
test_ransacked_exploits.py::TestPayloadGeneration::test_oai_5g_payloads[CVE-2024-24450] PASSED [5%]
...
test_ransacked_exploits.py::TestPerformance::test_single_payload_generation_speed PASSED [96%]
test_ransacked_exploits.py::TestPerformance::test_bulk_payload_generation_speed PASSED [97%]
test_ransacked_exploits.py::TestIntegration::test_exploit_engine_compatibility PASSED [98%]
test_ransacked_exploits.py::TestIntegration::test_payload_serialization PASSED [99%]

==================== 108 passed in 12.34s ====================
```

### Exploit Chain Testing

**Chain Execution Test:**
```python
def test_exploit_chain_execution():
    """Test exploit chain dry-run execution"""
    from exploit_chain_examples import chain_1_reconnaissance_crash
    
    chain = chain_1_reconnaissance_crash()
    result = chain.execute(target_ip="127.0.0.1", dry_run=True)
    
    assert result['success'] is True
    assert result['total_steps'] == 3
    assert result['executed_steps'] == 3
    assert result['success_rate'] == 1.0
```

**Manual Chain Testing:**
```bash
# Test all chains
python exploit_chain_examples.py

# Test specific chain
python -c "
from exploit_chain_examples import chain_1_reconnaissance_crash
chain = chain_1_reconnaissance_crash()
result = chain.execute('127.0.0.1', dry_run=True)
print(f'Success Rate: {result[\"success_rate\"]*100:.1f}%')
print(f'Steps: {result[\"executed_steps\"]}/{result[\"total_steps\"]}')
"
```

### GUI Testing

**Manual GUI Testing Checklist:**

✓ **Load GUI**: Navigate to `/exploits/ransacked`
✓ **Statistics**: Verify total CVEs = 96
✓ **Filtering**: Test implementation/protocol filters
✓ **Search**: Search for "CVE-2024-24445"
✓ **Multi-Select**: Click multiple cards, verify counter
✓ **Details Modal**: Click "View Details", verify payload preview
✓ **Execution Modal**: Click "Execute Selected", verify form
✓ **Dry Run**: Execute with dry_run=true, verify results
✓ **Clear Selection**: Click "Clear Selection", verify all deselected

**API Testing:**
```bash
# Test payload listing
curl http://localhost:5000/api/ransacked/payloads | jq '.total'
# Expected: 96

# Test payload details
curl http://localhost:5000/api/ransacked/payload/CVE-2024-24445 | jq '.cve_id'
# Expected: "CVE-2024-24445"

# Test statistics
curl http://localhost:5000/api/ransacked/stats | jq '.by_protocol'
# Expected: {"NGAP": 12, "S1AP": 62, "NAS": 15, "GTP": 3, "GTP-U": 2, ...}
```

### Performance Benchmarks

**Expected Performance:**

| Operation | Target | Typical |
|-----------|--------|---------|
| Single payload generation | <10ms | ~5ms |
| All 96 payloads | <2s | ~1.5s |
| Test suite execution | <20s | ~12s |
| Chain execution (3 steps) | <1s | ~700ms |
| API response (list) | <100ms | ~50ms |
| GUI page load | <500ms | ~300ms |

**Benchmarking:**
```bash
# Run performance tests
pytest falconone/tests/test_ransacked_exploits.py::TestPerformance -v --benchmark-only

# Benchmark with pytest-benchmark
pytest falconone/tests/test_ransacked_exploits.py --benchmark-autosave
```

### Continuous Integration

**GitHub Actions Workflow:**
```yaml
name: RANSacked Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-benchmark pytest-cov
      - name: Run RANSacked tests
        run: |
          pytest falconone/tests/test_ransacked_exploits.py -v --cov=falconone.exploit
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Troubleshooting Tests

**Import Errors:**
```bash
# Ensure falconone package is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest falconone/tests/test_ransacked_exploits.py -v
```

**Slow Tests:**
```bash
# Skip performance benchmarks
pytest falconone/tests/test_ransacked_exploits.py -v -m "not benchmark"
```

**Verbose Output:**
```bash
# Show print statements and full error tracebacks
pytest falconone/tests/test_ransacked_exploits.py -v -s --tb=long
```

---

## Contributing

### Git Workflow

**1. Fork and Clone**
```bash
git clone https://github.com/yourusername/falconone.git
cd falconone
git remote add upstream https://github.com/original/falconone.git
```

**2. Create Feature Branch**
```bash
git checkout -b feature/my-new-feature
```

**3. Make Changes**
```bash
# Edit files
git add .
git commit -m "feat: add new exploit engine"
```

**4. Push and Create PR**
```bash
git push origin feature/my-new-feature
# Open pull request on GitHub
```

### Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```
feat(ai): add SHAP explainability
fix(sdr): resolve USRP initialization bug
docs(api): update authentication endpoints
test(exploit): add DOS attack tests
```

### Code Review Process

**All PRs must:**
1. Pass CI/CD pipeline
2. Have 80%+ code coverage
3. Pass linting (Black, isort, flake8)
4. Have at least one approval
5. Include tests
6. Update documentation

**Review Checklist:**
- [ ] Code follows style guide
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No security vulnerabilities
- [ ] Performance impact acceptable

---

## Code Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) with these additions:

**Line Length:**
```python
# Maximum 100 characters (not 79)
```

**Imports:**
```python
# Standard library
import os
import sys

# Third-party
import numpy as np
import flask

# Local
from falconone.core import SignalBus
from falconone.utils import logger
```

**Type Hints:**
```python
def process_signal(signal_data: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Process signal data.
    
    Args:
        signal_data: IQ samples as numpy array
        threshold: Detection threshold (0.0-1.0)
    
    Returns:
        dict: Processing results
    """
    pass
```

**Docstrings:**
```python
class SignalClassifier:
    """
    Classifies cellular network signals using machine learning.
    
    This classifier uses a trained model to identify network types
    (5G, LTE, GSM) from IQ samples.
    
    Attributes:
        model: Trained sklearn model
        version: Model version string
        input_dim: Expected input dimension
    
    Example:
        >>> classifier = SignalClassifier.load('models/v2.pkl')
        >>> prediction = classifier.predict(iq_samples)
        >>> print(prediction['network_type'])
        '5G_NR'
    """
    pass
```

### Linting Tools

**Black (Formatter):**
```bash
black falconone/
```

**isort (Import Sorting):**
```bash
isort falconone/
```

**flake8 (Linting):**
```bash
flake8 falconone/ --max-line-length=100
```

**pylint (Static Analysis):**
```bash
pylint falconone/ --max-line-length=100
```

**mypy (Type Checking):**
```bash
mypy falconone/ --ignore-missing-imports
```

### Pre-commit Hooks

**Install:**
```bash
pip install pre-commit
pre-commit install
```

**.pre-commit-config.yaml:**
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]
```

---

## API Development

### Creating a New Endpoint

**1. Define Model (models.py):**
```python
from sqlalchemy import Column, Integer, String
from falconone.utils.database import Base

class Widget(Base):
    __tablename__ = 'widgets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    value = Column(Integer, default=0)
```

**2. Create API Namespace (api/widgets.py):**
```python
from flask import request
from flask_restx import Namespace, Resource, fields
from falconone.models import Widget
from falconone.utils.database import db_session

api = Namespace('widgets', description='Widget operations')

widget_model = api.model('Widget', {
    'id': fields.Integer(required=True),
    'name': fields.String(required=True),
    'value': fields.Integer()
})

@api.route('/')
class WidgetList(Resource):
    @api.doc('list_widgets')
    @api.marshal_list_with(widget_model)
    def get(self):
        """List all widgets"""
        return Widget.query.all()
    
    @api.doc('create_widget')
    @api.expect(widget_model)
    @api.marshal_with(widget_model, code=201)
    def post(self):
        """Create a new widget"""
        data = request.json
        widget = Widget(name=data['name'], value=data.get('value', 0))
        db_session.add(widget)
        db_session.commit()
        return widget, 201

@api.route('/<int:id>')
class WidgetResource(Resource):
    @api.doc('get_widget')
    @api.marshal_with(widget_model)
    def get(self, id):
        """Get widget by ID"""
        return Widget.query.get_or_404(id)
    
    @api.doc('delete_widget')
    def delete(self, id):
        """Delete widget"""
        widget = Widget.query.get_or_404(id)
        db_session.delete(widget)
        db_session.commit()
        return {'message': 'Widget deleted'}, 200
```

**3. Register Namespace (main.py):**
```python
from flask import Flask
from flask_restx import Api
from falconone.api.widgets import api as widgets_ns

app = Flask(__name__)
api = Api(app, version='3.0', title='FalconOne API')

api.add_namespace(widgets_ns, path='/api/widgets')
```

### Authentication Decorator

```python
from functools import wraps
from flask import request, jsonify
import jwt

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'Token required'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

# Usage
@api.route('/protected')
class ProtectedResource(Resource):
    @token_required
    def get(self, current_user):
        return {'message': f'Hello {current_user.username}'}
```

---

## Database Schema

### Core Tables

**targets:**
```sql
CREATE TABLE targets (
    id SERIAL PRIMARY KEY,
    imsi VARCHAR(15) NOT NULL UNIQUE,
    imei VARCHAR(15),
    msisdn VARCHAR(20),
    network_type VARCHAR(10),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP
);
```

**suci_captures:**
```sql
CREATE TABLE suci_captures (
    id SERIAL PRIMARY KEY,
    target_id INTEGER REFERENCES targets(id),
    suci VARCHAR(100),
    decrypted_imsi VARCHAR(15),
    capture_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    frequency DOUBLE PRECISION,
    rssi DOUBLE PRECISION
);
```

**exploit_operations:**
```sql
CREATE TABLE exploit_operations (
    id SERIAL PRIMARY KEY,
    exploit_id VARCHAR(50),
    target_id INTEGER REFERENCES targets(id),
    user_id INTEGER REFERENCES users(id),
    parameters JSONB,
    status VARCHAR(20),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    result JSONB
);
```

### Migrations

**Using Alembic:**
```bash
# Initialize
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Add widgets table"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## Deployment

### Production Checklist

- [ ] Set `DEBUG=False`
- [ ] Use strong `SECRET_KEY` and `JWT_SECRET_KEY`
- [ ] Enable HTTPS
- [ ] Configure firewall rules
- [ ] Set up database backups
- [ ] Enable monitoring (Prometheus/Grafana)
- [ ] Configure logging (ELK Stack)
- [ ] Set resource limits (CPU, memory)
- [ ] Enable auto-scaling
- [ ] Configure CDN for static files
- [ ] Set up disaster recovery

### Docker Deployment

**Build Image:**
```bash
docker build -t falconone:latest .
```

**Run Container:**
```bash
docker run -d \
  --name falconone \
  -p 5000:5000 \
  -e DATABASE_URL=postgresql://... \
  -e REDIS_URL=redis://... \
  falconone:latest
```

### Kubernetes Deployment

See [k8s-deployment.yaml](../k8s-deployment.yaml) for full configuration.

**Deploy:**
```bash
kubectl apply -f k8s-deployment.yaml
```

**Scale:**
```bash
kubectl scale deployment falconone --replicas=5
```

### AWS ECS Deployment

See [terraform/aws/main.tf](../terraform/aws/main.tf) for infrastructure as code.

**Deploy:**
```bash
cd terraform/aws
terraform init
terraform plan
terraform apply
```

---

## Additional Resources

**Documentation:**
- User Manual: [USER_MANUAL.md](USER_MANUAL.md)
- API Docs: [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- Cloud Deployment: [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md)

**Community:**
- GitHub: https://github.com/yourusername/falconone
- Discord: https://discord.gg/falconone
- Mailing List: dev@falconone.example.com

**External Resources:**
- 3GPP Specifications: https://www.3gpp.org/
- O-RAN Alliance: https://www.o-ran.org/
- USRP Documentation: https://files.ettus.com/manual/

---

**Developer Guide Version:** 3.0.0  
**Last Updated:** December 31, 2025  
**Maintainer:** FalconOne Development Team
