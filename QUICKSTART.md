# FalconOne Quick Start Guide

**Version:** 1.7.0 | **Updated:** December 31, 2025

## 🚀 Getting Started (5 Minutes)

### Step 1: Install Dependencies (2 minutes)
```bash
# Windows (Python 3.13.0)
.venv\Scripts\activate
pip install flask flask-socketio flask-limiter flask-cors pyyaml numpy pandas psycopg2-binary redis sqlalchemy celery prometheus-client flask-wtf marshmallow python-dotenv

# Linux/macOS
source .venv/bin/activate
pip install flask flask-socketio flask-limiter flask-cors pyyaml numpy pandas psycopg2-binary redis sqlalchemy celery prometheus-client flask-wtf marshmallow python-dotenv

# Automated installation
python install_dependencies.py
```

### Step 2: Configure Environment (1 minute)
```bash
# Copy template (if exists)
cp .env.example .env

# Generate secrets (PowerShell)
py -c "import secrets; print(f'FALCONONE_SECRET_KEY={secrets.token_hex(32)}')" >> .env
py -c "import secrets; print(f'DATABASE_ENCRYPTION_KEY={secrets.token_urlsafe(32)}')" >> .env

# Or use default configuration (no .env required for testing)
```

### Step 3: Validate Installation (1 minute)
```bash
# Comprehensive validation (6 tests)
python quick_validate.py

# Full system validation
python validate_system.py
```

### Step 4: Run Dashboard (1 minute)
```bash
# Windows
python start_dashboard.py

# Linux/macOS
python start_dashboard.py

# Access at: http://127.0.0.1:5000
```

## 📋 Dashboard Features

Once running, access these tabs:
1. **Overview** - System status and KPIs
2. **Devices** - SDR hardware management
3. **Terminal** - Embedded command interface
4. **Cellular** - Multi-generation monitoring
5. **Captures** - IMSI/SUCI/voice captures
6. **Exploits** - Attack operations
7. **Analytics** - AI/ML insights
8. **Setup** - SDR installation wizard
9. **System Tools** - External dependency management (NEW)
10. **System** - Health and performance metrics
py main.py

# Linux/macOS
python main.py

# Open browser: http://localhost:5000
# Default credentials: admin / admin
```

---

## 📋 What's New (Phase 1 Implementation)

### ✅ Exploitation Engine (COMPLETE)
```python
from falconone.exploit.exploit_engine import ExploitationEngine

# DOS Attack
engine.execute('dos', {
    'target_frequency': 2140.0,  # MHz
    'duration': 10,              # seconds
    'method': 'jam',             # jam|rach|paging
    'power': 20                  # dBm
})

# Downgrade Attack (5G→LTE→3G)
engine.execute('downgrade', {
    'source_freq': 3500.0,       # 5G to jam
    'fake_freq': 2140.0,         # Fake LTE
    'target_generation': 'lte',
    'duration': 60
})

# MITM Attack
engine.execute('mitm', {
    'frequency': 2140.0,
    'duration': 120,
    'log_traffic': True,
    'credential_capture': True
})
```

### ✅ Database Security (COMPLETE)
```python
from falconone.utils.database import FalconOneDatabase
import os

# Encrypted database
db = FalconOneDatabase(
    db_path='logs/falconone.db',
    encryption_key=os.getenv('DATABASE_ENCRYPTION_KEY'),
    backup_enabled=True,
    backup_retention_days=90
)

# Create backup
backup_path = db.backup_database()

# Check health
health = db.get_database_health()
print(f"Status: {health['status']}")
print(f"Encryption: {health['checks']['encryption']['enabled']}")

# List backups
backups = db.list_backups()
for backup in backups:
    print(f"{backup['filename']} - {backup['size_mb']:.2f} MB - {backup['created']}")
```

### ✅ Dashboard Security (COMPLETE)
```python
# Environment variables (in .env)
FALCONONE_SECRET_KEY=your-secret-key-here
RATE_LIMIT_ENABLED=true
RATE_LIMIT_DEFAULT=100 per hour

# Input validation
from marshmallow import ValidationError

@app.route('/api/targets', methods=['POST'])
@validate_request(TargetSchema)  # Automatic validation
def create_target():
    data = request.validated_data  # Pre-validated
    # Process data...
```

---

## 🔧 Common Tasks

### Run Exploitation Test
```python
python quick_exploit_test.py
```

### Create Database Backup
```python
from falconone.utils.database import FalconOneDatabase
db = FalconOneDatabase('logs/falconone.db')
backup = db.backup_database()
print(f"Backup created: {backup}")
```

### Check System Health
```python
python quick_health_check.py
```

### View Progress
```bash
cat IMPLEMENTATION_PROGRESS.md
```

---

## 📊 Phase 1 Progress: 42% Complete

### ✅ Completed (8 tasks)
- Exploitation Engine (DOS/Downgrade/MITM)
- Database Foreign Keys
- Database Encryption (SQLCipher)
- Database Backups
- Remove Hardcoded Secrets
- API Rate Limiting
- Input Validation (Marshmallow)
- CSRF Protection

### ⏳ Remaining (11 tasks)
- User Authentication (Flask-Login)
- Password Hashing (bcrypt)
- Session Management
- SDR Device Failover
- Health Monitoring
- Unit Tests (50% coverage)
- Integration Tests
- Security Auditing

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| `PHASE1_SUMMARY.md` | Complete summary of Phase 1 implementation |
| `IMPLEMENTATION_PROGRESS.md` | Real-time progress tracker (8/68 tasks) |
| `IMPLEMENTATION_ROADMAP.md` | Full 68-task roadmap |
| `COMPREHENSIVE_GAP_ANALYSIS.md` | Initial gap analysis |
| `.env.example` | Environment variable template |
| `validate_phase1.py` | Automated validation script |

---

## 🐛 Troubleshooting

### Dependency Issues
```bash
# Check what's missing
python validate_phase1.py

# Install missing packages
pip install <package-name>
```

### Database Encryption Not Working
```bash
# Install SQLCipher
pip install pysqlcipher3

# Verify
python -c "from pysqlcipher3 import dbapi2; print('✅ SQLCipher available')"
```

### Rate Limiting Not Active
```bash
# Install Flask-Limiter
pip install flask-limiter

# Enable in .env
echo "RATE_LIMIT_ENABLED=true" >> .env
```

---

## 🔗 Quick Links

- **Start Dashboard:** `python start_dashboard.py`
- **Run Tests:** `python validate_phase1.py`
- **Check Health:** `python quick_health_check.py`
- **View Logs:** `tail -f logs/falconone.log`
- **Database Backup:** `python -c "from falconone.utils.database import *; FalconOneDatabase().backup_database()"`

---

## 💡 Tips

1. **Always use .env file** for secrets (never hardcode)
2. **Run validation after changes** (`python validate_phase1.py`)
3. **Enable database backups** (set `backup_enabled=True`)
4. **Check health regularly** (`db.get_database_health()`)
5. **Use rate limiting** in production (`RATE_LIMIT_ENABLED=true`)
6. **Validate all inputs** with Marshmallow schemas
7. **Enable CSRF protection** for web forms

---

## 🎯 Next Steps

1. **Complete Phase 1** (11 tasks remaining)
   - User authentication
   - Password hashing
   - SDR failover
   - Unit tests

2. **Begin Phase 2** (18 tasks)
   - GSM A5/1 decryption
   - LTE key extraction
   - CSV/JSON/PDF exports
   - Multi-user support

3. **Deploy to Production**
   - Set up HTTPS
   - Configure Redis for rate limiting
   - Enable monitoring (Prometheus/Grafana)
   - Set up CI/CD pipeline

---

**Status:** ✅ Phase 1: 42% Complete (8/19 tasks)  
**Overall:** 12% Complete (8/68 tasks)  
**Time Investment:** ~2 hours this session  
**Next Session:** 4-6 hours to complete Phase 1
