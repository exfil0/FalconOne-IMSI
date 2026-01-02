#!/usr/bin/env python3
"""
FalconOne System Validation Script
Comprehensive check of database, modules, and functionality
"""

import sys
import sqlite3
import importlib
from pathlib import Path

print("=" * 70)
print("FALCONONE SYSTEM VALIDATION")
print("=" * 70)

# ==================== DATABASE VALIDATION ====================
print("\n[1] DATABASE SCHEMA VALIDATION")
print("-" * 70)

try:
    conn = sqlite3.connect('logs/falconone.db')
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"✅ Database connected: logs/falconone.db")
    print(f"✅ Total tables: {len(tables)}")
    print("\nTables:")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  - {table}: {count} rows")
    
    # Check users table structure
    print("\n📋 Users Table Schema:")
    cursor.execute("PRAGMA table_info(users)")
    cols = cursor.fetchall()
    for col in cols:
        print(f"  - {col[1]} ({col[2]})")
    
    conn.close()
    print("\n✅ Database validation: PASS")
    
except Exception as e:
    print(f"\n❌ Database validation: FAIL - {e}")

# ==================== MODULE IMPORT VALIDATION ====================
print("\n[2] CORE MODULE IMPORT VALIDATION")
print("-" * 70)

modules_to_test = [
    # AI modules
    ('falconone.ai.signal_classifier', 'SignalClassifier'),
    ('falconone.ai.suci_deconcealment', 'SUCIDeconcealmentEngine'),
    ('falconone.ai.kpi_monitor', 'KPIMonitor'),
    ('falconone.ai.ric_optimizer', 'RICOptimizer'),
    ('falconone.ai.graph_topology', 'GNNTopologyInference'),
    ('falconone.ai.federated_coordinator', 'FederatedCoordinator'),
    ('falconone.ai.payload_generator', 'PayloadGenerator'),
    ('falconone.ai.model_zoo', 'ModelZoo'),
    
    # Core modules
    ('falconone.core.main', 'FalconOneCore'),
    ('falconone.core.orchestrator', 'Orchestrator'),
    
    # Monitoring modules
    ('falconone.monitoring.fiveg_monitor', 'FiveGMonitor'),
    ('falconone.monitoring.lte_monitor', 'LTEMonitor'),
    ('falconone.monitoring.gsm_monitor', 'GSMMonitor'),
    
    # Exploit modules
    ('falconone.exploit.exploit_engine', 'ExploitationEngine'),
    ('falconone.exploit.message_injector', 'MessageInjector'),
    
    # Geolocation modules
    ('falconone.geolocation.locator', 'Geolocator'),
    ('falconone.geolocation.precision_geolocation', 'PrecisionGeolocation'),
    
    # Crypto modules
    ('falconone.crypto.analyzer', 'CryptographicAnalyzer'),
    ('falconone.crypto.quantum_resistant', 'QuantumResistantCrypto'),
    
    # Utils
    ('falconone.utils.database', 'FalconOneDatabase'),
]

passed = 0
failed = 0
warnings = 0

for module_path, class_name in modules_to_test:
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, class_name):
            print(f"✅ {module_path}.{class_name}")
            passed += 1
        else:
            print(f"⚠️  {module_path} (class {class_name} not found)")
            warnings += 1
    except ImportError as e:
        print(f"❌ {module_path}: {e}")
        failed += 1
    except Exception as e:
        print(f"⚠️  {module_path}: {e}")
        warnings += 1

print(f"\n✅ Passed: {passed}/{len(modules_to_test)}")
print(f"⚠️  Warnings: {warnings}/{len(modules_to_test)}")
print(f"❌ Failed: {failed}/{len(modules_to_test)}")

# ==================== DEPENDENCY CHECK ====================
print("\n[3] AI/ML DEPENDENCY CHECK")
print("-" * 70)

dependencies = [
    ('tensorflow', 'TensorFlow'),
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('ray', 'Ray'),
    ('gym', 'Gym'),
]

for module_name, display_name in dependencies:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {display_name}: {version}")
    except ImportError:
        print(f"❌ {display_name}: Not installed")

# ==================== SUMMARY ====================
print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

sys.exit(0)
