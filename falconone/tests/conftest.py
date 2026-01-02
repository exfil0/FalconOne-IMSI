"""
FalconOne pytest configuration and fixtures
Provides shared fixtures for testing across all modules
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, MagicMock
from datetime import datetime


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def sample_signal_data():
    """Generate sample IQ signal data for testing"""
    return {
        'iq_data': np.random.randn(1000) + 1j * np.random.randn(1000),
        'sample_rate': 2.4e6,
        'center_frequency': 900e6,
        'gain': 40.0,
        'timestamp': datetime.now(),
    }


@pytest.fixture
def sample_kpi_report():
    """Sample KPI report for O-RAN testing"""
    from falconone.oran import KPIReport
    
    return KPIReport(
        node_id='test_gnb_1',
        cell_id=1,
        timestamp=datetime.now(),
        prb_utilization_dl=0.65,
        prb_utilization_ul=0.45,
        active_ues=15,
        throughput_dl_mbps=120.5,
        throughput_ul_mbps=45.8,
        avg_cqi=10.2,
        handover_success_rate=0.95,
        latency_ms=12.5,
        packet_loss_rate=0.01,
    )


@pytest.fixture
def mock_sdr():
    """Mock SDR device for testing"""
    sdr = Mock()
    sdr.sample_rate = 2.4e6
    sdr.center_freq = 900e6
    sdr.gain = 40.0
    sdr.read_samples = Mock(return_value=np.random.randn(1024) + 1j * np.random.randn(1024))
    sdr.write_samples = Mock()
    return sdr


@pytest.fixture
def mock_ric():
    """Mock Near-RT RIC for xApp testing"""
    ric = Mock()
    ric.e2_interface = Mock()
    ric.sdl_set = Mock(return_value=True)
    ric.sdl_get = Mock(return_value={'key': 'value'})
    ric.install_policy = Mock(return_value=True)
    return ric


@pytest.fixture
def sample_model():
    """Sample ML model for testing"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 3, 100)
        model.fit(X, y)
        return model
    except ImportError:
        pytest.skip("scikit-learn not available")


@pytest.fixture
def sample_training_data():
    """Sample training data for ML testing"""
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 3, 100)
    return X, y


@pytest.fixture
def mock_database():
    """Mock database for testing"""
    db = Mock()
    db.execute_query = Mock(return_value=[])
    db.create_target = Mock(return_value=1)
    db.get_target = Mock(return_value={'id': 1, 'imsi': '001010000000001', 'imei': '123456789012345'})
    db.list_targets = Mock(return_value=[])
    return db


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility"""
    np.random.seed(42)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")


# Test collection configuration
def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location"""
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
        else:
            item.add_marker(pytest.mark.unit)
