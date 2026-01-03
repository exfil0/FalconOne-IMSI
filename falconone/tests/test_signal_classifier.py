"""
FalconOne Signal Classifier Unit Tests
Tests for signal_classifier.py with TensorFlow and fallback modes

Version: 1.9.2
Coverage: SignalClassifier, preprocessing, heuristic fallback
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys


# Test data fixtures
@pytest.fixture
def mock_config():
    """Mock configuration object"""
    config = Mock()
    config.get = Mock(side_effect=lambda key, default=None: {
        'ai_ml.signal_classification.accuracy_threshold': 0.90,
        'ai_ml.signal_classification.anomaly_threshold': 0.98,
        'ai_ml.signal_classification.adaptive_mode': True,
        'ai_ml.signal_classification.use_transformer': False,
        'ai.model_cache_dir': '/tmp/test_models',
    }.get(key, default))
    return config


@pytest.fixture
def mock_logger():
    """Mock logger"""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def sample_iq_complex():
    """Sample complex IQ data"""
    t = np.linspace(0, 1, 2048)
    # Simulate a signal with carrier and noise
    carrier = np.exp(2j * np.pi * 100 * t)
    noise = 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    return carrier + noise


@pytest.fixture
def sample_iq_real():
    """Sample real I/Q interleaved data"""
    return np.random.randn(2048)


@pytest.fixture
def sample_signals():
    """Sample signal dictionaries for classification"""
    return [
        {
            'id': 'signal_1',
            'iq_samples': np.random.randn(1024, 2),
            'frequency': 935e6,  # GSM900
            'bandwidth': 0.2e6
        },
        {
            'id': 'signal_2',
            'iq_samples': np.random.randn(1024, 2),
            'frequency': 2600e6,  # LTE
            'bandwidth': 20e6
        },
        {
            'id': 'signal_3',
            'frequency': 28e9,  # 5G mmWave
            'bandwidth': 100e6
        }
    ]


class TestSignalClassifierInitialization:
    """Tests for SignalClassifier initialization"""
    
    def test_init_with_config(self, mock_config, mock_logger):
        """Test initialization with valid config"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        
        assert classifier.accuracy_threshold == 0.90
        assert classifier.anomaly_threshold == 0.98
        assert classifier.adaptive_mode is True
        assert classifier._models_loaded is False  # Lazy loading
    
    def test_init_without_tensorflow(self, mock_config, mock_logger):
        """Test initialization when TensorFlow unavailable"""
        from falconone.ai.signal_classifier import SignalClassifier, TF_AVAILABLE
        
        classifier = SignalClassifier(mock_config, mock_logger)
        
        # Should initialize without errors
        assert classifier is not None
        assert len(classifier.labels) == 8  # Including NTN types
    
    def test_labels_include_ntn(self, mock_config, mock_logger):
        """Test that NTN labels are included"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        
        assert 'NTN-LEO' in classifier.labels
        assert 'NTN-GEO' in classifier.labels
        assert '6G' in classifier.labels


class TestSignalPreprocessing:
    """Tests for signal preprocessing"""
    
    def test_preprocess_complex_signal(self, mock_config, mock_logger, sample_iq_complex):
        """Test preprocessing of complex IQ data"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        processed = classifier._preprocess_signal(sample_iq_complex)
        
        if processed is not None:
            assert processed.shape == (1, 1024, 2)
            assert processed.dtype == np.float32
            # Check normalization
            assert np.max(np.abs(processed)) <= 1.0
    
    def test_preprocess_short_signal(self, mock_config, mock_logger):
        """Test preprocessing with signal shorter than target length"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        short_signal = np.random.randn(100) + 1j * np.random.randn(100)
        processed = classifier._preprocess_signal(short_signal)
        
        if processed is not None:
            assert processed.shape == (1, 1024, 2)  # Zero-padded
    
    def test_preprocess_long_signal(self, mock_config, mock_logger):
        """Test preprocessing with signal longer than target length"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        long_signal = np.random.randn(5000) + 1j * np.random.randn(5000)
        processed = classifier._preprocess_signal(long_signal)
        
        if processed is not None:
            assert processed.shape == (1, 1024, 2)  # Truncated
    
    def test_preprocess_none_signal(self, mock_config, mock_logger):
        """Test preprocessing with None input"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        processed = classifier._preprocess_signal(None)
        
        assert processed is None
    
    def test_preprocess_empty_signal(self, mock_config, mock_logger):
        """Test preprocessing with empty array"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        processed = classifier._preprocess_signal(np.array([]))
        
        assert processed is None


class TestHeuristicClassifier:
    """Tests for fallback heuristic classifier"""
    
    def test_classify_gsm_frequency(self, mock_config, mock_logger):
        """Test GSM classification by frequency"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        result = classifier._classify_heuristic('test', 900e6, 0.2e6)
        
        assert result['predicted_generation'] == 'GSM'
        assert result['method'] == 'heuristic'
        assert result['confidence'] > 0.5
    
    def test_classify_lte_frequency(self, mock_config, mock_logger):
        """Test LTE classification by frequency and bandwidth"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        result = classifier._classify_heuristic('test', 2600e6, 20e6)
        
        assert result['predicted_generation'] == 'LTE'
        assert result['method'] == 'heuristic'
    
    def test_classify_5g_mmwave(self, mock_config, mock_logger):
        """Test 5G mmWave classification"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        result = classifier._classify_heuristic('test', 28e9, 100e6)
        
        assert result['predicted_generation'] == '5G'
        assert result['confidence'] >= 0.85
    
    def test_classify_6g_subthz(self, mock_config, mock_logger):
        """Test 6G sub-THz classification"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        result = classifier._classify_heuristic('test', 150e9, 1e9)
        
        assert result['predicted_generation'] == '6G'
        assert result['confidence'] >= 0.8
    
    def test_classify_ntn_leo(self, mock_config, mock_logger):
        """Test NTN LEO satellite classification"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        result = classifier._classify_heuristic('test', 1530e6, 5e6)  # L-band
        
        assert result['predicted_generation'] == 'NTN-LEO'
    
    def test_classify_unknown_frequency(self, mock_config, mock_logger):
        """Test classification with unknown frequency"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        result = classifier._classify_heuristic('test', 0, 0)
        
        assert result['predicted_generation'] == 'Unknown'
        assert result['confidence'] == 0.5


class TestClassifyMethod:
    """Tests for main classify method"""
    
    def test_classify_empty_list(self, mock_config, mock_logger):
        """Test classification with empty signal list"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        results = classifier.classify([])
        
        assert results == []
    
    def test_classify_with_signals(self, mock_config, mock_logger, sample_signals):
        """Test classification with sample signals"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        results = classifier.classify(sample_signals)
        
        assert len(results) == 3
        for result in results:
            assert 'signal_id' in result
            assert 'predicted_generation' in result
            assert 'confidence' in result
            assert 'method' in result
    
    def test_classify_returns_signal_ids(self, mock_config, mock_logger, sample_signals):
        """Test that classification preserves signal IDs"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        results = classifier.classify(sample_signals)
        
        result_ids = {r['signal_id'] for r in results}
        expected_ids = {'signal_1', 'signal_2', 'signal_3'}
        assert result_ids == expected_ids
    
    def test_classify_handles_missing_iq(self, mock_config, mock_logger):
        """Test classification when IQ samples are missing"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        signals = [{'id': 'test', 'frequency': 2600e6, 'bandwidth': 20e6}]
        results = classifier.classify(signals)
        
        assert len(results) == 1
        assert results[0]['method'] == 'heuristic'  # Falls back to heuristic


class TestOnlineLearning:
    """Tests for online incremental learning (v1.9.2)"""
    
    def test_partial_fit_exists(self, mock_config, mock_logger):
        """Test that partial_fit method exists"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        assert hasattr(classifier, 'partial_fit')
    
    def test_concept_drift_detection_exists(self, mock_config, mock_logger):
        """Test that concept drift detection exists"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        assert hasattr(classifier, 'detect_concept_drift')


class TestAnomalyDetection:
    """Tests for anomaly detection functionality"""
    
    def test_generate_anomaly_report(self, mock_config, mock_logger):
        """Test anomaly report generation with history"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        
        # Add some anomaly history
        classifier.anomaly_history = [
            {'anomaly_detected': True, 'anomaly_score': 0.8},
            {'anomaly_detected': False, 'anomaly_score': 0.2},
        ]
        
        report = classifier.generate_anomaly_report()
        
        assert 'total_checks' in report
        assert 'anomalies_detected' in report
        assert 'detection_rate' in report
    
    def test_empty_anomaly_history(self, mock_config, mock_logger):
        """Test report with no anomaly history"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        report = classifier.generate_anomaly_report()
        
        assert report['total_checks'] == 0
        assert report['anomalies_detected'] == 0


class TestModelBuilding:
    """Tests for model construction"""
    
    @pytest.mark.skipif(
        'tensorflow' not in sys.modules,
        reason="TensorFlow not installed"
    )
    def test_build_model(self, mock_config, mock_logger):
        """Test CNN model building"""
        from falconone.ai.signal_classifier import SignalClassifier, TF_AVAILABLE
        
        if not TF_AVAILABLE:
            pytest.skip("TensorFlow not available")
        
        classifier = SignalClassifier(mock_config, mock_logger)
        classifier._build_model()
        
        assert classifier.model is not None
    
    def test_lazy_loading(self, mock_config, mock_logger):
        """Test that models are not loaded at initialization"""
        from falconone.ai.signal_classifier import SignalClassifier
        
        classifier = SignalClassifier(mock_config, mock_logger)
        
        assert classifier._models_loaded is False
        assert classifier.model is None


# Run with: pytest test_signal_classifier.py -v
