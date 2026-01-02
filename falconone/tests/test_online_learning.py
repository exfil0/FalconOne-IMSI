"""
Unit tests for Online Learning module
Tests incremental updates, drift detection, model versioning, and rollback
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


@pytest.fixture
def online_learner():
    """Create OnlineLearner instance"""
    from falconone.ai.online_learning import OnlineLearner
    return OnlineLearner(logger=Mock())


@pytest.fixture
def training_data():
    """Generate sample training data"""
    X = np.random.rand(200, 10)
    y = np.random.randint(0, 3, 200)
    return X, y


class TestOnlineLearner:
    """Test suite for OnlineLearner class"""
    
    def test_initialization(self, online_learner):
        """Test OnlineLearner initialization"""
        assert online_learner is not None
        assert online_learner.model is None
        assert online_learner.num_updates == 0
        assert online_learner.current_version == 0
    
    def test_initialize_model(self, online_learner):
        """Test model initialization"""
        online_learner.initialize_model(n_features=10, n_classes=3)
        
        assert online_learner.model is not None
        assert online_learner.scaler is not None
        assert online_learner.current_version == 1
    
    def test_initialize_with_pretrained(self, online_learner):
        """Test initialization with pretrained model"""
        from sklearn.linear_model import SGDClassifier
        
        pretrained = SGDClassifier(loss='log_loss')
        pretrained.partial_fit(
            np.random.rand(50, 10),
            np.random.randint(0, 3, 50),
            classes=[0, 1, 2]
        )
        
        online_learner.initialize_model(
            n_features=10,
            n_classes=3,
            pretrained_model=pretrained
        )
        
        assert online_learner.model is not None
        assert online_learner.current_version == 1
    
    def test_partial_fit_warmup_phase(self, online_learner, training_data):
        """Test warmup phase accumulation"""
        X, y = training_data
        
        online_learner.initialize_model(n_features=10, n_classes=3)
        
        # Add samples during warmup (default: 100 samples)
        for i in range(50):
            metrics = online_learner.partial_fit(X[i:i+1], y[i:i+1])
            assert metrics['in_warmup'] == True
        
        assert online_learner.num_updates == 0  # No updates during warmup
    
    def test_partial_fit_after_warmup(self, online_learner, training_data):
        """Test training after warmup phase"""
        X, y = training_data
        
        online_learner.initialize_model(n_features=10, n_classes=3)
        online_learner.warmup_samples = 10  # Reduce for testing
        
        # Complete warmup
        metrics = online_learner.partial_fit(X[:15], y[:15])
        
        assert metrics['in_warmup'] == False
        assert online_learner.num_updates == 1
        assert metrics['learning_rate'] > 0
    
    def test_predict(self, online_learner, training_data):
        """Test prediction"""
        X, y = training_data
        
        online_learner.initialize_model(n_features=10, n_classes=3)
        online_learner.warmup_samples = 10
        online_learner.partial_fit(X[:50], y[:50])
        
        predictions = online_learner.predict(X[50:60])
        
        assert predictions is not None
        assert len(predictions) == 10
        assert all(p in [0, 1, 2] for p in predictions)
    
    def test_predict_proba(self, online_learner, training_data):
        """Test probability prediction"""
        X, y = training_data
        
        online_learner.initialize_model(n_features=10, n_classes=3)
        online_learner.warmup_samples = 10
        online_learner.partial_fit(X[:50], y[:50])
        
        probas = online_learner.predict_proba(X[50:60])
        
        assert probas is not None
        assert probas.shape == (10, 3)
        # Probabilities should sum to 1
        assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_drift_detection(self, online_learner, training_data):
        """Test concept drift detection"""
        X, y = training_data
        
        online_learner.initialize_model(n_features=10, n_classes=3)
        online_learner.warmup_samples = 10
        online_learner.partial_fit(X[:50], y[:50])
        
        with patch.object(online_learner.drift_detector, 'update') as mock_update:
            mock_update.return_value = True  # Simulate drift detection
            
            metrics = online_learner.partial_fit(X[50:60], y[50:60])
            
            assert metrics['drift_detected'] == True
    
    def test_snapshot_creation(self, online_learner, training_data):
        """Test model snapshot creation"""
        X, y = training_data
        
        online_learner.initialize_model(n_features=10, n_classes=3)
        online_learner.warmup_samples = 10
        
        online_learner.partial_fit(X[:50], y[:50], create_snapshot=True)
        
        assert len(online_learner.snapshots) == 1
        snapshot = online_learner.snapshots[0]
        assert snapshot.version == 2  # v1 from init, v2 from first fit
        assert snapshot.num_samples >= 50
    
    def test_rollback(self, online_learner, training_data):
        """Test model rollback"""
        X, y = training_data
        
        online_learner.initialize_model(n_features=10, n_classes=3)
        online_learner.warmup_samples = 10
        
        # Create multiple snapshots
        online_learner.partial_fit(X[:50], y[:50], create_snapshot=True)
        version_2 = online_learner.current_version
        
        online_learner.partial_fit(X[50:100], y[50:100], create_snapshot=True)
        version_3 = online_learner.current_version
        
        # Rollback to version 2
        success = online_learner.rollback(version=version_2)
        
        assert success == True
        assert online_learner.current_version == version_2
        assert online_learner.rollback_count == 1
    
    def test_rollback_latest(self, online_learner, training_data):
        """Test rollback to latest snapshot"""
        X, y = training_data
        
        online_learner.initialize_model(n_features=10, n_classes=3)
        online_learner.warmup_samples = 10
        
        online_learner.partial_fit(X[:50], y[:50], create_snapshot=True)
        online_learner.partial_fit(X[50:100], y[50:100])  # No snapshot
        
        # Rollback to latest (should restore first snapshot)
        success = online_learner.rollback()
        
        assert success == True
    
    def test_rollback_invalid_version(self, online_learner):
        """Test rollback with invalid version"""
        online_learner.initialize_model(n_features=10, n_classes=3)
        
        success = online_learner.rollback(version=999)
        
        assert success == False
    
    def test_adaptive_learning_rate(self, online_learner, training_data):
        """Test adaptive learning rate adjustment"""
        X, y = training_data
        
        online_learner.initialize_model(n_features=10, n_classes=3)
        online_learner.warmup_samples = 10
        
        initial_lr = online_learner.learning_rate
        
        # Simulate declining performance
        online_learner.performance_history = [0.9, 0.85, 0.80, 0.75, 0.70]
        
        metrics = online_learner.partial_fit(X[:20], y[:20])
        
        # Learning rate should decrease
        assert online_learner.learning_rate < initial_lr
    
    def test_max_snapshots_limit(self, online_learner, training_data):
        """Test snapshot limit enforcement"""
        X, y = training_data
        
        online_learner.initialize_model(n_features=10, n_classes=3)
        online_learner.warmup_samples = 10
        online_learner.max_snapshots = 3
        
        # Create more snapshots than limit
        for i in range(5):
            start_idx = i * 20
            online_learner.partial_fit(
                X[start_idx:start_idx+20],
                y[start_idx:start_idx+20],
                create_snapshot=True
            )
        
        # Should only keep last 3 snapshots
        assert len(online_learner.snapshots) == 3
    
    def test_save_and_load(self, online_learner, training_data):
        """Test model saving and loading"""
        X, y = training_data
        
        online_learner.initialize_model(n_features=10, n_classes=3)
        online_learner.warmup_samples = 10
        online_learner.partial_fit(X[:50], y[:50])
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name
        
        try:
            # Save
            online_learner.save(filepath)
            assert os.path.exists(filepath)
            
            # Load into new instance
            new_learner = online_learner.__class__(logger=Mock())
            new_learner.load(filepath)
            
            assert new_learner.model is not None
            assert new_learner.num_updates == online_learner.num_updates
            
            # Predictions should match
            pred1 = online_learner.predict(X[50:55])
            pred2 = new_learner.predict(X[50:55])
            assert np.array_equal(pred1, pred2)
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_get_metrics(self, online_learner, training_data):
        """Test metrics retrieval"""
        X, y = training_data
        
        online_learner.initialize_model(n_features=10, n_classes=3)
        online_learner.warmup_samples = 10
        online_learner.partial_fit(X[:50], y[:50])
        online_learner.partial_fit(X[50:100], y[50:100])
        
        metrics = online_learner.get_metrics()
        
        assert metrics['num_updates'] == 2
        assert metrics['total_samples'] >= 100
        assert 'accuracy' in metrics
        assert 'learning_rate' in metrics
    
    def test_feature_scaling(self, online_learner, training_data):
        """Test feature scaling with StandardScaler"""
        X, y = training_data
        
        online_learner.initialize_model(n_features=10, n_classes=3, use_scaler=True)
        online_learner.warmup_samples = 10
        
        online_learner.partial_fit(X[:50], y[:50])
        
        assert online_learner.scaler is not None
        # Scaler should be fitted
        assert hasattr(online_learner.scaler, 'mean_')
    
    def test_performance_tracking(self, online_learner, training_data):
        """Test performance history tracking"""
        X, y = training_data
        
        online_learner.initialize_model(n_features=10, n_classes=3)
        online_learner.warmup_samples = 10
        
        for i in range(3):
            start_idx = i * 30
            online_learner.partial_fit(X[start_idx:start_idx+30], y[start_idx:start_idx+30])
        
        assert len(online_learner.performance_history) == 3
        assert all(0.0 <= acc <= 1.0 for acc in online_learner.performance_history)
    
    def test_classes_initialization(self, online_learner):
        """Test that classes are properly set"""
        online_learner.initialize_model(n_features=5, n_classes=4)
        
        assert online_learner.n_classes == 4
        assert online_learner.classes == [0, 1, 2, 3]
    
    def test_predict_before_training(self, online_learner):
        """Test prediction before any training"""
        with pytest.raises(ValueError):
            online_learner.predict(np.random.rand(10, 5))
    
    def test_partial_fit_before_initialization(self, online_learner):
        """Test partial_fit before initialization"""
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 3, 10)
        
        with pytest.raises(ValueError):
            online_learner.partial_fit(X, y)


class TestModelSnapshot:
    """Test ModelSnapshot dataclass"""
    
    def test_snapshot_creation(self):
        """Test creating ModelSnapshot"""
        from falconone.ai.online_learning import ModelSnapshot
        import time
        
        snapshot = ModelSnapshot(
            version=1,
            model_state=b'model_data',
            timestamp=time.time(),
            metrics={'accuracy': 0.85},
            num_samples=100
        )
        
        assert snapshot.version == 1
        assert snapshot.num_samples == 100
        assert snapshot.metrics['accuracy'] == 0.85


class TestLearningMetrics:
    """Test LearningMetrics dataclass"""
    
    def test_metrics_creation(self):
        """Test creating LearningMetrics"""
        from falconone.ai.online_learning import LearningMetrics
        
        metrics = LearningMetrics(
            num_updates=10,
            total_samples=500,
            accuracy=0.88,
            learning_rate=0.01,
            concept_drifts_detected=2,
            model_rollbacks=1
        )
        
        assert metrics.num_updates == 10
        assert metrics.accuracy == 0.88
        assert metrics.concept_drifts_detected == 2


@pytest.mark.slow
class TestOnlineLearningIntegration:
    """Integration tests with real data streams"""
    
    def test_continuous_learning_stream(self, online_learner):
        """Test continuous learning with data stream"""
        online_learner.initialize_model(n_features=10, n_classes=3)
        online_learner.warmup_samples = 20
        
        # Simulate data stream
        for batch_idx in range(10):
            X_batch = np.random.rand(20, 10)
            y_batch = np.random.randint(0, 3, 20)
            
            metrics = online_learner.partial_fit(X_batch, y_batch)
            
            if not metrics['in_warmup']:
                assert metrics['accuracy'] >= 0.0
        
        assert online_learner.num_updates > 0
    
    def test_concept_drift_adaptation(self, online_learner):
        """Test adaptation to concept drift"""
        online_learner.initialize_model(n_features=5, n_classes=2)
        online_learner.warmup_samples = 20
        
        # Initial concept
        X1 = np.random.rand(100, 5)
        y1 = (X1[:, 0] > 0.5).astype(int)
        
        online_learner.partial_fit(X1[:50], y1[:50])
        acc1 = online_learner.partial_fit(X1[50:], y1[50:])['accuracy']
        
        # Concept drift: flip the relationship
        X2 = np.random.rand(100, 5)
        y2 = (X2[:, 0] < 0.5).astype(int)
        
        # Performance should initially drop, then recover
        initial_acc = online_learner.partial_fit(X2[:30], y2[:30])['accuracy']
        final_acc = online_learner.partial_fit(X2[70:], y2[70:])['accuracy']
        
        # Model should adapt (final accuracy should improve)
        assert online_learner.drift_count > 0 or final_acc > initial_acc
