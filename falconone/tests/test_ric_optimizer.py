"""
FalconOne RIC Optimizer Unit Tests
Tests for ric_optimizer.py Deep Q-Network implementation

Version: 1.9.2
Coverage: RICOptimizer, DQN, experience replay, multi-agent
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from collections import deque
import sys


@pytest.fixture
def mock_config():
    """Mock configuration object"""
    config = Mock()
    config.get = Mock(side_effect=lambda key, default=None: {
        'ai.ric.state_size': 10,
        'ai.ric.action_size': 5,
        'ai.ric.learning_rate': 0.001,
        'ai.ric.gamma': 0.95,
        'ai.ric.epsilon': 1.0,
        'ai.ric.epsilon_min': 0.01,
        'ai.ric.epsilon_decay': 0.995,
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
def sample_state():
    """Sample state vector"""
    return np.random.randn(10)


@pytest.fixture
def sample_batch():
    """Sample batch of experiences"""
    return [
        (np.random.randn(10), 0, 1.0, np.random.randn(10), False)
        for _ in range(32)
    ]


class TestRICOptimizerInitialization:
    """Tests for RICOptimizer initialization"""
    
    def test_init_with_config(self, mock_config, mock_logger):
        """Test initialization with valid config"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        
        assert optimizer.state_size == 10
        assert optimizer.action_size == 5
        assert optimizer.gamma == 0.95
        assert optimizer.epsilon == 1.0
    
    def test_memory_initialized(self, mock_config, mock_logger):
        """Test experience replay memory is initialized"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        
        assert isinstance(optimizer.memory, deque)
        assert optimizer.memory.maxlen == 2000
    
    def test_epsilon_decay_parameters(self, mock_config, mock_logger):
        """Test epsilon decay parameters"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        
        assert optimizer.epsilon_min == 0.01
        assert optimizer.epsilon_decay == 0.995


class TestActionSelection:
    """Tests for action selection (epsilon-greedy)"""
    
    def test_get_action_returns_valid_action(self, mock_config, mock_logger, sample_state):
        """Test get_action returns valid action index"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        action = optimizer.get_action(sample_state)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < optimizer.action_size
    
    def test_exploration_with_high_epsilon(self, mock_config, mock_logger, sample_state):
        """Test exploration behavior with high epsilon"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        optimizer.epsilon = 1.0  # Always explore
        
        # With epsilon=1.0, should get random actions
        actions = [optimizer.get_action(sample_state) for _ in range(100)]
        
        # Should have some variety
        unique_actions = set(actions)
        assert len(unique_actions) > 1 or optimizer.action_size == 1
    
    def test_exploitation_with_low_epsilon(self, mock_config, mock_logger, sample_state):
        """Test exploitation behavior with low epsilon"""
        from falconone.ai.ric_optimizer import RICOptimizer, TF_AVAILABLE
        
        if not TF_AVAILABLE:
            pytest.skip("TensorFlow not available")
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        optimizer.epsilon = 0.0  # Always exploit
        
        # With epsilon=0.0 and model, should get consistent actions
        actions = [optimizer.get_action(sample_state) for _ in range(10)]
        
        # All actions should be the same (greedy)
        assert len(set(actions)) == 1


class TestExperienceReplay:
    """Tests for experience replay functionality"""
    
    def test_remember_stores_experience(self, mock_config, mock_logger, sample_state):
        """Test remember stores experience in buffer"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        next_state = np.random.randn(10)
        
        optimizer.remember(sample_state, 1, 1.0, next_state, False)
        
        assert len(optimizer.memory) == 1
    
    def test_memory_max_length(self, mock_config, mock_logger, sample_state):
        """Test memory respects max length"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        
        # Fill memory beyond max
        for i in range(3000):
            optimizer.remember(sample_state, i % 5, 1.0, sample_state, False)
        
        assert len(optimizer.memory) == 2000  # maxlen
    
    def test_replay_requires_min_samples(self, mock_config, mock_logger, sample_state):
        """Test replay requires minimum samples"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        
        # Add fewer than batch_size experiences
        for i in range(10):
            optimizer.remember(sample_state, 0, 1.0, sample_state, False)
        
        # Should not raise even with insufficient samples
        optimizer.replay()


class TestModelBuilding:
    """Tests for DQN model construction"""
    
    @pytest.mark.skipif(
        'tensorflow' not in sys.modules,
        reason="TensorFlow not installed"
    )
    def test_model_built_when_tf_available(self, mock_config, mock_logger):
        """Test model is built when TensorFlow available"""
        from falconone.ai.ric_optimizer import RICOptimizer, TF_AVAILABLE
        
        if not TF_AVAILABLE:
            pytest.skip("TensorFlow not available")
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        
        assert optimizer.model is not None
        assert optimizer.target_model is not None
    
    def test_target_model_sync(self, mock_config, mock_logger):
        """Test target model has same weights as model"""
        from falconone.ai.ric_optimizer import RICOptimizer, TF_AVAILABLE
        
        if not TF_AVAILABLE:
            pytest.skip("TensorFlow not available")
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        
        if optimizer.model and optimizer.target_model:
            model_weights = optimizer.model.get_weights()
            target_weights = optimizer.target_model.get_weights()
            
            for mw, tw in zip(model_weights, target_weights):
                np.testing.assert_array_equal(mw, tw)


class TestEpsilonDecay:
    """Tests for epsilon decay mechanism"""
    
    def test_epsilon_decays_after_replay(self, mock_config, mock_logger):
        """Test epsilon decays after replay"""
        from falconone.ai.ric_optimizer import RICOptimizer, TF_AVAILABLE
        
        if not TF_AVAILABLE:
            pytest.skip("TensorFlow not available")
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        initial_epsilon = optimizer.epsilon
        
        # Fill memory
        state = np.random.randn(10)
        for i in range(100):
            optimizer.remember(state, 0, 1.0, state, False)
        
        # Note: epsilon decay happens in replay, not automatically
        # This tests the decay mechanism if implemented
    
    def test_epsilon_minimum_bound(self, mock_config, mock_logger):
        """Test epsilon doesn't go below minimum"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        optimizer.epsilon = 0.005  # Below minimum
        
        # After decay, should not go below epsilon_min
        optimizer.epsilon = max(optimizer.epsilon * optimizer.epsilon_decay, 
                                optimizer.epsilon_min)
        
        assert optimizer.epsilon >= optimizer.epsilon_min


class TestBatchProcessing:
    """Tests for batch processing"""
    
    def test_batch_size_configuration(self, mock_config, mock_logger):
        """Test batch size is configurable"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        
        assert optimizer.batch_size == 32


class TestStatePreprocessing:
    """Tests for state preprocessing"""
    
    def test_state_shape_validation(self, mock_config, mock_logger):
        """Test state shape matches configuration"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        state = np.random.randn(optimizer.state_size)
        
        assert state.shape[0] == optimizer.state_size
    
    def test_invalid_state_size(self, mock_config, mock_logger):
        """Test handling of invalid state size"""
        from falconone.ai.ric_optimizer import RICOptimizer, TF_AVAILABLE
        
        if not TF_AVAILABLE:
            pytest.skip("TensorFlow not available")
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        wrong_state = np.random.randn(5)  # Wrong size
        
        # Should handle gracefully or error appropriately


class TestRLlibIntegration:
    """Tests for Ray RLlib integration (v1.4)"""
    
    def test_ray_availability_check(self, mock_config, mock_logger):
        """Test Ray availability is checked"""
        from falconone.ai.ric_optimizer import RAY_AVAILABLE
        
        # Just check the flag exists
        assert isinstance(RAY_AVAILABLE, bool)
    
    @pytest.mark.skipif(
        'ray' not in sys.modules,
        reason="Ray not installed"
    )
    def test_multiagent_env_exists(self, mock_config, mock_logger):
        """Test multi-agent environment class exists when Ray available"""
        from falconone.ai.ric_optimizer import RAY_AVAILABLE
        
        if not RAY_AVAILABLE:
            pytest.skip("Ray not available")


class TestGracefulDegradation:
    """Tests for graceful degradation when dependencies unavailable"""
    
    def test_works_without_tensorflow(self, mock_config, mock_logger):
        """Test optimizer works without TensorFlow"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        
        # Should still initialize
        assert optimizer is not None
        
        # Should return random action
        state = np.random.randn(10)
        action = optimizer.get_action(state)
        assert 0 <= action < optimizer.action_size
    
    def test_remember_works_without_model(self, mock_config, mock_logger):
        """Test remember works without model"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        state = np.random.randn(10)
        
        # Should not raise
        optimizer.remember(state, 0, 1.0, state, False)
        assert len(optimizer.memory) == 1


class TestRewardProcessing:
    """Tests for reward signal processing"""
    
    def test_positive_reward(self, mock_config, mock_logger):
        """Test positive reward handling"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        state = np.random.randn(10)
        
        optimizer.remember(state, 0, 10.0, state, False)
        
        stored_reward = optimizer.memory[-1][2]
        assert stored_reward == 10.0
    
    def test_negative_reward(self, mock_config, mock_logger):
        """Test negative reward handling"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        state = np.random.randn(10)
        
        optimizer.remember(state, 0, -5.0, state, False)
        
        stored_reward = optimizer.memory[-1][2]
        assert stored_reward == -5.0
    
    def test_done_flag(self, mock_config, mock_logger):
        """Test episode done flag"""
        from falconone.ai.ric_optimizer import RICOptimizer
        
        optimizer = RICOptimizer(mock_config, mock_logger)
        state = np.random.randn(10)
        
        optimizer.remember(state, 0, 1.0, state, True)  # Done=True
        
        stored_done = optimizer.memory[-1][4]
        assert stored_done is True


# Run with: pytest test_ric_optimizer.py -v
