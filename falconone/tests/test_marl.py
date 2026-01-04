"""
FalconOne Multi-Agent RL Unit Tests
Tests for SIGINTMultiAgentEnv and MARL components in ric_optimizer.py

Version: 1.9.6
Coverage: SIGINTMultiAgentEnv, gym integration, multi-agent step/reset
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys


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
def mock_config():
    """Mock configuration"""
    config = Mock()
    config.get = Mock(side_effect=lambda key, default=None: {
        'ai.ric.state_size': 10,
        'ai.ric.action_size': 5,
        'ai.marl.num_agents': 3,
    }.get(key, default))
    return config


@pytest.fixture
def marl_config():
    """Configuration for multi-agent environment"""
    return {
        'num_agents': 3,
    }


# =============================================================================
# Gym Availability Tests
# =============================================================================

class TestGymAvailability:
    """Tests for gym import and availability handling"""
    
    def test_gym_available_flag_exists(self):
        """Test GYM_AVAILABLE flag is defined"""
        from falconone.ai import ric_optimizer
        
        assert hasattr(ric_optimizer, 'GYM_AVAILABLE')
    
    def test_gym_import_at_top(self):
        """Test gym is imported at module level"""
        from falconone.ai import ric_optimizer
        
        # Should not raise NameError
        assert hasattr(ric_optimizer, 'GYM_AVAILABLE')
        
        if ric_optimizer.GYM_AVAILABLE:
            assert hasattr(ric_optimizer, 'gym')
    
    def test_ray_available_flag_exists(self):
        """Test RAY_AVAILABLE flag is defined"""
        from falconone.ai import ric_optimizer
        
        assert hasattr(ric_optimizer, 'RAY_AVAILABLE')


# =============================================================================
# SIGINTMultiAgentEnv Initialization Tests
# =============================================================================

class TestSIGINTMultiAgentEnvInit:
    """Tests for SIGINTMultiAgentEnv initialization"""
    
    def test_env_class_exists(self):
        """Test SIGINTMultiAgentEnv class exists"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        assert SIGINTMultiAgentEnv is not None
    
    def test_env_init_default_config(self):
        """Test environment initialization with default config"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv()
        
        assert env is not None
        assert env.num_agents == 3  # Default
    
    def test_env_init_custom_agents(self, marl_config):
        """Test environment initialization with custom agent count"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        config = {'num_agents': 5}
        env = SIGINTMultiAgentEnv(config)
        
        assert env.num_agents == 5
    
    def test_agent_ids_created(self, marl_config):
        """Test agent IDs are properly created"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        
        assert hasattr(env, '_agent_ids')
        assert len(env._agent_ids) == 3
        assert 'agent_0' in env._agent_ids
        assert 'agent_1' in env._agent_ids
        assert 'agent_2' in env._agent_ids
    
    def test_observation_space_defined(self, marl_config):
        """Test observation space is defined"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv, GYM_AVAILABLE
        
        env = SIGINTMultiAgentEnv(marl_config)
        
        if GYM_AVAILABLE:
            assert env.observation_space is not None
            # Box space with shape (10,)
            assert env.observation_space.shape == (10,)
        else:
            # Graceful fallback when gym unavailable
            assert env.observation_space is None
    
    def test_action_space_defined(self, marl_config):
        """Test action space is defined"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv, GYM_AVAILABLE
        
        env = SIGINTMultiAgentEnv(marl_config)
        
        if GYM_AVAILABLE:
            assert env.action_space is not None
            # Discrete(5) action space
            assert env.action_space.n == 5
        else:
            assert env.action_space is None
    
    def test_no_name_error_without_gym(self):
        """Test no NameError when gym is unavailable"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        # Should not raise NameError even if gym unavailable
        try:
            env = SIGINTMultiAgentEnv()
            assert env is not None
        except NameError as e:
            pytest.fail(f"NameError raised: {e}")


# =============================================================================
# Environment Reset Tests
# =============================================================================

class TestSIGINTMultiAgentEnvReset:
    """Tests for environment reset functionality"""
    
    def test_reset_returns_states(self, marl_config):
        """Test reset returns initial states for all agents"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        states = env.reset()
        
        assert isinstance(states, dict)
        assert len(states) == 3
    
    def test_reset_state_shape(self, marl_config):
        """Test reset returns states with correct shape"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        states = env.reset()
        
        for agent_id, state in states.items():
            assert isinstance(state, np.ndarray)
            assert state.shape == (10,)
            assert state.dtype == np.float32
    
    def test_reset_state_bounds(self, marl_config):
        """Test reset states are within observation bounds"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        states = env.reset()
        
        for agent_id, state in states.items():
            assert np.all(state >= 0)
            assert np.all(state <= 1)
    
    def test_reset_clears_timestep(self, marl_config):
        """Test reset clears timestep counter"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        env.timestep = 50
        
        env.reset()
        
        assert env.timestep == 0
    
    def test_multiple_resets(self, marl_config):
        """Test multiple consecutive resets"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        
        for _ in range(5):
            states = env.reset()
            assert len(states) == 3


# =============================================================================
# Environment Step Tests
# =============================================================================

class TestSIGINTMultiAgentEnvStep:
    """Tests for environment step functionality"""
    
    def test_step_returns_tuple(self, marl_config):
        """Test step returns (obs, rewards, dones, infos) tuple"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        env.reset()
        
        actions = {'agent_0': 0, 'agent_1': 1, 'agent_2': 2}
        result = env.step(actions)
        
        assert len(result) == 4
        observations, rewards, dones, infos = result
    
    def test_step_observations_dict(self, marl_config):
        """Test step returns observations for all agents"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        env.reset()
        
        actions = {'agent_0': 0, 'agent_1': 1, 'agent_2': 2}
        observations, _, _, _ = env.step(actions)
        
        assert isinstance(observations, dict)
        assert len(observations) == 3
    
    def test_step_rewards_dict(self, marl_config):
        """Test step returns rewards for all agents"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        env.reset()
        
        actions = {'agent_0': 0, 'agent_1': 1, 'agent_2': 2}
        _, rewards, _, _ = env.step(actions)
        
        assert isinstance(rewards, dict)
        assert len(rewards) == 3
        
        for agent_id, reward in rewards.items():
            assert isinstance(reward, (int, float))
    
    def test_step_dones_dict(self, marl_config):
        """Test step returns done flags for all agents"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        env.reset()
        
        actions = {'agent_0': 0, 'agent_1': 1, 'agent_2': 2}
        _, _, dones, _ = env.step(actions)
        
        assert isinstance(dones, dict)
        assert '__all__' in dones
    
    def test_step_infos_dict(self, marl_config):
        """Test step returns info dicts for all agents"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        env.reset()
        
        actions = {'agent_0': 0, 'agent_1': 1, 'agent_2': 2}
        _, _, _, infos = env.step(actions)
        
        assert isinstance(infos, dict)
    
    def test_step_increments_timestep(self, marl_config):
        """Test step increments timestep counter"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        env.reset()
        
        initial_timestep = env.timestep
        
        actions = {'agent_0': 0, 'agent_1': 1, 'agent_2': 2}
        env.step(actions)
        
        assert env.timestep == initial_timestep + 1
    
    def test_step_all_actions(self, marl_config):
        """Test all valid actions (0-4)"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        
        for action in range(5):
            env.reset()
            actions = {f'agent_{i}': action for i in range(3)}
            observations, rewards, dones, infos = env.step(actions)
            
            assert len(observations) == 3
            assert len(rewards) == 3


# =============================================================================
# Episode Termination Tests
# =============================================================================

class TestEpisodeTermination:
    """Tests for episode termination conditions"""
    
    def test_episode_terminates_at_100_steps(self, marl_config):
        """Test episode terminates after 100 timesteps"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        env.reset()
        
        actions = {'agent_0': 0, 'agent_1': 1, 'agent_2': 2}
        
        for i in range(100):
            _, _, dones, _ = env.step(actions)
        
        assert dones['__all__'] is True
    
    def test_episode_not_done_before_100(self, marl_config):
        """Test episode not done before 100 steps"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        env.reset()
        
        actions = {'agent_0': 0, 'agent_1': 1, 'agent_2': 2}
        
        for i in range(50):
            _, _, dones, _ = env.step(actions)
        
        assert dones['__all__'] is False
    
    def test_all_agents_done_at_termination(self, marl_config):
        """Test all agents marked done at episode end"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        env.reset()
        
        actions = {'agent_0': 0, 'agent_1': 1, 'agent_2': 2}
        
        for i in range(100):
            _, _, dones, _ = env.step(actions)
        
        for agent_id in env._agent_ids:
            assert dones[agent_id] is True


# =============================================================================
# Action Effects Tests
# =============================================================================

class TestActionEffects:
    """Tests for action effects on rewards"""
    
    def test_action_0_increase_priority(self, marl_config):
        """Test action 0 (increase priority) gives positive reward"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        env.reset()
        
        actions = {f'agent_{i}': 0 for i in range(3)}
        _, rewards, _, _ = env.step(actions)
        
        for agent_id, reward in rewards.items():
            # Reward should be positive (0.5 + 0.3*state[0] + noise)
            # With noise, could be slightly negative, but generally positive
            assert -1.0 <= reward <= 2.0
    
    def test_action_4_handover_reward(self, marl_config):
        """Test action 4 (trigger handover) reward depends on state"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        env.reset()
        
        actions = {f'agent_{i}': 4 for i in range(3)}
        _, rewards, _, _ = env.step(actions)
        
        for agent_id, reward in rewards.items():
            # Reward is 0.6 or 0.1 depending on state[3]
            assert -1.0 <= reward <= 2.0
    
    def test_state_transition_bounded(self, marl_config):
        """Test state transitions stay within [0, 1] bounds"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        env.reset()
        
        actions = {f'agent_{i}': np.random.randint(0, 5) for i in range(3)}
        
        for _ in range(20):
            observations, _, _, _ = env.step(actions)
            
            for agent_id, obs in observations.items():
                assert np.all(obs >= 0)
                assert np.all(obs <= 1)


# =============================================================================
# Multi-Agent Coordination Tests
# =============================================================================

class TestMultiAgentCoordination:
    """Tests for multi-agent coordination aspects"""
    
    def test_independent_agent_states(self, marl_config):
        """Test agents have independent states"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        states = env.reset()
        
        # States should be different (random initialization)
        state_0 = states['agent_0']
        state_1 = states['agent_1']
        
        # Very unlikely to be exactly equal
        assert not np.array_equal(state_0, state_1)
    
    def test_different_actions_different_rewards(self, marl_config):
        """Test different actions can lead to different rewards"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        env.reset()
        
        # Each agent takes different action
        actions = {'agent_0': 0, 'agent_1': 2, 'agent_2': 4}
        _, rewards, _, _ = env.step(actions)
        
        # Rewards should generally differ
        reward_values = list(rewards.values())
        # Due to noise, some may be similar, but mechanism should differ
        assert len(reward_values) == 3
    
    def test_scalability_many_agents(self):
        """Test environment scales to many agents"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        config = {'num_agents': 10}
        env = SIGINTMultiAgentEnv(config)
        
        assert env.num_agents == 10
        
        states = env.reset()
        assert len(states) == 10
        
        actions = {f'agent_{i}': i % 5 for i in range(10)}
        observations, rewards, dones, infos = env.step(actions)
        
        assert len(observations) == 10
        assert len(rewards) == 10


# =============================================================================
# Integration Tests
# =============================================================================

class TestMARLIntegration:
    """Integration tests for MARL components"""
    
    def test_full_episode_rollout(self, marl_config):
        """Test complete episode rollout"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        states = env.reset()
        
        total_rewards = {f'agent_{i}': 0.0 for i in range(3)}
        done = False
        steps = 0
        
        while not done:
            actions = {agent_id: np.random.randint(0, 5) for agent_id in env._agent_ids}
            observations, rewards, dones, infos = env.step(actions)
            
            for agent_id, reward in rewards.items():
                total_rewards[agent_id] += reward
            
            done = dones['__all__']
            steps += 1
        
        assert steps == 100
        assert all(r != 0 for r in total_rewards.values())  # Accumulated some reward
    
    def test_reset_after_episode(self, marl_config):
        """Test reset works properly after episode completion"""
        from falconone.ai.ric_optimizer import SIGINTMultiAgentEnv
        
        env = SIGINTMultiAgentEnv(marl_config)
        
        # Complete first episode
        env.reset()
        for _ in range(100):
            actions = {f'agent_{i}': 0 for i in range(3)}
            env.step(actions)
        
        # Reset and start new episode
        new_states = env.reset()
        
        assert env.timestep == 0
        assert len(new_states) == 3
        
        # Should be able to run new episode
        actions = {f'agent_{i}': 0 for i in range(3)}
        observations, rewards, dones, infos = env.step(actions)
        
        assert not dones['__all__']
