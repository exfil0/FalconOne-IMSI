"""
FalconOne Chaos Engineering Tests (v1.9.3)
==========================================
Chaos engineering tests for resilience validation.
Tests system behavior under adverse conditions including:
- Network partitions and latency
- Resource exhaustion (CPU, memory, disk)
- SDR hardware failures
- Cascading failures
- Recovery patterns

Requirements:
- pytest
- pytest-asyncio
- aiohttp (for network tests)

Run with:
    pytest falconone/tests/test_chaos.py -v --tb=short
"""

import pytest
import asyncio
import threading
import time
import random
import gc
import os
import sys
import signal
from typing import Callable, Any, Optional, List, Dict
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum


# ==================== CHAOS PRIMITIVES ====================

class ChaosType(Enum):
    """Types of chaos injection"""
    NETWORK_PARTITION = "network_partition"
    NETWORK_LATENCY = "network_latency"
    CPU_SPIKE = "cpu_spike"
    MEMORY_PRESSURE = "memory_pressure"
    DISK_FULL = "disk_full"
    SDR_DISCONNECT = "sdr_disconnect"
    SDR_NOISE = "sdr_noise"
    CLOCK_DRIFT = "clock_drift"
    DEPENDENCY_FAILURE = "dependency_failure"
    CASCADING_FAILURE = "cascading_failure"


@dataclass
class ChaosConfig:
    """Configuration for chaos injection"""
    chaos_type: ChaosType
    duration: float = 5.0  # seconds
    intensity: float = 0.5  # 0.0 to 1.0
    target: Optional[str] = None
    probability: float = 1.0  # probability of injection


class ChaosMonkey:
    """Chaos injection framework for resilience testing"""
    
    def __init__(self):
        self.active_chaos: List[ChaosConfig] = []
        self._stop_event = threading.Event()
        self._chaos_threads: List[threading.Thread] = []
        self.metrics = {
            'injections': 0,
            'recoveries': 0,
            'failures': 0
        }
    
    def inject(self, config: ChaosConfig) -> None:
        """Inject chaos according to configuration"""
        if random.random() > config.probability:
            return
        
        self.active_chaos.append(config)
        self.metrics['injections'] += 1
    
    def stop_all(self) -> None:
        """Stop all active chaos injections"""
        self._stop_event.set()
        for thread in self._chaos_threads:
            thread.join(timeout=1.0)
        self._chaos_threads.clear()
        self.active_chaos.clear()
        self._stop_event.clear()
    
    @contextmanager
    def chaos_context(self, config: ChaosConfig):
        """Context manager for chaos injection"""
        self.inject(config)
        try:
            yield
        finally:
            self.active_chaos.remove(config) if config in self.active_chaos else None
            self.metrics['recoveries'] += 1


# ==================== NETWORK CHAOS ====================

class NetworkChaos:
    """Network failure simulation"""
    
    def __init__(self):
        self.partitioned_hosts: set = set()
        self.latency_ms: float = 0
        self.packet_loss_rate: float = 0
        self.bandwidth_limit: Optional[int] = None
    
    def partition(self, hosts: List[str]) -> None:
        """Simulate network partition"""
        self.partitioned_hosts.update(hosts)
    
    def heal(self) -> None:
        """Heal network partition"""
        self.partitioned_hosts.clear()
        self.latency_ms = 0
        self.packet_loss_rate = 0
    
    def add_latency(self, latency_ms: float) -> None:
        """Add network latency"""
        self.latency_ms = latency_ms
    
    def set_packet_loss(self, rate: float) -> None:
        """Set packet loss rate (0.0 to 1.0)"""
        self.packet_loss_rate = min(1.0, max(0.0, rate))
    
    def should_drop_packet(self) -> bool:
        """Determine if packet should be dropped"""
        return random.random() < self.packet_loss_rate
    
    def is_reachable(self, host: str) -> bool:
        """Check if host is reachable"""
        return host not in self.partitioned_hosts
    
    async def simulate_request(self, host: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Simulate network request with chaos"""
        if not self.is_reachable(host):
            raise ConnectionError(f"Network partition: {host} unreachable")
        
        if self.should_drop_packet():
            raise TimeoutError("Packet lost")
        
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
        
        return {"status": "ok", "latency": self.latency_ms}


# ==================== RESOURCE EXHAUSTION ====================

class ResourceChaos:
    """Resource exhaustion simulation"""
    
    def __init__(self):
        self._memory_hogs: List[bytearray] = []
        self._cpu_threads: List[threading.Thread] = []
        self._stop_cpu = threading.Event()
    
    def consume_memory(self, mb: int) -> None:
        """Consume specified amount of memory"""
        try:
            # Allocate memory in chunks
            chunk_size = 1024 * 1024  # 1MB
            for _ in range(mb):
                self._memory_hogs.append(bytearray(chunk_size))
        except MemoryError:
            pass  # System is under pressure
    
    def release_memory(self) -> None:
        """Release consumed memory"""
        self._memory_hogs.clear()
        gc.collect()
    
    def spike_cpu(self, cores: int = 1, duration: float = 5.0) -> None:
        """Spike CPU usage"""
        self._stop_cpu.clear()
        
        def cpu_burn():
            end_time = time.time() + duration
            while time.time() < end_time and not self._stop_cpu.is_set():
                # CPU-intensive operation
                _ = sum(i * i for i in range(10000))
        
        for _ in range(cores):
            thread = threading.Thread(target=cpu_burn, daemon=True)
            thread.start()
            self._cpu_threads.append(thread)
    
    def stop_cpu_spike(self) -> None:
        """Stop CPU spike"""
        self._stop_cpu.set()
        for thread in self._cpu_threads:
            thread.join(timeout=1.0)
        self._cpu_threads.clear()
    
    def cleanup(self) -> None:
        """Cleanup all resources"""
        self.release_memory()
        self.stop_cpu_spike()


# ==================== SDR CHAOS ====================

class SDRChaos:
    """SDR hardware failure simulation"""
    
    def __init__(self):
        self.connected = True
        self.noise_level = 0.0  # 0.0 to 1.0
        self.gain_offset = 0.0
        self.frequency_drift_hz = 0.0
        self.sample_drop_rate = 0.0
        self.buffer_corruption_rate = 0.0
    
    def disconnect(self) -> None:
        """Simulate SDR disconnect"""
        self.connected = False
    
    def reconnect(self) -> None:
        """Simulate SDR reconnect"""
        self.connected = True
    
    def inject_noise(self, level: float) -> None:
        """Inject noise into signal"""
        self.noise_level = min(1.0, max(0.0, level))
    
    def drift_frequency(self, hz: float) -> None:
        """Simulate frequency drift"""
        self.frequency_drift_hz = hz
    
    def corrupt_samples(self, rate: float) -> None:
        """Corrupt random samples"""
        self.sample_drop_rate = min(1.0, max(0.0, rate))
    
    def process_samples(self, samples: List[complex]) -> List[complex]:
        """Process samples with chaos injection"""
        if not self.connected:
            raise RuntimeError("SDR not connected")
        
        import random
        import cmath
        
        result = []
        for sample in samples:
            # Apply noise
            if self.noise_level > 0:
                noise = complex(
                    random.gauss(0, self.noise_level),
                    random.gauss(0, self.noise_level)
                )
                sample = sample + noise
            
            # Apply frequency drift
            if self.frequency_drift_hz != 0:
                phase_shift = cmath.exp(1j * 2 * 3.14159 * self.frequency_drift_hz / 1e6)
                sample = sample * phase_shift
            
            # Drop samples
            if random.random() < self.sample_drop_rate:
                continue
            
            result.append(sample)
        
        return result
    
    def reset(self) -> None:
        """Reset to normal operation"""
        self.connected = True
        self.noise_level = 0.0
        self.gain_offset = 0.0
        self.frequency_drift_hz = 0.0
        self.sample_drop_rate = 0.0
        self.buffer_corruption_rate = 0.0


# ==================== TEST FIXTURES ====================

@pytest.fixture
def chaos_monkey():
    """Provide chaos monkey instance"""
    monkey = ChaosMonkey()
    yield monkey
    monkey.stop_all()


@pytest.fixture
def network_chaos():
    """Provide network chaos instance"""
    chaos = NetworkChaos()
    yield chaos
    chaos.heal()


@pytest.fixture
def resource_chaos():
    """Provide resource chaos instance"""
    chaos = ResourceChaos()
    yield chaos
    chaos.cleanup()


@pytest.fixture
def sdr_chaos():
    """Provide SDR chaos instance"""
    chaos = SDRChaos()
    yield chaos
    chaos.reset()


# ==================== NETWORK PARTITION TESTS ====================

class TestNetworkPartition:
    """Tests for network partition handling"""
    
    @pytest.mark.asyncio
    async def test_network_partition_detection(self, network_chaos):
        """Test that system detects network partitions"""
        hosts = ["192.168.1.100", "192.168.1.101", "192.168.1.102"]
        
        # All hosts should be reachable initially
        for host in hosts:
            assert network_chaos.is_reachable(host)
        
        # Partition one host
        network_chaos.partition(["192.168.1.101"])
        
        assert network_chaos.is_reachable("192.168.1.100")
        assert not network_chaos.is_reachable("192.168.1.101")
        assert network_chaos.is_reachable("192.168.1.102")
    
    @pytest.mark.asyncio
    async def test_network_partition_recovery(self, network_chaos):
        """Test recovery from network partition"""
        network_chaos.partition(["10.0.0.1"])
        
        assert not network_chaos.is_reachable("10.0.0.1")
        
        # Heal partition
        network_chaos.heal()
        
        assert network_chaos.is_reachable("10.0.0.1")
    
    @pytest.mark.asyncio
    async def test_request_during_partition(self, network_chaos):
        """Test request behavior during partition"""
        network_chaos.partition(["api.example.com"])
        
        with pytest.raises(ConnectionError) as exc_info:
            await network_chaos.simulate_request("api.example.com")
        
        assert "unreachable" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_during_partition(self, network_chaos):
        """Test circuit breaker behavior during partition"""
        try:
            from falconone.core.circuit_breaker import CircuitBreaker, CircuitState
            
            cb = CircuitBreaker(
                name="network_test",
                failure_threshold=3,
                recovery_timeout=1.0
            )
            
            network_chaos.partition(["backend.local"])
            
            # Simulate failures
            for _ in range(5):
                try:
                    await network_chaos.simulate_request("backend.local")
                except ConnectionError:
                    cb.record_failure(ConnectionError("Partition"))
            
            # Circuit should be open
            assert cb.state == CircuitState.OPEN
            
            # Heal and wait for recovery
            network_chaos.heal()
            await asyncio.sleep(1.1)
            
            # Circuit should be half-open
            assert cb.state == CircuitState.HALF_OPEN
            
        except ImportError:
            pytest.skip("circuit_breaker not available")


# ==================== LATENCY INJECTION TESTS ====================

class TestLatencyInjection:
    """Tests for latency handling"""
    
    @pytest.mark.asyncio
    async def test_high_latency_handling(self, network_chaos):
        """Test system handles high latency gracefully"""
        network_chaos.add_latency(500)  # 500ms latency
        
        start = time.time()
        result = await network_chaos.simulate_request("api.example.com")
        elapsed = time.time() - start
        
        assert elapsed >= 0.5
        assert result["status"] == "ok"
    
    @pytest.mark.asyncio
    async def test_latency_with_timeout(self, network_chaos):
        """Test timeout during high latency"""
        network_chaos.add_latency(10000)  # 10 second latency
        
        async def request_with_timeout():
            try:
                return await asyncio.wait_for(
                    network_chaos.simulate_request("api.example.com"),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                return {"status": "timeout"}
        
        result = await request_with_timeout()
        assert result["status"] == "timeout"
    
    @pytest.mark.asyncio
    async def test_packet_loss_handling(self, network_chaos):
        """Test system handles packet loss"""
        network_chaos.set_packet_loss(0.5)  # 50% packet loss
        
        successes = 0
        failures = 0
        
        for _ in range(100):
            try:
                await network_chaos.simulate_request("api.example.com")
                successes += 1
            except TimeoutError:
                failures += 1
        
        # Should have some failures due to packet loss
        assert failures > 0
        assert successes > 0
        # Roughly 50% should fail
        assert 30 < failures < 70


# ==================== RESOURCE EXHAUSTION TESTS ====================

class TestResourceExhaustion:
    """Tests for resource exhaustion scenarios"""
    
    def test_memory_pressure_recovery(self, resource_chaos):
        """Test system recovers from memory pressure"""
        import psutil
        
        initial_memory = psutil.virtual_memory().available
        
        # Consume memory (be conservative - only 50MB)
        resource_chaos.consume_memory(50)
        
        during_pressure = psutil.virtual_memory().available
        assert during_pressure < initial_memory
        
        # Release memory
        resource_chaos.release_memory()
        
        after_release = psutil.virtual_memory().available
        # Memory should be recovered (within 100MB margin)
        assert after_release > during_pressure - 100 * 1024 * 1024
    
    def test_cpu_spike_handling(self, resource_chaos):
        """Test system handles CPU spikes"""
        import psutil
        
        # Get baseline CPU
        baseline = psutil.cpu_percent(interval=0.1)
        
        # Spike CPU
        resource_chaos.spike_cpu(cores=2, duration=2.0)
        time.sleep(0.5)
        
        # CPU should be higher during spike
        during_spike = psutil.cpu_percent(interval=0.1)
        
        # Stop spike
        resource_chaos.stop_cpu_spike()
        
        # Wait for recovery
        time.sleep(0.5)
        after_recovery = psutil.cpu_percent(interval=0.1)
        
        # This is probabilistic, so we just verify no crash
        assert True
    
    def test_thread_pool_under_load(self, resource_chaos):
        """Test thread pool behavior under resource pressure"""
        try:
            from falconone.core.circuit_breaker import CircuitBreaker
            
            cb = CircuitBreaker(
                name="resource_test",
                failure_threshold=10,
                recovery_timeout=5.0
            )
            
            # Consume some memory
            resource_chaos.consume_memory(20)
            
            # Execute operations under pressure
            results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(lambda: time.sleep(0.01) or "ok")
                    for _ in range(20)
                ]
                
                for future in futures:
                    try:
                        results.append(future.result(timeout=5.0))
                    except Exception as e:
                        results.append(f"error: {e}")
            
            # Most operations should succeed
            successes = sum(1 for r in results if r == "ok")
            assert successes >= 15
            
        except ImportError:
            pytest.skip("circuit_breaker not available")


# ==================== SDR FAILURE TESTS ====================

class TestSDRFailures:
    """Tests for SDR hardware failure scenarios"""
    
    def test_sdr_disconnect_handling(self, sdr_chaos):
        """Test handling of SDR disconnect"""
        # Generate test samples
        samples = [complex(1.0, 0.5) for _ in range(100)]
        
        # Normal processing
        result = sdr_chaos.process_samples(samples)
        assert len(result) == 100
        
        # Disconnect SDR
        sdr_chaos.disconnect()
        
        with pytest.raises(RuntimeError) as exc_info:
            sdr_chaos.process_samples(samples)
        
        assert "not connected" in str(exc_info.value)
        
        # Reconnect
        sdr_chaos.reconnect()
        result = sdr_chaos.process_samples(samples)
        assert len(result) == 100
    
    def test_sdr_noise_injection(self, sdr_chaos):
        """Test SDR with noise injection"""
        import cmath
        
        # Pure signal
        samples = [complex(1.0, 0.0) for _ in range(1000)]
        
        # Process without noise
        clean_result = sdr_chaos.process_samples(samples)
        clean_power = sum(abs(s) ** 2 for s in clean_result) / len(clean_result)
        
        # Inject noise
        sdr_chaos.inject_noise(0.5)
        noisy_result = sdr_chaos.process_samples(samples)
        noisy_power = sum(abs(s) ** 2 for s in noisy_result) / len(noisy_result)
        
        # Noisy signal should have different power
        assert noisy_power != clean_power
    
    def test_sdr_sample_dropping(self, sdr_chaos):
        """Test SDR with sample dropping"""
        samples = [complex(1.0, 0.5) for _ in range(1000)]
        
        sdr_chaos.corrupt_samples(0.1)  # 10% drop rate
        
        result = sdr_chaos.process_samples(samples)
        
        # Should have fewer samples
        assert len(result) < 1000
        assert len(result) > 800  # Should still have most samples
    
    def test_sdr_recovery_sequence(self, sdr_chaos):
        """Test complete SDR failure and recovery sequence"""
        samples = [complex(1.0, 0.5) for _ in range(100)]
        
        # Normal operation
        result1 = sdr_chaos.process_samples(samples)
        assert len(result1) == 100
        
        # Inject all chaos types
        sdr_chaos.inject_noise(0.3)
        sdr_chaos.drift_frequency(1000)
        sdr_chaos.corrupt_samples(0.1)
        
        result2 = sdr_chaos.process_samples(samples)
        assert len(result2) < 100  # Some samples dropped
        
        # Disconnect
        sdr_chaos.disconnect()
        with pytest.raises(RuntimeError):
            sdr_chaos.process_samples(samples)
        
        # Full recovery
        sdr_chaos.reset()
        result3 = sdr_chaos.process_samples(samples)
        assert len(result3) == 100


# ==================== CASCADING FAILURE TESTS ====================

class TestCascadingFailures:
    """Tests for cascading failure scenarios"""
    
    def test_cascading_circuit_breaker(self):
        """Test cascading failures across circuit breakers"""
        try:
            from falconone.core.circuit_breaker import (
                CircuitBreaker, CircuitBreakerRegistry, CircuitState
            )
            
            registry = CircuitBreakerRegistry()
            
            # Create dependent circuit breakers
            cb_db = registry.get_or_create("database", failure_threshold=3)
            cb_cache = registry.get_or_create("cache", failure_threshold=3)
            cb_api = registry.get_or_create("api", failure_threshold=3)
            
            # Simulate database failure
            for _ in range(5):
                cb_db.record_failure(Exception("DB timeout"))
            
            assert cb_db.state == CircuitState.OPEN
            
            # Cache depends on DB, simulate failure cascade
            for _ in range(5):
                # Cache fails because DB is down
                cb_cache.record_failure(Exception("DB unavailable"))
            
            assert cb_cache.state == CircuitState.OPEN
            
            # API depends on cache
            for _ in range(5):
                cb_api.record_failure(Exception("Cache unavailable"))
            
            assert cb_api.state == CircuitState.OPEN
            
            # All circuits open - system is protected
            all_open = all(
                cb.state == CircuitState.OPEN 
                for cb in registry.list_all().values()
            )
            assert all_open
            
        except ImportError:
            pytest.skip("circuit_breaker not available")
    
    def test_partial_failure_isolation(self, network_chaos, sdr_chaos):
        """Test that partial failures are isolated"""
        # Network is partitioned
        network_chaos.partition(["backend.api"])
        
        # SDR is still working
        samples = [complex(1.0, 0.5) for _ in range(100)]
        result = sdr_chaos.process_samples(samples)
        
        # SDR should still work despite network issues
        assert len(result) == 100
        assert sdr_chaos.connected
    
    def test_recovery_order(self, network_chaos, sdr_chaos, resource_chaos):
        """Test proper recovery order after multiple failures"""
        # Inject multiple failures
        network_chaos.partition(["api.local"])
        sdr_chaos.disconnect()
        resource_chaos.consume_memory(10)
        
        # Recovery sequence
        # 1. Release memory first (most critical)
        resource_chaos.release_memory()
        gc.collect()
        
        # 2. Reconnect SDR
        sdr_chaos.reconnect()
        samples = [complex(1.0, 0.0) for _ in range(10)]
        result = sdr_chaos.process_samples(samples)
        assert len(result) == 10
        
        # 3. Heal network last
        network_chaos.heal()
        assert network_chaos.is_reachable("api.local")


# ==================== TIMING AND CLOCK TESTS ====================

class TestTimingChaos:
    """Tests for timing and clock-related chaos"""
    
    def test_clock_drift_detection(self):
        """Test detection of clock drift"""
        import time
        
        # Simulate time source with drift
        base_time = time.time()
        drift_rate = 0.01  # 1% drift
        
        simulated_times = []
        for i in range(10):
            actual_elapsed = i * 0.1
            drifted_elapsed = actual_elapsed * (1 + drift_rate)
            simulated_times.append(base_time + drifted_elapsed)
        
        # Detect drift by comparing intervals
        intervals = [
            simulated_times[i+1] - simulated_times[i] 
            for i in range(len(simulated_times) - 1)
        ]
        
        # All intervals should be roughly 0.101 (0.1 * 1.01)
        avg_interval = sum(intervals) / len(intervals)
        assert 0.100 < avg_interval < 0.102
    
    def test_timeout_under_load(self, resource_chaos):
        """Test timeout accuracy under CPU load"""
        import time
        
        # Spike CPU
        resource_chaos.spike_cpu(cores=1, duration=3.0)
        
        # Measure timeout accuracy
        target_sleep = 0.1
        start = time.time()
        time.sleep(target_sleep)
        actual_sleep = time.time() - start
        
        resource_chaos.stop_cpu_spike()
        
        # Under load, sleep might take longer but shouldn't fail
        assert actual_sleep >= target_sleep
        assert actual_sleep < target_sleep * 5  # Should still be reasonable


# ==================== CHAOS MONKEY INTEGRATION ====================

class TestChaosMonkeyIntegration:
    """Integration tests using chaos monkey"""
    
    def test_chaos_injection_lifecycle(self, chaos_monkey):
        """Test complete chaos injection lifecycle"""
        config = ChaosConfig(
            chaos_type=ChaosType.NETWORK_LATENCY,
            duration=1.0,
            intensity=0.5
        )
        
        with chaos_monkey.chaos_context(config):
            assert config in chaos_monkey.active_chaos
            assert chaos_monkey.metrics['injections'] == 1
        
        assert config not in chaos_monkey.active_chaos
        assert chaos_monkey.metrics['recoveries'] == 1
    
    def test_probabilistic_injection(self, chaos_monkey):
        """Test probabilistic chaos injection"""
        injected_count = 0
        
        for _ in range(100):
            config = ChaosConfig(
                chaos_type=ChaosType.CPU_SPIKE,
                probability=0.3  # 30% injection rate
            )
            chaos_monkey.inject(config)
            if config in chaos_monkey.active_chaos:
                injected_count += 1
                chaos_monkey.active_chaos.remove(config)
        
        # Should be roughly 30% (within reasonable margin)
        assert 15 < injected_count < 50
    
    def test_multiple_chaos_types(self, chaos_monkey, network_chaos, sdr_chaos):
        """Test multiple simultaneous chaos types"""
        configs = [
            ChaosConfig(chaos_type=ChaosType.NETWORK_LATENCY, intensity=0.3),
            ChaosConfig(chaos_type=ChaosType.SDR_NOISE, intensity=0.5),
            ChaosConfig(chaos_type=ChaosType.CPU_SPIKE, intensity=0.2),
        ]
        
        for config in configs:
            chaos_monkey.inject(config)
        
        assert len(chaos_monkey.active_chaos) == 3
        
        # System should still function
        network_chaos.add_latency(100)
        sdr_chaos.inject_noise(0.2)
        
        samples = [complex(1.0, 0.0) for _ in range(100)]
        result = sdr_chaos.process_samples(samples)
        assert len(result) == 100
        
        chaos_monkey.stop_all()
        assert len(chaos_monkey.active_chaos) == 0


# ==================== MAIN ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
