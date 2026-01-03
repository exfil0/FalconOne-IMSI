"""
FalconOne Circuit Breaker Pattern Implementation
Provides resilience and fault tolerance for distributed operations

Version 1.9.3: Extended circuit breakers with retry logic and AI adaptation
"""

import time
import threading
import random
import logging
from typing import Callable, Optional, Any, Dict, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import functools


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject all calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Statistics for circuit breaker monitoring"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    avg_response_time_ms: float = 0.0
    state_changes: int = 0


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay_ms: float = 100.0
    max_delay_ms: float = 10000.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    retry_on_exceptions: tuple = (Exception,)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff"""
        delay = self.base_delay_ms * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay_ms)
        
        if self.jitter:
            jitter = delay * self.jitter_factor * random.random()
            delay = delay + jitter
        
        return delay / 1000.0  # Convert to seconds


class CircuitBreaker:
    """
    Circuit breaker with retry logic and adaptive thresholds.
    
    Features:
    - Three-state circuit (CLOSED, OPEN, HALF_OPEN)
    - Configurable failure threshold and recovery timeout
    - Exponential backoff with jitter for retries
    - Adaptive thresholds based on success rate
    - Thread-safe operation
    - Metrics and monitoring support
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
        adaptive: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Identifier for this circuit
            failure_threshold: Failures before opening circuit
            success_threshold: Successes in HALF_OPEN before closing
            timeout_seconds: Time before attempting recovery
            retry_config: Retry configuration (None disables retries)
            adaptive: Enable adaptive threshold adjustment
            logger: Optional logger instance
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.retry_config = retry_config or RetryConfig()
        self.adaptive = adaptive
        self.logger = logger or logging.getLogger(__name__)
        
        # State
        self._state = CircuitState.CLOSED
        self._state_lock = threading.RLock()
        self._opened_at: Optional[float] = None
        
        # Statistics
        self.stats = CircuitStats()
        self._response_times: deque = deque(maxlen=100)
        
        # Adaptive thresholds
        self._initial_failure_threshold = failure_threshold
        self._adaptation_window: deque = deque(maxlen=50)
        
        # Callbacks
        self._on_state_change: List[Callable] = []
        self._on_failure: List[Callable] = []
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        with self._state_lock:
            self._check_recovery()
            return self._state
    
    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN
    
    def _check_recovery(self):
        """Check if circuit should transition to HALF_OPEN"""
        if self._state == CircuitState.OPEN and self._opened_at:
            elapsed = time.time() - self._opened_at
            if elapsed >= self.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state with logging and callbacks"""
        old_state = self._state
        if old_state == new_state:
            return
        
        self._state = new_state
        self.stats.state_changes += 1
        
        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
        elif new_state == CircuitState.CLOSED:
            self._opened_at = None
            self.stats.consecutive_failures = 0
        
        self.logger.info(
            f"Circuit '{self.name}' state change: {old_state.value} -> {new_state.value}"
        )
        
        for callback in self._on_state_change:
            try:
                callback(self.name, old_state, new_state)
            except Exception as e:
                self.logger.error(f"State change callback error: {e}")
    
    def _record_success(self, response_time_ms: float):
        """Record successful call"""
        with self._state_lock:
            self.stats.total_calls += 1
            self.stats.successful_calls += 1
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0
            self.stats.last_success_time = time.time()
            
            self._response_times.append(response_time_ms)
            self.stats.avg_response_time_ms = sum(self._response_times) / len(self._response_times)
            
            if self.adaptive:
                self._adaptation_window.append(True)
                self._adapt_thresholds()
            
            if self._state == CircuitState.HALF_OPEN:
                if self.stats.consecutive_successes >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
    
    def _record_failure(self, exception: Exception):
        """Record failed call"""
        with self._state_lock:
            self.stats.total_calls += 1
            self.stats.failed_calls += 1
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0
            self.stats.last_failure_time = time.time()
            
            if self.adaptive:
                self._adaptation_window.append(False)
                self._adapt_thresholds()
            
            for callback in self._on_failure:
                try:
                    callback(self.name, exception)
                except Exception as e:
                    self.logger.error(f"Failure callback error: {e}")
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self.stats.consecutive_failures >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def _adapt_thresholds(self):
        """Adapt thresholds based on recent success rate"""
        if len(self._adaptation_window) < 20:
            return
        
        success_rate = sum(self._adaptation_window) / len(self._adaptation_window)
        
        # If success rate is high, can be more lenient
        if success_rate > 0.95:
            self.failure_threshold = min(
                self.failure_threshold + 1,
                self._initial_failure_threshold * 2
            )
        # If success rate is low, be more aggressive
        elif success_rate < 0.5:
            self.failure_threshold = max(
                self.failure_threshold - 1,
                2
            )
    
    def _reject_call(self):
        """Reject call when circuit is open"""
        self.stats.rejected_calls += 1
        raise CircuitOpenError(
            f"Circuit '{self.name}' is OPEN - rejecting call"
        )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through the circuit breaker with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
            Exception: If all retries fail
        """
        current_state = self.state
        
        if current_state == CircuitState.OPEN:
            self._reject_call()
        
        last_exception = None
        max_attempts = 1 + (self.retry_config.max_retries if self._state == CircuitState.CLOSED else 0)
        
        for attempt in range(max_attempts):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                response_time_ms = (time.time() - start_time) * 1000
                
                self._record_success(response_time_ms)
                return result
                
            except self.retry_config.retry_on_exceptions as e:
                last_exception = e
                self._record_failure(e)
                
                # Check if circuit just opened
                if self.state == CircuitState.OPEN:
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' opened after failure"
                    ) from e
                
                # Retry with backoff if not last attempt
                if attempt < max_attempts - 1:
                    delay = self.retry_config.calculate_delay(attempt)
                    self.logger.debug(
                        f"Retry {attempt + 1}/{self.retry_config.max_retries} "
                        f"for '{self.name}' after {delay:.2f}s"
                    )
                    time.sleep(delay)
        
        raise last_exception
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for wrapping functions with circuit breaker"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def on_state_change(self, callback: Callable):
        """Register state change callback"""
        self._on_state_change.append(callback)
    
    def on_failure(self, callback: Callable):
        """Register failure callback"""
        self._on_failure.append(callback)
    
    def reset(self):
        """Reset circuit to closed state"""
        with self._state_lock:
            self._transition_to(CircuitState.CLOSED)
            self.stats = CircuitStats()
            self._response_times.clear()
            self._adaptation_window.clear()
            self.failure_threshold = self._initial_failure_threshold
    
    def get_health(self) -> Dict[str, Any]:
        """Get circuit health metrics"""
        return {
            'name': self.name,
            'state': self.state.value,
            'stats': {
                'total_calls': self.stats.total_calls,
                'successful_calls': self.stats.successful_calls,
                'failed_calls': self.stats.failed_calls,
                'rejected_calls': self.stats.rejected_calls,
                'consecutive_failures': self.stats.consecutive_failures,
                'avg_response_time_ms': round(self.stats.avg_response_time_ms, 2),
            },
            'thresholds': {
                'failure_threshold': self.failure_threshold,
                'success_threshold': self.success_threshold,
                'timeout_seconds': self.timeout_seconds,
            },
            'adaptive': self.adaptive,
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._circuits: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def get_or_create(
        self,
        name: str,
        **kwargs
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker"""
        with self._lock:
            if name not in self._circuits:
                self._circuits[name] = CircuitBreaker(name, logger=self.logger, **kwargs)
            return self._circuits[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self._circuits.get(name)
    
    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health metrics for all circuits"""
        return {name: cb.get_health() for name, cb in self._circuits.items()}
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for cb in self._circuits.values():
            cb.reset()


# Global registry
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get or create a circuit breaker from global registry"""
    return _global_registry.get_or_create(name, **kwargs)


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout_seconds: float = 30.0,
    adaptive: bool = False
):
    """
    Decorator to wrap function with circuit breaker.
    
    Usage:
        @circuit_breaker("my_service", failure_threshold=3)
        def call_external_api():
            ...
    """
    def decorator(func: Callable) -> Callable:
        cb = get_circuit_breaker(
            name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout_seconds,
            adaptive=adaptive
        )
        return cb(func)
    return decorator
