"""
FalconOne Circuit Breaker Pattern Implementation
Provides fault tolerance and graceful degradation for all operations
Version 1.9.1 - Enhanced reliability with context managers

Features:
- Generic circuit breaker for any operation
- Context managers for subprocess handling
- Automatic retry with exponential backoff
- Health monitoring and metrics
- Thread-safe implementation
"""

import threading
import time
import logging
import subprocess
import contextlib
from typing import Callable, Any, Optional, Dict, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    timeout_seconds: float = 60.0       # Time before attempting recovery
    half_open_max_calls: int = 3        # Max calls in half-open state
    excluded_exceptions: tuple = ()     # Exceptions that don't trip breaker


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracking for circuit breaker"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: int = 0


class CircuitBreaker:
    """
    Generic circuit breaker for fault tolerance
    
    Usage:
        cb = CircuitBreaker("sdr_operations", config=CircuitBreakerConfig(failure_threshold=3))
        
        @cb.protect
        def risky_operation():
            # ... operation that might fail
            
        # Or with context manager:
        with cb:
            result = risky_operation()
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None,
                 logger: Optional[logging.Logger] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = logger or logging.getLogger(f"CircuitBreaker.{name}")
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.RLock()
        
        # Metrics
        self.metrics = CircuitBreakerMetrics()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state with automatic timeout check"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if timeout has passed for recovery attempt
                if self._last_failure_time:
                    elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                    if elapsed >= self.config.timeout_seconds:
                        self._transition_to(CircuitState.HALF_OPEN)
            return self._state
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state"""
        old_state = self._state
        self._state = new_state
        self.metrics.state_changes += 1
        
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        
        self.logger.info(f"Circuit '{self.name}' transitioned: {old_state.value} -> {new_state.value}")
    
    def _record_success(self):
        """Record successful call"""
        with self._lock:
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.metrics.last_success_time = datetime.now()
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
    
    def _record_failure(self, exception: Exception):
        """Record failed call"""
        with self._lock:
            # Check if exception is excluded
            if isinstance(exception, self.config.excluded_exceptions):
                return
            
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.metrics.last_failure_time = datetime.now()
            self._last_failure_time = datetime.now()
            self._failure_count += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def _can_execute(self) -> bool:
        """Check if execution is allowed"""
        state = self.state  # Property checks timeout
        
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.OPEN:
            self.metrics.rejected_calls += 1
            return False
        else:  # HALF_OPEN
            with self._lock:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
    
    def protect(self, func: Callable) -> Callable:
        """Decorator to protect a function with circuit breaker"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if not self._can_execute():
            raise CircuitBreakerOpenError(f"Circuit '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise
    
    def __enter__(self):
        """Context manager entry"""
        if not self._can_execute():
            raise CircuitBreakerOpenError(f"Circuit '{self.name}' is OPEN")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type is None:
            self._record_success()
        else:
            self._record_failure(exc_val)
        return False  # Don't suppress exceptions
    
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self.logger.info(f"Circuit '{self.name}' manually reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status and metrics"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self._failure_count,
            'success_count': self._success_count,
            'metrics': {
                'total_calls': self.metrics.total_calls,
                'successful_calls': self.metrics.successful_calls,
                'failed_calls': self.metrics.failed_calls,
                'rejected_calls': self.metrics.rejected_calls,
                'last_failure_time': self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
                'last_success_time': self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
            }
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._breakers: Dict[str, CircuitBreaker] = {}
            return cls._instance
    
    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get existing or create new circuit breaker"""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers"""
        return {name: cb.get_status() for name, cb in self._breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for cb in self._breakers.values():
            cb.reset()


# Global registry instance
circuit_registry = CircuitBreakerRegistry()


@contextlib.contextmanager
def subprocess_context(cmd: list, timeout: float = 30.0, 
                       circuit_name: Optional[str] = None,
                       cleanup_callback: Optional[Callable] = None):
    """
    Context manager for safe subprocess execution with circuit breaker protection
    
    Usage:
        with subprocess_context(['ls', '-la'], timeout=10, circuit_name='file_ops') as proc:
            stdout, stderr = proc.communicate()
    
    Args:
        cmd: Command to execute
        timeout: Timeout in seconds
        circuit_name: Optional circuit breaker name
        cleanup_callback: Optional cleanup function called on error
    
    Yields:
        subprocess.Popen instance
    """
    proc = None
    cb = circuit_registry.get_or_create(circuit_name) if circuit_name else None
    
    try:
        if cb and not cb._can_execute():
            raise CircuitBreakerOpenError(f"Circuit '{circuit_name}' is OPEN")
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        yield proc
        
        # Wait for completion with timeout
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        
        if cb:
            if proc.returncode == 0:
                cb._record_success()
            else:
                cb._record_failure(subprocess.CalledProcessError(proc.returncode, cmd))
        
    except Exception as e:
        if cb:
            cb._record_failure(e)
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait()
        if cleanup_callback:
            try:
                cleanup_callback()
            except Exception as cleanup_error:
                logging.warning(f"Cleanup callback failed: {cleanup_error}")
        raise
    
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def with_retry(max_retries: int = 3, 
               base_delay: float = 1.0,
               max_delay: float = 30.0,
               exponential: bool = True,
               exceptions: tuple = (Exception,)):
    """
    Decorator for automatic retry with exponential backoff
    
    Usage:
        @with_retry(max_retries=3, base_delay=1.0)
        def unstable_operation():
            # ... operation that might fail
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential: Whether to use exponential backoff
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        if exponential:
                            delay = min(base_delay * (2 ** attempt), max_delay)
                        else:
                            delay = base_delay
                        
                        # Add jitter (±25%)
                        import random
                        jitter = delay * 0.25 * (2 * random.random() - 1)
                        delay += jitter
                        
                        logging.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )
                        time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


def protected_loop(circuit_name: str, 
                   interval: float = 1.0,
                   max_consecutive_failures: int = 5):
    """
    Decorator for protected loop execution with circuit breaker
    
    Usage:
        @protected_loop("monitoring_loop", interval=2.0)
        def monitor_iteration():
            # ... single iteration logic
            return should_continue  # Return False to stop
    
    Args:
        circuit_name: Name for the circuit breaker
        interval: Sleep interval between iterations
        max_consecutive_failures: Max failures before breaking loop
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cb = circuit_registry.get_or_create(circuit_name)
            consecutive_failures = 0
            
            while True:
                try:
                    with cb:
                        result = func(*args, **kwargs)
                        consecutive_failures = 0
                        
                        if result is False:
                            break
                        
                except CircuitBreakerOpenError:
                    logging.warning(f"Circuit '{circuit_name}' is open, waiting...")
                    time.sleep(cb.config.timeout_seconds / 2)
                    continue
                    
                except Exception as e:
                    consecutive_failures += 1
                    logging.error(f"Error in {func.__name__}: {e}")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logging.error(f"Max consecutive failures reached in {func.__name__}")
                        raise
                
                time.sleep(interval)
        
        return wrapper
    return decorator


class AsyncCircuitBreaker:
    """
    Async-compatible circuit breaker for async/await patterns
    
    Usage:
        async_cb = AsyncCircuitBreaker("async_ops")
        
        async with async_cb:
            result = await async_operation()
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self._sync_cb = CircuitBreaker(name, config)
        self.name = name
    
    async def __aenter__(self):
        if not self._sync_cb._can_execute():
            raise CircuitBreakerOpenError(f"Circuit '{self.name}' is OPEN")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._sync_cb._record_success()
        else:
            self._sync_cb._record_failure(exc_val)
        return False
    
    @property
    def state(self) -> CircuitState:
        return self._sync_cb.state
    
    def get_status(self) -> Dict[str, Any]:
        return self._sync_cb.get_status()
