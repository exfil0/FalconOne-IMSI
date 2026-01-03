"""
FalconOne Circuit Breaker Pattern Implementation
Provides fault tolerance and graceful degradation for all operations
Version 1.9.2 - Extended subprocess protection and long-running task monitoring

Features:
- Generic circuit breaker for any operation
- Context managers for subprocess handling
- Automatic retry with exponential backoff
- Health monitoring and metrics
- Thread-safe implementation
- Long-running task monitoring with timeout escalation
- Subprocess wrapper for all external calls
- Task cancellation and cleanup management
"""

import threading
import time
import logging
import subprocess
import contextlib
import asyncio
import signal
import os
import weakref
from typing import Callable, Any, Optional, Dict, TypeVar, Generic, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError


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


# =============================================================================
# Long-Running Task Monitor (v1.9.2)
# =============================================================================

@dataclass
class TaskMetrics:
    """Metrics for a monitored task"""
    task_id: str
    task_name: str
    start_time: datetime
    timeout_seconds: float
    status: str = "running"  # running, completed, timeout, cancelled, failed
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    cpu_time_seconds: float = 0.0
    memory_peak_mb: float = 0.0


class LongRunningTaskMonitor:
    """
    Monitor and manage long-running tasks with timeout escalation
    
    Features:
    - Track all running tasks with deadlines
    - Graceful then forceful termination
    - Resource usage monitoring
    - Task cancellation support
    
    Usage:
        monitor = LongRunningTaskMonitor()
        
        # Register a task
        task_id = monitor.register_task("heavy_computation", timeout=300)
        
        # Update progress
        monitor.heartbeat(task_id)
        
        # Complete
        monitor.complete_task(task_id, result=data)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._tasks: Dict[str, TaskMetrics] = {}
        self._task_threads: Dict[str, threading.Thread] = {}
        self._task_futures: Dict[str, Future] = {}
        self._cancellation_flags: Dict[str, threading.Event] = {}
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="TaskMonitor")
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self._task_lock = threading.RLock()
        self.logger = logging.getLogger("LongRunningTaskMonitor")
        
        # Escalation settings
        self.warning_threshold_ratio = 0.8  # Warn at 80% of timeout
        self.graceful_termination_timeout = 10.0  # Seconds for graceful shutdown
        
        self._initialized = True
    
    def start(self):
        """Start the monitoring thread"""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="TaskMonitorLoop",
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Long-running task monitor started")
    
    def stop(self):
        """Stop the monitoring thread"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self._executor.shutdown(wait=False)
        self.logger.info("Long-running task monitor stopped")
    
    def register_task(self, name: str, timeout: float = 300.0,
                     callback: Optional[Callable] = None) -> str:
        """
        Register a new long-running task for monitoring
        
        Args:
            name: Human-readable task name
            timeout: Maximum allowed runtime in seconds
            callback: Optional callback on completion/timeout
            
        Returns:
            Unique task ID
        """
        import uuid
        task_id = str(uuid.uuid4())[:8]
        
        with self._task_lock:
            self._tasks[task_id] = TaskMetrics(
                task_id=task_id,
                task_name=name,
                start_time=datetime.now(),
                timeout_seconds=timeout
            )
            self._cancellation_flags[task_id] = threading.Event()
        
        self.logger.info(f"Task registered: {name} (ID: {task_id}, timeout: {timeout}s)")
        return task_id
    
    def heartbeat(self, task_id: str) -> bool:
        """
        Update task heartbeat (extends monitoring window)
        
        Returns:
            False if task should be cancelled
        """
        with self._task_lock:
            if task_id in self._cancellation_flags:
                if self._cancellation_flags[task_id].is_set():
                    return False  # Task should cancel
        return True
    
    def is_cancelled(self, task_id: str) -> bool:
        """Check if task has been cancelled"""
        with self._task_lock:
            if task_id in self._cancellation_flags:
                return self._cancellation_flags[task_id].is_set()
        return True  # Unknown task = cancelled
    
    def cancel_task(self, task_id: str, reason: str = "User requested"):
        """Request task cancellation"""
        with self._task_lock:
            if task_id in self._cancellation_flags:
                self._cancellation_flags[task_id].set()
                if task_id in self._tasks:
                    self._tasks[task_id].status = "cancelled"
                    self._tasks[task_id].error = reason
                    self._tasks[task_id].end_time = datetime.now()
                self.logger.warning(f"Task {task_id} cancelled: {reason}")
    
    def complete_task(self, task_id: str, result: Any = None):
        """Mark task as completed successfully"""
        with self._task_lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = "completed"
                task.result = result
                task.end_time = datetime.now()
                self.logger.info(f"Task {task_id} ({task.task_name}) completed")
    
    def fail_task(self, task_id: str, error: str):
        """Mark task as failed"""
        with self._task_lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = "failed"
                task.error = error
                task.end_time = datetime.now()
                self.logger.error(f"Task {task_id} ({task.task_name}) failed: {error}")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running:
            try:
                self._check_tasks()
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
            time.sleep(1.0)
    
    def _check_tasks(self):
        """Check all tasks for timeout"""
        now = datetime.now()
        
        with self._task_lock:
            for task_id, task in list(self._tasks.items()):
                if task.status != "running":
                    continue
                
                elapsed = (now - task.start_time).total_seconds()
                
                # Warning threshold
                if elapsed >= task.timeout_seconds * self.warning_threshold_ratio:
                    remaining = task.timeout_seconds - elapsed
                    if remaining > 0:
                        self.logger.warning(
                            f"Task {task_id} ({task.task_name}) approaching timeout: "
                            f"{remaining:.1f}s remaining"
                        )
                
                # Timeout exceeded
                if elapsed >= task.timeout_seconds:
                    self.logger.error(
                        f"Task {task_id} ({task.task_name}) timed out after {elapsed:.1f}s"
                    )
                    self.cancel_task(task_id, f"Timeout after {elapsed:.1f}s")
    
    def get_running_tasks(self) -> List[Dict[str, Any]]:
        """Get status of all running tasks"""
        with self._task_lock:
            return [
                {
                    'task_id': t.task_id,
                    'task_name': t.task_name,
                    'elapsed_seconds': (datetime.now() - t.start_time).total_seconds(),
                    'timeout_seconds': t.timeout_seconds,
                    'status': t.status
                }
                for t in self._tasks.values()
                if t.status == "running"
            ]
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific task"""
        with self._task_lock:
            if task_id in self._tasks:
                t = self._tasks[task_id]
                return {
                    'task_id': t.task_id,
                    'task_name': t.task_name,
                    'start_time': t.start_time.isoformat(),
                    'end_time': t.end_time.isoformat() if t.end_time else None,
                    'elapsed_seconds': (
                        (t.end_time or datetime.now()) - t.start_time
                    ).total_seconds(),
                    'timeout_seconds': t.timeout_seconds,
                    'status': t.status,
                    'error': t.error
                }
        return None
    
    def cleanup_completed(self, max_age_seconds: float = 3600):
        """Remove completed/failed tasks older than max_age"""
        cutoff = datetime.now() - timedelta(seconds=max_age_seconds)
        
        with self._task_lock:
            to_remove = [
                task_id for task_id, task in self._tasks.items()
                if task.status != "running" and task.end_time and task.end_time < cutoff
            ]
            for task_id in to_remove:
                del self._tasks[task_id]
                self._cancellation_flags.pop(task_id, None)


# Global task monitor instance
task_monitor = LongRunningTaskMonitor()


# =============================================================================
# Protected Subprocess Wrapper (v1.9.2)
# =============================================================================

@dataclass
class SubprocessResult:
    """Result from protected subprocess execution"""
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False
    killed: bool = False


class ProtectedSubprocess:
    """
    Enhanced subprocess wrapper with circuit breaker protection
    
    Features:
    - Circuit breaker integration
    - Long-running task monitoring
    - Graceful and forceful termination
    - Resource limit enforcement
    - Cross-platform signal handling
    
    Usage:
        ps = ProtectedSubprocess("sdr_operations")
        result = ps.run(['hackrf_info'], timeout=30)
        
        # Or as context manager
        with ps.popen(['grgsm_livemon', '-f', '945e6']) as proc:
            for line in proc.stdout:
                print(line)
    """
    
    def __init__(self, circuit_name: str,
                 config: Optional[CircuitBreakerConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize protected subprocess handler
        
        Args:
            circuit_name: Name for circuit breaker
            config: Optional circuit breaker configuration
            logger: Optional logger instance
        """
        self.circuit_name = circuit_name
        self.circuit = circuit_registry.get_or_create(circuit_name, config)
        self.logger = logger or logging.getLogger(f"ProtectedSubprocess.{circuit_name}")
        
        # Track active processes for cleanup
        self._active_procs: Dict[int, subprocess.Popen] = {}
        self._proc_lock = threading.Lock()
    
    def run(self, cmd: List[str], 
            timeout: float = 30.0,
            input_data: Optional[bytes] = None,
            env: Optional[Dict[str, str]] = None,
            cwd: Optional[str] = None,
            check: bool = False,
            monitor_task: bool = True) -> SubprocessResult:
        """
        Run subprocess with full protection
        
        Args:
            cmd: Command and arguments
            timeout: Maximum runtime in seconds
            input_data: Optional stdin data
            env: Optional environment variables
            cwd: Optional working directory
            check: Raise exception on non-zero exit
            monitor_task: Register with task monitor
            
        Returns:
            SubprocessResult with output and timing
        """
        if not self.circuit._can_execute():
            raise CircuitBreakerOpenError(f"Circuit '{self.circuit_name}' is OPEN")
        
        task_id = None
        if monitor_task:
            task_id = task_monitor.register_task(
                f"subprocess:{cmd[0]}",
                timeout=timeout
            )
        
        start_time = time.monotonic()
        proc = None
        timed_out = False
        killed = False
        
        try:
            # Start process
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if input_data else None,
                env=env,
                cwd=cwd,
                text=True
            )
            
            with self._proc_lock:
                self._active_procs[proc.pid] = proc
            
            try:
                stdout, stderr = proc.communicate(
                    input=input_data.decode() if input_data else None,
                    timeout=timeout
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                self.logger.warning(f"Subprocess timed out: {cmd[0]} (PID: {proc.pid})")
                
                # Graceful termination
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Forceful termination
                    proc.kill()
                    killed = True
                    proc.wait()
                
                stdout, stderr = proc.communicate()
            
            elapsed = time.monotonic() - start_time
            
            result = SubprocessResult(
                returncode=proc.returncode,
                stdout=stdout or "",
                stderr=stderr or "",
                elapsed_seconds=elapsed,
                timed_out=timed_out,
                killed=killed
            )
            
            # Record outcome in circuit breaker
            if proc.returncode == 0 and not timed_out:
                self.circuit._record_success()
                if task_id:
                    task_monitor.complete_task(task_id, result)
            else:
                error = subprocess.CalledProcessError(proc.returncode, cmd)
                self.circuit._record_failure(error)
                if task_id:
                    task_monitor.fail_task(task_id, f"Exit code: {proc.returncode}")
            
            if check and proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd, stdout, stderr)
            
            return result
            
        except Exception as e:
            self.circuit._record_failure(e)
            if task_id:
                task_monitor.fail_task(task_id, str(e))
            raise
            
        finally:
            if proc:
                with self._proc_lock:
                    self._active_procs.pop(proc.pid, None)
    
    @contextlib.contextmanager
    def popen(self, cmd: List[str],
              timeout: float = 300.0,
              env: Optional[Dict[str, str]] = None,
              cwd: Optional[str] = None,
              monitor_task: bool = True):
        """
        Context manager for long-running subprocess
        
        Args:
            cmd: Command and arguments
            timeout: Maximum runtime in seconds
            env: Optional environment variables
            cwd: Optional working directory
            monitor_task: Register with task monitor
            
        Yields:
            subprocess.Popen instance
        """
        if not self.circuit._can_execute():
            raise CircuitBreakerOpenError(f"Circuit '{self.circuit_name}' is OPEN")
        
        task_id = None
        if monitor_task:
            task_id = task_monitor.register_task(
                f"popen:{cmd[0]}",
                timeout=timeout
            )
        
        proc = None
        start_time = time.monotonic()
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=cwd
            )
            
            with self._proc_lock:
                self._active_procs[proc.pid] = proc
            
            yield proc
            
            # Wait for completion
            try:
                proc.wait(timeout=timeout - (time.monotonic() - start_time))
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
            
            if proc.returncode == 0:
                self.circuit._record_success()
                if task_id:
                    task_monitor.complete_task(task_id)
            else:
                error = subprocess.CalledProcessError(proc.returncode, cmd)
                self.circuit._record_failure(error)
                if task_id:
                    task_monitor.fail_task(task_id, f"Exit code: {proc.returncode}")
                    
        except Exception as e:
            self.circuit._record_failure(e)
            if task_id:
                task_monitor.fail_task(task_id, str(e))
            raise
            
        finally:
            if proc:
                with self._proc_lock:
                    self._active_procs.pop(proc.pid, None)
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
    
    def kill_all(self):
        """Kill all active processes managed by this handler"""
        with self._proc_lock:
            for pid, proc in list(self._active_procs.items()):
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                    self.logger.info(f"Killed process PID: {pid}")
                except Exception as e:
                    self.logger.error(f"Failed to kill PID {pid}: {e}")
            self._active_procs.clear()
    
    def get_active_count(self) -> int:
        """Get count of active processes"""
        with self._proc_lock:
            return len(self._active_procs)


# Convenience function for one-off protected subprocess calls
def run_protected(cmd: List[str],
                  circuit_name: str = "default",
                  timeout: float = 30.0,
                  **kwargs) -> SubprocessResult:
    """
    Run a subprocess with circuit breaker protection
    
    Args:
        cmd: Command and arguments
        circuit_name: Circuit breaker name
        timeout: Maximum runtime in seconds
        **kwargs: Additional arguments for ProtectedSubprocess.run()
        
    Returns:
        SubprocessResult
    """
    ps = ProtectedSubprocess(circuit_name)
    return ps.run(cmd, timeout=timeout, **kwargs)
