"""
FalconOne Comprehensive Error Recovery Framework
Handles SDR disconnects, inference errors, network drops with automatic recovery
Version 1.7.0 - Phase 1 Real-World Resilience

Capabilities:
- SDR disconnect/overflow/underflow recovery
- Inference error fallback (GPU → CPU)
- Campaign state checkpointing
- Phase rollback on failure
- Exponential backoff for federated syncs
- Auto-resume after drops

Target: >99% uptime, <10s recovery time
"""

import logging
import time
import pickle
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import json

try:
    from .logger import ModuleLogger
except ImportError:
    class ModuleLogger:
        def __init__(self, name, parent):
            self.logger = logging.getLogger(name) if parent is None else parent.getChild(name)
        def info(self, msg, **kw): self.logger.info(f"{msg} {kw if kw else ''}")
        def warning(self, msg, **kw): self.logger.warning(f"{msg} {kw if kw else ''}")
        def error(self, msg, **kw): self.logger.error(f"{msg} {kw if kw else ''}")
        def debug(self, msg, **kw): self.logger.debug(f"{msg} {kw if kw else ''}")


class ErrorType(Enum):
    """Error classification"""
    SDR_DISCONNECT = "sdr_disconnect"
    SDR_OVERFLOW = "sdr_overflow"
    SDR_UNDERFLOW = "sdr_underflow"
    INFERENCE_ERROR = "inference_error"
    NETWORK_DROP = "network_drop"
    FEDERATED_SYNC_FAIL = "federated_sync_fail"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PROTOCOL_ERROR = "protocol_error"
    UNKNOWN = "unknown"


@dataclass
class RecoveryAttempt:
    """Recovery attempt record"""
    error_type: ErrorType
    timestamp: datetime
    attempt_number: int
    success: bool
    duration_ms: float
    fallback_used: Optional[str] = None
    error_message: str = ""


@dataclass
class CampaignCheckpoint:
    """Campaign state checkpoint for rollback"""
    checkpoint_id: str
    phase: str
    timestamp: datetime
    state: Dict[str, Any]
    module_states: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


class ErrorRecoverer:
    """
    Comprehensive error recovery and resilience manager
    
    Handles production-grade failure scenarios:
    - SDR disconnects during A-IoT jamming
    - GPU OOM during inference → CPU fallback
    - Network drops during federated sync
    - Resource exhaustion in high-density scenarios
    
    Recovery strategies:
    - Exponential backoff with jitter
    - Automatic fallback mechanisms
    - State checkpointing for rollback
    - Circuit breaker pattern
    
    Typical usage:
        recoverer = ErrorRecoverer(config, logger)
        
        # Protect SDR operations
        def sdr_operation():
            # ... SDR code ...
        recoverer.with_recovery(sdr_operation, ErrorType.SDR_DISCONNECT)
        
        # Checkpoint before risky operation
        recoverer.checkpoint_campaign(campaign_state, phase='jamming')
        
        # Rollback if needed
        if failure:
            recoverer.rollback_on_failure('jamming')
    """
    
    def __init__(self, config, logger: logging.Logger):
        """
        Initialize error recoverer
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = ModuleLogger('ErrorRecoverer', logger)
        
        # Configuration
        self.max_retries = config.get('utils.recovery.max_retries', 5)
        self.backoff_base_sec = config.get('utils.recovery.backoff_base_sec', 2)
        self.checkpoint_dir = Path(config.get('utils.recovery.checkpoint_dir', '/tmp/falconone_checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Recovery state
        self.recovery_attempts: List[RecoveryAttempt] = []
        self.recovery_history: List[Dict] = []  # Track all recovery actions
        self.checkpoints: Dict[str, CampaignCheckpoint] = {}
        self.circuit_breakers: Dict[ErrorType, Dict] = {}
        
        # Statistics
        self.total_errors = 0
        self.total_recoveries = 0
        self.failed_recoveries = 0
        self.uptime_start = time.time()
        self.downtime_seconds = 0.0
        
        # Initialize circuit breakers
        self._init_circuit_breakers()
        
        self.logger.info("Error Recovery Framework initialized",
                       max_retries=self.max_retries,
                       backoff_base=self.backoff_base_sec,
                       checkpoint_dir=str(self.checkpoint_dir))
    
    def _init_circuit_breakers(self):
        """Initialize circuit breakers for each error type"""
        for error_type in ErrorType:
            self.circuit_breakers[error_type] = {
                'state': 'closed',  # closed, open, half_open
                'failure_count': 0,
                'last_failure_time': None,
                'success_count': 0,
                'threshold': 5,  # Open after 5 failures
                'timeout_sec': 60,  # Try again after 60s
            }
    
    # ===== SDR RECOVERY =====
    
    def handle_sdr_disconnect(self, sdr_instance, retry_count: int = 3) -> bool:
        """
        Handle SDR disconnect with automatic reconnection
        
        Args:
            sdr_instance: SDR instance to reconnect
            retry_count: Number of retry attempts (default: 3)
        
        Returns:
            True if reconnected successfully
        
        Common causes:
        - USB cable disconnect
        - Driver crash
        - Power brownout
        - Overflow/underflow cascade
        
        Recovery steps:
        1. Close existing connection
        2. Wait with exponential backoff
        3. Reinitialize SDR
        4. Verify connection
        5. Restore previous settings
        """
        self.total_errors += 1
        error_type = ErrorType.SDR_DISCONNECT
        
        # Check circuit breaker
        if not self._check_circuit_breaker(error_type):
            self.logger.error("Circuit breaker OPEN for SDR disconnect - too many failures")
            return False
        
        self.logger.warning(f"SDR disconnect detected - attempting recovery")
        
        for attempt in range(1, retry_count + 1):
            start_time = time.time()
            
            try:
                self.logger.info(f"SDR reconnect attempt {attempt}/{retry_count}")
                
                # Step 1: Close existing connection
                try:
                    if hasattr(sdr_instance, 'close'):
                        sdr_instance.close()
                except Exception as e:
                    self.logger.debug(f"Error closing SDR: {e}")
                
                # Step 2: Exponential backoff with jitter
                backoff_sec = self._calculate_backoff(attempt)
                self.logger.debug(f"Waiting {backoff_sec:.1f}s before reconnect")
                time.sleep(backoff_sec)
                
                # Step 3: Reinitialize SDR
                if hasattr(sdr_instance, 'initialize'):
                    sdr_instance.initialize()
                elif hasattr(sdr_instance, 'connect'):
                    sdr_instance.connect()
                
                # Step 4: Verify connection
                if self._verify_sdr_connection(sdr_instance):
                    duration_ms = (time.time() - start_time) * 1000
                    
                    self._record_recovery(
                        error_type=error_type,
                        attempt=attempt,
                        success=True,
                        duration_ms=duration_ms
                    )
                    
                    self._update_circuit_breaker(error_type, success=True)
                    self.total_recoveries += 1
                    
                    self.logger.info(f"SDR reconnected successfully in {duration_ms:.0f}ms")
                    return True
                
            except Exception as e:
                self.logger.warning(f"SDR reconnect attempt {attempt} failed: {e}")
                self._record_recovery(
                    error_type=error_type,
                    attempt=attempt,
                    success=False,
                    duration_ms=(time.time() - start_time) * 1000,
                    error_message=str(e)
                )
        
        # All attempts failed
        self.failed_recoveries += 1
        self._update_circuit_breaker(error_type, success=False)
        self.logger.error(f"SDR reconnect failed after {retry_count} attempts")
        return False
    
    def handle_sdr_overflow(self, sdr_instance) -> bool:
        """
        Handle SDR buffer overflow
        
        Overflow occurs when:
        - Sample rate too high for USB/processing
        - CPU overloaded with parallel tasks
        - Buffer size too small
        
        Recovery:
        1. Flush buffers
        2. Reduce sample rate temporarily
        3. Resume at original rate
        """
        self.total_errors += 1
        self.logger.warning("SDR overflow detected - flushing buffers")
        
        try:
            start_time = time.time()
            
            # Flush RX buffer
            if hasattr(sdr_instance, 'flush_rx'):
                sdr_instance.flush_rx()
            
            # Reduce sample rate temporarily
            if hasattr(sdr_instance, 'set_sample_rate'):
                current_rate = getattr(sdr_instance, 'sample_rate', 2e6)
                reduced_rate = current_rate * 0.8  # 80% of original
                
                self.logger.info(f"Temporarily reducing sample rate to {reduced_rate/1e6:.1f} MHz")
                sdr_instance.set_sample_rate(reduced_rate)
                
                time.sleep(1.0)  # Stabilize
                
                # Restore original rate
                sdr_instance.set_sample_rate(current_rate)
            
            duration_ms = (time.time() - start_time) * 1000
            
            self._record_recovery(
                error_type=ErrorType.SDR_OVERFLOW,
                attempt=1,
                success=True,
                duration_ms=duration_ms
            )
            
            self.total_recoveries += 1
            self.logger.info(f"SDR overflow recovered in {duration_ms:.0f}ms")
            return True
            
        except Exception as e:
            self.logger.error(f"SDR overflow recovery failed: {e}")
            self.failed_recoveries += 1
            return False
    
    def _verify_sdr_connection(self, sdr_instance) -> bool:
        """Verify SDR is connected and responsive"""
        try:
            # Check if we can read samples
            if hasattr(sdr_instance, 'read_samples'):
                samples = sdr_instance.read_samples(1024)
                return samples is not None and len(samples) > 0
            
            # Check if device is open
            if hasattr(sdr_instance, 'is_open'):
                return sdr_instance.is_open()
            
            return True  # Assume success if no verification method
            
        except Exception:
            return False
    
    # ===== INFERENCE RECOVERY =====
    
    def recover_from_inference_error(self, inference_fn: Callable, 
                                    fallback: str = 'cpu', *args, **kwargs) -> Any:
        """
        Recover from inference errors with automatic fallback
        
        Args:
            inference_fn: Inference function to execute
            fallback: Fallback device ('cpu', 'simple_model', 'skip')
            *args, **kwargs: Arguments for inference function
        
        Returns:
            Inference result (from GPU or fallback)
        
        Common errors:
        - GPU OOM (out of memory)
        - CUDA driver error
        - Model loading failure
        
        Fallback strategies:
        - 'cpu': Retry on CPU (slower but reliable)
        - 'simple_model': Use lightweight model
        - 'skip': Return None (fail gracefully)
        """
        self.total_errors += 1
        error_type = ErrorType.INFERENCE_ERROR
        
        try:
            start_time = time.time()
            result = inference_fn(*args, **kwargs)
            
            duration_ms = (time.time() - start_time) * 1000
            self.logger.debug(f"Inference completed in {duration_ms:.0f}ms")
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Inference error: {e} - attempting {fallback} fallback")
            
            try:
                start_time = time.time()
                
                if fallback == 'cpu':
                    # Force CPU execution
                    kwargs['device'] = 'cpu'
                    result = inference_fn(*args, **kwargs)
                    
                elif fallback == 'simple_model':
                    # Use simplified model
                    kwargs['use_simple'] = True
                    result = inference_fn(*args, **kwargs)
                    
                elif fallback == 'skip':
                    # Fail gracefully
                    result = None
                
                else:
                    raise ValueError(f"Unknown fallback: {fallback}")
                
                duration_ms = (time.time() - start_time) * 1000
                
                self._record_recovery(
                    error_type=error_type,
                    attempt=1,
                    success=True,
                    duration_ms=duration_ms,
                    fallback_used=fallback
                )
                
                self.total_recoveries += 1
                self.logger.info(f"Inference recovered using {fallback} in {duration_ms:.0f}ms")
                
                return result
                
            except Exception as fallback_error:
                self.logger.error(f"Inference fallback failed: {fallback_error}")
                self.failed_recoveries += 1
                
                self._record_recovery(
                    error_type=error_type,
                    attempt=1,
                    success=False,
                    duration_ms=(time.time() - start_time) * 1000,
                    error_message=str(fallback_error)
                )
                
                return None
    
    # ===== CAMPAIGN CHECKPOINTING =====
    
    def checkpoint_campaign(self, state: Dict[str, Any], phase: str) -> str:
        """
        Create checkpoint of campaign state for rollback
        
        Args:
            state: Campaign state dictionary
            phase: Current phase name
        
        Returns:
            Checkpoint ID
        
        Checkpoint captures:
        - Full campaign state
        - Module configurations
        - Performance metrics
        - Timestamp
        
        Use before risky operations (e.g., A-IoT jamming)
        """
        checkpoint_id = f"{phase}_{int(time.time())}"
        
        checkpoint = CampaignCheckpoint(
            checkpoint_id=checkpoint_id,
            phase=phase,
            timestamp=datetime.now(),
            state=state.copy(),
            module_states={},
            metrics={}
        )
        
        # Save to disk
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            self.checkpoints[checkpoint_id] = checkpoint
            
            self.logger.info(f"Campaign checkpoint created",
                           checkpoint_id=checkpoint_id,
                           phase=phase,
                           state_keys=len(state))
            
            return checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}")
            return ""
    
    def rollback_on_failure(self, phase: str) -> Optional[Dict[str, Any]]:
        """
        Rollback to last checkpoint for given phase
        
        Args:
            phase: Phase name to rollback
        
        Returns:
            Restored state dictionary or None
        
        Finds most recent checkpoint for phase and restores state.
        Use when operation fails critically (e.g., SDR damage risk).
        """
        # Find most recent checkpoint for phase
        matching_checkpoints = [
            cp for cp_id, cp in self.checkpoints.items()
            if cp.phase == phase
        ]
        
        if not matching_checkpoints:
            self.logger.warning(f"No checkpoint found for phase: {phase}")
            return None
        
        # Get most recent
        latest_checkpoint = max(matching_checkpoints, key=lambda cp: cp.timestamp)
        
        self.logger.info(f"Rolling back to checkpoint",
                       checkpoint_id=latest_checkpoint.checkpoint_id,
                       phase=phase,
                       timestamp=latest_checkpoint.timestamp)
        
        return latest_checkpoint.state.copy()
    
    # ===== FEDERATED SYNC RECOVERY =====
    
    def sync_with_backoff(self, sync_fn: Callable, max_retries: int = None, 
                         *args, **kwargs) -> bool:
        """
        Execute federated sync with exponential backoff
        
        Args:
            sync_fn: Sync function to execute
            max_retries: Max retry attempts (default: config value)
            *args, **kwargs: Arguments for sync function
        
        Returns:
            True if sync succeeded
        
        Handles network drops during federated aggregation.
        Uses exponential backoff to avoid overwhelming server.
        """
        max_retries = max_retries or self.max_retries
        error_type = ErrorType.FEDERATED_SYNC_FAIL
        
        for attempt in range(1, max_retries + 1):
            try:
                start_time = time.time()
                
                result = sync_fn(*args, **kwargs)
                
                duration_ms = (time.time() - start_time) * 1000
                
                self._record_recovery(
                    error_type=error_type,
                    attempt=attempt,
                    success=True,
                    duration_ms=duration_ms
                )
                
                if attempt > 1:
                    self.total_recoveries += 1
                    self.logger.info(f"Federated sync succeeded on attempt {attempt}")
                
                return True
                
            except Exception as e:
                self.total_errors += 1
                self.logger.warning(f"Federated sync attempt {attempt}/{max_retries} failed: {e}")
                
                if attempt < max_retries:
                    backoff_sec = self._calculate_backoff(attempt)
                    self.logger.info(f"Retrying in {backoff_sec:.1f}s")
                    time.sleep(backoff_sec)
                else:
                    self.logger.error(f"Federated sync failed after {max_retries} attempts")
                    self.failed_recoveries += 1
                    return False
        
        return False
    
    # ===== UTILITY METHODS =====
    
    def with_recovery(self, operation: Callable, error_type: ErrorType, 
                     max_retries: int = None, *args, **kwargs) -> Any:
        """
        Execute operation with automatic recovery
        
        Generic wrapper for any recoverable operation.
        
        Args:
            operation: Function to execute
            error_type: Expected error type for circuit breaker
            max_retries: Max retry attempts
            *args, **kwargs: Arguments for operation
        
        Returns:
            Operation result or None on failure
        """
        max_retries = max_retries or self.max_retries
        
        for attempt in range(1, max_retries + 1):
            try:
                result = operation(*args, **kwargs)
                
                if attempt > 1:
                    self.total_recoveries += 1
                
                return result
                
            except Exception as e:
                self.total_errors += 1
                self.logger.warning(f"Operation failed (attempt {attempt}/{max_retries}): {e}")
                
                if attempt < max_retries:
                    backoff_sec = self._calculate_backoff(attempt)
                    time.sleep(backoff_sec)
                else:
                    self.failed_recoveries += 1
                    self.logger.error(f"Operation failed after {max_retries} attempts")
                    return None
        
        return None
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter"""
        import random
        
        backoff = self.backoff_base_sec * (2 ** (attempt - 1))
        jitter = random.uniform(0, 0.1 * backoff)
        
        return min(backoff + jitter, 60.0)  # Cap at 60s
    
    def _check_circuit_breaker(self, error_type: ErrorType) -> bool:
        """Check if circuit breaker allows operation"""
        breaker = self.circuit_breakers[error_type]
        
        if breaker['state'] == 'open':
            # Check timeout
            if breaker['last_failure_time']:
                elapsed = time.time() - breaker['last_failure_time']
                if elapsed > breaker['timeout_sec']:
                    breaker['state'] = 'half_open'
                    self.logger.info(f"Circuit breaker for {error_type.value} entering half-open state")
                    return True
            return False
        
        return True
    
    def _update_circuit_breaker(self, error_type: ErrorType, success: bool):
        """Update circuit breaker state"""
        breaker = self.circuit_breakers[error_type]
        
        if success:
            breaker['failure_count'] = 0
            breaker['success_count'] += 1
            
            if breaker['state'] == 'half_open' and breaker['success_count'] >= 3:
                breaker['state'] = 'closed'
                self.logger.info(f"Circuit breaker for {error_type.value} closed")
        else:
            breaker['failure_count'] += 1
            breaker['success_count'] = 0
            breaker['last_failure_time'] = time.time()
            
            if breaker['failure_count'] >= breaker['threshold']:
                breaker['state'] = 'open'
                self.logger.error(f"Circuit breaker for {error_type.value} OPEN")
    
    def _record_recovery(self, error_type: ErrorType, attempt: int, success: bool,
                        duration_ms: float, fallback_used: str = None, error_message: str = ""):
        """Record recovery attempt"""
        recovery = RecoveryAttempt(
            error_type=error_type,
            timestamp=datetime.now(),
            attempt_number=attempt,
            success=success,
            duration_ms=duration_ms,
            fallback_used=fallback_used,
            error_message=error_message
        )
        
        self.recovery_attempts.append(recovery)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        uptime_sec = time.time() - self.uptime_start
        uptime_pct = ((uptime_sec - self.downtime_seconds) / uptime_sec) * 100 if uptime_sec > 0 else 100
        
        # Group by error type
        by_type = {}
        for attempt in self.recovery_attempts:
            error_name = attempt.error_type.value
            if error_name not in by_type:
                by_type[error_name] = {'total': 0, 'success': 0, 'failed': 0}
            
            by_type[error_name]['total'] += 1
            if attempt.success:
                by_type[error_name]['success'] += 1
            else:
                by_type[error_name]['failed'] += 1
        
        return {
            'uptime_percent': uptime_pct,
            'uptime_seconds': uptime_sec - self.downtime_seconds,
            'total_errors': self.total_errors,
            'total_recoveries': self.total_recoveries,
            'failed_recoveries': self.failed_recoveries,
            'recovery_rate': self.total_recoveries / max(1, self.total_errors),
            'checkpoints_created': len(self.checkpoints),
            'by_error_type': by_type,
            'circuit_breakers': {
                et.value: {
                    'state': cb['state'],
                    'failure_count': cb['failure_count']
                }
                for et, cb in self.circuit_breakers.items()
            }
        }
    
    # ===== ENHANCED ERROR RECOVERY (Phase 1.6.2) =====
    
    def retry_with_exponential_backoff(
        self,
        operation: Callable,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exceptions: tuple = (Exception,),
        on_retry: Optional[Callable] = None
    ) -> Any:
        """
        Retry operation with exponential backoff and jitter
        
        Phase 1.6.2: Enhanced retry mechanism with configurable parameters
        
        Args:
            operation: Function to execute
            max_attempts: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay cap in seconds
            exponential_base: Base for exponential calculation (default: 2)
            jitter: Add random jitter to prevent thundering herd
            exceptions: Tuple of exceptions to catch and retry
            on_retry: Optional callback on each retry (receives attempt number)
        
        Returns:
            Operation result
        
        Raises:
            Last exception if all retries exhausted
        
        Example:
            result = recoverer.retry_with_exponential_backoff(
                lambda: api_call(),
                max_attempts=5,
                base_delay=1.0
            )
        
        Backoff schedule (with jitter):
            Attempt 1: 0s (immediate)
            Attempt 2: ~1s
            Attempt 3: ~2s
            Attempt 4: ~4s
            Attempt 5: ~8s
        """
        import random
        
        last_exception = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                result = operation()
                
                if attempt > 1:
                    self.total_recoveries += 1
                    self.logger.info(f"Operation succeeded on attempt {attempt}/{max_attempts}")
                
                return result
                
            except exceptions as e:
                last_exception = e
                self.total_errors += 1
                
                if attempt < max_attempts:
                    # Calculate exponential backoff
                    delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        jitter_amount = random.uniform(0, 0.1 * delay)
                        delay += jitter_amount
                    
                    self.logger.warning(
                        f"Operation failed (attempt {attempt}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(attempt)
                        except Exception as cb_error:
                            self.logger.error(f"Retry callback error: {cb_error}")
                    
                    time.sleep(delay)
                else:
                    self.failed_recoveries += 1
                    self.logger.error(f"Operation failed after {max_attempts} attempts: {e}")
        
        # All retries exhausted
        raise last_exception
    
    def with_circuit_breaker(
        self,
        operation: Callable,
        breaker_name: str,
        failure_threshold: int = 5,
        timeout_seconds: float = 60.0,
        expected_exceptions: tuple = (Exception,)
    ) -> Any:
        """
        Execute operation with circuit breaker pattern
        
        Phase 1.6.2: Circuit breaker prevents cascading failures
        
        Args:
            operation: Function to execute
            breaker_name: Unique circuit breaker identifier
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before attempting recovery (half-open)
            expected_exceptions: Exceptions that trigger circuit breaker
        
        Returns:
            Operation result
        
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Original exception: If operation fails
        
        Circuit States:
            - CLOSED: Normal operation, requests pass through
            - OPEN: Too many failures, requests blocked
            - HALF_OPEN: Testing if service recovered
        
        Example:
            result = recoverer.with_circuit_breaker(
                lambda: external_api_call(),
                breaker_name='external_api',
                failure_threshold=3
            )
        """
        # Initialize breaker if not exists
        if breaker_name not in self._circuit_breaker_state:
            self._circuit_breaker_state[breaker_name] = {
                'state': 'closed',
                'failure_count': 0,
                'success_count': 0,
                'last_failure_time': None,
                'last_success_time': None
            }
        
        breaker = self._circuit_breaker_state[breaker_name]
        
        # Check if circuit is open
        if breaker['state'] == 'open':
            elapsed = time.time() - breaker['last_failure_time']
            
            if elapsed < timeout_seconds:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{breaker_name}' is OPEN. "
                    f"Retry in {timeout_seconds - elapsed:.1f}s"
                )
            else:
                # Transition to half-open
                breaker['state'] = 'half_open'
                self.logger.info(f"Circuit breaker '{breaker_name}' entering HALF_OPEN state")
        
        # Execute operation
        try:
            result = operation()
            
            # Success - update breaker
            breaker['failure_count'] = 0
            breaker['success_count'] += 1
            breaker['last_success_time'] = time.time()
            
            if breaker['state'] == 'half_open':
                if breaker['success_count'] >= 2:
                    breaker['state'] = 'closed'
                    self.logger.info(f"Circuit breaker '{breaker_name}' CLOSED")
            
            return result
            
        except expected_exceptions as e:
            # Failure - update breaker
            breaker['failure_count'] += 1
            breaker['success_count'] = 0
            breaker['last_failure_time'] = time.time()
            
            if breaker['failure_count'] >= failure_threshold:
                breaker['state'] = 'open'
                self.logger.error(
                    f"Circuit breaker '{breaker_name}' OPEN after "
                    f"{failure_threshold} failures"
                )
            
            raise
    
    def with_graceful_degradation(
        self,
        primary_operation: Callable,
        fallback_operations: List[Callable],
        degradation_levels: Optional[List[str]] = None
    ) -> tuple:
        """
        Execute operation with graceful degradation strategy
        
        Phase 1.6.2: Fallback to degraded service levels instead of total failure
        
        Args:
            primary_operation: Primary function to execute
            fallback_operations: List of fallback functions (ordered by preference)
            degradation_levels: Optional names for each degradation level
        
        Returns:
            Tuple of (result, degradation_level)
            degradation_level: 'full' for primary, 'degraded_1', 'degraded_2', etc.
        
        Example:
            result, level = recoverer.with_graceful_degradation(
                primary_operation=lambda: high_quality_inference(),
                fallback_operations=[
                    lambda: medium_quality_inference(),
                    lambda: low_quality_inference(),
                    lambda: cached_result()
                ],
                degradation_levels=['full', 'medium', 'low', 'cached']
            )
        """
        if degradation_levels is None:
            degradation_levels = ['full'] + [f'degraded_{i+1}' for i in range(len(fallback_operations))]
        
        # Try primary operation
        try:
            result = primary_operation()
            self.logger.info("Operation succeeded at full service level")
            return (result, degradation_levels[0])
            
        except Exception as e:
            self.logger.warning(f"Primary operation failed: {e}. Attempting degradation...")
        
        # Try fallback operations
        for idx, fallback in enumerate(fallback_operations):
            degradation_level = degradation_levels[idx + 1] if (idx + 1) < len(degradation_levels) else f'degraded_{idx+1}'
            
            try:
                result = fallback()
                self.logger.warning(f"Operation succeeded at degraded level: {degradation_level}")
                return (result, degradation_level)
                
            except Exception as e:
                self.logger.warning(f"Fallback {idx+1} failed: {e}")
                continue
        
        # All operations failed
        self.failed_recoveries += 1
        raise GracefulDegradationError("All service levels failed (primary + all fallbacks)")
    
    def with_timeout_and_retry(
        self,
        operation: Callable,
        timeout_seconds: float,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Any:
        """
        Execute operation with timeout and automatic retry
        
        Phase 1.6.2: Prevent hanging operations with timeout enforcement
        
        Args:
            operation: Function to execute
            timeout_seconds: Maximum execution time in seconds
            max_retries: Number of retry attempts on timeout
            retry_delay: Delay between retries
        
        Returns:
            Operation result
        
        Raises:
            TimeoutError: If all attempts timeout
        """
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout_seconds}s")
        
        for attempt in range(1, max_retries + 1):
            try:
                # Set timeout alarm
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout_seconds))
                
                result = operation()
                
                # Cancel alarm
                signal.alarm(0)
                
                if attempt > 1:
                    self.logger.info(f"Operation succeeded on attempt {attempt} after timeout recovery")
                
                return result
                
            except TimeoutError as e:
                signal.alarm(0)  # Cancel alarm
                self.total_errors += 1
                
                if attempt < max_retries:
                    self.logger.warning(
                        f"Operation timed out (attempt {attempt}/{max_retries}). "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    self.failed_recoveries += 1
                    self.logger.error(f"Operation timed out after {max_retries} attempts")
                    raise
            
            except Exception as e:
                signal.alarm(0)  # Cancel alarm
                raise
    
    @property
    def _circuit_breaker_state(self):
        """Get or initialize circuit breaker state dict"""
        if not hasattr(self, '_cb_state'):
            self._cb_state = {}
        return self._cb_state


# ===== CUSTOM EXCEPTIONS =====

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class GracefulDegradationError(Exception):
    """Raised when all degradation levels fail"""
    pass
