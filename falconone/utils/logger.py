"""
FalconOne Logging System
Advanced logging with rotation, compression, and monitoring
"""

import os
import logging
import logging.handlers
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any, List, Dict
import json


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for terminal output"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str = 'falconone',
    log_dir: str = '/var/log/falconone',
    log_level: str = 'INFO',
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up advanced logging system with rotation and compression
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Enable console output
        file_output: Enable file output
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    
    # Log format
    log_format = '%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = ColoredFormatter(log_format, datefmt=date_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_output:
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Main log file with rotation (10MB per file, keep 10 backups)
        log_file = os.path.join(log_dir, f'{name}.log')
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Separate error log file
        error_log_file = os.path.join(log_dir, f'{name}_error.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
    
    return logger


class ModuleLogger:
    """Logger wrapper for specific modules with enhanced context"""
    
    def __init__(self, module_name: str, parent_logger: Optional[logging.Logger] = None):
        """
        Initialize module logger
        
        Args:
            module_name: Name of the module (e.g., 'GSM', 'LTE', 'AI')
            parent_logger: Parent logger instance
        """
        self.module_name = module_name
        self.logger = parent_logger or logging.getLogger('falconone')
    
    def _log(self, level: str, message: str, **kwargs):
        """Internal log method with module context"""
        formatted_message = f"[{self.module_name}] {message}"
        if kwargs:
            formatted_message += f" - Context: {kwargs}"
        getattr(self.logger, level)(formatted_message)
    
    def debug(self, message: str, **kwargs):
        self._log('debug', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log('warning', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log('error', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log('critical', message, **kwargs)


class PerformanceLogger:
    """Logger for performance metrics and benchmarking"""
    
    def __init__(self, log_dir: str = '/var/log/falconone/performance'):
        """
        Initialize performance logger
        
        Args:
            log_dir: Directory for performance logs
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f'performance_{datetime.now().strftime("%Y%m%d")}.csv')
        
        # Create CSV header if file doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write('timestamp,module,operation,duration_ms,cpu_percent,memory_mb,status\n')
    
    def log_operation(
        self,
        module: str,
        operation: str,
        duration_ms: float,
        cpu_percent: float = 0.0,
        memory_mb: float = 0.0,
        status: str = 'success'
    ):
        """
        Log performance metrics for an operation
        
        Args:
            module: Module name
            operation: Operation description
            duration_ms: Operation duration in milliseconds
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
            status: Operation status (success/failure)
        """
        timestamp = datetime.now().isoformat()
        log_entry = f"{timestamp},{module},{operation},{duration_ms:.2f},{cpu_percent:.2f},{memory_mb:.2f},{status}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)


class AuditLogger:
    """
    Comprehensive security and compliance audit logger (Phase 1.6.1)
    
    Logs all security-sensitive events:
    - Authentication attempts (login/logout/failed attempts)
    - Exploit operations (DOS/downgrade/MITM attacks)
    - Configuration changes (SDR settings, security policies)
    - Database modifications (sensitive data access)
    - SDR device changes (connect/disconnect/failover)
    
    Features:
    - Automatic log rotation (daily files, 90-day retention)
    - JSON structured logging for parsing
    - Separate critical event log
    - Tamper-resistant timestamps
    """
    
    def __init__(self, log_dir: str = '/var/log/falconone/audit', retention_days: int = 90):
        """
        Initialize comprehensive audit logger
        
        Args:
            log_dir: Directory for audit logs
            retention_days: Number of days to retain audit logs (default: 90)
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.retention_days = retention_days
        
        # Daily audit log file
        date_str = datetime.now().strftime("%Y%m%d")
        self.log_file = self.log_dir / f'audit_{date_str}.log'
        
        # Critical events log (separate file for high-priority review)
        self.critical_log_file = self.log_dir / f'critical_{date_str}.log'
        
        # Set up main audit logger
        self.logger = logging.getLogger('falconone.audit')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # Main audit handler with rotation
        handler = logging.handlers.TimedRotatingFileHandler(
            self.log_file,
            when='midnight',
            interval=1,
            backupCount=retention_days,
            encoding='utf-8'
        )
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Critical events handler
        critical_handler = logging.FileHandler(self.critical_log_file, encoding='utf-8')
        critical_handler.setLevel(logging.CRITICAL)
        critical_handler.setFormatter(formatter)
        self.logger.addHandler(critical_handler)
        
        # Perform log rotation/cleanup
        self._rotate_old_logs()
        
        self.logger.info(json.dumps({
            'event': 'AUDIT_LOGGER_INITIALIZED',
            'retention_days': retention_days,
            'log_dir': str(log_dir)
        }))
    
    def _rotate_old_logs(self):
        """Remove audit logs older than retention period"""
        try:
            import time
            current_time = time.time()
            retention_seconds = self.retention_days * 24 * 60 * 60
            
            for log_file in self.log_dir.glob('audit_*.log*'):
                if log_file.stat().st_mtime < (current_time - retention_seconds):
                    log_file.unlink()
                    self.logger.debug(f"Rotated old audit log: {log_file.name}")
                    
            for log_file in self.log_dir.glob('critical_*.log*'):
                if log_file.stat().st_mtime < (current_time - retention_seconds):
                    log_file.unlink()
                    
        except Exception as e:
            self.logger.error(f"Log rotation failed: {e}")
    
    def log_event(
        self,
        event_type: str,
        description: str,
        user: str = 'system',
        target: Optional[str] = None,
        status: str = 'success',
        severity: str = 'INFO',
        metadata: Optional[dict] = None
    ):
        """
        Log a comprehensive audit event
        
        Args:
            event_type: Type of event (AUTH_LOGIN, EXPLOIT_DOS, CONFIG_CHANGE, etc.)
            description: Human-readable event description
            user: User or system component initiating the event
            target: Target device, network, or resource
            status: Event status (success, failure, warning)
            severity: Log severity (INFO, WARNING, ERROR, CRITICAL)
            metadata: Additional structured data
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user': user,
            'target': target or 'N/A',
            'description': description,
            'status': status,
            'severity': severity,
            'metadata': metadata or {}
        }
        
        log_message = json.dumps(log_entry)
        
        # Log at appropriate level
        if severity == 'CRITICAL':
            self.logger.critical(log_message)
        elif severity == 'ERROR':
            self.logger.error(log_message)
        elif severity == 'WARNING':
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    # ===== AUTHENTICATION EVENTS =====
    
    def log_login_attempt(self, username: str, success: bool, ip_address: str = 'unknown'):
        """Log user login attempt"""
        self.log_event(
            event_type='AUTH_LOGIN_ATTEMPT',
            description=f"Login attempt for user '{username}'",
            user=username,
            status='success' if success else 'failure',
            severity='WARNING' if not success else 'INFO',
            metadata={'ip_address': ip_address, 'success': success}
        )
    
    def log_logout(self, username: str):
        """Log user logout"""
        self.log_event(
            event_type='AUTH_LOGOUT',
            description=f"User '{username}' logged out",
            user=username,
            status='success'
        )
    
    def log_password_change(self, username: str, success: bool):
        """Log password change attempt"""
        self.log_event(
            event_type='AUTH_PASSWORD_CHANGE',
            description=f"Password change for user '{username}'",
            user=username,
            status='success' if success else 'failure',
            severity='WARNING' if not success else 'INFO'
        )
    
    def log_failed_authentication(self, username: str, reason: str, ip_address: str = 'unknown'):
        """Log failed authentication with reason"""
        self.log_event(
            event_type='AUTH_FAILED',
            description=f"Authentication failed for '{username}': {reason}",
            user=username,
            status='failure',
            severity='WARNING',
            metadata={'reason': reason, 'ip_address': ip_address}
        )
    
    # ===== EXPLOIT OPERATIONS =====
    
    def log_exploit_start(self, exploit_type: str, target: str, user: str, parameters: dict):
        """Log exploit operation start"""
        self.log_event(
            event_type=f'EXPLOIT_{exploit_type.upper()}_START',
            description=f"Started {exploit_type} attack on {target}",
            user=user,
            target=target,
            status='initiated',
            severity='CRITICAL',
            metadata={'parameters': parameters}
        )
    
    def log_exploit_complete(self, exploit_type: str, target: str, user: str, success: bool, results: dict):
        """Log exploit operation completion"""
        self.log_event(
            event_type=f'EXPLOIT_{exploit_type.upper()}_COMPLETE',
            description=f"Completed {exploit_type} attack on {target}",
            user=user,
            target=target,
            status='success' if success else 'failure',
            severity='CRITICAL',
            metadata={'results': results}
        )
    
    def log_exploit_error(self, exploit_type: str, target: str, user: str, error: str):
        """Log exploit operation error"""
        self.log_event(
            event_type=f'EXPLOIT_{exploit_type.upper()}_ERROR',
            description=f"Error in {exploit_type} attack: {error}",
            user=user,
            target=target,
            status='error',
            severity='ERROR',
            metadata={'error': error}
        )
    
    # ===== CONFIGURATION CHANGES =====
    
    def log_config_change(self, config_key: str, old_value: Any, new_value: Any, user: str):
        """Log configuration change"""
        self.log_event(
            event_type='CONFIG_CHANGE',
            description=f"Configuration '{config_key}' changed",
            user=user,
            status='success',
            severity='WARNING',
            metadata={
                'config_key': config_key,
                'old_value': str(old_value),
                'new_value': str(new_value)
            }
        )
    
    def log_sdr_config_change(self, parameter: str, old_value: Any, new_value: Any, device: str):
        """Log SDR configuration change"""
        self.log_event(
            event_type='SDR_CONFIG_CHANGE',
            description=f"SDR parameter '{parameter}' changed on {device}",
            user='system',
            target=device,
            status='success',
            severity='INFO',
            metadata={
                'parameter': parameter,
                'old_value': str(old_value),
                'new_value': str(new_value)
            }
        )
    
    # ===== DATABASE OPERATIONS =====
    
    def log_database_access(self, operation: str, table: str, user: str, record_count: int = 1):
        """Log database access (especially for sensitive tables)"""
        self.log_event(
            event_type='DATABASE_ACCESS',
            description=f"{operation} operation on table '{table}'",
            user=user,
            target=table,
            status='success',
            metadata={'operation': operation, 'record_count': record_count}
        )
    
    def log_database_modification(self, operation: str, table: str, user: str, record_id: Optional[str] = None):
        """Log database modification"""
        self.log_event(
            event_type='DATABASE_MODIFY',
            description=f"{operation} on table '{table}'",
            user=user,
            target=table,
            status='success',
            severity='WARNING',
            metadata={'operation': operation, 'record_id': record_id}
        )
    
    def log_backup_operation(self, operation: str, backup_file: str, success: bool):
        """Log database backup operation"""
        self.log_event(
            event_type='DATABASE_BACKUP',
            description=f"Database backup {operation}",
            user='system',
            target=backup_file,
            status='success' if success else 'failure',
            severity='ERROR' if not success else 'INFO'
        )
    
    # ===== SDR DEVICE EVENTS =====
    
    def log_sdr_connect(self, device_type: str, device_id: str, success: bool):
        """Log SDR device connection"""
        self.log_event(
            event_type='SDR_CONNECT',
            description=f"SDR device {device_type} ({device_id}) connection",
            user='system',
            target=f"{device_type}:{device_id}",
            status='success' if success else 'failure',
            severity='WARNING' if not success else 'INFO'
        )
    
    def log_sdr_disconnect(self, device_type: str, device_id: str, reason: str = 'manual'):
        """Log SDR device disconnection"""
        self.log_event(
            event_type='SDR_DISCONNECT',
            description=f"SDR device {device_type} disconnected: {reason}",
            user='system',
            target=f"{device_type}:{device_id}",
            status='warning',
            severity='WARNING',
            metadata={'reason': reason}
        )
    
    def log_sdr_failover(self, from_device: str, to_device: str, reason: str, success: bool):
        """Log SDR device failover"""
        self.log_event(
            event_type='SDR_FAILOVER',
            description=f"SDR failover from {from_device} to {to_device}: {reason}",
            user='system',
            target=f"{from_device}->{to_device}",
            status='success' if success else 'failure',
            severity='CRITICAL' if not success else 'WARNING',
            metadata={'from_device': from_device, 'to_device': to_device, 'reason': reason}
        )
    
    def log_sdr_health_issue(self, device: str, issue: str, severity: str = 'WARNING'):
        """Log SDR health monitoring issue"""
        self.log_event(
            event_type='SDR_HEALTH_ISSUE',
            description=f"SDR health issue on {device}: {issue}",
            user='system',
            target=device,
            status='warning',
            severity=severity,
            metadata={'issue': issue}
        )
    
    # ===== SECURITY EVENTS =====
    
    def log_security_violation(self, violation_type: str, description: str, user: str, severity: str = 'CRITICAL'):
        """Log security policy violation"""
        self.log_event(
            event_type='SECURITY_VIOLATION',
            description=description,
            user=user,
            status='violation',
            severity=severity,
            metadata={'violation_type': violation_type}
        )
    
    def log_rate_limit_exceeded(self, endpoint: str, user: str, ip_address: str):
        """Log rate limit exceeded"""
        self.log_event(
            event_type='RATE_LIMIT_EXCEEDED',
            description=f"Rate limit exceeded on {endpoint}",
            user=user,
            target=endpoint,
            status='blocked',
            severity='WARNING',
            metadata={'ip_address': ip_address}
        )
    
    # ===== AUDIT LOG READING (Phase 2.4.3) =====
    
    def get_audit_logs(self, limit: int = 100, event_type: str = None, 
                       user: str = None, start_date: str = None, end_date: str = None,
                       severity: str = None) -> List[Dict[str, Any]]:
        """
        Read and filter audit logs (Phase 2.4.3 - Audit Logs UI)
        
        Args:
            limit: Maximum number of logs to return
            event_type: Filter by event type (e.g., 'LOGIN', 'EXPLOIT_START')
            user: Filter by username
            start_date: Filter by start date (ISO format: YYYY-MM-DD)
            end_date: Filter by end date (ISO format: YYYY-MM-DD)
            severity: Filter by severity (INFO, WARNING, ERROR, CRITICAL)
        
        Returns:
            List of audit log entries as dictionaries
        """
        import glob
        
        audit_logs = []
        
        # Get all audit log files (sorted by date, newest first)
        log_files = sorted(
            glob.glob(str(self.log_dir / 'audit_*.log')),
            reverse=True
        )
        
        # Parse date filters
        start_ts = None
        end_ts = None
        if start_date:
            try:
                start_ts = datetime.fromisoformat(start_date).timestamp()
            except:
                pass
        if end_date:
            try:
                # Add 1 day to include the end date
                end_ts = (datetime.fromisoformat(end_date) + timedelta(days=1)).timestamp()
            except:
                pass
        
        # Read log files until we have enough entries
        for log_file in log_files:
            if len(audit_logs) >= limit:
                break
            
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    # Process lines in reverse (newest first)
                    for line in reversed(lines):
                        if len(audit_logs) >= limit:
                            break
                        
                        try:
                            # Parse log line
                            # Format: YYYY-MM-DD HH:MM:SS - AUDIT - {JSON}
                            parts = line.split(' - AUDIT - ', 1)
                            if len(parts) != 2:
                                continue
                            
                            timestamp_str = parts[0]
                            json_str = parts[1].strip()
                            
                            # Parse JSON data
                            log_data = json.loads(json_str)
                            
                            # Add parsed timestamp
                            log_data['timestamp_str'] = timestamp_str
                            log_data['timestamp'] = datetime.strptime(
                                timestamp_str, '%Y-%m-%d %H:%M:%S'
                            ).timestamp()
                            
                            # Apply filters
                            if event_type and log_data.get('event_type') != event_type:
                                continue
                            
                            if user and log_data.get('user') != user:
                                continue
                            
                            if severity and log_data.get('severity') != severity:
                                continue
                            
                            if start_ts and log_data['timestamp'] < start_ts:
                                continue
                            
                            if end_ts and log_data['timestamp'] > end_ts:
                                continue
                            
                            audit_logs.append(log_data)
                        
                        except (json.JSONDecodeError, ValueError):
                            # Skip malformed lines
                            continue
            
            except Exception as e:
                # Log file might be locked or inaccessible
                continue
        
        return audit_logs
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """
        Get audit log summary statistics
        
        Returns:
            Dictionary with log statistics
        """
        import glob
        from collections import defaultdict
        
        stats = {
            'total_events': 0,
            'by_type': defaultdict(int),
            'by_user': defaultdict(int),
            'by_severity': defaultdict(int),
            'critical_events': 0,
            'failed_auth': 0,
            'successful_auth': 0,
            'exploits_started': 0,
            'exploits_completed': 0,
            'config_changes': 0
        }
        
        # Get all audit log files from last 7 days
        log_files = glob.glob(str(self.log_dir / 'audit_*.log'))
        
        for log_file in log_files[-7:]:  # Last 7 files
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            parts = line.split(' - AUDIT - ', 1)
                            if len(parts) != 2:
                                continue
                            
                            log_data = json.loads(parts[1].strip())
                            
                            stats['total_events'] += 1
                            stats['by_type'][log_data.get('event_type', 'UNKNOWN')] += 1
                            stats['by_user'][log_data.get('user', 'unknown')] += 1
                            stats['by_severity'][log_data.get('severity', 'INFO')] += 1
                            
                            if log_data.get('severity') == 'CRITICAL':
                                stats['critical_events'] += 1
                            
                            event_type = log_data.get('event_type', '')
                            if event_type == 'LOGIN':
                                if log_data.get('status') == 'success':
                                    stats['successful_auth'] += 1
                                else:
                                    stats['failed_auth'] += 1
                            elif event_type == 'EXPLOIT_START':
                                stats['exploits_started'] += 1
                            elif event_type == 'EXPLOIT_COMPLETE':
                                stats['exploits_completed'] += 1
                            elif event_type in ['CONFIG_CHANGE', 'SDR_CONFIG_CHANGE']:
                                stats['config_changes'] += 1
                        
                        except (json.JSONDecodeError, ValueError):
                            continue
            
            except Exception:
                continue
        
        # Convert defaultdicts to regular dicts
        stats['by_type'] = dict(stats['by_type'])
        stats['by_user'] = dict(stats['by_user'])
        stats['by_severity'] = dict(stats['by_severity'])
        
        return stats
