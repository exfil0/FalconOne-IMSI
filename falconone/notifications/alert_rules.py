"""
Alert Rules Engine
Configure and manage notification rules

Version 1.0: Phase 2.5.3 - Email Alerts
- Rule-based alert triggering
- User notification preferences
- Alert suppression and rate limiting
"""

from typing import List, Dict, Callable, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Alert rule configuration"""
    
    name: str
    condition: Callable[[Dict], bool]  # Function that checks if alert should fire
    alert_type: str
    severity: str  # CRITICAL, WARNING, INFO
    recipients: List[str]
    enabled: bool = True
    cooldown_minutes: int = 30  # Minimum time between same alerts
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    max_triggers_per_hour: int = 10
    metadata: Dict = field(default_factory=dict)
    
    def should_trigger(self, event_data: Dict) -> bool:
        """
        Check if rule should trigger alert
        
        Args:
            event_data: Event data to evaluate
        
        Returns:
            bool: True if alert should be sent
        """
        if not self.enabled:
            return False
        
        # Check cooldown period
        if self.last_triggered:
            time_since_last = datetime.utcnow() - self.last_triggered
            if time_since_last < timedelta(minutes=self.cooldown_minutes):
                logger.debug(f"Rule {self.name} in cooldown period")
                return False
        
        # Check rate limit
        if self.trigger_count >= self.max_triggers_per_hour:
            logger.warning(f"Rule {self.name} exceeded rate limit")
            return False
        
        # Evaluate condition
        try:
            return self.condition(event_data)
        except Exception as e:
            logger.error(f"Error evaluating rule {self.name}: {e}")
            return False
    
    def mark_triggered(self):
        """Mark rule as triggered"""
        self.last_triggered = datetime.utcnow()
        self.trigger_count += 1
    
    def reset_trigger_count(self):
        """Reset hourly trigger count"""
        self.trigger_count = 0


class AlertRulesEngine:
    """Manages alert rules and triggers notifications"""
    
    def __init__(self, email_manager):
        """
        Initialize alert rules engine
        
        Args:
            email_manager: EmailAlertManager instance
        """
        self.email_manager = email_manager
        self.rules: Dict[str, AlertRule] = {}
        self._setup_default_rules()
        logger.info("AlertRulesEngine initialized")
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        
        # Rule: DOS Attack Detected
        self.add_rule(AlertRule(
            name='dos_attack_detected',
            condition=lambda data: data.get('attack_type') == 'dos' and data.get('confidence', 0) > 0.8,
            alert_type='dos_detected',
            severity='CRITICAL',
            recipients=['admin@example.com'],
            cooldown_minutes=15
        ))
        
        # Rule: MITM Attack Detected
        self.add_rule(AlertRule(
            name='mitm_attack_detected',
            condition=lambda data: data.get('attack_type') == 'mitm' and data.get('rogue_bs', False),
            alert_type='mitm_detected',
            severity='CRITICAL',
            recipients=['admin@example.com', 'security@example.com'],
            cooldown_minutes=10
        ))
        
        # Rule: Downgrade Attack Detected
        self.add_rule(AlertRule(
            name='downgrade_attack_detected',
            condition=lambda data: data.get('attack_type') == 'downgrade' and data.get('forced_downgrade', False),
            alert_type='downgrade_attack',
            severity='WARNING',
            recipients=['admin@example.com'],
            cooldown_minutes=30
        ))
        
        # Rule: SDR Failure
        self.add_rule(AlertRule(
            name='sdr_failure',
            condition=lambda data: data.get('event_type') == 'sdr_failure' and not data.get('failover_successful', False),
            alert_type='sdr_failure',
            severity='WARNING',
            recipients=['admin@example.com', 'ops@example.com'],
            cooldown_minutes=5
        ))
        
        # Rule: High Signal Anomaly
        self.add_rule(AlertRule(
            name='signal_anomaly_high',
            condition=lambda data: data.get('anomaly_score', 0) > 0.9,
            alert_type='anomalous_traffic',
            severity='WARNING',
            recipients=['admin@example.com'],
            cooldown_minutes=60
        ))
        
        # Rule: Authentication Failure Spike
        self.add_rule(AlertRule(
            name='auth_failure_spike',
            condition=lambda data: data.get('failed_logins', 0) > 10 and data.get('time_window', 0) < 300,
            alert_type='authentication_spike',
            severity='WARNING',
            recipients=['admin@example.com', 'security@example.com'],
            cooldown_minutes=20
        ))
        
        # Rule: Exploit Completion (Informational)
        self.add_rule(AlertRule(
            name='exploit_completed',
            condition=lambda data: data.get('event_type') == 'exploit_complete' and data.get('success', False),
            alert_type='exploit_completion',
            severity='INFO',
            recipients=['admin@example.com'],
            cooldown_minutes=0,  # No cooldown for informational alerts
            max_triggers_per_hour=100
        ))
        
        # Rule: Database Security Alert
        self.add_rule(AlertRule(
            name='database_security_alert',
            condition=lambda data: data.get('event_type') == 'db_security' and data.get('severity') == 'high',
            alert_type='database_security',
            severity='CRITICAL',
            recipients=['admin@example.com', 'dba@example.com'],
            cooldown_minutes=15
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove alert rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def enable_rule(self, rule_name: str):
        """Enable alert rule"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            logger.info(f"Enabled alert rule: {rule_name}")
    
    def disable_rule(self, rule_name: str):
        """Disable alert rule"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            logger.info(f"Disabled alert rule: {rule_name}")
    
    def evaluate_event(self, event_data: Dict) -> List[str]:
        """
        Evaluate event against all rules
        
        Args:
            event_data: Event data to evaluate
        
        Returns:
            List of rule names that triggered
        """
        triggered_rules = []
        
        for rule_name, rule in self.rules.items():
            if rule.should_trigger(event_data):
                logger.info(f"Rule triggered: {rule_name}")
                
                # Send alert email
                details = {
                    'severity': rule.severity,
                    'timestamp': datetime.utcnow().isoformat(),
                    **event_data
                }
                
                success = self.email_manager.send_security_alert(
                    to_emails=rule.recipients,
                    alert_type=rule.alert_type,
                    details=details
                )
                
                if success:
                    rule.mark_triggered()
                    triggered_rules.append(rule_name)
                else:
                    logger.error(f"Failed to send alert for rule: {rule_name}")
        
        return triggered_rules
    
    def get_rule(self, rule_name: str) -> Optional[AlertRule]:
        """Get rule by name"""
        return self.rules.get(rule_name)
    
    def list_rules(self) -> List[Dict]:
        """List all rules with their status"""
        return [
            {
                'name': rule.name,
                'alert_type': rule.alert_type,
                'severity': rule.severity,
                'enabled': rule.enabled,
                'cooldown_minutes': rule.cooldown_minutes,
                'trigger_count': rule.trigger_count,
                'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None,
                'recipients': rule.recipients
            }
            for rule in self.rules.values()
        ]
    
    def update_recipients(self, rule_name: str, recipients: List[str]):
        """Update rule recipients"""
        if rule_name in self.rules:
            self.rules[rule_name].recipients = recipients
            logger.info(f"Updated recipients for rule {rule_name}: {recipients}")
    
    def reset_all_trigger_counts(self):
        """Reset all rule trigger counts (run hourly)"""
        for rule in self.rules.values():
            rule.reset_trigger_count()
        logger.info("Reset all rule trigger counts")
