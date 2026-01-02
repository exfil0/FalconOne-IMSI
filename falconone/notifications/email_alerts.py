"""
Email Alert System
SMTP-based notifications for security events and system status

Version 1.0: Phase 2.5.3 - Email Alerts
- SMTP configuration
- Email templates for different alert types
- Batch notifications
- User preference management
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Optional
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class EmailAlertManager:
    """Manages email alert configuration and sending"""
    
    def __init__(self, smtp_host: str = None, smtp_port: int = None,
                 smtp_user: str = None, smtp_password: str = None,
                 from_email: str = None):
        """
        Initialize email alert manager
        
        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            smtp_user: SMTP username
            smtp_password: SMTP password
            from_email: Sender email address
        """
        # Load from environment if not provided
        self.smtp_host = smtp_host or os.getenv('SMTP_HOST', 'localhost')
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = smtp_user or os.getenv('SMTP_USER', '')
        self.smtp_password = smtp_password or os.getenv('SMTP_PASSWORD', '')
        self.from_email = from_email or os.getenv('FROM_EMAIL', 'falconone@example.com')
        self.use_tls = os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'
        
        logger.info(f"EmailAlertManager initialized: {self.smtp_host}:{self.smtp_port}")
    
    def send_email(self, to_emails: List[str], subject: str, 
                   body_text: str, body_html: str = None,
                   attachments: List[Dict] = None) -> bool:
        """
        Send email notification
        
        Args:
            to_emails: List of recipient email addresses
            subject: Email subject
            body_text: Plain text body
            body_html: HTML body (optional)
            attachments: List of attachments [{filename, data}]
        
        Returns:
            bool: Success status
        """
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = subject
            msg['Date'] = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S +0000')
            
            # Add text body
            msg.attach(MIMEText(body_text, 'plain'))
            
            # Add HTML body if provided
            if body_html:
                msg.attach(MIMEText(body_html, 'html'))
            
            # Add attachments
            if attachments:
                for attachment in attachments:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment['data'])
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', 
                                  f"attachment; filename= {attachment['filename']}")
                    msg.attach(part)
            
            # Connect and send
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
                if self.use_tls:
                    server.starttls()
                
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {len(to_emails)} recipients: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send_security_alert(self, to_emails: List[str], alert_type: str, 
                           details: Dict) -> bool:
        """
        Send security alert email
        
        Args:
            to_emails: Recipients
            alert_type: Type of alert (dos_detected, mitm_detected, etc.)
            details: Alert details dictionary
        
        Returns:
            bool: Success status
        """
        subject = f"🚨 FalconOne Security Alert: {alert_type.replace('_', ' ').title()}"
        
        body_text = f"""
FalconOne Security Alert
========================

Alert Type: {alert_type}
Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
Severity: {details.get('severity', 'HIGH')}

Details:
--------
{self._format_details_text(details)}

Recommended Actions:
-------------------
{self._get_recommended_actions(alert_type)}

This is an automated alert from FalconOne.
        """
        
        body_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .alert {{ background-color: #fff3cd; border: 1px solid #ffc107; padding: 20px; border-radius: 5px; }}
                .critical {{ background-color: #f8d7da; border-color: #dc3545; }}
                .warning {{ background-color: #fff3cd; border-color: #ffc107; }}
                .info {{ background-color: #d1ecf1; border-color: #0dcaf0; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ text-align: left; padding: 12px; border: 1px solid #ddd; }}
                th {{ background-color: #343a40; color: white; }}
            </style>
        </head>
        <body>
            <div class="alert {details.get('severity', 'warning').lower()}">
                <h2>🚨 FalconOne Security Alert</h2>
                <h3>{alert_type.replace('_', ' ').title()}</h3>
                <p><strong>Timestamp:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <p><strong>Severity:</strong> {details.get('severity', 'HIGH')}</p>
                
                <h4>Alert Details:</h4>
                <table>
                    {self._format_details_html(details)}
                </table>
                
                <h4>Recommended Actions:</h4>
                <p>{self._get_recommended_actions(alert_type)}</p>
            </div>
            <p style="color: #6c757d; font-size: 0.9em; margin-top: 30px;">
                This is an automated alert from FalconOne.
            </p>
        </body>
        </html>
        """
        
        return self.send_email(to_emails, subject, body_text, body_html)
    
    def send_system_status(self, to_emails: List[str], status: Dict) -> bool:
        """
        Send system status report
        
        Args:
            to_emails: Recipients
            status: System status dictionary
        
        Returns:
            bool: Success status
        """
        subject = f"FalconOne System Status Report - {datetime.utcnow().strftime('%Y-%m-%d')}"
        
        body_text = f"""
FalconOne System Status Report
==============================

Report Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

System Health:
--------------
Overall Status: {status.get('overall_status', 'UNKNOWN')}
Uptime: {status.get('uptime', 'N/A')}

Components:
-----------
{self._format_status_components(status.get('components', {}))}

Recent Activity:
----------------
Scans Completed: {status.get('scans_completed', 0)}
Alerts Triggered: {status.get('alerts_triggered', 0)}
Tasks Running: {status.get('tasks_running', 0)}

This is an automated report from FalconOne.
        """
        
        return self.send_email(to_emails, subject, body_text)
    
    def send_exploit_completion(self, to_emails: List[str], exploit_type: str, 
                               results: Dict) -> bool:
        """
        Send exploit completion notification
        
        Args:
            to_emails: Recipients
            exploit_type: Type of exploit
            results: Exploit results
        
        Returns:
            bool: Success status
        """
        subject = f"FalconOne Exploit Completed: {exploit_type}"
        
        success = results.get('success', False)
        status_emoji = "✅" if success else "❌"
        
        body_text = f"""
FalconOne Exploit Completion
============================

{status_emoji} Status: {'SUCCESS' if success else 'FAILED'}
Exploit Type: {exploit_type}
Completion Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
Duration: {results.get('duration', 'N/A')}

Results:
--------
{self._format_details_text(results)}

This is an automated notification from FalconOne.
        """
        
        return self.send_email(to_emails, subject, body_text)
    
    def send_sdr_failure(self, to_emails: List[str], error_details: Dict) -> bool:
        """
        Send SDR failure alert
        
        Args:
            to_emails: Recipients
            error_details: Error information
        
        Returns:
            bool: Success status
        """
        subject = "🔧 FalconOne SDR Failure Alert"
        
        body_text = f"""
FalconOne SDR Failure Alert
===========================

An SDR failure has been detected.

Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
SDR Device: {error_details.get('device', 'Unknown')}
Error Type: {error_details.get('error_type', 'Unknown')}

Error Details:
--------------
{error_details.get('message', 'No details available')}

Failover Status: {error_details.get('failover_status', 'N/A')}

Recommended Actions:
-------------------
1. Check SDR device connection
2. Verify USB/network connectivity
3. Review SDR logs in /logs/sdr/
4. Attempt manual device reset
5. Contact support if issue persists

This is an automated alert from FalconOne.
        """
        
        return self.send_email(to_emails, subject, body_text)
    
    def _format_details_text(self, details: Dict) -> str:
        """Format details dictionary as text"""
        lines = []
        for key, value in details.items():
            if key not in ['severity', 'timestamp']:
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
        return '\n'.join(lines)
    
    def _format_details_html(self, details: Dict) -> str:
        """Format details dictionary as HTML table rows"""
        rows = []
        for key, value in details.items():
            if key not in ['severity', 'timestamp']:
                rows.append(f"<tr><th>{key.replace('_', ' ').title()}</th><td>{value}</td></tr>")
        return '\n'.join(rows)
    
    def _format_status_components(self, components: Dict) -> str:
        """Format component status as text"""
        lines = []
        for component, status in components.items():
            status_symbol = "✓" if status == "healthy" else "✗"
            lines.append(f"{status_symbol} {component}: {status}")
        return '\n'.join(lines)
    
    def _get_recommended_actions(self, alert_type: str) -> str:
        """Get recommended actions for alert type"""
        actions = {
            'dos_detected': '1. Investigate signal source\n2. Enable frequency blacklisting\n3. Alert network operator\n4. Document incident',
            'mitm_detected': '1. Isolate affected network\n2. Identify rogue base station\n3. Notify authorities\n4. Update detection signatures',
            'downgrade_attack': '1. Enable forced LTE/5G mode\n2. Check SIM security settings\n3. Review authentication logs\n4. Update security policies',
            'suci_concealment_failure': '1. Verify 5G SUCI implementation\n2. Check home network keys\n3. Review cryptographic algorithms\n4. Update carrier settings',
            'anomalous_traffic': '1. Analyze traffic patterns\n2. Check for unauthorized devices\n3. Review firewall rules\n4. Investigate data exfiltration',
        }
        return actions.get(alert_type, 'Review security logs and investigate the incident.')


def send_alert_email(to_emails: List[str], alert_type: str, details: Dict,
                    smtp_config: Dict = None) -> bool:
    """
    Convenience function to send alert email
    
    Args:
        to_emails: Recipients
        alert_type: Alert type
        details: Alert details
        smtp_config: Optional SMTP configuration override
    
    Returns:
        bool: Success status
    """
    manager = EmailAlertManager(**(smtp_config or {}))
    return manager.send_security_alert(to_emails, alert_type, details)
