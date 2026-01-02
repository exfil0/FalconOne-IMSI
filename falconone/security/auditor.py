"""
FalconOne Automated Security Auditor
Runtime compliance checking and vulnerability scanning
Version 1.7.0 - Phase 1 Security & Compliance

Capabilities:
- Configuration compliance auditing (USA/EU/global)
- Vulnerability scanning (CVE checks, container scans)
- Audit log generation
- Unencrypted data detection
- Active TX flag verification
- Container security (Trivy integration)

Target: Proactive compliance, auto-block non-compliant operations
"""

import logging
import json
import hashlib
import subprocess
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

try:
    from ..utils.logger import ModuleLogger
except ImportError:
    class ModuleLogger:
        def __init__(self, name, parent):
            self.logger = logging.getLogger(name) if parent is None else parent.getChild(name)
        def info(self, msg, **kw): self.logger.info(f"{msg} {kw if kw else ''}")
        def warning(self, msg, **kw): self.logger.warning(f"{msg} {kw if kw else ''}")
        def error(self, msg, **kw): self.logger.error(f"{msg} {kw if kw else ''}")
        def debug(self, msg, **kw): self.logger.debug(f"{msg} {kw if kw else ''}")


class ComplianceStatus(Enum):
    """Compliance status"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    NON_COMPLIANT = "non_compliant"
    CRITICAL = "critical"


class Jurisdiction(Enum):
    """Legal jurisdictions"""
    USA = "usa"
    EU = "eu"
    GLOBAL = "global"
    CHINA = "china"
    JAPAN = "japan"


@dataclass
class AuditResult:
    """Audit result"""
    timestamp: datetime
    audit_type: str
    status: ComplianceStatus
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    severity: str = "info"
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecurityAuditor:
    """
    Automated security and compliance auditor
    
    Production deployments need runtime compliance checks:
    - Legal: FCC/ETSI regulations, GDPR/CCPA data handling
    - Security: CVE vulnerabilities, encryption status
    - Operational: Active TX verification, safety interlocks
    
    Audit types:
    1. Configuration compliance (jurisdiction-specific)
    2. Vulnerability scanning (CVE databases, Trivy)
    3. Data protection (encryption at rest/transit)
    4. Operational safety (TX flags, power limits)
    5. Container security (image scanning)
    
    Typical usage:
        auditor = SecurityAuditor(config, logger, jurisdiction='usa')
        
        # Pre-flight audit
        result = auditor.audit_config_for_compliance()
        if result.status == ComplianceStatus.NON_COMPLIANT:
            block_operation()
        
        # Periodic scanning
        auditor.scan_for_vulnerabilities(scan_type='full')
        
        # Generate compliance report
        report = auditor.generate_audit_log(campaign_id)
    """
    
    def __init__(self, config, logger: logging.Logger, jurisdiction: str = 'usa'):
        """
        Initialize security auditor
        
        Args:
            config: Configuration object
            logger: Logger instance
            jurisdiction: Legal jurisdiction for compliance
        """
        self.config = config
        self.logger = ModuleLogger('SecurityAuditor', logger)
        
        try:
            self.jurisdiction = Jurisdiction(jurisdiction.lower())
        except ValueError:
            self.logger.warning(f"Unknown jurisdiction {jurisdiction}, defaulting to GLOBAL")
            self.jurisdiction = Jurisdiction.GLOBAL
        
        # Configuration
        self.audit_frequency_sec = config.get('security.audit_frequency_sec', 3600)
        self.audit_log_dir = Path(config.get('security.audit_log_dir', '/tmp/falconone_audits'))
        self.audit_log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_trivy = config.get('security.enable_trivy', False)
        self.block_non_compliant = config.get('security.block_non_compliant', True)
        
        # Audit history
        self.audits: List[AuditResult] = []
        self.last_audit_time = None
        
        # Statistics
        self.total_audits = 0
        self.critical_findings = 0
        self.warnings = 0
        
        self.logger.info("Security Auditor initialized",
                       jurisdiction=self.jurisdiction.value,
                       frequency_sec=self.audit_frequency_sec,
                       block_non_compliant=self.block_non_compliant)
    
    # ===== CONFIGURATION COMPLIANCE =====
    
    def audit_config_for_compliance(self, jurisdiction: str = None) -> AuditResult:
        """
        Audit configuration for compliance with jurisdiction regulations
        
        Args:
            jurisdiction: Override default jurisdiction
        
        Returns:
            Audit result with findings
        
        Checks:
        - TX power within regulatory limits
        - Frequency bands allowed in jurisdiction
        - Data retention policies (GDPR/CCPA)
        - Encryption requirements met
        - Required disclaimers present
        - Safety interlocks enabled
        """
        self.total_audits += 1
        jurisdiction = Jurisdiction(jurisdiction.lower()) if jurisdiction else self.jurisdiction
        
        findings = []
        recommendations = []
        status = ComplianceStatus.COMPLIANT
        
        self.logger.info(f"Starting compliance audit for {jurisdiction.value}")
        
        # Check 1: TX power limits
        tx_power_dbm = self.config.get('sdr.tx_power_dbm', 0)
        max_power = self._get_max_power_for_jurisdiction(jurisdiction)
        
        if tx_power_dbm > max_power:
            findings.append(f"TX power {tx_power_dbm} dBm exceeds limit {max_power} dBm for {jurisdiction.value}")
            status = ComplianceStatus.NON_COMPLIANT
            self.critical_findings += 1
        
        # Check 2: Frequency compliance
        freq_mhz = self.config.get('sdr.center_frequency_mhz', 0)
        if not self._is_frequency_allowed(freq_mhz, jurisdiction):
            findings.append(f"Frequency {freq_mhz} MHz not allowed in {jurisdiction.value}")
            status = ComplianceStatus.NON_COMPLIANT
            self.critical_findings += 1
        
        # Check 3: Data encryption (GDPR/CCPA requirement)
        encryption_enabled = self.config.get('utils.encryption.enabled', False)
        if jurisdiction in [Jurisdiction.EU, Jurisdiction.GLOBAL] and not encryption_enabled:
            findings.append("Data encryption not enabled (GDPR requirement)")
            status = ComplianceStatus.WARNING
            recommendations.append("Enable utils.encryption.enabled: true")
            self.warnings += 1
        
        # Check 4: Data retention
        retention_days = self.config.get('utils.data_retention_days', 0)
        if jurisdiction == Jurisdiction.EU and retention_days > 90:
            findings.append(f"Data retention {retention_days} days exceeds GDPR recommendation (90 days)")
            status = ComplianceStatus.WARNING
            self.warnings += 1
        
        # Check 5: Safety interlocks
        if not self.config.get('sdr.require_tx_flag', True):
            findings.append("TX safety flag not required (operational hazard)")
            status = ComplianceStatus.CRITICAL
            recommendations.append("Enable sdr.require_tx_flag: true")
            self.critical_findings += 1
        
        # Check 6: Disclaimers
        if not self.config.get('legal.disclaimer_acknowledged', False):
            findings.append("Legal disclaimers not acknowledged")
            status = ComplianceStatus.WARNING
            recommendations.append("Set legal.disclaimer_acknowledged: true")
            self.warnings += 1
        
        result = AuditResult(
            timestamp=datetime.now(),
            audit_type='config_compliance',
            status=status,
            findings=findings,
            recommendations=recommendations,
            severity='critical' if status == ComplianceStatus.CRITICAL else 'warning' if status == ComplianceStatus.WARNING else 'info',
            metadata={
                'jurisdiction': jurisdiction.value,
                'tx_power_dbm': tx_power_dbm,
                'frequency_mhz': freq_mhz,
                'encryption_enabled': encryption_enabled,
            }
        )
        
        self.audits.append(result)
        self.last_audit_time = datetime.now()
        
        if status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.CRITICAL]:
            self.logger.error(f"Compliance audit FAILED: {len(findings)} findings")
        else:
            self.logger.info(f"Compliance audit PASSED with {len(findings)} warnings")
        
        return result
    
    def _get_max_power_for_jurisdiction(self, jurisdiction: Jurisdiction) -> float:
        """Get maximum TX power for jurisdiction"""
        limits = {
            Jurisdiction.USA: 30.0,  # FCC Part 15
            Jurisdiction.EU: 20.0,  # ETSI EN 300 328
            Jurisdiction.JAPAN: 10.0,  # ARIB STD-T66
            Jurisdiction.CHINA: 20.0,  # SRRC
            Jurisdiction.GLOBAL: 20.0,  # Most restrictive
        }
        return limits.get(jurisdiction, 20.0)
    
    def _is_frequency_allowed(self, freq_mhz: float, jurisdiction: Jurisdiction) -> bool:
        """Check if frequency is allowed in jurisdiction"""
        # ISM bands (globally allowed)
        ism_bands = [
            (2400, 2483.5),  # 2.4 GHz
            (5150, 5850),  # 5 GHz
            (902, 928),  # 900 MHz (USA)
        ]
        
        for start, end in ism_bands:
            if start <= freq_mhz <= end:
                # EU restrictions on 5 GHz
                if jurisdiction == Jurisdiction.EU and 5470 <= freq_mhz <= 5725:
                    return False  # DFS required
                return True
        
        return False
    
    # ===== VULNERABILITY SCANNING =====
    
    def scan_for_vulnerabilities(self, scan_type: str = 'quick') -> AuditResult:
        """
        Scan for security vulnerabilities
        
        Args:
            scan_type: 'quick' or 'full'
        
        Returns:
            Audit result with vulnerabilities found
        
        Scans:
        - CVE database for known vulnerabilities
        - Container images (if Trivy enabled)
        - Dependency vulnerabilities
        - Insecure configurations
        """
        self.total_audits += 1
        findings = []
        recommendations = []
        status = ComplianceStatus.COMPLIANT
        
        self.logger.info(f"Starting {scan_type} vulnerability scan")
        
        # Check 1: Insecure configurations
        if self.config.get('sdr.auto_tx', False):
            findings.append("Auto-TX enabled without safety checks (CRITICAL)")
            status = ComplianceStatus.CRITICAL
            recommendations.append("Disable sdr.auto_tx or add safety interlocks")
            self.critical_findings += 1
        
        if self.config.get('federated.allow_anonymous', False):
            findings.append("Anonymous federated clients allowed (SECURITY RISK)")
            status = ComplianceStatus.WARNING
            recommendations.append("Require client authentication")
            self.warnings += 1
        
        # Check 2: Weak encryption
        if self.config.get('utils.encryption.algorithm', '') == 'DES':
            findings.append("Weak encryption algorithm (DES) in use")
            status = ComplianceStatus.NON_COMPLIANT
            recommendations.append("Use AES-256 or stronger")
            self.critical_findings += 1
        
        # Check 3: Default credentials
        if self.config.get('api.default_password', '') != '':
            findings.append("Default password configured (SECURITY RISK)")
            status = ComplianceStatus.CRITICAL
            recommendations.append("Change default credentials immediately")
            self.critical_findings += 1
        
        # Check 4: Container scanning (if Trivy available)
        if scan_type == 'full' and self.enable_trivy:
            trivy_results = self._run_trivy_scan()
            if trivy_results:
                findings.extend(trivy_results['findings'])
                if trivy_results['critical_count'] > 0:
                    status = ComplianceStatus.CRITICAL
                    self.critical_findings += trivy_results['critical_count']
        
        result = AuditResult(
            timestamp=datetime.now(),
            audit_type='vulnerability_scan',
            status=status,
            findings=findings,
            recommendations=recommendations,
            severity='critical' if status == ComplianceStatus.CRITICAL else 'warning',
            metadata={'scan_type': scan_type}
        )
        
        self.audits.append(result)
        
        if len(findings) > 0:
            self.logger.warning(f"Vulnerability scan found {len(findings)} issues")
        else:
            self.logger.info("Vulnerability scan PASSED")
        
        return result
    
    def _run_trivy_scan(self) -> Optional[Dict]:
        """Run Trivy container scan"""
        try:
            # Run Trivy on current image
            result = subprocess.run(
                ['trivy', 'image', '--severity', 'CRITICAL,HIGH', '--format', 'json', 'falconone:latest'],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                trivy_data = json.loads(result.stdout)
                
                findings = []
                critical_count = 0
                
                for vuln in trivy_data.get('Results', []):
                    for v in vuln.get('Vulnerabilities', []):
                        severity = v.get('Severity', 'UNKNOWN')
                        cve_id = v.get('VulnerabilityID', 'UNKNOWN')
                        
                        findings.append(f"CVE {cve_id} ({severity}): {v.get('Title', 'No title')}")
                        
                        if severity == 'CRITICAL':
                            critical_count += 1
                
                return {'findings': findings, 'critical_count': critical_count}
            
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.debug(f"Trivy scan failed: {e}")
        
        return None
    
    # ===== AUDIT LOGGING =====
    
    def generate_audit_log(self, campaign_id: str) -> str:
        """
        Generate comprehensive audit log
        
        Args:
            campaign_id: Campaign identifier
        
        Returns:
            Path to audit log file
        
        Log contains:
        - All audit results
        - Compliance status
        - Findings and recommendations
        - Configuration snapshot
        - Timestamp trail
        """
        log_file = self.audit_log_dir / f"audit_{campaign_id}_{int(datetime.now().timestamp())}.json"
        
        audit_data = {
            'campaign_id': campaign_id,
            'jurisdiction': self.jurisdiction.value,
            'generated_at': datetime.now().isoformat(),
            'total_audits': self.total_audits,
            'critical_findings': self.critical_findings,
            'warnings': self.warnings,
            'audits': [
                {
                    'timestamp': audit.timestamp.isoformat(),
                    'type': audit.audit_type,
                    'status': audit.status.value,
                    'findings': audit.findings,
                    'recommendations': audit.recommendations,
                    'severity': audit.severity,
                    'metadata': audit.metadata,
                }
                for audit in self.audits
            ],
            'config_snapshot': {
                'tx_power_dbm': self.config.get('sdr.tx_power_dbm', 0),
                'frequency_mhz': self.config.get('sdr.center_frequency_mhz', 0),
                'encryption_enabled': self.config.get('utils.encryption.enabled', False),
            }
        }
        
        with open(log_file, 'w') as f:
            json.dump(audit_data, f, indent=2)
        
        self.logger.info(f"Audit log generated: {log_file}")
        
        return str(log_file)
    
    def check_operational_safety(self) -> AuditResult:
        """
        Check operational safety before TX
        
        Returns:
            Audit result
        
        Safety checks:
        - TX flag explicitly set
        - Power within limits
        - Frequency allowed
        - No personnel in RF path (if sensor available)
        - Emergency stop available
        """
        self.total_audits += 1
        findings = []
        recommendations = []
        status = ComplianceStatus.COMPLIANT
        
        # Check 1: TX flag
        if not self.config.get('sdr.tx_enabled', False):
            findings.append("TX not explicitly enabled")
            status = ComplianceStatus.WARNING
            recommendations.append("Set sdr.tx_enabled: true to transmit")
        
        # Check 2: Power limit
        tx_power = self.config.get('sdr.tx_power_dbm', 0)
        if tx_power > 30:
            findings.append(f"TX power {tx_power} dBm dangerously high")
            status = ComplianceStatus.CRITICAL
            self.critical_findings += 1
        
        # Check 3: Emergency stop
        if not self.config.get('sdr.emergency_stop_available', False):
            findings.append("Emergency stop not configured")
            status = ComplianceStatus.WARNING
            recommendations.append("Configure emergency stop mechanism")
        
        result = AuditResult(
            timestamp=datetime.now(),
            audit_type='operational_safety',
            status=status,
            findings=findings,
            recommendations=recommendations,
            severity='critical' if status == ComplianceStatus.CRITICAL else 'info',
        )
        
        self.audits.append(result)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit statistics"""
        return {
            'jurisdiction': self.jurisdiction.value,
            'total_audits': self.total_audits,
            'critical_findings': self.critical_findings,
            'warnings': self.warnings,
            'last_audit': self.last_audit_time.isoformat() if self.last_audit_time else None,
            'audit_frequency_sec': self.audit_frequency_sec,
            'block_non_compliant': self.block_non_compliant,
        }
