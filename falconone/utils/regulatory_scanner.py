"""
FalconOne Regulatory Compliance Scanner
Pre-flight compliance checks for FCC/ETSI/global regulations
Version 1.6.2 - December 29, 2025

Capabilities:
- Scan local regulations (FCC Part 15, ETSI EN 300 328, etc.)
- Auto-limit TX power per band/location
- Simulate operations for compliance pre-check
- Flag potential violations before transmission
- Generate compliance reports

Reference: FCC CFR Title 47, ETSI harmonized standards
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    from .logger import ModuleLogger
except ImportError:
    class ModuleLogger:
        def __init__(self, name, parent):
            self.logger = logging.getLogger(name) if parent is None else parent.getChild(name)
        def info(self, msg, **kw): self.logger.info(f"{msg} {kw if kw else ''}")
        def warning(self, msg, **kw): self.logger.warning(f"{msg} {kw if kw else ''}")
        def error(self, msg, **kw): self.logger.error(f"{msg} {kw if kw else ''}")


class RegulatoryZone(Enum):
    """Regulatory zone classification"""
    FCC_USA = "fcc_usa"
    ETSI_EU = "etsi_eu"
    ARIB_JAPAN = "arib_japan"
    MIC_KOREA = "mic_korea"
    ACMA_AUSTRALIA = "acma_australia"
    GLOBAL = "global"


@dataclass
class FrequencyBand:
    """
    Frequency band specification
    
    Attributes:
        band_name: Band identifier (e.g., '2.4GHz ISM')
        start_mhz: Start frequency
        end_mhz: End frequency
        max_eirp_dbm: Maximum EIRP in dBm
        max_power_dbm: Maximum transmit power
        duty_cycle_pct: Maximum duty cycle (%)
        restrictions: Additional restrictions
    """
    band_name: str
    start_mhz: float
    end_mhz: float
    max_eirp_dbm: float
    max_power_dbm: float
    duty_cycle_pct: float = 100.0
    restrictions: List[str] = field(default_factory=list)


@dataclass
class ComplianceCheck:
    """
    Compliance check result
    
    Attributes:
        compliant: Compliance status
        zone: Regulatory zone
        band: Frequency band
        violations: List of violations
        warnings: List of warnings
        recommended_power_dbm: Recommended TX power
        timestamp: Check timestamp
    """
    compliant: bool
    zone: RegulatoryZone
    band: FrequencyBand
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommended_power_dbm: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class RegulatoryScanner:
    """
    Regulatory compliance scanner and simulator
    
    Prevents legal violations by:
    - Pre-checking TX parameters against local regulations
    - Auto-limiting power to maximum allowed
    - Simulating operations for compliance validation
    - Flagging restricted bands/modes
    
    Regulatory databases:
    - FCC Part 15 (USA): 2.4GHz ISM, 5GHz U-NII
    - ETSI EN 300 328 (EU): 2.4GHz, 5GHz
    - ARIB STD-T66 (Japan): 2.4GHz, 5GHz
    - Global ISM bands
    
    Typical usage:
        scanner = RegulatoryScanner(config, logger, zone='fcc_usa')
        check = scanner.check_compliance(freq_mhz=2437, power_dbm=30)
        if not check.compliant:
            power_dbm = check.recommended_power_dbm
        scanner.simulate_operation(freq_mhz, power_dbm, duration_ms)
    """
    
    def __init__(self, config, logger: logging.Logger, zone: str = 'fcc_usa'):
        """
        Initialize regulatory scanner
        
        Args:
            config: Configuration object
            logger: Logger instance
            zone: Regulatory zone (default: FCC USA)
        """
        self.config = config
        self.logger = ModuleLogger('RegulatoryScanner', logger)
        
        try:
            self.zone = RegulatoryZone(zone)
        except ValueError:
            self.logger.warning(f"Unknown zone {zone}, defaulting to FCC USA")
            self.zone = RegulatoryZone.FCC_USA
        
        # Configuration
        self.enabled = config.get('regulatory.enabled', True)
        self.strict_mode = config.get('regulatory.strict_mode', True)
        
        # Load regulatory database
        self.bands: Dict[RegulatoryZone, List[FrequencyBand]] = self._load_regulatory_db()
        
        # Compliance history
        self.checks: List[ComplianceCheck] = []
        self.violations_count = 0
        self.warnings_count = 0
        
        self.logger.info("Regulatory scanner initialized",
                       zone=self.zone.value,
                       enabled=self.enabled,
                       strict=self.strict_mode)
    
    def _load_regulatory_db(self) -> Dict[RegulatoryZone, List[FrequencyBand]]:
        """Load regulatory database"""
        db = {}
        
        # FCC (USA)
        db[RegulatoryZone.FCC_USA] = [
            FrequencyBand("2.4GHz ISM", 2400, 2483.5, max_eirp_dbm=36, max_power_dbm=30,
                         restrictions=["DFS not required", "Indoor/outdoor"]),
            FrequencyBand("5GHz U-NII-1", 5150, 5250, max_eirp_dbm=24, max_power_dbm=17,
                         restrictions=["Indoor only", "DFS required 5250-5350"]),
            FrequencyBand("5GHz U-NII-2A", 5250, 5350, max_eirp_dbm=24, max_power_dbm=17,
                         restrictions=["DFS required", "TPC required"]),
            FrequencyBand("5GHz U-NII-2C", 5470, 5725, max_eirp_dbm=30, max_power_dbm=24,
                         restrictions=["DFS required", "TPC required"]),
            FrequencyBand("5GHz U-NII-3", 5725, 5850, max_eirp_dbm=36, max_power_dbm=30,
                         restrictions=["No DFS", "Indoor/outdoor"]),
            FrequencyBand("900MHz ISM", 902, 928, max_eirp_dbm=36, max_power_dbm=30,
                         restrictions=["Frequency hopping or direct sequence"]),
        ]
        
        # ETSI (EU)
        db[RegulatoryZone.ETSI_EU] = [
            FrequencyBand("2.4GHz ISM", 2400, 2483.5, max_eirp_dbm=20, max_power_dbm=20,
                         restrictions=["ETSI EN 300 328", "Adaptive frequency agility"]),
            FrequencyBand("5GHz Band A", 5150, 5350, max_eirp_dbm=23, max_power_dbm=17,
                         restrictions=["Indoor only", "TPC/DFS required"]),
            FrequencyBand("5GHz Band B", 5470, 5725, max_eirp_dbm=30, max_power_dbm=24,
                         restrictions=["TPC/DFS required", "Outdoor allowed"]),
        ]
        
        # ARIB (Japan)
        db[RegulatoryZone.ARIB_JAPAN] = [
            FrequencyBand("2.4GHz", 2400, 2497, max_eirp_dbm=10, max_power_dbm=10,
                         duty_cycle_pct=100, restrictions=["ARIB STD-T66"]),
            FrequencyBand("5GHz Band 1", 5150, 5250, max_eirp_dbm=11, max_power_dbm=11,
                         restrictions=["Indoor only"]),
        ]
        
        # MIC (Korea)
        db[RegulatoryZone.MIC_KOREA] = [
            FrequencyBand("2.4GHz", 2400, 2483.5, max_eirp_dbm=23, max_power_dbm=20,
                         restrictions=["KC certification required"]),
            FrequencyBand("5GHz", 5150, 5350, max_eirp_dbm=23, max_power_dbm=17,
                         restrictions=["Indoor only", "DFS required"]),
        ]
        
        # ACMA (Australia)
        db[RegulatoryZone.ACMA_AUSTRALIA] = [
            FrequencyBand("2.4GHz ISM", 2400, 2483.5, max_eirp_dbm=30, max_power_dbm=24,
                         restrictions=["AS/NZS 4268"]),
            FrequencyBand("5GHz", 5150, 5350, max_eirp_dbm=24, max_power_dbm=17,
                         restrictions=["Indoor only", "DFS required"]),
        ]
        
        # Global ISM bands (most restrictive)
        db[RegulatoryZone.GLOBAL] = [
            FrequencyBand("2.4GHz ISM", 2400, 2483.5, max_eirp_dbm=20, max_power_dbm=20,
                         restrictions=["Global ISM", "Lowest common denominator"]),
            FrequencyBand("5GHz", 5150, 5350, max_eirp_dbm=20, max_power_dbm=17,
                         restrictions=["Indoor only globally"]),
        ]
        
        return db
    
    def check_compliance(self, freq_mhz: float, power_dbm: float,
                        bandwidth_mhz: float = 20, duty_cycle_pct: float = 100) -> ComplianceCheck:
        """
        Check compliance for transmission parameters
        
        Args:
            freq_mhz: Center frequency in MHz
            power_dbm: Transmit power in dBm
            bandwidth_mhz: Signal bandwidth
            duty_cycle_pct: Duty cycle percentage
        
        Returns:
            Compliance check result
        """
        self.logger.info(f"Checking compliance",
                       freq_mhz=freq_mhz,
                       power_dbm=power_dbm,
                       zone=self.zone.value)
        
        # Find matching band
        band = self._find_band(freq_mhz)
        
        if not band:
            # Frequency not in any allowed band
            check = ComplianceCheck(
                compliant=False,
                zone=self.zone,
                band=FrequencyBand("Unknown", freq_mhz, freq_mhz, 0, 0),
                violations=[f"Frequency {freq_mhz} MHz not in any allowed band"],
                recommended_power_dbm=0
            )
            self.violations_count += 1
            self.checks.append(check)
            return check
        
        # Check compliance
        violations = []
        warnings = []
        
        # Power check
        if power_dbm > band.max_power_dbm:
            violations.append(f"Power {power_dbm} dBm exceeds max {band.max_power_dbm} dBm")
        
        if power_dbm > band.max_eirp_dbm:
            violations.append(f"EIRP {power_dbm} dBm exceeds max {band.max_eirp_dbm} dBm")
        
        # Duty cycle check
        if duty_cycle_pct > band.duty_cycle_pct:
            violations.append(f"Duty cycle {duty_cycle_pct}% exceeds max {band.duty_cycle_pct}%")
        
        # Bandwidth check (warn if excessive)
        if bandwidth_mhz > (band.end_mhz - band.start_mhz) / 2:
            warnings.append(f"Bandwidth {bandwidth_mhz} MHz may cause adjacent channel interference")
        
        # Restriction warnings
        for restriction in band.restrictions:
            if "indoor only" in restriction.lower() and not self._is_indoor():
                warnings.append(f"Band requires indoor use: {restriction}")
            if "dfs" in restriction.lower():
                warnings.append(f"DFS required but not implemented: {restriction}")
        
        # Determine recommended power
        recommended_power = min(power_dbm, band.max_power_dbm, band.max_eirp_dbm)
        
        # Compliance determination
        compliant = len(violations) == 0
        if self.strict_mode and len(warnings) > 0:
            compliant = False
        
        check = ComplianceCheck(
            compliant=compliant,
            zone=self.zone,
            band=band,
            violations=violations,
            warnings=warnings,
            recommended_power_dbm=recommended_power
        )
        
        if not compliant:
            self.violations_count += 1
        if len(warnings) > 0:
            self.warnings_count += 1
        
        self.checks.append(check)
        
        if not compliant:
            self.logger.warning(f"Compliance FAILED",
                              violations=len(violations),
                              warnings=len(warnings))
        else:
            self.logger.info(f"Compliance PASSED")
        
        return check
    
    def simulate_operation(self, freq_mhz: float, power_dbm: float,
                          duration_ms: float, bandwidth_mhz: float = 20) -> Dict[str, Any]:
        """
        Simulate operation for compliance validation
        
        Args:
            freq_mhz: Center frequency
            power_dbm: Transmit power
            duration_ms: Operation duration
            bandwidth_mhz: Signal bandwidth
        
        Returns:
            Simulation result with compliance status
        """
        self.logger.info(f"Simulating operation",
                       freq_mhz=freq_mhz,
                       power_dbm=power_dbm,
                       duration_ms=duration_ms)
        
        # Check compliance
        check = self.check_compliance(freq_mhz, power_dbm, bandwidth_mhz)
        
        # Simulate duty cycle
        duty_cycle_pct = (duration_ms / 1000) * 100  # Assuming 1s window
        
        # Simulate emissions
        spectral_mask_compliant = self._check_spectral_mask(freq_mhz, bandwidth_mhz)
        
        # Simulate interference
        interference_risk = self._assess_interference_risk(freq_mhz, power_dbm)
        
        result = {
            'compliant': check.compliant,
            'frequency_mhz': freq_mhz,
            'power_dbm': power_dbm,
            'duration_ms': duration_ms,
            'duty_cycle_pct': duty_cycle_pct,
            'violations': check.violations,
            'warnings': check.warnings,
            'spectral_mask_compliant': spectral_mask_compliant,
            'interference_risk': interference_risk,
            'recommended_power_dbm': check.recommended_power_dbm,
            'safe_to_transmit': check.compliant and spectral_mask_compliant and interference_risk < 0.5,
        }
        
        self.logger.info(f"Simulation complete",
                       safe=result['safe_to_transmit'],
                       interference=f"{interference_risk:.2f}")
        
        return result
    
    def auto_limit_power(self, freq_mhz: float, requested_power_dbm: float) -> float:
        """
        Auto-limit power to maximum allowed
        
        Args:
            freq_mhz: Center frequency
            requested_power_dbm: Requested power
        
        Returns:
            Limited power (dBm)
        """
        band = self._find_band(freq_mhz)
        
        if not band:
            self.logger.warning(f"Frequency {freq_mhz} MHz not allowed, limiting to 0 dBm")
            return 0.0
        
        limited_power = min(requested_power_dbm, band.max_power_dbm, band.max_eirp_dbm)
        
        if limited_power < requested_power_dbm:
            self.logger.info(f"Power limited from {requested_power_dbm} to {limited_power} dBm")
        
        return limited_power
    
    def _find_band(self, freq_mhz: float) -> Optional[FrequencyBand]:
        """Find band for frequency"""
        bands = self.bands.get(self.zone, [])
        
        for band in bands:
            if band.start_mhz <= freq_mhz <= band.end_mhz:
                return band
        
        return None
    
    def _is_indoor(self) -> bool:
        """Check if operating indoors (simplified)"""
        # In production, use GPS, signal strength, or user configuration
        return self.config.get('regulatory.indoor', True)
    
    def _check_spectral_mask(self, freq_mhz: float, bandwidth_mhz: float) -> bool:
        """Check spectral mask compliance (simplified)"""
        # In production, analyze actual signal spectrum
        # For now, assume compliant if bandwidth is reasonable
        return bandwidth_mhz <= 40  # Typical max for most bands
    
    def _assess_interference_risk(self, freq_mhz: float, power_dbm: float) -> float:
        """Assess interference risk (0-1)"""
        # Simplified risk assessment
        risk = 0.0
        
        # High power = higher risk
        if power_dbm > 20:
            risk += 0.3
        
        # Crowded bands = higher risk
        if 2400 <= freq_mhz <= 2483.5:  # 2.4GHz ISM (crowded)
            risk += 0.4
        elif 5150 <= freq_mhz <= 5350:  # 5GHz (less crowded)
            risk += 0.2
        
        return min(1.0, risk)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate compliance report"""
        return {
            'regulatory_zone': self.zone.value,
            'enabled': self.enabled,
            'strict_mode': self.strict_mode,
            'total_checks': len(self.checks),
            'violations': self.violations_count,
            'warnings': self.warnings_count,
            'compliance_rate': (len(self.checks) - self.violations_count) / max(1, len(self.checks)),
            'recent_checks': [
                {
                    'timestamp': check.timestamp.isoformat(),
                    'compliant': check.compliant,
                    'band': check.band.band_name,
                    'violations': len(check.violations),
                    'warnings': len(check.warnings),
                }
                for check in self.checks[-10:]
            ],
        }
    
    def get_allowed_bands(self) -> List[FrequencyBand]:
        """Get list of allowed bands for current zone"""
        return self.bands.get(self.zone, [])
