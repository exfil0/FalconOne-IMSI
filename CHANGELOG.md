# Changelog

All notable changes to the FalconOne Intelligence Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.8.0] - 2025-01-XX

### Added - RANSacked Vulnerability Auditor Integration

#### Core Features (Phase 1-3)
- **RANSacked Vulnerability Database**: Comprehensive CVE tracking for 7 5G implementations
  - 97 total CVE signatures spanning 2020-2025
  - Coverage: Open5GS (14), OpenAirInterface (18), srsRAN (15), Magma (12), free5GC (13), Amarisoft (11), UERANSIM (14)
  - CVE metadata: CVSS scores (2.4-9.8), attack vectors (L2/L3/DoS/AuthBypass), severity classifications
  
- **Implementation Scanning API**: `/api/audit/ransacked/scan`
  - Scans 5G implementations for known vulnerabilities by name and version
  - Returns matching CVEs with detailed metadata (description, impact, mitigation)
  - Version-aware filtering with wildcard support (*) for all versions
  - Risk scoring algorithm based on CVSS scores
  
- **Packet-Level Auditing API**: `/api/audit/ransacked/audit-packet`
  - Deep packet inspection for vulnerability patterns in NAS/NGAP/RRC protocols
  - Protocol detection and vulnerability pattern matching
  - Hex/Base64 payload support
  - Real-time vulnerability classification
  
- **Statistics API**: `/api/audit/ransacked/stats`
  - CVE database statistics (total count, implementations, CVSS averages)
  - Top implementations by CVE count
  - Top CVEs by severity
  
- **Dashboard UI Integration**: New "RANSacked Audit" tab
  - Implementation scanner with version filtering
  - Packet auditor with hex/base64 input
  - Real-time statistics display (auto-refresh every 5 seconds)
  - Export functionality (JSON/CSV download for scan results)
  - Responsive table views with sorting and filtering

#### Security Hardening (Phase 5)
- **XSS Protection**: HTML escaping for all user-generated and CVE database content
  - 23 data fields sanitized across scan results and packet audit displays
  - `escapeHtml()` JavaScript function for client-side protection
  - Prevents script injection even if CVE database is compromised
  
- **Enhanced Rate Limiting**:
  - `/api/audit/ransacked/scan`: 10 requests/minute per user
  - `/api/audit/ransacked/audit-packet`: 20 requests/minute per user
  - `/api/audit/ransacked/stats`: 60 requests/minute per user
  - DDoS protection for all RANSacked endpoints
  
- **Comprehensive Audit Logging**:
  - `[AUDIT]` tagged logs for SIEM integration
  - Includes IP addresses, usernames, timestamps, request details
  - Scan logs: implementation, version, CVE count, risk score
  - Packet audit logs: protocol, packet size, vulnerability count, risk level
  - Forensics-ready for security incident investigation

#### Performance Optimization (Phase 7)
- **LRU Caching for Scans**:
  - `functools.lru_cache` with 128-entry capacity
  - 10x performance improvement for cached scans (10-15ms → <1ms)
  - Automatic cache eviction (Least Recently Used policy)
  - Memory overhead: ~70 KB for full cache (128 entries × 550 bytes/entry)
  - Expected cache hit rate: 85% in production workloads
  
- **Version Parsing Hardening**:
  - Try/except wrapper for malformed version strings
  - Warning logs for debugging version comparison failures
  - Graceful fallback (returns 0 for unparseable versions)

#### Deployment Configuration (Phase 8)
- **Docker Updates**:
  - Version 1.8.0 metadata with RANSacked feature descriptions
  - Environment variables: `RANSACKED_CACHE_SIZE`, `RANSACKED_RATE_LIMIT_*`
  - Health check updated to test `/api/audit/ransacked/stats` endpoint
  - Volume mounts for audit logs: `/app/logs/audit`
  - Resource limits: 1024M memory, 1.0 CPU (production), 512M memory, 0.5 CPU (reserved)
  
- **docker-compose.yml Updates**:
  - Version 1.8.0 image tags for all services
  - RANSacked environment configuration
  - Health check using RANSacked stats endpoint
  - Audit log volume mapping
  - Resource allocation (memory/CPU limits)
  
- **Kubernetes Manifest Updates**:
  - Version 1.8.0 namespace and deployment labels
  - ConfigMap with RANSacked configuration (cache size, rate limits)
  - Liveness/readiness probes updated to `/api/audit/ransacked/stats`
  - All deployment images updated to 1.8.0

### Changed

- **Datetime Handling**: Updated all `datetime.utcnow()` calls to `datetime.now(timezone.utc)` for Python 3.13+ compatibility
- **Dashboard Health Checks**: Replaced generic health endpoints with RANSacked-specific health checks
- **Security Compliance**: Improved from 75% (6/8 controls) to 94% (7.5/8 controls)

### Fixed

- **CVE-2025-8869**: Upgraded pip from 24.2 to 25.3 to address tar extraction vulnerability
  - Risk: LOW (Python 3.13 PEP 706 provides mitigation)
  - Status: Zero remaining vulnerabilities per `pip-audit`
  
- **Version Comparison Silent Failures**: Added warning logging for malformed version strings
  - Improves debugging and error visibility
  - No functional impact (already returned 0 for failed comparisons)

### Security

- **Dependency Audit**: Full scan of 153 Python packages
  - Tool: `pip-audit 2.7.3`
  - Result: Zero critical/high/medium vulnerabilities
  - Production-ready from dependency security perspective
  
- **Security Compliance Score**: 94% (7.5/8 controls passing)
  - ✅ Input validation, access control, secure storage, transport security
  - ✅ XSS prevention, rate limiting, audit logging
  - ⚠️ API key authentication (optional for internal deployments)

### Testing

- **Unit Tests**: 31/31 passing, 0 warnings
  - Test file: `falconone/tests/test_ransacked_audit.py`
  - Coverage: Database initialization, CVE retrieval, scan implementation, packet audit, statistics
  
- **API Integration Tests**: All RANSacked endpoints validated
  - Scan API with Open5GS v2.7.0 (5 CVEs expected)
  - Packet audit with NAS registration request
  - Statistics API response validation

### Documentation

Created 4 comprehensive reports (2,232 total lines):

1. **RANSACKED_SECURITY_REVIEW.md** (421 lines)
   - Security assessment of all RANSacked components
   - Identified 3 HIGH, 2 MEDIUM, 2 LOW priority issues
   - Attack vectors, remediation strategies, compliance status

2. **RANSACKED_PHASE_5_SECURITY_HARDENING.md** (618 lines)
   - Implementation details for XSS protection, rate limiting, audit logging
   - Before/after code comparisons
   - Performance impact analysis (<2ms overhead)
   - Testing validation procedures

3. **DEPENDENCY_SECURITY_AUDIT.md** (512 lines)
   - Full audit of 153 Python packages
   - CVE-2025-8869 remediation details
   - Supply chain security analysis
   - Monitoring and incident response procedures

4. **RANSACKED_PERFORMANCE_OPTIMIZATION.md** (681 lines)
   - LRU caching implementation and benchmarking
   - Performance metrics (10x improvement)
   - Memory analysis (70 KB overhead)
   - Cache hit rate projections (85%)
   - Future optimization recommendations

### Performance Metrics

- **Scan Performance**:
  - Cold scan (uncached): 10-15ms
  - Warm scan (cached): <1ms
  - Improvement: 10x faster for cached requests
  
- **Memory Usage**:
  - CVE database in memory: ~53 KB (97 CVEs × 550 bytes)
  - LRU cache overhead: ~70 KB (128 entries)
  - Total RANSacked memory footprint: ~123 KB
  
- **Cache Efficiency**:
  - Cache size: 128 entries (LRU eviction)
  - Expected hit rate: 85% (based on common scan patterns)
  - Miss rate: 15% (first-time scans, rare implementations)

### Known Limitations

- **API Key Authentication**: Not implemented (optional for internal deployments)
- **Manual UI Testing**: Requires user interaction, not automated
- **CVE Database Updates**: Manual process (no automated CVE feed integration)
- **Rate Limiting Scope**: Per-user only (no global rate limiting)

### Migration Notes

For users upgrading from v1.7.x or earlier:

1. **Environment Variables**: Add new RANSacked configuration:
   ```bash
   export RANSACKED_CACHE_SIZE=128
   export RANSACKED_RATE_LIMIT_SCAN=10
   export RANSACKED_RATE_LIMIT_PACKET=20
   export RANSACKED_RATE_LIMIT_STATS=60
   ```

2. **Docker**: Update image tags from `1.3` or `2.0` to `1.8.0`
   ```bash
   docker-compose pull
   docker-compose up -d
   ```

3. **Kubernetes**: Apply updated manifests
   ```bash
   kubectl apply -f k8s-deployment.yaml
   kubectl rollout status deployment/falconone-agent -n falconone
   ```

4. **Logs Directory**: Ensure audit log directory exists
   ```bash
   mkdir -p logs/audit
   chmod 755 logs/audit
   ```

5. **Health Check Updates**: If using custom health checks, update URLs:
   - Old: `/health`, `/ready`
   - New: `/api/audit/ransacked/stats`

### Upgrade Path

- **From 1.0.x-1.7.x**: Direct upgrade to 1.8.0 (backward compatible)
- **Database**: No schema changes required
- **Configuration**: Add RANSacked environment variables (optional, defaults provided)
- **Testing**: Run `pytest falconone/tests/test_ransacked_audit.py` to verify

---

## [1.7.0] - 2024-XX-XX

### Added
- Initial platform release
- Multi-generation IMSI/TMSI catching (2G/3G/4G/5G)
- AI/ML signal classification
- SDR integration (USRP, LimeSDR, BladeRF, RTL-SDR)
- Dashboard UI with real-time monitoring
- Celery task queue for distributed scanning
- PostgreSQL database with audit logging
- Redis caching and message broker

---

## Version History

- **1.8.0** (Current): RANSacked integration complete
- **1.7.0**: Initial platform release
- **1.0.0-1.6.0**: Internal development versions

---

## Contributing

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for contribution guidelines.

## Security

For security issues, see [SECURITY.md](SECURITY.md) or contact security@falconone.io.
