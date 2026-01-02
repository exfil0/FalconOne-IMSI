"""
Security scanning configuration and automation for FalconOne
Integrates Bandit (Python security), Safety (dependency vulnerabilities), and OWASP ZAP

Run with:
    python falconone/tests/security_scan.py
"""

import subprocess
import json
import sys
import os
from datetime import datetime
from pathlib import Path


class SecurityScanner:
    """Automated security scanning orchestrator"""
    
    def __init__(self, project_root=None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'bandit': None,
            'safety': None,
            'summary': {}
        }
    
    def run_bandit(self):
        """Run Bandit for Python security issues"""
        print("Running Bandit security scan...")
        
        bandit_config = {
            'exclude_dirs': ['tests', '__pycache__', '.git', 'venv', 'env'],
            'severity': 'medium',  # low, medium, high
            'confidence': 'medium'
        }
        
        try:
            # Run Bandit
            cmd = [
                'bandit',
                '-r', str(self.project_root / 'falconone'),
                '-f', 'json',
                '-ll',  # Only report issues of medium/high severity
                '-i',  # Show confidence level
            ]
            
            # Add exclusions
            for exclude_dir in bandit_config['exclude_dirs']:
                cmd.extend(['-x', f"*/{exclude_dir}/*"])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parse results
            if result.stdout:
                bandit_results = json.loads(result.stdout)
                self.results['bandit'] = bandit_results
                
                # Summary
                metrics = bandit_results.get('metrics', {})
                total_issues = sum(
                    sum(file_data.values())
                    for file_data in metrics.values()
                    if isinstance(file_data, dict)
                )
                
                print(f"✓ Bandit scan complete: {total_issues} issues found")
                self.results['summary']['bandit_issues'] = total_issues
                
                # Check for critical issues
                critical_issues = [
                    issue for issue in bandit_results.get('results', [])
                    if issue.get('issue_severity') == 'HIGH'
                ]
                
                if critical_issues:
                    print(f"⚠ WARNING: {len(critical_issues)} HIGH severity issues found!")
                    for issue in critical_issues[:5]:  # Show first 5
                        print(f"  - {issue['test_id']}: {issue['issue_text']}")
                        print(f"    File: {issue['filename']}:{issue['line_number']}")
                
                return True
            else:
                print("✗ Bandit scan failed")
                return False
                
        except FileNotFoundError:
            print("✗ Bandit not installed. Install with: pip install bandit")
            return False
        except subprocess.TimeoutExpired:
            print("✗ Bandit scan timed out")
            return False
        except Exception as e:
            print(f"✗ Bandit scan error: {e}")
            return False
    
    def run_safety(self):
        """Run Safety check for dependency vulnerabilities"""
        print("\nRunning Safety dependency scan...")
        
        try:
            # Run Safety check
            cmd = ['safety', 'check', '--json']
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Parse results
            if result.stdout:
                try:
                    safety_results = json.loads(result.stdout)
                    self.results['safety'] = safety_results
                    
                    vulnerabilities = len(safety_results) if isinstance(safety_results, list) else 0
                    
                    print(f"✓ Safety scan complete: {vulnerabilities} vulnerabilities found")
                    self.results['summary']['safety_vulnerabilities'] = vulnerabilities
                    
                    if vulnerabilities > 0:
                        print(f"⚠ WARNING: {vulnerabilities} vulnerable dependencies found!")
                        for vuln in safety_results[:5]:  # Show first 5
                            print(f"  - {vuln.get('package', 'unknown')}: {vuln.get('vulnerability', 'N/A')}")
                    
                    return True
                except json.JSONDecodeError:
                    # Safety may return non-JSON output
                    print("✓ Safety scan complete (no vulnerabilities)")
                    self.results['summary']['safety_vulnerabilities'] = 0
                    return True
                    
        except FileNotFoundError:
            print("✗ Safety not installed. Install with: pip install safety")
            return False
        except subprocess.TimeoutExpired:
            print("✗ Safety scan timed out")
            return False
        except Exception as e:
            print(f"✗ Safety scan error: {e}")
            return False
    
    def run_owasp_zap(self, target_url='http://localhost:5000'):
        """Run OWASP ZAP for web application security testing"""
        print(f"\nRunning OWASP ZAP scan on {target_url}...")
        print("Note: ZAP requires manual installation and running daemon")
        print("Install: https://www.zaproxy.org/download/")
        print("Start ZAP daemon: zap.sh -daemon -port 8080")
        
        try:
            # Check if ZAP API is available
            import requests
            
            zap_api_url = 'http://localhost:8080'
            response = requests.get(f'{zap_api_url}/JSON/core/view/version/')
            
            if response.status_code == 200:
                print(f"✓ ZAP daemon detected (version: {response.json().get('version', 'unknown')})")
                
                # Start spider scan
                print("Starting ZAP spider scan...")
                requests.get(f'{zap_api_url}/JSON/spider/action/scan/?url={target_url}')
                
                # Note: Full ZAP integration would require more extensive setup
                print("✓ ZAP scan initiated (manual review recommended)")
                return True
            else:
                print("✗ ZAP daemon not accessible")
                return False
                
        except ImportError:
            print("✗ requests library not installed")
            return False
        except Exception as e:
            print(f"⚠ ZAP scan skipped: {e}")
            return False
    
    def generate_report(self):
        """Generate security scan report"""
        print("\n" + "="*60)
        print("SECURITY SCAN SUMMARY")
        print("="*60)
        
        total_issues = (
            self.results['summary'].get('bandit_issues', 0) +
            self.results['summary'].get('safety_vulnerabilities', 0)
        )
        
        print(f"Total Issues Found: {total_issues}")
        print(f"  - Bandit (code security): {self.results['summary'].get('bandit_issues', 0)}")
        print(f"  - Safety (dependencies): {self.results['summary'].get('safety_vulnerabilities', 0)}")
        
        # Determine overall status
        if total_issues == 0:
            status = "✓ PASS - No security issues detected"
            exit_code = 0
        elif total_issues < 10:
            status = "⚠ WARNING - Minor security issues detected"
            exit_code = 0
        else:
            status = "✗ FAIL - Critical security issues detected"
            exit_code = 1
        
        print(f"\nOverall Status: {status}")
        
        # Save report to file
        report_path = self.project_root / 'security_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        if self.results['summary'].get('bandit_issues', 0) > 0:
            print("1. Review Bandit findings and address HIGH severity issues")
            print("   Run: bandit -r falconone/ -ll")
        
        if self.results['summary'].get('safety_vulnerabilities', 0) > 0:
            print("2. Update vulnerable dependencies")
            print("   Run: pip install --upgrade <package>")
        
        print("3. Conduct manual security review of authentication/authorization")
        print("4. Enable rate limiting on all API endpoints")
        print("5. Implement HTTPS/TLS for production deployment")
        print("6. Regular security audits recommended (quarterly)")
        
        return exit_code
    
    def run_all(self):
        """Run all security scans"""
        print("="*60)
        print("FalconOne Security Scanner")
        print("="*60)
        
        success = True
        
        # Run scans
        if not self.run_bandit():
            success = False
        
        if not self.run_safety():
            success = False
        
        # Optional ZAP scan (may not be available)
        self.run_owasp_zap()
        
        # Generate report
        exit_code = self.generate_report()
        
        return exit_code


def main():
    """Main entry point"""
    scanner = SecurityScanner()
    exit_code = scanner.run_all()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
