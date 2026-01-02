#!/usr/bin/env python3
"""
Comprehensive Dashboard Audit Script
Checks for all potential issues preventing UI display
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("FALCONONE DASHBOARD COMPREHENSIVE AUDIT")
print("=" * 70)

# Test 1: Import Check
print("\n[TEST 1] Checking imports...")
try:
    from falconone.ui.dashboard import app, DashboardServer
    print("OK - Dashboard imports successful")
except Exception as e:
    print(f"ERROR - Import failed: {e}")
    sys.exit(1)

# Test 2: Route Registration
print("\n[TEST 2] Checking route registration...")
routes = []
for rule in app.url_map.iter_rules():
    routes.append(f"{rule.endpoint}: {rule.rule}")
print(f"OK - Found {len(routes)} registered routes:")
for route in sorted(routes)[:10]:  # Show first 10
    print(f"  - {route}")

# Test 3: Dashboard Initialization
print("\n[TEST 3] Testing dashboard initialization...")
try:
    import logging
    logger = logging.getLogger('TestLogger')
    config = {
        'dashboard': {
            'host': '127.0.0.1',
            'port': 5000,
            'refresh_rate_ms': 100
        }
    }
    dashboard = DashboardServer(config=config, logger=logger, core_system=None)
    print(f"OK - Dashboard created (refresh_rate: {dashboard.refresh_rate_ms}ms)")
except Exception as e:
    print(f"ERROR - Dashboard creation failed: {e}")
    sys.exit(1)

# Test 4: Template Validation
print("\n[TEST 4] Validating HTML template...")
try:
    with open('falconone/ui/dashboard.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find template
    import re
    template_match = re.search(r'DASHBOARD_HTML_TEMPLATE = """(.*?)"""', content, re.DOTALL)
    
    if not template_match:
        print("ERROR - Template not found in dashboard.py")
        sys.exit(1)
    
    template = template_match.group(1)
    print(f"OK - Template found ({len(template):,} characters)")
    
    # Critical elements check
    checks = {
        '<html': 'HTML root tag',
        '<head>': 'Head section',
        '<body': 'Body tag',
        'class="sidebar"': 'Sidebar navigation',
        'class="main-content"': 'Main content area',
        'class="tab-content"': 'Tab content containers',
        'function showTab': 'Tab switching function',
        'DOMContentLoaded': 'Page load initialization',
        'Chart': 'Chart.js usage',
    }
    
    missing = []
    for check, description in checks.items():
        if check in template:
            print(f"  OK - {description}")
        else:
            print(f"  ERROR - MISSING: {description}")
            missing.append(description)
    
    if missing:
        print(f"\nCRITICAL: {len(missing)} essential elements missing!")
    
    # Check for syntax issues
    single_quotes = template.count("'")
    double_quotes = template.count('"')
    backticks = template.count('`')
    print(f"\n  Quote counts: ' ({single_quotes}), \" ({double_quotes}), ` ({backticks})")
    
except Exception as e:
    print(f"ERROR - Template validation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test Server Response
print("\n[TEST 5] Testing server response...")
try:
    with app.test_client() as client:
        response = client.get('/')
        print(f"OK - Status code: {response.status_code}")
        print(f"OK - Content length: {len(response.data):,} bytes")
        print(f"OK - Content type: {response.content_type}")
        
        # Check response content
        html = response.data.decode('utf-8')
        
        critical_elements = [
            ('<html', 'HTML tag'),
            ('<body', 'Body tag'),
            ('class="sidebar"', 'Sidebar'),
            ('class="main-content"', 'Main content'),
            ('id="tab-overview"', 'Overview tab'),
            ('function showTab', 'Tab function'),
        ]
        
        print("\n  Content validation:")
        errors = []
        for element, description in critical_elements:
            if element in html:
                print(f"    OK - {description} present")
            else:
                print(f"    ERROR - {description} MISSING")
                errors.append(description)
        
        if errors:
            print(f"\n  CRITICAL: {len(errors)} elements missing from rendered HTML!")
        
        # Save sample for inspection
        with open('dashboard_response_sample.html', 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"\n  OK - Saved full response to dashboard_response_sample.html ({len(html):,} bytes)")
        
except Exception as e:
    print(f"ERROR - Server test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: JavaScript Validation
print("\n[TEST 6] Extracting and validating JavaScript...")
try:
    # Extract JavaScript from template
    js_pattern = re.compile(r'<script[^>]*>(.*?)</script>', re.DOTALL)
    js_blocks = js_pattern.findall(template)
    
    print(f"OK - Found {len(js_blocks)} JavaScript blocks")
    
    total_js_size = sum(len(block) for block in js_blocks)
    print(f"  Total JavaScript: {total_js_size:,} characters")
    
    # Check for common JS issues
    js_issues = []
    for i, block in enumerate(js_blocks, 1):
        if len(block) > 1000:  # Only check substantial blocks
            issues = []
            
            # Check for unclosed brackets
            if block.count('{') != block.count('}'):
                issues.append("Unmatched curly braces")
            if block.count('[') != block.count(']'):
                issues.append("Unmatched square brackets")
            if block.count('(') != block.count(')'):
                issues.append("Unmatched parentheses")
            
            if issues:
                print(f"  ERROR - Block {i}: {', '.join(issues)}")
                js_issues.extend(issues)
            else:
                print(f"  OK - Block {i}: No bracket issues ({len(block):,} chars)")
    
    if js_issues:
        print(f"\n  CRITICAL: Found {len(js_issues)} JavaScript syntax issues!")
    
except Exception as e:
    print(f"ERROR - JavaScript validation failed: {e}")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
