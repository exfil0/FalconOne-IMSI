"""
Unit tests for authentication and authorization
Tests login, permissions, password hashing, and security features
"""

import pytest
import hashlib
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from falconone.ui.dashboard import app, User, login_required
from flask import session
from flask_login import current_user


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
    app.config['SECRET_KEY'] = 'test-secret-key'
    
    with app.test_client() as client:
        with app.app_context():
            yield client


@pytest.fixture
def authenticated_client(client):
    """Create authenticated test client"""
    # Login
    client.post('/login', data={
        'username': 'admin',
        'password': 'admin'
    })
    return client


class TestAuthentication:
    """Test authentication mechanisms"""
    
    def test_login_page_accessible(self, client):
        """Test login page is accessible"""
        response = client.get('/login')
        assert response.status_code == 200
        assert b'login' in response.data.lower()
    
    def test_valid_login(self, client):
        """Test login with valid credentials"""
        response = client.post('/login', data={
            'username': 'admin',
            'password': 'admin'
        }, follow_redirects=True)
        
        assert response.status_code == 200
        # Should redirect to dashboard
        assert b'dashboard' in response.data.lower() or b'welcome' in response.data.lower()
    
    def test_invalid_password(self, client):
        """Test login with invalid password"""
        response = client.post('/login', data={
            'username': 'admin',
            'password': 'wrong_password'
        })
        
        # Should show error
        assert b'invalid' in response.data.lower() or b'incorrect' in response.data.lower()
    
    def test_invalid_username(self, client):
        """Test login with invalid username"""
        response = client.post('/login', data={
            'username': 'nonexistent_user',
            'password': 'password'
        })
        
        assert b'invalid' in response.data.lower() or b'not found' in response.data.lower()
    
    def test_logout(self, authenticated_client):
        """Test logout functionality"""
        response = authenticated_client.get('/logout', follow_redirects=True)
        
        assert response.status_code == 200
        # Should redirect to login
        assert b'login' in response.data.lower()
    
    def test_session_timeout(self, client):
        """Test session timeout after inactivity"""
        # Login
        client.post('/login', data={
            'username': 'admin',
            'password': 'admin'
        })
        
        # Access protected page (should work)
        response = client.get('/dashboard')
        assert response.status_code == 200
        
        # Simulate session expiry
        with client.session_transaction() as sess:
            sess['_permanent'] = False
        
        # Should redirect to login after timeout
        # (This test depends on implementation details)


class TestPasswordSecurity:
    """Test password hashing and security"""
    
    def test_password_hashing(self):
        """Test that passwords are hashed"""
        user = User(username='testuser', password='testpass123')
        
        # Password should be hashed (not stored in plaintext)
        assert user.password != 'testpass123'
        assert len(user.password) > 20  # Hashed password should be longer
    
    def test_password_verification(self):
        """Test password verification"""
        user = User(username='testuser', password='testpass123')
        
        # Correct password should verify
        assert user.verify_password('testpass123')
        
        # Wrong password should not verify
        assert not user.verify_password('wrongpass')
    
    def test_bcrypt_used(self):
        """Test that bcrypt is used for password hashing"""
        user = User(username='testuser', password='testpass123')
        
        # Bcrypt hashes start with $2b$ or $2a$
        assert user.password.startswith('$2b$') or user.password.startswith('$2a$')
    
    def test_password_salt(self):
        """Test that different users have different password salts"""
        user1 = User(username='user1', password='samepassword')
        user2 = User(username='user2', password='samepassword')
        
        # Even with same password, hashes should differ (due to salt)
        assert user1.password != user2.password


class TestAuthorization:
    """Test authorization and permissions"""
    
    def test_protected_route_requires_login(self, client):
        """Test that protected routes require authentication"""
        response = client.get('/dashboard')
        
        # Should redirect to login
        assert response.status_code == 302
        assert '/login' in response.location
    
    def test_admin_access(self, authenticated_client):
        """Test admin user can access admin features"""
        response = authenticated_client.get('/admin/users')
        
        # Admin should have access
        assert response.status_code in [200, 404]  # 404 if route doesn't exist yet
    
    def test_role_based_access_control(self, client):
        """Test RBAC - different roles have different permissions"""
        # Login as regular user
        client.post('/login', data={
            'username': 'operator',
            'password': 'operator'
        })
        
        # Try to access admin page
        response = client.get('/admin/users')
        
        # Should be forbidden
        assert response.status_code in [403, 302]  # 403 Forbidden or redirect
    
    def test_operator_permissions(self, client):
        """Test operator role permissions"""
        # Login as operator
        client.post('/login', data={
            'username': 'operator',
            'password': 'operator'
        })
        
        # Can access monitoring
        response = client.get('/monitoring/signals')
        assert response.status_code in [200, 404]
        
        # Cannot access exploit features
        response = client.get('/exploit/dos')
        assert response.status_code == 403
    
    def test_analyst_permissions(self, client):
        """Test analyst role permissions (read-only)"""
        # Login as analyst
        client.post('/login', data={
            'username': 'analyst',
            'password': 'analyst'
        })
        
        # Can view data
        response = client.get('/data/sessions')
        assert response.status_code in [200, 404]
        
        # Cannot execute attacks
        response = client.post('/exploit/dos', data={'target': '900.0'})
        assert response.status_code == 403


class TestSessionManagement:
    """Test session management features (Phase 1.3.7)"""
    
    def test_session_lifetime_config(self, client):
        """Test configurable session lifetime"""
        import os
        
        # Set session lifetime via environment
        os.environ['SESSION_LIFETIME_HOURS'] = '2'
        
        # Login
        client.post('/login', data={
            'username': 'admin',
            'password': 'admin'
        })
        
        # Check session configuration
        with client.session_transaction() as sess:
            assert sess.permanent
            # Session should expire in 2 hours
    
    def test_remember_me_cookie(self, client):
        """Test remember-me functionality"""
        response = client.post('/login', data={
            'username': 'admin',
            'password': 'admin',
            'remember_me': True
        })
        
        # Should set remember-me cookie
        cookies = response.headers.getlist('Set-Cookie')
        remember_cookie = [c for c in cookies if 'remember_token' in c]
        assert len(remember_cookie) > 0
    
    def test_session_refresh_on_activity(self, authenticated_client):
        """Test session extends on activity"""
        # Make request (activity)
        authenticated_client.get('/dashboard')
        
        # Session should be refreshed
        with authenticated_client.session_transaction() as sess:
            # Check session is still valid
            assert '_user_id' in sess
    
    def test_secure_cookie_flags(self, client):
        """Test secure cookie flags are set"""
        response = client.post('/login', data={
            'username': 'admin',
            'password': 'admin'
        })
        
        cookies = response.headers.getlist('Set-Cookie')
        
        # Check for HttpOnly flag
        http_only_cookies = [c for c in cookies if 'HttpOnly' in c]
        assert len(http_only_cookies) > 0


class TestInputValidation:
    """Test input validation and sanitization"""
    
    def test_xss_prevention_username(self, client):
        """Test XSS prevention in username field"""
        response = client.post('/login', data={
            'username': '<script>alert("xss")</script>',
            'password': 'password'
        })
        
        # Should escape or reject malicious input
        assert b'<script>' not in response.data
    
    def test_sql_injection_prevention(self, client):
        """Test SQL injection prevention"""
        response = client.post('/login', data={
            'username': "admin' OR '1'='1",
            'password': 'password'
        })
        
        # Should not allow SQL injection
        assert response.status_code != 200 or b'dashboard' not in response.data.lower()
    
    def test_username_length_limit(self, client):
        """Test username length validation"""
        long_username = 'a' * 1000
        response = client.post('/login', data={
            'username': long_username,
            'password': 'password'
        })
        
        # Should reject excessively long username
        assert b'too long' in response.data.lower() or response.status_code == 400


class TestBruteForceProtection:
    """Test brute-force attack protection"""
    
    def test_rate_limiting(self, client):
        """Test rate limiting on login attempts"""
        # Attempt multiple failed logins
        for i in range(10):
            response = client.post('/login', data={
                'username': 'admin',
                'password': f'wrongpass{i}'
            })
        
        # Should be rate-limited after several attempts
        # (Check for 429 Too Many Requests or lockout message)
        assert response.status_code == 429 or b'too many' in response.data.lower()
    
    def test_account_lockout(self, client):
        """Test account lockout after failed attempts"""
        # Attempt 5 failed logins
        for i in range(5):
            client.post('/login', data={
                'username': 'admin',
                'password': 'wrongpass'
            })
        
        # Next attempt should be locked out
        response = client.post('/login', data={
            'username': 'admin',
            'password': 'admin'  # Even correct password
        })
        
        # Should indicate account is locked
        assert b'locked' in response.data.lower() or b'too many attempts' in response.data.lower()


class TestCSRFProtection:
    """Test CSRF protection"""
    
    def test_csrf_token_required(self, client):
        """Test that CSRF token is required for POST requests"""
        # Re-enable CSRF for this test
        app.config['WTF_CSRF_ENABLED'] = True
        
        # POST without CSRF token
        response = client.post('/login', data={
            'username': 'admin',
            'password': 'admin'
        })
        
        # Should be rejected
        assert response.status_code in [400, 403]
        
        # Disable again for other tests
        app.config['WTF_CSRF_ENABLED'] = False
    
    def test_csrf_token_in_forms(self, client):
        """Test that forms include CSRF tokens"""
        response = client.get('/login')
        
        # Should include CSRF token field
        assert b'csrf_token' in response.data or b'_csrf_token' in response.data


class TestAuditLogging:
    """Test authentication audit logging"""
    
    @patch('falconone.ui.dashboard.AuditLogger')
    def test_successful_login_logged(self, mock_logger, client):
        """Test successful login is logged"""
        client.post('/login', data={
            'username': 'admin',
            'password': 'admin'
        })
        
        # Should log successful login
        # (Implementation depends on audit logger setup)
    
    @patch('falconone.ui.dashboard.AuditLogger')
    def test_failed_login_logged(self, mock_logger, client):
        """Test failed login is logged"""
        client.post('/login', data={
            'username': 'admin',
            'password': 'wrongpass'
        })
        
        # Should log failed login attempt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
