"""
Load testing for FalconOne Dashboard API using Locust
Tests API performance under various load conditions

Run with:
    locust -f falconone/tests/locustfile.py --host=http://localhost:5000
    
Or headless:
    locust -f falconone/tests/locustfile.py --host=http://localhost:5000 --users 100 --spawn-rate 10 --run-time 60s --headless
"""

from locust import HttpUser, task, between, events
import json
import random
import time


class DashboardUser(HttpUser):
    """Simulated dashboard user"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Called when user starts - login"""
        self.login()
    
    def login(self):
        """Authenticate user"""
        response = self.client.post("/api/login", json={
            "username": "admin",
            "password": "admin"
        })
        
        if response.status_code == 200:
            self.authenticated = True
        else:
            self.authenticated = False
    
    @task(10)
    def view_targets(self):
        """View targets list (common operation)"""
        if not hasattr(self, 'authenticated') or not self.authenticated:
            return
        
        self.client.get("/api/targets")
    
    @task(5)
    def view_specific_target(self):
        """View specific target details"""
        if not hasattr(self, 'authenticated') or not self.authenticated:
            return
        
        # Random target ID (assumes some targets exist)
        target_id = random.randint(1, 100)
        self.client.get(f"/api/targets/{target_id}")
    
    @task(3)
    def create_target(self):
        """Create new target"""
        if not hasattr(self, 'authenticated') or not self.authenticated:
            return
        
        target_data = {
            "imsi": f"001010{random.randint(0, 999999):012d}",
            "imei": f"{random.randint(0, 999999999999999):015d}",
            "msisdn": f"+1{random.randint(1000000000, 9999999999)}"
        }
        
        self.client.post("/api/targets", json=target_data)
    
    @task(2)
    def view_signals(self):
        """View detected signals"""
        if not hasattr(self, 'authenticated') or not self.authenticated:
            return
        
        self.client.get("/api/signals")
    
    @task(2)
    def view_exploits(self):
        """View exploit operations"""
        if not hasattr(self, 'authenticated') or not self.authenticated:
            return
        
        self.client.get("/api/exploits")
    
    @task(1)
    def execute_scan(self):
        """Execute frequency scan"""
        if not hasattr(self, 'authenticated') or not self.authenticated:
            return
        
        scan_config = {
            "start_freq": 900e6,
            "end_freq": 1800e6,
            "step": 200e3
        }
        
        self.client.post("/api/scan", json=scan_config)
    
    @task(8)
    def view_dashboard(self):
        """View main dashboard"""
        if not hasattr(self, 'authenticated') or not self.authenticated:
            return
        
        self.client.get("/api/dashboard/stats")


class AdminUser(HttpUser):
    """Simulated admin user performing heavy operations"""
    
    wait_time = between(2, 5)
    
    def on_start(self):
        """Login as admin"""
        response = self.client.post("/api/login", json={
            "username": "admin",
            "password": "admin"
        })
        self.authenticated = response.status_code == 200
    
    @task(5)
    def view_all_users(self):
        """View all users (admin only)"""
        if not self.authenticated:
            return
        
        self.client.get("/api/users")
    
    @task(2)
    def create_user(self):
        """Create new user (admin only)"""
        if not self.authenticated:
            return
        
        user_data = {
            "username": f"user_{random.randint(1000, 9999)}",
            "password": "password123",
            "role": random.choice(["operator", "viewer"]),
            "full_name": f"Test User {random.randint(1, 1000)}"
        }
        
        self.client.post("/api/users", json=user_data)
    
    @task(3)
    def execute_exploit(self):
        """Execute exploit (admin/operator only)"""
        if not self.authenticated:
            return
        
        exploit_config = {
            "exploit_type": "dos_attack",
            "target_frequency": 900e6,
            "duration_sec": 10,
            "power_dbm": 20
        }
        
        self.client.post("/api/exploits", json=exploit_config)
    
    @task(1)
    def export_data(self):
        """Export data to CSV"""
        if not self.authenticated:
            return
        
        self.client.get("/api/export/targets?format=csv")
    
    @task(1)
    def backup_database(self):
        """Trigger database backup"""
        if not self.authenticated:
            return
        
        self.client.post("/api/admin/backup")


class APIStressTest(HttpUser):
    """High-frequency API calls for stress testing"""
    
    wait_time = between(0.1, 0.5)  # Minimal wait time
    
    @task
    def rapid_api_calls(self):
        """Rapid consecutive API calls"""
        endpoints = [
            "/api/targets",
            "/api/signals",
            "/api/exploits",
            "/api/dashboard/stats"
        ]
        
        endpoint = random.choice(endpoints)
        self.client.get(endpoint)


class WebSocketUser(HttpUser):
    """Simulated WebSocket user for real-time updates"""
    
    wait_time = between(1, 2)
    
    @task
    def poll_updates(self):
        """Poll for real-time updates"""
        self.client.get("/api/updates/poll")


# Custom event handlers for detailed metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts"""
    print("Load test starting...")
    environment.stats.clear_all()


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops"""
    print("\n=== Load Test Results ===")
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Total failures: {environment.stats.total.num_failures}")
    print(f"Average response time: {environment.stats.total.avg_response_time:.2f}ms")
    print(f"Min response time: {environment.stats.total.min_response_time:.2f}ms")
    print(f"Max response time: {environment.stats.total.max_response_time:.2f}ms")
    print(f"Requests per second: {environment.stats.total.current_rps:.2f}")
    
    # Calculate success rate
    total = environment.stats.total.num_requests
    failures = environment.stats.total.num_failures
    success_rate = ((total - failures) / total * 100) if total > 0 else 0
    print(f"Success rate: {success_rate:.2f}%")


# Load test scenarios
"""
Scenario 1: Normal Load (10 users)
    locust -f locustfile.py --host=http://localhost:5000 --users 10 --spawn-rate 2 --run-time 60s

Scenario 2: Peak Load (50 users)
    locust -f locustfile.py --host=http://localhost:5000 --users 50 --spawn-rate 5 --run-time 120s

Scenario 3: Stress Test (100 users)
    locust -f locustfile.py --host=http://localhost:5000 --users 100 --spawn-rate 10 --run-time 300s

Scenario 4: Spike Test (ramp up quickly)
    locust -f locustfile.py --host=http://localhost:5000 --users 200 --spawn-rate 50 --run-time 60s

Performance Targets:
- Average response time: < 200ms
- 95th percentile: < 500ms
- 99th percentile: < 1000ms
- Success rate: > 99%
- Throughput: > 100 RPS
"""
