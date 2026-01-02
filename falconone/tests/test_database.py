"""
Unit tests for database module
Tests CRUD operations, encryption, backups, and security features
"""

import pytest
import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from falconone.utils.database import (
    DatabaseManager, SessionRecord, SignalCapture, DetectionEvent
)


@pytest.fixture
def temp_db_path():
    """Create temporary database path for testing"""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_falconone.db")
    yield db_path
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def db_manager(temp_db_path):
    """Create database manager instance for testing"""
    manager = DatabaseManager(temp_db_path, enable_encryption=True)
    yield manager
    manager.close()


class TestDatabaseInitialization:
    """Test database initialization and schema creation"""
    
    def test_create_database(self, temp_db_path):
        """Test database file creation"""
        manager = DatabaseManager(temp_db_path)
        assert os.path.exists(temp_db_path)
        manager.close()
    
    def test_encryption_enabled(self, temp_db_path):
        """Test SQLCipher encryption is enabled"""
        manager = DatabaseManager(temp_db_path, enable_encryption=True)
        
        # Try to read with wrong password (should fail)
        manager.close()
        
        # Database should be encrypted (not plain SQLite format)
        with open(temp_db_path, 'rb') as f:
            header = f.read(16)
            # SQLCipher encrypted file doesn't have "SQLite format 3" header
            assert header != b'SQLite format 3\x00'
        
        manager.close()
    
    def test_tables_created(self, db_manager):
        """Test all required tables are created"""
        tables = db_manager.get_table_names()
        
        expected_tables = [
            'sessions', 'signal_captures', 'detection_events',
            'exploit_logs', 'network_cells', 'device_fingerprints'
        ]
        
        for table in expected_tables:
            assert table in tables, f"Table '{table}' not found"


class TestSessionRecords:
    """Test session CRUD operations"""
    
    def test_create_session(self, db_manager):
        """Test creating a new session record"""
        session = SessionRecord(
            session_id="test-session-001",
            start_time=datetime.now(),
            technology="LTE",
            frequency=1850.0,
            bandwidth=10.0
        )
        
        session_id = db_manager.insert_session(session)
        assert session_id is not None
        assert session_id > 0
    
    def test_read_session(self, db_manager):
        """Test reading session record"""
        # Insert test session
        session = SessionRecord(
            session_id="test-session-002",
            start_time=datetime.now(),
            technology="5G NR",
            frequency=3500.0,
            bandwidth=100.0
        )
        db_manager.insert_session(session)
        
        # Read session
        retrieved = db_manager.get_session_by_id("test-session-002")
        assert retrieved is not None
        assert retrieved.technology == "5G NR"
        assert retrieved.frequency == 3500.0
    
    def test_update_session(self, db_manager):
        """Test updating session record"""
        # Insert test session
        session = SessionRecord(
            session_id="test-session-003",
            start_time=datetime.now(),
            technology="GSM",
            frequency=900.0,
            bandwidth=0.2
        )
        db_manager.insert_session(session)
        
        # Update session
        db_manager.update_session(
            "test-session-003",
            end_time=datetime.now(),
            status="completed"
        )
        
        # Verify update
        retrieved = db_manager.get_session_by_id("test-session-003")
        assert retrieved.status == "completed"
        assert retrieved.end_time is not None
    
    def test_delete_session(self, db_manager):
        """Test deleting session record"""
        # Insert test session
        session = SessionRecord(
            session_id="test-session-004",
            start_time=datetime.now(),
            technology="UMTS",
            frequency=2100.0,
            bandwidth=5.0
        )
        db_manager.insert_session(session)
        
        # Delete session
        db_manager.delete_session("test-session-004")
        
        # Verify deletion
        retrieved = db_manager.get_session_by_id("test-session-004")
        assert retrieved is None


class TestSignalCaptures:
    """Test signal capture CRUD operations"""
    
    def test_insert_capture(self, db_manager):
        """Test inserting signal capture"""
        capture = SignalCapture(
            session_id="test-session-005",
            timestamp=datetime.now(),
            frequency=2600.0,
            technology="LTE",
            signal_strength=-75.5,
            bandwidth=20.0,
            raw_data=b'\x00\x01\x02\x03'
        )
        
        capture_id = db_manager.insert_capture(capture)
        assert capture_id is not None
        assert capture_id > 0
    
    def test_query_captures_by_frequency(self, db_manager):
        """Test querying captures by frequency range"""
        # Insert multiple captures
        for i in range(5):
            capture = SignalCapture(
                session_id=f"test-session-{i}",
                timestamp=datetime.now(),
                frequency=2600.0 + (i * 10),  # 2600, 2610, 2620, 2630, 2640
                technology="LTE",
                signal_strength=-70.0 - i,
                bandwidth=20.0
            )
            db_manager.insert_capture(capture)
        
        # Query frequency range
        captures = db_manager.get_captures_by_frequency_range(2610.0, 2630.0)
        assert len(captures) == 3  # Should get 2610, 2620, 2630
    
    def test_query_captures_by_technology(self, db_manager):
        """Test querying captures by technology"""
        # Insert mixed technology captures
        technologies = ["GSM", "UMTS", "LTE", "5G NR", "LTE"]
        for i, tech in enumerate(technologies):
            capture = SignalCapture(
                session_id=f"test-session-tech-{i}",
                timestamp=datetime.now(),
                frequency=900.0 + (i * 100),
                technology=tech,
                signal_strength=-80.0,
                bandwidth=5.0
            )
            db_manager.insert_capture(capture)
        
        # Query LTE captures
        lte_captures = db_manager.get_captures_by_technology("LTE")
        assert len(lte_captures) == 2


class TestDetectionEvents:
    """Test detection event operations"""
    
    def test_insert_detection(self, db_manager):
        """Test inserting detection event"""
        event = DetectionEvent(
            session_id="test-session-detection",
            timestamp=datetime.now(),
            event_type="IMSI_CATCHER",
            severity="HIGH",
            description="Fake base station detected",
            confidence=0.95,
            indicators={
                "lai_mismatch": True,
                "encryption_downgrade": True
            }
        )
        
        event_id = db_manager.insert_detection(event)
        assert event_id is not None
        assert event_id > 0
    
    def test_query_high_severity_events(self, db_manager):
        """Test querying high-severity events"""
        severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL", "LOW", "HIGH"]
        for i, severity in enumerate(severities):
            event = DetectionEvent(
                session_id=f"test-session-sev-{i}",
                timestamp=datetime.now(),
                event_type="ANOMALY",
                severity=severity,
                description=f"Test event {i}",
                confidence=0.8
            )
            db_manager.insert_detection(event)
        
        # Query high+ severity
        high_events = db_manager.get_detections_by_severity(["HIGH", "CRITICAL"])
        assert len(high_events) == 3  # 2 HIGH + 1 CRITICAL


class TestDatabaseBackup:
    """Test database backup functionality"""
    
    def test_create_backup(self, db_manager, temp_db_path):
        """Test creating database backup"""
        # Insert some data
        session = SessionRecord(
            session_id="backup-test",
            start_time=datetime.now(),
            technology="5G NR",
            frequency=3700.0,
            bandwidth=100.0
        )
        db_manager.insert_session(session)
        
        # Create backup
        backup_dir = os.path.join(os.path.dirname(temp_db_path), "backups")
        backup_path = db_manager.create_backup(backup_dir)
        
        assert os.path.exists(backup_path)
        assert os.path.getsize(backup_path) > 0
    
    def test_restore_from_backup(self, db_manager, temp_db_path):
        """Test restoring database from backup"""
        # Insert test data
        session = SessionRecord(
            session_id="restore-test",
            start_time=datetime.now(),
            technology="LTE",
            frequency=1800.0,
            bandwidth=20.0
        )
        db_manager.insert_session(session)
        
        # Create backup
        backup_dir = os.path.join(os.path.dirname(temp_db_path), "backups")
        backup_path = db_manager.create_backup(backup_dir)
        
        # Delete data
        db_manager.delete_session("restore-test")
        assert db_manager.get_session_by_id("restore-test") is None
        
        # Restore from backup
        db_manager.restore_from_backup(backup_path)
        
        # Verify data restored
        restored = db_manager.get_session_by_id("restore-test")
        assert restored is not None
        assert restored.technology == "LTE"


class TestDatabaseExport:
    """Test data export functionality (Phase 2.3)"""
    
    def test_export_to_csv(self, db_manager, temp_db_path):
        """Test exporting data to CSV"""
        # Insert test data
        for i in range(5):
            capture = SignalCapture(
                session_id=f"csv-export-{i}",
                timestamp=datetime.now(),
                frequency=2600.0,
                technology="LTE",
                signal_strength=-70.0,
                bandwidth=20.0
            )
            db_manager.insert_capture(capture)
        
        # Export to CSV
        export_dir = os.path.join(os.path.dirname(temp_db_path), "exports")
        os.makedirs(export_dir, exist_ok=True)
        csv_path = os.path.join(export_dir, "captures.csv")
        
        db_manager.export_to_csv("signal_captures", csv_path)
        
        assert os.path.exists(csv_path)
        assert os.path.getsize(csv_path) > 0
    
    def test_export_to_json(self, db_manager, temp_db_path):
        """Test exporting data to JSON"""
        # Insert test data
        session = SessionRecord(
            session_id="json-export",
            start_time=datetime.now(),
            technology="5G NR",
            frequency=3500.0,
            bandwidth=100.0
        )
        db_manager.insert_session(session)
        
        # Export to JSON
        export_dir = os.path.join(os.path.dirname(temp_db_path), "exports")
        os.makedirs(export_dir, exist_ok=True)
        json_path = os.path.join(export_dir, "sessions.json")
        
        db_manager.export_to_json("sessions", json_path)
        
        assert os.path.exists(json_path)
        assert os.path.getsize(json_path) > 0


class TestForeignKeyConstraints:
    """Test foreign key constraint enforcement"""
    
    def test_cascade_delete_session(self, db_manager):
        """Test cascading delete when session is removed"""
        # Create session with captures
        session = SessionRecord(
            session_id="cascade-test",
            start_time=datetime.now(),
            technology="LTE",
            frequency=2600.0,
            bandwidth=20.0
        )
        db_manager.insert_session(session)
        
        # Add captures linked to session
        for i in range(3):
            capture = SignalCapture(
                session_id="cascade-test",
                timestamp=datetime.now(),
                frequency=2600.0,
                technology="LTE",
                signal_strength=-70.0,
                bandwidth=20.0
            )
            db_manager.insert_capture(capture)
        
        # Verify captures exist
        captures_before = db_manager.get_captures_by_session("cascade-test")
        assert len(captures_before) == 3
        
        # Delete session (should cascade)
        db_manager.delete_session("cascade-test")
        
        # Verify captures also deleted
        captures_after = db_manager.get_captures_by_session("cascade-test")
        assert len(captures_after) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
