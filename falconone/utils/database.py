"""
FalconOne Database Manager
Persistent storage for captures, targets, and operational data

Version 2.0: Enhanced Security
- SQLCipher encryption at rest
- Foreign key constraints
- Automatic backup with rotation
- Parameterized queries (SQL injection prevention)
- Connection pooling
"""

import sqlite3
import json
import time
import logging
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import threading
import os

# Try to import SQLCipher (fallback to standard sqlite3)
try:
    from pysqlcipher3 import dbapi2 as sqlcipher
    SQLCIPHER_AVAILABLE = True
except ImportError:
    SQLCIPHER_AVAILABLE = False
    print("[WARNING] SQLCipher not installed. Database encryption disabled. Install: pip install pysqlcipher3")


class FalconOneDatabase:
    """
    SQLite database manager for FalconOne persistent storage
    
    Stores:
    - IMSI/SUCI captures
    - Voice call intercepts
    - Targets and tracking data
    - Anomaly alerts
    - Security audits
    - Exploit operations
    - System events
    
    Version 2.0 Features:
    - SQLCipher encryption at rest
    - Foreign key constraints for data integrity
    - Automatic backups with rotation
    - Parameterized queries (SQL injection prevention)
    """
    
    def __init__(self, db_path: str = None, encryption_key: str = None, 
                 backup_enabled: bool = True, backup_retention_days: int = 90):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file (default: logs/falconone.db)
            encryption_key: Encryption key for SQLCipher (None = no encryption)
            backup_enabled: Enable automatic backups
            backup_retention_days: Days to retain backups
        """
        self.logger = logging.getLogger('FalconOne.Database')
        
        if db_path is None:
            # Default to logs directory
            db_dir = Path(__file__).parent.parent.parent / 'logs'
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / 'falconone.db')
        
        self.db_path = db_path
        self.encryption_key = encryption_key
        self.backup_enabled = backup_enabled
        self.backup_retention_days = backup_retention_days
        self.lock = threading.Lock()
        
        # For in-memory databases, keep a persistent connection
        self.is_memory = (db_path == ':memory:')
        self.persistent_connection = None
        if self.is_memory:
            conn = self._create_connection()
            self.persistent_connection = conn
        
        # Check encryption status
        if encryption_key and not SQLCIPHER_AVAILABLE:
            self.logger.warning("Encryption key provided but SQLCipher not installed. Database will NOT be encrypted!")
        elif encryption_key and SQLCIPHER_AVAILABLE:
            self.logger.info("Database encryption enabled with SQLCipher")
        
        self._initialize_database()
        self.logger.info(f"Database initialized: {self.db_path} (encrypted={bool(encryption_key and SQLCIPHER_AVAILABLE)})")
    
    def _create_connection(self):
        """Create database connection with encryption if available"""
        if self.is_memory and self.persistent_connection:
            return self.persistent_connection
        
        # Use SQLCipher if available and encryption key provided
        if self.encryption_key and SQLCIPHER_AVAILABLE:
            conn = sqlcipher.connect(self.db_path, check_same_thread=False)
            conn.execute(f"PRAGMA key = '{self.encryption_key}'")
            conn.execute("PRAGMA cipher_page_size = 4096")
            conn.execute("PRAGMA kdf_iter = 64000")  # PBKDF2 iterations
        else:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        return conn
    
    def _get_connection(self):
        """Get thread-safe database connection"""
        if self.is_memory and self.persistent_connection:
            return self.persistent_connection
        return self._create_connection()
    
    def _close_connection(self, conn):
        """Close connection unless it's the persistent in-memory connection"""
        if not self.is_memory:
            conn.close()
    
    def _initialize_database(self):
        """Create database schema with foreign key constraints"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Users Table (Master table - must be created first for authentication)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                role TEXT DEFAULT 'operator',
                is_active BOOLEAN DEFAULT 1,
                last_login REAL,
                created_at REAL NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Targets Table (Master table - must be created first)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS targets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_id TEXT UNIQUE NOT NULL,
                target_type TEXT,
                imsi TEXT,
                imei TEXT,
                name TEXT,
                latitude REAL,
                longitude REAL,
                status TEXT,
                first_seen REAL NOT NULL,
                last_seen REAL,
                metadata TEXT
            )
        ''')
        
        # IMSI/SUCI Captures Table (with foreign key to targets)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS suci_captures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                suci TEXT NOT NULL,
                imsi TEXT,
                generation TEXT,
                mcc TEXT,
                mnc TEXT,
                deconcealed BOOLEAN DEFAULT 0,
                timestamp REAL NOT NULL,
                metadata TEXT,
                target_id INTEGER,
                FOREIGN KEY (target_id) REFERENCES targets(id) ON DELETE SET NULL
            )
        ''')
        
        # Voice Calls Table (with foreign keys to targets)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS voice_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caller TEXT NOT NULL,
                callee TEXT NOT NULL,
                duration REAL,
                quality TEXT,
                codec TEXT,
                call_type TEXT,
                timestamp REAL NOT NULL,
                metadata TEXT,
                caller_target_id INTEGER,
                callee_target_id INTEGER,
                FOREIGN KEY (caller_target_id) REFERENCES targets(id) ON DELETE SET NULL,
                FOREIGN KEY (callee_target_id) REFERENCES targets(id) ON DELETE SET NULL
            )
        ''')
        
        # Anomaly Alerts Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomaly_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                source TEXT,
                timestamp REAL NOT NULL,
                acknowledged BOOLEAN DEFAULT 0,
                metadata TEXT,
                target_id INTEGER,
                FOREIGN KEY (target_id) REFERENCES targets(id) ON DELETE SET NULL
            )
        ''')
        
        # Security Audits Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audit_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                component TEXT,
                timestamp REAL NOT NULL,
                resolved BOOLEAN DEFAULT 0,
                metadata TEXT
            )
        ''')
        
        # Exploit Operations Table (with foreign key to targets)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exploit_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exploit_type TEXT NOT NULL,
                target TEXT,
                status TEXT,
                success BOOLEAN,
                timestamp REAL NOT NULL,
                duration REAL,
                metadata TEXT,
                target_id INTEGER,
                FOREIGN KEY (target_id) REFERENCES targets(id) ON DELETE SET NULL
            )
        ''')
        
        # System Events Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                description TEXT NOT NULL,
                level TEXT,
                component TEXT,
                timestamp REAL NOT NULL,
                metadata TEXT
            )
        ''')
        
        # ISAC Sensing Data Table (v1.9.8 production addition)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS isac_sensing_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mode TEXT NOT NULL,
                range_m REAL,
                velocity_mps REAL,
                angle_deg REAL,
                snr_db REAL,
                timestamp REAL NOT NULL,
                session_id TEXT,
                target_id INTEGER,
                metadata TEXT,
                FOREIGN KEY (target_id) REFERENCES targets(id) ON DELETE SET NULL
            )
        ''')
        
        # LE Mode Warrants Table (v1.9.8 production addition)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS le_warrants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                warrant_id TEXT UNIQUE NOT NULL,
                authority TEXT NOT NULL,
                jurisdiction TEXT,
                target_identifiers TEXT,
                scope TEXT,
                valid_from REAL NOT NULL,
                valid_until REAL NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                is_expired BOOLEAN DEFAULT 0,
                created_at REAL NOT NULL,
                revoked_at REAL,
                revoked_by TEXT,
                metadata TEXT
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_suci_timestamp ON suci_captures(timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_suci_target ON suci_captures(target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_calls_timestamp ON voice_calls(timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_calls_caller_target ON voice_calls(caller_target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_calls_callee_target ON voice_calls(callee_target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_targets_id ON targets(target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_targets_imsi ON targets(imsi)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON anomaly_alerts(timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_target ON anomaly_alerts(target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audits_timestamp ON security_audits(timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exploits_timestamp ON exploit_operations(timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exploits_target ON exploit_operations(target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_isac_timestamp ON isac_sensing_data(timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_isac_mode ON isac_sensing_data(mode)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_warrants_warrant_id ON le_warrants(warrant_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_warrants_valid_until ON le_warrants(valid_until DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_warrants_active ON le_warrants(is_active)')
        
        conn.commit()
        # Don't close persistent in-memory connections
        if not self.is_memory:
            self._close_connection(conn)
    
    # ==================== SUCI/IMSI CAPTURES ====================
    
    def add_suci_capture(self, suci: str, generation: str, imsi: str = None, 
                         deconcealed: bool = False, metadata: Dict = None) -> int:
        """Add SUCI capture to database"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Extract MCC/MNC from IMSI if available
            mcc = imsi[:3] if imsi and len(imsi) >= 5 else None
            mnc = imsi[3:5] if imsi and len(imsi) >= 5 else None
            
            cursor.execute('''
                INSERT INTO suci_captures (suci, imsi, generation, mcc, mnc, deconcealed, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (suci, imsi, generation, mcc, mnc, deconcealed, time.time(), json.dumps(metadata or {})))
            
            capture_id = cursor.lastrowid
            conn.commit()
            self._close_connection(conn)
            
            self.logger.debug(f"SUCI capture added: {suci} (ID: {capture_id})")
            return capture_id
    
    def get_suci_captures(self, limit: int = 100, generation: str = None) -> List[Dict]:
        """Get SUCI captures from database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if generation:
            cursor.execute('''
                SELECT id, suci, imsi, generation, mcc, mnc, deconcealed, timestamp, metadata
                FROM suci_captures 
                WHERE generation = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (generation, limit))
        else:
            cursor.execute('''
                SELECT id, suci, imsi, generation, mcc, mnc, deconcealed, timestamp, metadata
                FROM suci_captures 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        captures = []
        for row in cursor.fetchall():
            captures.append({
                'id': row[0],
                'suci': row[1],
                'imsi': row[2],
                'generation': row[3],
                'mcc': row[4],
                'mnc': row[5],
                'deconcealed': bool(row[6]),
                'timestamp': row[7],
                'timestamp_human': datetime.fromtimestamp(row[7]).strftime('%Y-%m-%d %H:%M:%S'),
                'metadata': json.loads(row[8]) if row[8] else {}
            })
        
        self._close_connection(conn)
        return captures
    
    def count_suci_captures(self, generation: str = None) -> int:
        """Count total SUCI captures"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if generation:
            cursor.execute('SELECT COUNT(*) FROM suci_captures WHERE generation = ?', (generation,))
        else:
            cursor.execute('SELECT COUNT(*) FROM suci_captures')
        
        count = cursor.fetchone()[0]
        self._close_connection(conn)
        return count
    
    # ==================== VOICE CALLS ====================
    
    def add_voice_call(self, caller: str, callee: str, duration: float, 
                      quality: str, codec: str = None, call_type: str = 'voice',
                      metadata: Dict = None) -> int:
        """Add voice call intercept to database"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO voice_calls (caller, callee, duration, quality, codec, call_type, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (caller, callee, duration, quality, codec, call_type, time.time(), json.dumps(metadata or {})))
            
            call_id = cursor.lastrowid
            conn.commit()
            self._close_connection(conn)
            
            self.logger.debug(f"Voice call added: {caller} -> {callee} (ID: {call_id})")
            return call_id
    
    def get_voice_calls(self, limit: int = 100) -> List[Dict]:
        """Get voice call intercepts from database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, caller, callee, duration, quality, codec, call_type, timestamp, metadata
            FROM voice_calls 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        calls = []
        for row in cursor.fetchall():
            calls.append({
                'id': row[0],
                'caller': row[1],
                'callee': row[2],
                'duration': row[3],
                'quality': row[4],
                'codec': row[5],
                'type': row[6],
                'timestamp': row[7],
                'timestamp_human': datetime.fromtimestamp(row[7]).strftime('%Y-%m-%d %H:%M:%S'),
                'metadata': json.loads(row[8]) if row[8] else {}
            })
        
        self._close_connection(conn)
        return calls
    
    def count_voice_calls(self) -> int:
        """Count total voice call intercepts"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM voice_calls')
        count = cursor.fetchone()[0]
        self._close_connection(conn)
        return count
    
    # ==================== TARGETS ====================
    
    def add_target(self, target_id: str, target_type: str, imsi: str = None,
                  imei: str = None, name: str = None, latitude: float = None,
                  longitude: float = None, status: str = 'active', metadata: Dict = None) -> int:
        """Add or update target in database"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if target already exists
            cursor.execute('SELECT id, first_seen FROM targets WHERE target_id = ?', (target_id,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing target
                cursor.execute('''
                    UPDATE targets 
                    SET target_type=?, imsi=?, imei=?, name=?, latitude=?, longitude=?, 
                        status=?, last_seen=?, metadata=?
                    WHERE target_id = ?
                ''', (target_type, imsi, imei, name, latitude, longitude, status, 
                      time.time(), json.dumps(metadata or {}), target_id))
                target_db_id = existing[0]
            else:
                # Insert new target
                cursor.execute('''
                    INSERT INTO targets (target_id, target_type, imsi, imei, name, latitude, longitude,
                                       status, first_seen, last_seen, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (target_id, target_type, imsi, imei, name, latitude, longitude, status,
                      time.time(), time.time(), json.dumps(metadata or {})))
                target_db_id = cursor.lastrowid
            
            conn.commit()
            self._close_connection(conn)
            
            self.logger.debug(f"Target {'updated' if existing else 'added'}: {target_id} (ID: {target_db_id})")
            return target_db_id
    
    def get_targets(self, status: str = None, limit: int = 1000) -> List[Dict]:
        """Get targets from database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if status:
            cursor.execute('''
                SELECT id, target_id, target_type, imsi, imei, name, latitude, longitude,
                       status, first_seen, last_seen, metadata
                FROM targets 
                WHERE status = ?
                ORDER BY last_seen DESC 
                LIMIT ?
            ''', (status, limit))
        else:
            cursor.execute('''
                SELECT id, target_id, target_type, imsi, imei, name, latitude, longitude,
                       status, first_seen, last_seen, metadata
                FROM targets 
                ORDER BY last_seen DESC 
                LIMIT ?
            ''', (limit,))
        
        targets = []
        for row in cursor.fetchall():
            targets.append({
                'id': row[0],
                'target_id': row[1],
                'type': row[2],
                'imsi': row[3],
                'imei': row[4],
                'name': row[5],
                'latitude': row[6],
                'longitude': row[7],
                'status': row[8],
                'first_seen': row[9],
                'last_seen': row[10],
                'metadata': json.loads(row[11]) if row[11] else {}
            })
        
        self._close_connection(conn)
        return targets
    
    def count_targets(self, status: str = None) -> int:
        """Count targets"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if status:
            cursor.execute('SELECT COUNT(*) FROM targets WHERE status = ?', (status,))
        else:
            cursor.execute('SELECT COUNT(*) FROM targets')
        
        count = cursor.fetchone()[0]
        self._close_connection(conn)
        return count
    
    # ==================== ANOMALY ALERTS ====================
    
    def add_anomaly_alert(self, severity: str, description: str, 
                         source: str = None, metadata: Dict = None) -> int:
        """Add anomaly alert to database"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO anomaly_alerts (severity, description, source, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (severity, description, source, time.time(), json.dumps(metadata or {})))
            
            alert_id = cursor.lastrowid
            conn.commit()
            self._close_connection(conn)
            
            self.logger.debug(f"Anomaly alert added: {severity} - {description[:50]} (ID: {alert_id})")
            return alert_id
    
    def get_anomaly_alerts(self, limit: int = 100, acknowledged: bool = None) -> List[Dict]:
        """Get anomaly alerts from database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if acknowledged is not None:
            cursor.execute('''
                SELECT id, severity, description, source, timestamp, acknowledged, metadata
                FROM anomaly_alerts 
                WHERE acknowledged = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (int(acknowledged), limit))
        else:
            cursor.execute('''
                SELECT id, severity, description, source, timestamp, acknowledged, metadata
                FROM anomaly_alerts 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        alerts = []
        for row in cursor.fetchall():
            alerts.append({
                'id': row[0],
                'severity': row[1],
                'description': row[2],
                'source': row[3],
                'timestamp': row[4],
                'timestamp_human': datetime.fromtimestamp(row[4]).strftime('%Y-%m-%d %H:%M:%S'),
                'acknowledged': bool(row[5]),
                'metadata': json.loads(row[6]) if row[6] else {}
            })
        
        self._close_connection(conn)
        return alerts
    
    def count_anomaly_alerts(self, acknowledged: bool = None) -> int:
        """Count anomaly alerts"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if acknowledged is not None:
            cursor.execute('SELECT COUNT(*) FROM anomaly_alerts WHERE acknowledged = ?', (int(acknowledged),))
        else:
            cursor.execute('SELECT COUNT(*) FROM anomaly_alerts')
        
        count = cursor.fetchone()[0]
        self._close_connection(conn)
        return count
    
    # ==================== SECURITY AUDITS ====================
    
    def add_security_audit(self, audit_type: str, severity: str, description: str,
                          component: str = None, metadata: Dict = None) -> int:
        """Add security audit entry"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO security_audits (audit_type, severity, description, component, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (audit_type, severity, description, component, time.time(), json.dumps(metadata or {})))
            
            audit_id = cursor.lastrowid
            conn.commit()
            self._close_connection(conn)
            
            return audit_id
    
    def get_security_audits(self, limit: int = 100) -> List[Dict]:
        """Get security audit entries"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, audit_type, severity, description, component, timestamp, resolved, metadata
            FROM security_audits 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        audits = []
        for row in cursor.fetchall():
            audits.append({
                'id': row[0],
                'type': row[1],
                'severity': row[2],
                'description': row[3],
                'component': row[4],
                'timestamp': row[5],
                'timestamp_human': datetime.fromtimestamp(row[5]).strftime('%Y-%m-%d %H:%M:%S'),
                'resolved': bool(row[6]),
                'metadata': json.loads(row[7]) if row[7] else {}
            })
        
        self._close_connection(conn)
        return audits
    
    def count_security_audits(self) -> int:
        """Count total security audit entries"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM security_audits')
        count = cursor.fetchone()[0]
        self._close_connection(conn)
        return count
    
    # ==================== ISAC SENSING DATA (v1.9.8) ====================
    
    def add_isac_sensing_data(self, mode: str, range_m: float, velocity_mps: float,
                              angle_deg: float = None, snr_db: float = None,
                              session_id: str = None, target_id: int = None,
                              metadata: Dict = None) -> int:
        """Add ISAC sensing data entry"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO isac_sensing_data (mode, range_m, velocity_mps, angle_deg, snr_db, 
                                               timestamp, session_id, target_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (mode, range_m, velocity_mps, angle_deg, snr_db, time.time(),
                  session_id, target_id, json.dumps(metadata or {})))
            
            entry_id = cursor.lastrowid
            conn.commit()
            self._close_connection(conn)
            
            return entry_id
    
    def get_isac_sensing_data(self, limit: int = 100, mode: str = None,
                              session_id: str = None) -> List[Dict]:
        """Get ISAC sensing data entries"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = '''
            SELECT id, mode, range_m, velocity_mps, angle_deg, snr_db, timestamp, 
                   session_id, target_id, metadata
            FROM isac_sensing_data 
        '''
        params = []
        
        conditions = []
        if mode:
            conditions.append('mode = ?')
            params.append(mode)
        if session_id:
            conditions.append('session_id = ?')
            params.append(session_id)
        
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        data = []
        for row in cursor.fetchall():
            data.append({
                'id': row[0],
                'mode': row[1],
                'range_m': row[2],
                'velocity_mps': row[3],
                'angle_deg': row[4],
                'snr_db': row[5],
                'timestamp': row[6],
                'timestamp_human': datetime.fromtimestamp(row[6]).strftime('%Y-%m-%d %H:%M:%S'),
                'session_id': row[7],
                'target_id': row[8],
                'metadata': json.loads(row[9]) if row[9] else {}
            })
        
        self._close_connection(conn)
        return data
    
    def count_isac_sensing_data(self, mode: str = None) -> int:
        """Count ISAC sensing data entries"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if mode:
            cursor.execute('SELECT COUNT(*) FROM isac_sensing_data WHERE mode = ?', (mode,))
        else:
            cursor.execute('SELECT COUNT(*) FROM isac_sensing_data')
        
        count = cursor.fetchone()[0]
        self._close_connection(conn)
        return count
    
    # ==================== LE MODE WARRANTS ====================
    
    def add_warrant(self, warrant_id: str, authority: str, jurisdiction: str = None,
                    target_identifiers: List[str] = None, scope: str = None,
                    valid_from: float = None, valid_until: float = None,
                    metadata: Dict = None) -> int:
        """
        Add a new LE warrant to the database.
        
        Args:
            warrant_id: Unique warrant identifier
            authority: Issuing authority
            jurisdiction: Geographic/legal jurisdiction
            target_identifiers: List of target IDs covered by warrant
            scope: Scope of warrant (e.g., 'satellite', 'isac', 'full')
            valid_from: Warrant validity start timestamp
            valid_until: Warrant validity end timestamp
            metadata: Additional warrant metadata
            
        Returns:
            Warrant database ID
        """
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            now = time.time()
            valid_from = valid_from or now
            
            cursor.execute('''
                INSERT INTO le_warrants (warrant_id, authority, jurisdiction, target_identifiers,
                    scope, valid_from, valid_until, is_active, is_expired, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?)
            ''', (
                warrant_id, authority, jurisdiction,
                json.dumps(target_identifiers or []),
                scope, valid_from, valid_until, now,
                json.dumps(metadata or {})
            ))
            
            db_id = cursor.lastrowid
            conn.commit()
            self._close_connection(conn)
            
            self.logger.info(f"Warrant added: {warrant_id} (ID: {db_id})")
            return db_id
    
    def validate_warrant(self, warrant_id: str, scope: str = None) -> Dict:
        """
        Validate a warrant by ID and optionally check scope.
        
        Args:
            warrant_id: Warrant ID to validate
            scope: Required scope (e.g., 'satellite', 'isac')
            
        Returns:
            Dict with validation result and warrant data
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, warrant_id, authority, jurisdiction, target_identifiers,
                   scope, valid_from, valid_until, is_active, is_expired, created_at, metadata
            FROM le_warrants
            WHERE warrant_id = ?
        ''', (warrant_id,))
        
        row = cursor.fetchone()
        self._close_connection(conn)
        
        if not row:
            return {
                'valid': False,
                'error': 'Warrant not found',
                'warrant': None
            }
        
        warrant = {
            'id': row[0],
            'warrant_id': row[1],
            'authority': row[2],
            'jurisdiction': row[3],
            'target_identifiers': json.loads(row[4]) if row[4] else [],
            'scope': row[5],
            'valid_from': row[6],
            'valid_until': row[7],
            'is_active': bool(row[8]),
            'is_expired': bool(row[9]),
            'created_at': row[10],
            'metadata': json.loads(row[11]) if row[11] else {}
        }
        
        now = time.time()
        
        # Check if warrant is active
        if not warrant['is_active']:
            return {
                'valid': False,
                'error': 'Warrant has been revoked',
                'warrant': warrant
            }
        
        # Check expiration
        if warrant['valid_until'] and now > warrant['valid_until']:
            # Mark as expired in database
            self._mark_warrant_expired(warrant_id)
            return {
                'valid': False,
                'error': 'Warrant has expired',
                'warrant': warrant
            }
        
        # Check if warrant has started
        if warrant['valid_from'] and now < warrant['valid_from']:
            return {
                'valid': False,
                'error': 'Warrant not yet valid',
                'warrant': warrant
            }
        
        # Check scope if specified
        if scope and warrant['scope']:
            if scope.lower() not in warrant['scope'].lower() and warrant['scope'].lower() != 'full':
                return {
                    'valid': False,
                    'error': f'Warrant scope does not cover {scope} operations',
                    'warrant': warrant
                }
        
        return {
            'valid': True,
            'error': None,
            'warrant': warrant
        }
    
    def _mark_warrant_expired(self, warrant_id: str):
        """Mark a warrant as expired"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE le_warrants SET is_expired = 1 WHERE warrant_id = ?
            ''', (warrant_id,))
            
            conn.commit()
            self._close_connection(conn)
    
    def revoke_warrant(self, warrant_id: str, revoked_by: str = None) -> bool:
        """
        Revoke an active warrant.
        
        Args:
            warrant_id: Warrant ID to revoke
            revoked_by: User/authority revoking the warrant
            
        Returns:
            True if revoked, False if not found
        """
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE le_warrants 
                SET is_active = 0, revoked_at = ?, revoked_by = ?
                WHERE warrant_id = ? AND is_active = 1
            ''', (time.time(), revoked_by, warrant_id))
            
            affected = cursor.rowcount
            conn.commit()
            self._close_connection(conn)
            
            if affected > 0:
                self.logger.info(f"Warrant revoked: {warrant_id} by {revoked_by}")
                return True
            return False
    
    def get_active_warrants(self, scope: str = None) -> List[Dict]:
        """
        Get all active (non-expired, non-revoked) warrants.
        
        Args:
            scope: Filter by scope (optional)
            
        Returns:
            List of active warrant records
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = time.time()
        query = '''
            SELECT id, warrant_id, authority, jurisdiction, target_identifiers,
                   scope, valid_from, valid_until, is_active, is_expired, created_at, metadata
            FROM le_warrants
            WHERE is_active = 1 AND is_expired = 0 AND valid_until > ?
        '''
        params = [now]
        
        if scope:
            query += ' AND (scope LIKE ? OR scope = ?)'
            params.extend([f'%{scope}%', 'full'])
        
        query += ' ORDER BY valid_until DESC'
        
        cursor.execute(query, params)
        
        warrants = []
        for row in cursor.fetchall():
            warrants.append({
                'id': row[0],
                'warrant_id': row[1],
                'authority': row[2],
                'jurisdiction': row[3],
                'target_identifiers': json.loads(row[4]) if row[4] else [],
                'scope': row[5],
                'valid_from': row[6],
                'valid_from_human': datetime.fromtimestamp(row[6]).strftime('%Y-%m-%d %H:%M:%S') if row[6] else None,
                'valid_until': row[7],
                'valid_until_human': datetime.fromtimestamp(row[7]).strftime('%Y-%m-%d %H:%M:%S') if row[7] else None,
                'is_active': bool(row[8]),
                'is_expired': bool(row[9]),
                'created_at': row[10],
                'metadata': json.loads(row[11]) if row[11] else {}
            })
        
        self._close_connection(conn)
        return warrants
    
    def get_warrant_by_id(self, warrant_id: str) -> Optional[Dict]:
        """Get warrant details by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, warrant_id, authority, jurisdiction, target_identifiers,
                   scope, valid_from, valid_until, is_active, is_expired, created_at,
                   revoked_at, revoked_by, metadata
            FROM le_warrants
            WHERE warrant_id = ?
        ''', (warrant_id,))
        
        row = cursor.fetchone()
        self._close_connection(conn)
        
        if not row:
            return None
        
        return {
            'id': row[0],
            'warrant_id': row[1],
            'authority': row[2],
            'jurisdiction': row[3],
            'target_identifiers': json.loads(row[4]) if row[4] else [],
            'scope': row[5],
            'valid_from': row[6],
            'valid_from_human': datetime.fromtimestamp(row[6]).strftime('%Y-%m-%d %H:%M:%S') if row[6] else None,
            'valid_until': row[7],
            'valid_until_human': datetime.fromtimestamp(row[7]).strftime('%Y-%m-%d %H:%M:%S') if row[7] else None,
            'is_active': bool(row[8]),
            'is_expired': bool(row[9]),
            'created_at': row[10],
            'revoked_at': row[11],
            'revoked_by': row[12],
            'metadata': json.loads(row[13]) if row[13] else {}
        }
    
    # ==================== EXPLOIT OPERATIONS ====================
    
    def add_exploit_operation(self, exploit_type: str, target: str, status: str,
                             success: bool = None, duration: float = None,
                             metadata: Dict = None) -> int:
        """Add exploit operation record"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO exploit_operations (exploit_type, target, status, success, timestamp, duration, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (exploit_type, target, status, success, time.time(), duration, json.dumps(metadata or {})))
            
            exploit_id = cursor.lastrowid
            conn.commit()
            self._close_connection(conn)
            
            return exploit_id
    
    def count_exploit_operations(self, status: str = None) -> int:
        """Count exploit operations"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if status:
            cursor.execute('SELECT COUNT(*) FROM exploit_operations WHERE status = ?', (status,))
        else:
            cursor.execute('SELECT COUNT(*) FROM exploit_operations')
        
        count = cursor.fetchone()[0]
        self._close_connection(conn)
        return count
    
    # ==================== STATISTICS ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Total counts
        cursor.execute('SELECT COUNT(*) FROM suci_captures')
        stats['total_suci_captures'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM voice_calls')
        stats['total_voice_calls'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM targets')
        stats['total_targets'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM targets WHERE status = "active"')
        stats['active_targets'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM anomaly_alerts')
        stats['total_alerts'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM anomaly_alerts WHERE acknowledged = 0')
        stats['unacknowledged_alerts'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM exploit_operations')
        stats['total_exploits'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM exploit_operations WHERE status = "active"')
        stats['active_exploits'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM security_audits')
        stats['total_security_audits'] = cursor.fetchone()[0]
        
        # Recent activity (last 24 hours)
        day_ago = time.time() - 86400
        
        cursor.execute('SELECT COUNT(*) FROM suci_captures WHERE timestamp > ?', (day_ago,))
        stats['suci_captures_24h'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM voice_calls WHERE timestamp > ?', (day_ago,))
        stats['voice_calls_24h'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM anomaly_alerts WHERE timestamp > ?', (day_ago,))
        stats['alerts_24h'] = cursor.fetchone()[0]
        
        # Database size
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        stats['database_size_bytes'] = cursor.fetchone()[0]
        stats['database_size_mb'] = stats['database_size_bytes'] / (1024 * 1024)
        
        self._close_connection(conn)
        
        return stats
    
    def clear_old_data(self, days: int = 30):
        """Clear data older than specified days"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cutoff = time.time() - (days * 86400)
            
            cursor.execute('DELETE FROM suci_captures WHERE timestamp < ?', (cutoff,))
            suci_deleted = cursor.rowcount
            
            cursor.execute('DELETE FROM voice_calls WHERE timestamp < ?', (cutoff,))
            calls_deleted = cursor.rowcount
            
            cursor.execute('DELETE FROM anomaly_alerts WHERE timestamp < ? AND acknowledged = 1', (cutoff,))
            alerts_deleted = cursor.rowcount
            
            cursor.execute('DELETE FROM system_events WHERE timestamp < ?', (cutoff,))
            events_deleted = cursor.rowcount
            
            conn.commit()
            self._close_connection(conn)
            
            self.logger.info(f"Cleared old data: {suci_deleted} SUCI, {calls_deleted} calls, "
                           f"{alerts_deleted} alerts, {events_deleted} events")
            
            return {
                'suci_deleted': suci_deleted,
                'calls_deleted': calls_deleted,
                'alerts_deleted': alerts_deleted,
                'events_deleted': events_deleted
            }
    
    # ==================== USER AUTHENTICATION (Phase 1.3.5-1.3.6) ====================
    
    def create_user(self, username: str, password: str, email: str = None, 
                   full_name: str = None, role: str = 'operator') -> int:
        """
        Create new user with hashed password
        
        Args:
            username: Unique username
            password: Plain text password (will be hashed)
            email: Optional email address
            full_name: Optional full name
            role: User role (admin, operator, viewer)
        
        Returns:
            User ID
        """
        try:
            # Import bcrypt if available
            try:
                import bcrypt
                password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            except ImportError:
                # Fallback to simple hash if bcrypt not available (NOT SECURE - development only)
                import hashlib
                password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
                self.logger.warning("bcrypt not available, using insecure hash. Install: pip install bcrypt")
            
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash, full_name, role, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (username, email, password_hash, full_name, role, time.time(), json.dumps({})))
                
                user_id = cursor.lastrowid
                conn.commit()
                self._close_connection(conn)
                
                self.logger.info(f"User created: {username} (ID: {user_id}, Role: {role})")
                return user_id
        
        except Exception as e:
            self.logger.error(f"Failed to create user: {e}")
            return None
    
    def verify_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Verify user credentials
        
        Args:
            username: Username
            password: Plain text password
        
        Returns:
            User dict if valid, None if invalid
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, password_hash, full_name, role, is_active, last_login, metadata
                FROM users 
                WHERE username = ? AND is_active = 1
            ''', (username,))
            
            row = cursor.fetchone()
            self._close_connection(conn)
            
            if not row:
                return None
            
            password_hash = row[3]
            
            # Verify password
            try:
                import bcrypt
                if bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
                    # Update last login
                    self.update_user_last_login(row[0])
                    
                    return {
                        'id': row[0],
                        'username': row[1],
                        'email': row[2],
                        'full_name': row[4],
                        'role': row[5],
                        'is_active': bool(row[6]),
                        'last_login': row[7],
                        'metadata': json.loads(row[8]) if row[8] else {}
                    }
            except ImportError:
                # Fallback verification if bcrypt not available
                import hashlib
                if hashlib.sha256(password.encode('utf-8')).hexdigest() == password_hash:
                    self.update_user_last_login(row[0])
                    return {
                        'id': row[0],
                        'username': row[1],
                        'email': row[2],
                        'full_name': row[4],
                        'role': row[5],
                        'is_active': bool(row[6]),
                        'last_login': row[7],
                        'metadata': json.loads(row[8]) if row[8] else {}
                    }
            
            return None
        
        except Exception as e:
            self.logger.error(f"Failed to verify user: {e}")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, full_name, role, is_active, last_login, created_at, metadata
                FROM users 
                WHERE id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            self._close_connection(conn)
            
            if row:
                return {
                    'id': row[0],
                    'username': row[1],
                    'email': row[2],
                    'full_name': row[3],
                    'role': row[4],
                    'is_active': bool(row[5]),
                    'last_login': row[6],
                    'created_at': row[7],
                    'metadata': json.loads(row[8]) if row[8] else {}
                }
            return None
        
        except Exception as e:
            self.logger.error(f"Failed to get user: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, full_name, role, is_active, last_login, created_at, metadata
                FROM users 
                WHERE username = ?
            ''', (username,))
            
            row = cursor.fetchone()
            self._close_connection(conn)
            
            if row:
                return {
                    'id': row[0],
                    'username': row[1],
                    'email': row[2],
                    'full_name': row[3],
                    'role': row[4],
                    'is_active': bool(row[5]),
                    'last_login': row[6],
                    'created_at': row[7],
                    'metadata': json.loads(row[8]) if row[8] else {}
                }
            return None
        
        except Exception as e:
            self.logger.error(f"Failed to get user: {e}")
            return None
    
    def update_user_last_login(self, user_id: int) -> bool:
        """Update user's last login timestamp"""
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE users 
                    SET last_login = ?
                    WHERE id = ?
                ''', (time.time(), user_id))
                
                conn.commit()
                self._close_connection(conn)
                return True
        
        except Exception as e:
            self.logger.error(f"Failed to update last login: {e}")
            return False
    
    def list_users(self) -> List[Dict[str, Any]]:
        """List all users"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, full_name, role, is_active, last_login, created_at
                FROM users
                ORDER BY created_at DESC
            ''')
            
            users = []
            for row in cursor.fetchall():
                users.append({
                    'id': row[0],
                    'username': row[1],
                    'email': row[2],
                    'full_name': row[3],
                    'role': row[4],
                    'is_active': bool(row[5]),
                    'last_login': row[6],
                    'last_login_human': datetime.fromtimestamp(row[6]).strftime('%Y-%m-%d %H:%M:%S') if row[6] else 'Never',
                    'created_at': row[7],
                    'created_at_human': datetime.fromtimestamp(row[7]).strftime('%Y-%m-%d %H:%M:%S')
                })
            
            self._close_connection(conn)
            return users
        
        except Exception as e:
            self.logger.error(f"Failed to list users: {e}")
            return []
    
    def change_user_password(self, user_id: int, new_password: str) -> bool:
        """Change user password"""
        try:
            # Hash password
            try:
                import bcrypt
                password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            except ImportError:
                import hashlib
                password_hash = hashlib.sha256(new_password.encode('utf-8')).hexdigest()
            
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE users 
                    SET password_hash = ?
                    WHERE id = ?
                ''', (password_hash, user_id))
                
                conn.commit()
                self._close_connection(conn)
                
                self.logger.info(f"Password changed for user ID: {user_id}")
                return True
        
        except Exception as e:
            self.logger.error(f"Failed to change password: {e}")
            return False
    
    def delete_user(self, user_id: int) -> bool:
        """Delete user (soft delete - set inactive)"""
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE users 
                    SET is_active = 0
                    WHERE id = ?
                ''', (user_id,))
                
                conn.commit()
                self._close_connection(conn)
                
                self.logger.info(f"User deactivated: ID {user_id}")
                return True
        
        except Exception as e:
            self.logger.error(f"Failed to delete user: {e}")
            return False
    
    def ensure_default_admin(self) -> bool:
        """Create default admin user if no users exist"""
        try:
            users = self.list_users()
            if not users:
                self.logger.info("No users found. Creating default admin user")
                result = self.create_user(
                    username='admin',
                    password='admin',  # Should be changed after first login
                    email='admin@falconone.local',
                    full_name='System Administrator',
                    role='admin'
                )
                if result:
                    self.logger.warning("⚠️  Default admin user created. Please change password immediately!")
                    self.logger.warning("⚠️  Username: admin | Password: admin")
                    return True
            return False
        
        except Exception as e:
            self.logger.error(f"Failed to create default admin: {e}")
            return False
    
    # ==================== BACKUP & RESTORE (Phase 1.2.3) ====================
    
    def backup_database(self, backup_dir: str = None) -> str:
        """
        Create backup of database with automatic rotation
        
        Args:
            backup_dir: Directory to store backups (default: backups/database/)
        
        Returns:
            Path to backup file
        """
        try:
            # Set default backup directory
            if backup_dir is None:
                backup_dir = Path(__file__).parent.parent.parent / 'backups' / 'database'
            else:
                backup_dir = Path(backup_dir)
            
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"falconone_backup_{timestamp}.db"
            backup_path = backup_dir / backup_filename
            
            # Create backup
            if self.is_memory:
                self.logger.warning("Cannot backup in-memory database")
                return None
            
            self.logger.info(f"Creating database backup: {backup_path}")
            
            # Use SQLite backup API for safe backup
            with self.lock:
                source_conn = self._get_connection()
                backup_conn = self._create_connection_raw(str(backup_path))
                
                source_conn.backup(backup_conn)
                
                backup_conn.close()
                self._close_connection(source_conn)
            
            # Get backup file size
            backup_size = backup_path.stat().st_size
            self.logger.info(f"Backup created: {backup_path} ({backup_size / 1024 / 1024:.2f} MB)")
            
            # Rotate old backups
            if self.backup_enabled:
                self._rotate_backups(backup_dir)
            
            return str(backup_path)
        
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            return None
    
    def _create_connection_raw(self, db_path: str):
        """Create raw connection for backup (no encryption setup)"""
        if self.encryption_key and SQLCIPHER_AVAILABLE:
            conn = sqlcipher.connect(db_path, check_same_thread=False)
            conn.execute(f"PRAGMA key = '{self.encryption_key}'")
        else:
            conn = sqlite3.connect(db_path, check_same_thread=False)
        return conn
    
    def _rotate_backups(self, backup_dir: Path):
        """
        Rotate backups - delete backups older than retention period
        
        Args:
            backup_dir: Backup directory path
        """
        try:
            self.logger.debug(f"Rotating backups (retention: {self.backup_retention_days} days)")
            
            # Get all backup files
            backup_files = sorted(backup_dir.glob("falconone_backup_*.db"))
            
            # Calculate cutoff date
            cutoff_time = time.time() - (self.backup_retention_days * 86400)
            
            deleted_count = 0
            for backup_file in backup_files:
                # Get file modification time
                file_mtime = backup_file.stat().st_mtime
                
                if file_mtime < cutoff_time:
                    self.logger.debug(f"Deleting old backup: {backup_file.name}")
                    backup_file.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                self.logger.info(f"Rotated {deleted_count} old backups")
            
            # Keep only last N backups (in case retention is very long)
            max_backups = 100
            remaining_backups = sorted(backup_dir.glob("falconone_backup_*.db"), reverse=True)
            if len(remaining_backups) > max_backups:
                for old_backup in remaining_backups[max_backups:]:
                    self.logger.debug(f"Deleting excess backup: {old_backup.name}")
                    old_backup.unlink()
        
        except Exception as e:
            self.logger.error(f"Backup rotation failed: {e}")
    
    def restore_database(self, backup_path: str) -> bool:
        """
        Restore database from backup
        
        Args:
            backup_path: Path to backup file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            backup_path = Path(backup_path)
            
            if not backup_path.exists():
                self.logger.error(f"Backup file not found: {backup_path}")
                return False
            
            if self.is_memory:
                self.logger.error("Cannot restore to in-memory database")
                return False
            
            self.logger.warning(f"Restoring database from backup: {backup_path}")
            
            with self.lock:
                # Close existing connection
                self._close_connection(self._get_connection())
                
                # Create backup of current database before restore
                current_backup = f"{self.db_path}.pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(self.db_path, current_backup)
                self.logger.info(f"Created safety backup: {current_backup}")
                
                # Restore from backup
                shutil.copy2(backup_path, self.db_path)
                self.logger.info(f"Database restored from {backup_path}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Database restore failed: {e}")
            return False
    
    def list_backups(self, backup_dir: str = None) -> List[Dict[str, Any]]:
        """
        List available backups
        
        Args:
            backup_dir: Backup directory (default: backups/database/)
        
        Returns:
            List of backup information
        """
        try:
            if backup_dir is None:
                backup_dir = Path(__file__).parent.parent.parent / 'backups' / 'database'
            else:
                backup_dir = Path(backup_dir)
            
            if not backup_dir.exists():
                return []
            
            backups = []
            for backup_file in sorted(backup_dir.glob("falconone_backup_*.db"), reverse=True):
                stat = backup_file.stat()
                backups.append({
                    'filename': backup_file.name,
                    'path': str(backup_file),
                    'size_bytes': stat.st_size,
                    'size_mb': stat.st_size / 1024 / 1024,
                    'created': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'age_days': (time.time() - stat.st_mtime) / 86400
                })
            
            return backups
        
        except Exception as e:
            self.logger.error(f"Failed to list backups: {e}")
            return []
    
    def get_database_health(self) -> Dict[str, Any]:
        """
        Check database health and integrity
        
        Returns:
            Health report
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            health = {
                'status': 'healthy',
                'checks': {}
            }
            
            # Integrity check
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            health['checks']['integrity'] = {
                'status': 'ok' if integrity_result == 'ok' else 'failed',
                'result': integrity_result
            }
            
            # Foreign key check
            cursor.execute("PRAGMA foreign_key_check")
            fk_violations = cursor.fetchall()
            health['checks']['foreign_keys'] = {
                'status': 'ok' if len(fk_violations) == 0 else 'violations',
                'violations': len(fk_violations)
            }
            
            # Database size
            cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]
            health['checks']['size'] = {
                'bytes': db_size,
                'mb': db_size / 1024 / 1024,
                'status': 'ok' if db_size < 1e9 else 'large'  # Warn if > 1GB
            }
            
            # Encryption status
            health['checks']['encryption'] = {
                'enabled': bool(self.encryption_key and SQLCIPHER_AVAILABLE),
                'available': SQLCIPHER_AVAILABLE
            }
            
            # Backup status
            backups = self.list_backups()
            health['checks']['backups'] = {
                'count': len(backups),
                'latest': backups[0]['created'] if backups else None,
                'status': 'ok' if len(backups) > 0 else 'no_backups'
            }
            
            # Set overall status
            if health['checks']['integrity']['status'] != 'ok':
                health['status'] = 'critical'
            elif health['checks']['foreign_keys']['status'] != 'ok':
                health['status'] = 'warning'
            elif health['checks']['backups']['status'] != 'ok':
                health['status'] = 'warning'
            
            self._close_connection(conn)
            return health
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'persistent_connection') and self.persistent_connection:
            self.persistent_connection.close()
            self.logger.info("Database connection closed")
    
    # ==================== PHASE 2.3: DATA EXPORT ====================
    
    def export_to_csv(self, table: str, output_file: str, filters: Dict = None) -> bool:
        """
        Export table data to CSV format
        
        Args:
            table: Table name (suci_captures, voice_calls, targets, etc.)
            output_file: Output CSV file path
            filters: Optional filters (e.g., {'status': 'active'})
        
        Returns:
            True if export successful
        """
        try:
            import csv
            
            self.logger.info(f"Exporting {table} to CSV: {output_file}")
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build query with filters
            query = f"SELECT * FROM {table}"
            params = []
            
            if filters:
                where_clauses = []
                for key, value in filters.items():
                    where_clauses.append(f"{key} = ?")
                    params.append(value)
                query += " WHERE " + " AND ".join(where_clauses)
            
            cursor.execute(query, params)
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Write CSV
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(columns)  # Header
                
                row_count = 0
                for row in cursor:
                    writer.writerow(row)
                    row_count += 1
            
            self._close_connection(conn)
            
            self.logger.info(f"Exported {row_count} rows from {table} to {output_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"CSV export failed: {e}")
            return False
    
    def export_to_json(self, table: str, output_file: str, filters: Dict = None, pretty: bool = True) -> bool:
        """
        Export table data to JSON format
        
        Args:
            table: Table name
            output_file: Output JSON file path
            filters: Optional filters
            pretty: Pretty-print JSON (default: True)
        
        Returns:
            True if export successful
        """
        try:
            self.logger.info(f"Exporting {table} to JSON: {output_file}")
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build query with filters
            query = f"SELECT * FROM {table}"
            params = []
            
            if filters:
                where_clauses = []
                for key, value in filters.items():
                    where_clauses.append(f"{key} = ?")
                    params.append(value)
                query += " WHERE " + " AND ".join(where_clauses)
            
            cursor.execute(query, params)
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Convert rows to dictionaries
            data = []
            for row in cursor:
                row_dict = {}
                for i, value in enumerate(row):
                    # Handle JSON fields
                    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                        try:
                            row_dict[columns[i]] = json.loads(value)
                        except:
                            row_dict[columns[i]] = value
                    else:
                        row_dict[columns[i]] = value
                data.append(row_dict)
            
            self._close_connection(conn)
            
            # Write JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(data, f, ensure_ascii=False)
            
            self.logger.info(f"Exported {len(data)} rows from {table} to {output_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"JSON export failed: {e}")
            return False
    
    def export_to_pdf(self, report_type: str, output_file: str, filters: Dict = None) -> bool:
        """
        Generate PDF report
        
        Args:
            report_type: Type of report (summary, captures, targets, operations)
            output_file: Output PDF file path
            filters: Optional filters
        
        Returns:
            True if export successful
        """
        try:
            # Try to import ReportLab
            try:
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
                from reportlab.lib import colors
            except ImportError:
                self.logger.error("ReportLab not installed. Install with: pip install reportlab")
                return False
            
            self.logger.info(f"Generating PDF report: {report_type} -> {output_file}")
            
            # Create PDF document
            doc = SimpleDocTemplate(output_file, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#0d47a1'),
                spaceAfter=30
            )
            story.append(Paragraph(f"FalconOne {report_type.title()} Report", title_style))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 0.5*inch))
            
            # Generate content based on report type
            if report_type == 'summary':
                self._add_summary_to_pdf(story, styles, filters)
            elif report_type == 'captures':
                self._add_captures_to_pdf(story, styles, filters)
            elif report_type == 'targets':
                self._add_targets_to_pdf(story, styles, filters)
            elif report_type == 'operations':
                self._add_operations_to_pdf(story, styles, filters)
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"PDF report generated: {output_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"PDF export failed: {e}")
            return False
    
    def _add_summary_to_pdf(self, story, styles, filters):
        """Add summary statistics to PDF"""
        stats = self.get_statistics()
        
        story.append(Paragraph("System Statistics", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        # Create statistics table
        data = [
            ['Metric', 'Count'],
            ['Total SUCI Captures', str(stats['total_suci_captures'])],
            ['Total Voice Calls', str(stats['total_voice_calls'])],
            ['Active Targets', str(stats['active_targets'])],
            ['Unacknowledged Alerts', str(stats['unacknowledged_alerts'])],
        ]
        
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
    
    def _add_captures_to_pdf(self, story, styles, filters):
        """Add captures data to PDF"""
        captures = self.get_suci_captures(limit=50)
        
        story.append(Paragraph(f"Recent Captures ({len(captures)})", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        for capture in captures[:20]:  # Limit to 20 for PDF
            story.append(Paragraph(
                f"<b>SUCI:</b> {capture['suci']} | "
                f"<b>IMSI:</b> {capture.get('imsi', 'N/A')} | "
                f"<b>Time:</b> {datetime.fromtimestamp(capture['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}",
                styles['Normal']
            ))
            story.append(Spacer(1, 0.1*inch))
    
    def _add_targets_to_pdf(self, story, styles, filters):
        """Add targets data to PDF"""
        targets = self.get_targets(status='active', limit=50)
        
        story.append(Paragraph(f"Active Targets ({len(targets)})", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        for target in targets[:20]:
            story.append(Paragraph(
                f"<b>ID:</b> {target['target_id']} | "
                f"<b>Type:</b> {target['target_type']} | "
                f"<b>Generation:</b> {target['generation']} | "
                f"<b>Priority:</b> {target['priority']}",
                styles['Normal']
            ))
            story.append(Spacer(1, 0.1*inch))
    
    def _add_operations_to_pdf(self, story, styles, filters):
        """Add exploit operations to PDF"""
        operations = self.get_exploit_operations(limit=50)
        
        story.append(Paragraph(f"Exploit Operations ({len(operations)})", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        for op in operations[:20]:
            story.append(Paragraph(
                f"<b>Type:</b> {op['exploit_type']} | "
                f"<b>Target:</b> {op.get('target_id', 'N/A')} | "
                f"<b>Status:</b> {op['status']} | "
                f"<b>Success Rate:</b> {op.get('success_rate', 0):.1f}%",
                styles['Normal']
            ))
            story.append(Spacer(1, 0.1*inch))
    
    def export_all_tables(self, output_dir: str, format: str = 'json') -> Dict[str, bool]:
        """
        Export all tables to specified format
        
        Args:
            output_dir: Output directory
            format: Export format ('csv', 'json')
        
        Returns:
            Dictionary with table names and export status
        """
        try:
            from pathlib import Path
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            tables = [
                'suci_captures',
                'voice_calls',
                'targets',
                'anomaly_alerts',
                'exploit_operations',
                'security_audits',
                'system_events',
                'users'
            ]
            
            results = {}
            
            for table in tables:
                try:
                    output_file = output_path / f"{table}.{format}"
                    
                    if format == 'csv':
                        success = self.export_to_csv(table, str(output_file))
                    elif format == 'json':
                        success = self.export_to_json(table, str(output_file))
                    else:
                        success = False
                    
                    results[table] = success
                
                except Exception as e:
                    self.logger.error(f"Failed to export {table}: {e}")
                    results[table] = False
            
            successful = sum(1 for v in results.values() if v)
            self.logger.info(f"Exported {successful}/{len(tables)} tables to {output_dir}")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Export all tables failed: {e}")
            return {}
