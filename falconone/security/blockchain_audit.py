"""
FalconOne Blockchain Audit Trail Module (v3.0)
Immutable audit logging using distributed ledger technology

Implements:
- Blockchain-based audit trail for security events
- Tamper-proof logging with cryptographic verification
- Distributed ledger for compliance and accountability
- Smart contract integration (Ethereum/Hyperledger)

Note: This is a simplified implementation. Production systems should use
      Hyperledger Fabric, Ethereum, or similar battle-tested platforms.

Version: 3.0.0
"""

import hashlib
import time
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging

try:
    from ..utils.logger import ModuleLogger
except ImportError:
    class ModuleLogger:
        def __init__(self, name, parent):
            self.logger = logging.getLogger(name) if parent else logging.getLogger(__name__)
        def info(self, msg, **kw): self.logger.info(f"{msg} {kw if kw else ''}")
        def warning(self, msg, **kw): self.logger.warning(f"{msg} {kw if kw else ''}")
        def error(self, msg, **kw): self.logger.error(f"{msg} {kw if kw else ''}")


@dataclass
class AuditEvent:
    """Audit event structure"""
    event_id: str
    event_type: str
    timestamp: float
    user_id: str
    action: str
    resource: str
    details: Dict[str, Any]
    severity: str = 'INFO'  # INFO, WARNING, ERROR, CRITICAL
    ip_address: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class Block:
    """Blockchain block"""
    index: int
    timestamp: float
    events: List[AuditEvent]
    previous_hash: str
    nonce: int = 0
    hash: str = None
    
    def __post_init__(self):
        if self.hash is None:
            self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_data = {
            'index': self.index,
            'timestamp': self.timestamp,
            'events': [event.to_dict() for event in self.events],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 4):
        """
        Mine block using Proof of Work
        
        Args:
            difficulty: Number of leading zeros required in hash
        """
        target = '0' * difficulty
        
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'events': [event.to_dict() for event in self.events],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'hash': self.hash,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ChainMetadata:
    """Blockchain metadata"""
    chain_id: str
    genesis_timestamp: float
    total_blocks: int
    total_events: int
    difficulty: int
    consensus_algorithm: str = 'PoW'
    network_nodes: List[str] = field(default_factory=list)


class BlockchainAuditTrail:
    """
    Blockchain-based audit trail
    
    Provides:
    - Immutable logging of security events
    - Tamper-proof audit trail with cryptographic verification
    - Block mining with Proof of Work
    - Chain validation and integrity checking
    - Query and search capabilities
    """
    
    def __init__(self, chain_id: str = 'falconone-audit', difficulty: int = 4,
                 logger=None):
        """
        Initialize blockchain audit trail
        
        Args:
            chain_id: Unique identifier for this blockchain
            difficulty: PoW difficulty (number of leading zeros)
            logger: Optional logger instance
        """
        self.logger = ModuleLogger('BlockchainAudit', logger)
        self.chain_id = chain_id
        self.difficulty = difficulty
        
        # Initialize blockchain
        self.chain: List[Block] = []
        self.pending_events: List[AuditEvent] = []
        self.block_size = 10  # Events per block
        
        # Create genesis block
        self._create_genesis_block()
        
        # Check for blockchain library availability
        self.web3_available = False
        self.fabric_available = False
        
        try:
            import web3
            self.web3_available = True
            self.logger.info("Web3 (Ethereum) library available")
        except ImportError:
            self.logger.warning("Web3 not available")
        
        try:
            import hfc  # Hyperledger Fabric SDK
            self.fabric_available = True
            self.logger.info("Hyperledger Fabric SDK available")
        except ImportError:
            self.logger.warning("Hyperledger Fabric SDK not available")
        
        self.statistics = {
            'total_blocks': 1,
            'total_events': 0,
            'blocks_mined': 0,
            'validation_checks': 0,
            'integrity_violations': 0,
        }
    
    def _create_genesis_block(self):
        """Create the genesis block"""
        genesis_event = AuditEvent(
            event_id='genesis',
            event_type='SYSTEM',
            timestamp=time.time(),
            user_id='system',
            action='CHAIN_INIT',
            resource='blockchain',
            details={'chain_id': self.chain_id},
            severity='INFO'
        )
        
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            events=[genesis_event],
            previous_hash='0',
            nonce=0
        )
        
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)
        
        self.logger.info(f"Genesis block created", 
                        hash=genesis_block.hash[:16],
                        difficulty=self.difficulty)
    
    def log_event(self, event_type: str, user_id: str, action: str,
                  resource: str, details: Dict[str, Any],
                  severity: str = 'INFO', ip_address: Optional[str] = None) -> AuditEvent:
        """
        Log an audit event
        
        Args:
            event_type: Type of event (AUTH, ACCESS, EXPLOIT, etc.)
            user_id: User performing action
            action: Action performed (LOGIN, SCAN, EXECUTE, etc.)
            resource: Resource affected
            details: Additional event details
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
            ip_address: Optional IP address
        
        Returns:
            Created AuditEvent
        """
        event = AuditEvent(
            event_id=hashlib.sha256(
                f"{event_type}{user_id}{action}{time.time()}".encode()
            ).hexdigest()[:16],
            event_type=event_type,
            timestamp=time.time(),
            user_id=user_id,
            action=action,
            resource=resource,
            details=details,
            severity=severity,
            ip_address=ip_address
        )
        
        self.pending_events.append(event)
        self.statistics['total_events'] += 1
        
        self.logger.info(f"Event logged: {event_type}/{action}",
                        event_id=event.event_id,
                        user=user_id)
        
        # Auto-mine block if pending events reach block size
        if len(self.pending_events) >= self.block_size:
            self.mine_pending_block()
        
        return event
    
    def mine_pending_block(self) -> Optional[Block]:
        """
        Mine a new block with pending events
        
        Returns:
            Mined Block or None if no pending events
        """
        if not self.pending_events:
            self.logger.warning("No pending events to mine")
            return None
        
        # Get last block hash
        previous_hash = self.chain[-1].hash
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            events=self.pending_events.copy(),
            previous_hash=previous_hash,
            nonce=0
        )
        
        # Mine block (PoW)
        self.logger.info(f"Mining block {new_block.index}...",
                        events=len(new_block.events),
                        difficulty=self.difficulty)
        
        start_time = time.time()
        new_block.mine_block(self.difficulty)
        mining_time = time.time() - start_time
        
        # Add to chain
        self.chain.append(new_block)
        self.pending_events = []
        
        self.statistics['total_blocks'] += 1
        self.statistics['blocks_mined'] += 1
        
        self.logger.info(f"Block mined successfully",
                        index=new_block.index,
                        hash=new_block.hash[:16],
                        nonce=new_block.nonce,
                        mining_time=f"{mining_time:.2f}s")
        
        return new_block
    
    def validate_chain(self) -> bool:
        """
        Validate entire blockchain integrity
        
        Returns:
            True if chain is valid, False if tampered
        """
        self.statistics['validation_checks'] += 1
        
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check 1: Verify current block hash
            if current_block.hash != current_block.calculate_hash():
                self.logger.error(f"Block {i} hash mismatch (tampered)")
                self.statistics['integrity_violations'] += 1
                return False
            
            # Check 2: Verify previous hash link
            if current_block.previous_hash != previous_block.hash:
                self.logger.error(f"Block {i} previous hash mismatch (chain broken)")
                self.statistics['integrity_violations'] += 1
                return False
            
            # Check 3: Verify PoW difficulty
            target = '0' * self.difficulty
            if not current_block.hash.startswith(target):
                self.logger.error(f"Block {i} does not meet PoW difficulty")
                self.statistics['integrity_violations'] += 1
                return False
        
        self.logger.info("Blockchain validation successful",
                        total_blocks=len(self.chain))
        return True
    
    def get_block(self, index: int) -> Optional[Block]:
        """Get block by index"""
        if 0 <= index < len(self.chain):
            return self.chain[index]
        return None
    
    def get_latest_block(self) -> Block:
        """Get the latest block"""
        return self.chain[-1]
    
    def query_events(self, event_type: Optional[str] = None,
                     user_id: Optional[str] = None,
                     action: Optional[str] = None,
                     severity: Optional[str] = None,
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None,
                     limit: int = 100) -> List[AuditEvent]:
        """
        Query audit events with filters
        
        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            action: Filter by action
            severity: Filter by severity
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            limit: Maximum results
        
        Returns:
            List of matching AuditEvents
        """
        results = []
        
        for block in self.chain:
            for event in block.events:
                # Apply filters
                if event_type and event.event_type != event_type:
                    continue
                if user_id and event.user_id != user_id:
                    continue
                if action and event.action != action:
                    continue
                if severity and event.severity != severity:
                    continue
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                
                results.append(event)
                
                if len(results) >= limit:
                    return results
        
        self.logger.info(f"Query returned {len(results)} events")
        return results
    
    def get_user_activity(self, user_id: str, limit: int = 50) -> List[AuditEvent]:
        """Get all activity for a specific user"""
        return self.query_events(user_id=user_id, limit=limit)
    
    def get_critical_events(self, limit: int = 50) -> List[AuditEvent]:
        """Get all critical severity events"""
        return self.query_events(severity='CRITICAL', limit=limit)
    
    def export_chain(self, format: str = 'json') -> str:
        """
        Export blockchain to string format
        
        Args:
            format: 'json' or 'csv'
        
        Returns:
            Serialized blockchain
        """
        if format == 'json':
            chain_data = {
                'chain_id': self.chain_id,
                'difficulty': self.difficulty,
                'total_blocks': len(self.chain),
                'total_events': self.statistics['total_events'],
                'blocks': [block.to_dict() for block in self.chain]
            }
            return json.dumps(chain_data, indent=2)
        
        elif format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['block_index', 'block_hash', 'event_id', 'event_type',
                           'timestamp', 'user_id', 'action', 'resource', 'severity'])
            
            # Write events
            for block in self.chain:
                for event in block.events:
                    writer.writerow([
                        block.index,
                        block.hash,
                        event.event_id,
                        event.event_type,
                        datetime.fromtimestamp(event.timestamp).isoformat(),
                        event.user_id,
                        event.action,
                        event.resource,
                        event.severity
                    ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_chain(self, chain_data: str, format: str = 'json'):
        """
        Import blockchain from string format
        
        Args:
            chain_data: Serialized blockchain
            format: 'json'
        """
        if format != 'json':
            raise ValueError(f"Unsupported format: {format}")
        
        data = json.loads(chain_data)
        
        # Validate chain metadata
        if data['chain_id'] != self.chain_id:
            self.logger.warning(f"Chain ID mismatch: {data['chain_id']} != {self.chain_id}")
        
        # Import blocks
        imported_chain = []
        for block_data in data['blocks']:
            events = [AuditEvent(**event_data) for event_data in block_data['events']]
            block = Block(
                index=block_data['index'],
                timestamp=block_data['timestamp'],
                events=events,
                previous_hash=block_data['previous_hash'],
                nonce=block_data['nonce'],
                hash=block_data['hash']
            )
            imported_chain.append(block)
        
        # Replace chain
        self.chain = imported_chain
        self.statistics['total_blocks'] = len(self.chain)
        
        # Validate imported chain
        if self.validate_chain():
            self.logger.info(f"Chain imported successfully",
                           blocks=len(self.chain),
                           events=data['total_events'])
        else:
            self.logger.error("Imported chain validation failed")
    
    def get_chain_metadata(self) -> ChainMetadata:
        """Get blockchain metadata"""
        return ChainMetadata(
            chain_id=self.chain_id,
            genesis_timestamp=self.chain[0].timestamp if self.chain else 0,
            total_blocks=len(self.chain),
            total_events=self.statistics['total_events'],
            difficulty=self.difficulty,
            consensus_algorithm='PoW',
            network_nodes=[]
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        return {
            **self.statistics,
            'chain_id': self.chain_id,
            'difficulty': self.difficulty,
            'pending_events': len(self.pending_events),
            'chain_valid': self.validate_chain(),
            'web3_available': self.web3_available,
            'fabric_available': self.fabric_available,
        }
    
    def get_merkle_root(self, block_index: int) -> Optional[str]:
        """
        Calculate Merkle root of events in a block
        
        Args:
            block_index: Block index
        
        Returns:
            Merkle root hash or None if block not found
        """
        block = self.get_block(block_index)
        if not block:
            return None
        
        # Get event hashes
        event_hashes = [
            hashlib.sha256(event.to_json().encode()).hexdigest()
            for event in block.events
        ]
        
        # Build Merkle tree
        while len(event_hashes) > 1:
            next_level = []
            for i in range(0, len(event_hashes), 2):
                if i + 1 < len(event_hashes):
                    combined = event_hashes[i] + event_hashes[i + 1]
                else:
                    combined = event_hashes[i] + event_hashes[i]
                
                next_hash = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(next_hash)
            
            event_hashes = next_level
        
        return event_hashes[0] if event_hashes else None
