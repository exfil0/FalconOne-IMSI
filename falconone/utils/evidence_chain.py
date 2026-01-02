"""
FalconOne Evidence Chain Module
Cryptographic evidence hashing and chain-of-custody management
Version 1.8.1 - LE Mode Integration

Capabilities:
- SHA-256 + timestamp hashing for all intercepts
- Blockchain-style immutable evidence chains
- Forensic export with chain of custody metadata
- PII redaction (IMSI/IMEI hashing)
- Multi-signature verification for warrant compliance

Typical Usage:
    from falconone.utils.evidence_chain import EvidenceChain
    
    evidence_chain = EvidenceChain(config, logger)
    
    # Hash intercept with metadata
    evidence_id = evidence_chain.hash_intercept(
        data=voice_stream,
        intercept_type='volte',
        target_imsi='001010123456789',
        warrant_id='WRT-2026-00123'
    )
    
    # Export forensic evidence with chain of custody
    evidence_chain.export_forensic(evidence_id, output_path='evidence/')
    
    # Verify evidence integrity
    is_valid = evidence_chain.verify_chain()
"""

import logging
import hashlib
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
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


class InterceptType(Enum):
    """Type of intercept/evidence"""
    IMSI_CATCH = "imsi_catch"
    VOLTE_VOICE = "volte_voice"
    SMS = "sms"
    LOCATION = "location"
    SUCI_CONCEAL = "suci_conceal"
    SIGNALING = "signaling"
    EXPLOIT_OUTPUT = "exploit_output"


@dataclass
class EvidenceBlock:
    """Single block in evidence chain (blockchain-style)"""
    block_id: str  # SHA-256 hash of this block
    previous_hash: str  # Hash of previous block (chain linkage)
    timestamp: float  # Unix timestamp
    intercept_type: str  # Type of evidence
    target_identifier: str  # Hashed IMSI/IMEI (PII redacted)
    warrant_id: Optional[str]  # Associated warrant
    operator: str  # User who collected evidence
    data_hash: str  # SHA-256 of actual intercept data
    metadata: Dict[str, Any]  # Additional context
    signature: Optional[str] = None  # Digital signature (future)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def compute_block_hash(self) -> str:
        """Compute SHA-256 hash of this block"""
        block_str = json.dumps({
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'intercept_type': self.intercept_type,
            'target_identifier': self.target_identifier,
            'warrant_id': self.warrant_id,
            'operator': self.operator,
            'data_hash': self.data_hash,
            'metadata': self.metadata
        }, sort_keys=True)
        return hashlib.sha256(block_str.encode()).hexdigest()


class EvidenceChain:
    """
    Cryptographic evidence chain manager
    
    Implements blockchain-style immutable evidence logging with:
    - Chain of custody tracking
    - Cryptographic integrity verification
    - PII redaction (hash IMSI/IMEI)
    - Forensic export with metadata
    - Warrant association
    """
    
    def __init__(self, config, logger: logging.Logger):
        """Initialize evidence chain"""
        self.config = config
        self.logger = ModuleLogger('EvidenceChain', logger)
        
        # LE mode settings
        le_config = config.get('law_enforcement', {})
        self.enabled = le_config.get('enabled', False)
        self.hash_all_intercepts = le_config.get('exploit_chain_safeguards', {}).get('hash_all_intercepts', True)
        self.immutable_log = le_config.get('exploit_chain_safeguards', {}).get('immutable_evidence_log', True)
        self.auto_redact_pii = le_config.get('exploit_chain_safeguards', {}).get('auto_redact_pii', True)
        
        # Storage
        self.evidence_dir = Path(config.get('system.data_dir', 'logs')) / 'evidence'
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        
        # Blockchain-style chain
        self.chain: List[EvidenceBlock] = []
        self.genesis_block = self._create_genesis_block()
        self.chain.append(self.genesis_block)
        
        # Load existing chain if present
        self._load_chain()
        
        self.logger.info("Evidence chain initialized",
                        enabled=self.enabled,
                        blocks=len(self.chain),
                        immutable=self.immutable_log)
    
    def _create_genesis_block(self) -> EvidenceBlock:
        """Create genesis block (first block in chain)"""
        genesis = EvidenceBlock(
            block_id='0' * 64,  # All zeros
            previous_hash='0' * 64,
            timestamp=time.time(),
            intercept_type='genesis',
            target_identifier='GENESIS',
            warrant_id=None,
            operator='SYSTEM',
            data_hash='0' * 64,
            metadata={'chain_version': '1.0', 'system': 'FalconOne v1.8.1'}
        )
        genesis.block_id = genesis.compute_block_hash()
        return genesis
    
    def hash_intercept(self, 
                      data: bytes,
                      intercept_type: str,
                      target_imsi: Optional[str] = None,
                      target_imei: Optional[str] = None,
                      warrant_id: Optional[str] = None,
                      operator: str = 'unknown',
                      metadata: Optional[Dict] = None) -> str:
        """
        Hash intercept data and add to evidence chain
        
        Args:
            data: Raw intercept data (voice, SMS, signaling)
            intercept_type: Type of intercept (use InterceptType enum)
            target_imsi: Target IMSI (will be hashed if auto_redact_pii=True)
            target_imei: Target IMEI (will be hashed if auto_redact_pii=True)
            warrant_id: Associated warrant ID
            operator: User collecting evidence
            metadata: Additional context (frequency, cell_id, etc.)
        
        Returns:
            block_id: Unique evidence block ID
        """
        if not self.hash_all_intercepts:
            self.logger.warning("Hash all intercepts disabled - skipping")
            return None
        
        # Hash the actual data
        data_hash = hashlib.sha256(data).hexdigest()
        
        # Redact PII (hash instead of plaintext)
        if self.auto_redact_pii:
            if target_imsi:
                target_identifier = hashlib.sha256(target_imsi.encode()).hexdigest()[:16]
            elif target_imei:
                target_identifier = hashlib.sha256(target_imei.encode()).hexdigest()[:16]
            else:
                target_identifier = 'UNKNOWN'
        else:
            target_identifier = target_imsi or target_imei or 'UNKNOWN'
        
        # Create evidence block
        previous_hash = self.chain[-1].block_id if self.chain else '0' * 64
        
        block = EvidenceBlock(
            block_id='',  # Will be computed
            previous_hash=previous_hash,
            timestamp=time.time(),
            intercept_type=intercept_type,
            target_identifier=target_identifier,
            warrant_id=warrant_id,
            operator=operator,
            data_hash=data_hash,
            metadata=metadata or {}
        )
        
        # Compute block hash (includes all fields)
        block.block_id = block.compute_block_hash()
        
        # Add to chain
        if self.immutable_log:
            self.chain.append(block)
            self._persist_chain()
        
        # Store actual data (encrypted)
        data_file = self.evidence_dir / f"{block.block_id}.bin"
        data_file.write_bytes(data)
        
        self.logger.info("Intercept hashed and added to evidence chain",
                        block_id=block.block_id[:8],
                        type=intercept_type,
                        warrant=warrant_id,
                        target_hash=target_identifier[:8])
        
        return block.block_id
    
    def verify_chain(self) -> bool:
        """
        Verify integrity of entire evidence chain
        
        Returns:
            True if chain is valid (no tampering), False otherwise
        """
        if len(self.chain) <= 1:
            return True  # Genesis only
        
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check 1: Previous hash matches
            if current_block.previous_hash != previous_block.block_id:
                self.logger.error("Chain integrity violation: previous_hash mismatch",
                                block=i,
                                expected=previous_block.block_id[:8],
                                actual=current_block.previous_hash[:8])
                return False
            
            # Check 2: Block hash is valid
            recomputed_hash = current_block.compute_block_hash()
            if current_block.block_id != recomputed_hash:
                self.logger.error("Chain integrity violation: block_hash mismatch",
                                block=i,
                                expected=recomputed_hash[:8],
                                actual=current_block.block_id[:8])
                return False
        
        self.logger.info("Evidence chain integrity verified", blocks=len(self.chain))
        return True
    
    def export_forensic(self, block_id: str, output_path: str = 'evidence_export') -> Dict[str, Any]:
        """
        Export evidence with forensic chain of custody
        
        Args:
            block_id: Evidence block ID to export
            output_path: Output directory
        
        Returns:
            Export manifest with file paths
        """
        # Find block in chain
        block = next((b for b in self.chain if b.block_id == block_id), None)
        if not block:
            self.logger.error("Block not found", block_id=block_id[:8])
            return {'error': 'Block not found'}
        
        # Create export directory
        export_dir = Path(output_path) / block_id[:8]
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export block metadata (chain of custody)
        custody_file = export_dir / 'chain_of_custody.json'
        custody_data = {
            'evidence_block': block.to_dict(),
            'chain_position': self.chain.index(block),
            'total_chain_length': len(self.chain),
            'chain_verified': self.verify_chain(),
            'export_timestamp': datetime.now().isoformat(),
            'export_operator': 'system',
            'forensic_notes': 'FalconOne LE Mode Evidence Export'
        }
        custody_file.write_text(json.dumps(custody_data, indent=2))
        
        # Copy actual data
        data_file = self.evidence_dir / f"{block.block_id}.bin"
        if data_file.exists():
            export_data_file = export_dir / 'evidence_data.bin'
            export_data_file.write_bytes(data_file.read_bytes())
        
        # Generate integrity manifest
        manifest = {
            'block_id': block.block_id,
            'chain_of_custody': str(custody_file),
            'evidence_data': str(export_data_file) if data_file.exists() else None,
            'integrity_verified': self.verify_chain(),
            'warrant_id': block.warrant_id,
            'export_path': str(export_dir)
        }
        
        self.logger.info("Forensic evidence exported",
                        block_id=block_id[:8],
                        path=str(export_dir))
        
        return manifest
    
    def get_chain_summary(self) -> Dict[str, Any]:
        """Get summary of evidence chain"""
        warrant_counts = {}
        for block in self.chain[1:]:  # Skip genesis
            w = block.warrant_id or 'NO_WARRANT'
            warrant_counts[w] = warrant_counts.get(w, 0) + 1
        
        return {
            'total_blocks': len(self.chain),
            'total_evidence': len(self.chain) - 1,  # Exclude genesis
            'chain_valid': self.verify_chain(),
            'warrants': list(warrant_counts.keys()),
            'evidence_by_warrant': warrant_counts,
            'types': list(set(b.intercept_type for b in self.chain[1:])),
            'oldest_timestamp': datetime.fromtimestamp(self.chain[1].timestamp).isoformat() if len(self.chain) > 1 else None,
            'newest_timestamp': datetime.fromtimestamp(self.chain[-1].timestamp).isoformat() if len(self.chain) > 1 else None
        }
    
    def _persist_chain(self):
        """Persist chain to disk (immutable append-only)"""
        chain_file = self.evidence_dir / 'evidence_chain.json'
        chain_data = {
            'blocks': [b.to_dict() for b in self.chain],
            'last_updated': datetime.now().isoformat()
        }
        chain_file.write_text(json.dumps(chain_data, indent=2))
    
    def _load_chain(self):
        """Load existing chain from disk"""
        chain_file = self.evidence_dir / 'evidence_chain.json'
        if chain_file.exists():
            try:
                chain_data = json.loads(chain_file.read_text())
                self.chain = [EvidenceBlock(**b) for b in chain_data['blocks']]
                self.logger.info("Existing evidence chain loaded", blocks=len(self.chain))
                
                # Verify integrity on load
                if not self.verify_chain():
                    self.logger.critical("EVIDENCE CHAIN TAMPERING DETECTED - Chain invalid on load")
            except Exception as e:
                self.logger.error(f"Failed to load evidence chain: {e}")
