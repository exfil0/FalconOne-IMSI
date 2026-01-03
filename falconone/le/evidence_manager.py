"""
FalconOne Evidence Manager
Law Enforcement Mode Evidence Chain Management
Version 1.8.1
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from falconone.utils.logger import setup_logger

logger = setup_logger(__name__)


class EvidenceManager:
    """
    LE Mode Evidence Chain Manager

    Manages evidence collection, validation, and chain-of-custody
    for law enforcement operations.
    """

    def __init__(self, evidence_dir: str = "evidence", config: Dict = None):
        """
        Initialize evidence manager.

        Args:
            evidence_dir: Directory for evidence storage
            config: Configuration dictionary
        """
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(exist_ok=True)
        self.config = config or {}

        # Evidence chain
        self.chain = []
        self.current_warrant = None

        logger.info(f"EvidenceManager initialized - dir: {evidence_dir}")

    def validate_warrant(self, warrant_data: Dict) -> bool:
        """
        Validate warrant data and format.

        Args:
            warrant_data: Warrant information

        Returns:
            True if valid
        """
        required_fields = ['warrant_id', 'authority', 'target', 'valid_until']

        for field in required_fields:
            if field not in warrant_data:
                logger.error(f"Warrant missing required field: {field}")
                return False

        # Check expiration
        valid_until = warrant_data.get('valid_until')
        if isinstance(valid_until, str):
            try:
                valid_until = datetime.fromisoformat(valid_until)
            except ValueError:
                logger.error("Invalid warrant expiration format")
                return False

        if datetime.now() > valid_until:
            logger.error("Warrant has expired")
            return False

        self.current_warrant = warrant_data
        logger.info(f"Warrant validated: {warrant_data['warrant_id']}")
        return True

    def add_evidence(self, evidence_type: str, data: Any,
                    metadata: Dict = None) -> str:
        """
        Add evidence to the chain.

        Args:
            evidence_type: Type of evidence
            data: Evidence data
            metadata: Additional metadata

        Returns:
            Evidence ID
        """
        evidence_id = self._generate_evidence_id()

        evidence_entry = {
            'id': evidence_id,
            'type': evidence_type,
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'metadata': metadata or {},
            'warrant_id': self.current_warrant['warrant_id'] if self.current_warrant else None,
            'hash': self._hash_data(data)
        }

        self.chain.append(evidence_entry)

        # Save to file
        self._save_evidence(evidence_entry)

        logger.info(f"Evidence added: {evidence_id} ({evidence_type})")
        return evidence_id

    def get_evidence_chain(self) -> List[Dict]:
        """Get the complete evidence chain."""
        return self.chain.copy()

    def export_chain(self, filepath: str) -> bool:
        """
        Export evidence chain to file.

        Args:
            filepath: Export file path

        Returns:
            True if successful
        """
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'warrant': self.current_warrant,
                    'chain': self.chain,
                    'export_timestamp': datetime.now().isoformat()
                }, f, indent=2, default=str)

            logger.info(f"Evidence chain exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export evidence chain: {e}")
            return False

    def _generate_evidence_id(self) -> str:
        """Generate unique evidence ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_part = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
        return f"EV_{timestamp}_{random_part}"

    def _hash_data(self, data: Any) -> str:
        """Generate hash of evidence data."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _save_evidence(self, evidence: Dict) -> None:
        """Save evidence entry to file."""
        filename = f"{evidence['id']}.json"
        filepath = self.evidence_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(evidence, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save evidence {evidence['id']}: {e}")
    
    def log_event(self, event_type: str, event_data: Dict) -> Optional[str]:
        """
        Log an event to the evidence chain.
        
        This is a convenience method for logging exploit/monitoring events
        that wraps add_evidence() with appropriate formatting.
        
        Args:
            event_type: Type of event (e.g., 'isac_waveform_exploit', 'ntn_beam_hijack')
            event_data: Event data including timestamp, success status, etc.
            
        Returns:
            Evidence hash if successful, None otherwise
        """
        if not self.current_warrant:
            logger.warning(f"log_event called without validated warrant for {event_type}")
            # Still log but mark as unwarranted
            event_data['warrant_status'] = 'no_warrant'
        
        try:
            # Ensure timestamp is present
            if 'timestamp' not in event_data:
                event_data['timestamp'] = datetime.now().timestamp()
            
            # Add evidence to chain
            evidence_id = self.add_evidence(
                evidence_type=event_type,
                data=event_data,
                metadata={
                    'source': 'automated_logging',
                    'event_category': self._categorize_event(event_type)
                }
            )
            
            # Return the hash for verification
            if self.chain:
                return self.chain[-1].get('hash')
            return evidence_id
            
        except Exception as e:
            logger.error(f"Failed to log event {event_type}: {e}")
            return None
    
    def _categorize_event(self, event_type: str) -> str:
        """
        Categorize event type for reporting.
        
        Args:
            event_type: Event type string
            
        Returns:
            Category string
        """
        if 'exploit' in event_type.lower():
            return 'exploitation'
        elif 'sensing' in event_type.lower() or 'monitor' in event_type.lower():
            return 'surveillance'
        elif 'intercept' in event_type.lower():
            return 'interception'
        elif 'quantum' in event_type.lower():
            return 'quantum_operation'
        else:
            return 'general'
    
    def verify_chain_integrity(self) -> bool:
        """
        Verify the integrity of the evidence chain.
        
        Checks that all evidence entries have valid hashes and
        that the chain has not been tampered with.
        
        Returns:
            True if chain is intact, False if tampering detected
        """
        for evidence in self.chain:
            stored_hash = evidence.get('hash')
            computed_hash = self._hash_data(evidence.get('data'))
            
            if stored_hash != computed_hash:
                logger.error(f"Evidence chain integrity violation: {evidence['id']}")
                return False
        
        logger.info(f"Evidence chain verified: {len(self.chain)} entries intact")
        return True
    
    def get_evidence_by_warrant(self, warrant_id: str) -> List[Dict]:
        """
        Get all evidence associated with a specific warrant.
        
        Args:
            warrant_id: Warrant ID to filter by
            
        Returns:
            List of evidence entries for the warrant
        """
        return [e for e in self.chain if e.get('warrant_id') == warrant_id]
    
    def get_evidence_summary(self) -> Dict:
        """
        Get summary statistics of evidence chain.
        
        Returns:
            Dict with chain statistics
        """
        if not self.chain:
            return {'count': 0, 'types': {}, 'warrant': None}
        
        type_counts = {}
        for evidence in self.chain:
            etype = evidence.get('type', 'unknown')
            type_counts[etype] = type_counts.get(etype, 0) + 1
        
        return {
            'count': len(self.chain),
            'types': type_counts,
            'warrant': self.current_warrant.get('warrant_id') if self.current_warrant else None,
            'first_entry': self.chain[0].get('timestamp') if self.chain else None,
            'last_entry': self.chain[-1].get('timestamp') if self.chain else None,
            'integrity_verified': self.verify_chain_integrity()
        }