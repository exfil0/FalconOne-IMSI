"""
FalconOne LE Mode Unit Tests
Test warrant validation, evidence chains, and exploit-listen chains
Version 1.8.1
"""

import pytest
import time
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from falconone.utils.evidence_chain import EvidenceChain, EvidenceBlock, InterceptType
from falconone.le.intercept_enhancer import InterceptEnhancer, ChainType


class TestEvidenceChain:
    """Test evidence chain cryptographic integrity"""
    
    @pytest.fixture
    def config(self):
        """Mock configuration"""
        return {
            'law_enforcement': {
                'enabled': True,
                'exploit_chain_safeguards': {
                    'hash_all_intercepts': True,
                    'immutable_evidence_log': True,
                    'auto_redact_pii': True
                }
            },
            'system': {
                'data_dir': tempfile.mkdtemp()
            }
        }
    
    @pytest.fixture
    def logger(self):
        """Mock logger"""
        import logging
        return logging.getLogger('test')
    
    @pytest.fixture
    def evidence_chain(self, config, logger):
        """Create evidence chain instance"""
        chain = EvidenceChain(config, logger)
        yield chain
        # Cleanup
        shutil.rmtree(config['system']['data_dir'], ignore_errors=True)
    
    def test_genesis_block_created(self, evidence_chain):
        """Test genesis block is created on initialization"""
        assert len(evidence_chain.chain) == 1
        assert evidence_chain.chain[0].intercept_type == 'genesis'
        assert evidence_chain.chain[0].previous_hash == '0' * 64
    
    def test_hash_intercept_creates_block(self, evidence_chain):
        """Test hashing intercept creates new block"""
        block_id = evidence_chain.hash_intercept(
            data=b"test_imsi_001010123456789",
            intercept_type=InterceptType.IMSI_CATCH.value,
            target_imsi="001010123456789",
            warrant_id="WRT-2026-00001",
            operator="test_user"
        )
        
        assert block_id is not None
        assert len(evidence_chain.chain) == 2  # Genesis + new block
        assert evidence_chain.chain[-1].block_id == block_id
    
    def test_pii_redaction(self, evidence_chain):
        """Test PII is redacted (IMSI hashed)"""
        imsi = "001010123456789"
        
        block_id = evidence_chain.hash_intercept(
            data=b"test_data",
            intercept_type=InterceptType.IMSI_CATCH.value,
            target_imsi=imsi,
            warrant_id="WRT-2026-00001"
        )
        
        block = evidence_chain.chain[-1]
        # Should NOT contain plaintext IMSI
        assert block.target_identifier != imsi
        # Should be a hash
        assert len(block.target_identifier) == 16  # Truncated SHA-256
    
    def test_chain_integrity_valid(self, evidence_chain):
        """Test chain integrity verification succeeds for valid chain"""
        # Add multiple blocks
        for i in range(5):
            evidence_chain.hash_intercept(
                data=f"test_data_{i}".encode(),
                intercept_type=InterceptType.IMSI_CATCH.value,
                target_imsi=f"00101012345678{i}",
                warrant_id="WRT-2026-00001"
            )
        
        assert evidence_chain.verify_chain() == True
    
    def test_chain_integrity_tampered(self, evidence_chain):
        """Test chain integrity verification detects tampering"""
        # Add blocks
        evidence_chain.hash_intercept(
            data=b"test_data_1",
            intercept_type=InterceptType.IMSI_CATCH.value,
            target_imsi="001010123456789"
        )
        evidence_chain.hash_intercept(
            data=b"test_data_2",
            intercept_type=InterceptType.IMSI_CATCH.value,
            target_imsi="001010123456790"
        )
        
        # Tamper with chain (change data hash of last block)
        evidence_chain.chain[-1].data_hash = "tampered_hash"
        
        # Should detect tampering
        assert evidence_chain.verify_chain() == False
    
    def test_export_forensic(self, evidence_chain):
        """Test forensic export with chain of custody"""
        block_id = evidence_chain.hash_intercept(
            data=b"test_voice_stream",
            intercept_type=InterceptType.VOLTE_VOICE.value,
            target_imsi="001010123456789",
            warrant_id="WRT-2026-00001",
            operator="officer_smith"
        )
        
        manifest = evidence_chain.export_forensic(block_id, output_path='test_export')
        
        assert manifest['block_id'] == block_id
        assert manifest['integrity_verified'] == True
        assert manifest['warrant_id'] == "WRT-2026-00001"
        assert Path(manifest['chain_of_custody']).exists()
        
        # Cleanup
        shutil.rmtree('test_export', ignore_errors=True)
    
    def test_chain_summary(self, evidence_chain):
        """Test chain summary statistics"""
        # Add blocks with different warrants
        evidence_chain.hash_intercept(b"data1", InterceptType.IMSI_CATCH.value, warrant_id="WRT-001")
        evidence_chain.hash_intercept(b"data2", InterceptType.VOLTE_VOICE.value, warrant_id="WRT-001")
        evidence_chain.hash_intercept(b"data3", InterceptType.SMS.value, warrant_id="WRT-002")
        
        summary = evidence_chain.get_chain_summary()
        
        assert summary['total_blocks'] == 4  # Genesis + 3
        assert summary['total_evidence'] == 3
        assert summary['chain_valid'] == True
        assert 'WRT-001' in summary['warrants']
        assert 'WRT-002' in summary['warrants']
        assert InterceptType.IMSI_CATCH.value in summary['types']


class TestInterceptEnhancer:
    """Test LE intercept enhancer chains"""
    
    @pytest.fixture
    def config(self):
        """Mock configuration"""
        return {
            'law_enforcement': {
                'enabled': True,
                'warrant_validation': {
                    'ocr_enabled': True,
                    'ocr_retries': 3
                },
                'exploit_chain_safeguards': {
                    'mandate_warrant_for_chains': True,
                    'hash_all_intercepts': True,
                    'immutable_evidence_log': True,
                    'auto_redact_pii': True
                },
                'fallback_mode': {
                    'if_warrant_invalid': 'passive_scan',
                    'timeout_seconds': 300
                }
            },
            'system': {
                'data_dir': tempfile.mkdtemp()
            }
        }
    
    @pytest.fixture
    def logger(self):
        """Mock logger"""
        import logging
        return logging.getLogger('test')
    
    @pytest.fixture
    def enhancer(self, config, logger):
        """Create intercept enhancer instance"""
        enh = InterceptEnhancer(config, logger, orchestrator=None)
        yield enh
        # Cleanup
        shutil.rmtree(config['system']['data_dir'], ignore_errors=True)
    
    def test_initialization(self, enhancer):
        """Test enhancer initializes correctly"""
        assert enhancer.le_mode_enabled == True
        assert enhancer.mandate_warrant == True
        assert enhancer.chains_executed == 0
    
    def test_enable_le_mode(self, enhancer):
        """Test enabling LE mode with warrant"""
        result = enhancer.enable_le_mode(
            warrant_id="WRT-2026-00123",
            warrant_metadata={
                'jurisdiction': 'Southern District NY',
                'case_number': '2026-CR-00123',
                'authorized_by': 'Judge Smith',
                'valid_until': (datetime.now() + timedelta(days=30)).isoformat(),
                'operator': 'officer_jones'
            }
        )
        
        assert result == True
        assert enhancer.current_warrant_id == "WRT-2026-00123"
        assert enhancer.warrant_metadata['jurisdiction'] == 'Southern District NY'
    
    def test_disable_le_mode(self, enhancer):
        """Test disabling LE mode"""
        enhancer.enable_le_mode("WRT-2026-00123")
        enhancer.disable_le_mode()
        
        assert enhancer.current_warrant_id is None
        assert enhancer.warrant_metadata == {}
    
    def test_warrant_required_blocks_execution(self, enhancer):
        """Test exploit chain blocked without warrant"""
        # No warrant enabled
        result = enhancer.chain_dos_with_imsi_catch(
            target_ip="192.168.1.100",
            dos_duration=5,
            listen_duration=10
        )
        
        # Should fallback to passive mode
        assert result['mode'] == 'passive'
        assert 'warrant' in result['warning'].lower()
    
    def test_warrant_expiry_blocks_execution(self, enhancer):
        """Test expired warrant blocks execution"""
        # Enable with expired warrant
        enhancer.enable_le_mode(
            warrant_id="WRT-2026-EXPIRED",
            warrant_metadata={
                'valid_until': '2020-01-01T00:00:00'  # Expired
            }
        )
        
        result = enhancer.chain_dos_with_imsi_catch(
            target_ip="192.168.1.100",
            dos_duration=5,
            listen_duration=10
        )
        
        # Should fallback or fail
        assert result['success'] == False or result.get('mode') == 'passive'
    
    def test_dos_imsi_chain_with_warrant(self, enhancer):
        """Test DoS+IMSI chain executes with valid warrant"""
        # Enable LE mode with valid warrant
        enhancer.enable_le_mode(
            warrant_id="WRT-2026-00123",
            warrant_metadata={
                'valid_until': (datetime.now() + timedelta(days=30)).isoformat(),
                'operator': 'officer_smith'
            }
        )
        
        result = enhancer.chain_dos_with_imsi_catch(
            target_ip="192.168.1.100",
            dos_duration=5,
            listen_duration=10
        )
        
        # Should execute (simulated)
        assert result['chain_type'] == ChainType.DOS_IMSI.value
        assert result['warrant_id'] == "WRT-2026-00123"
        assert 'steps' in result
        assert enhancer.chains_executed == 1
    
    def test_enhanced_volte_intercept_with_warrant(self, enhancer):
        """Test enhanced VoLTE intercept with valid warrant"""
        enhancer.enable_le_mode(
            warrant_id="WRT-2026-00123",
            warrant_metadata={
                'valid_until': (datetime.now() + timedelta(days=30)).isoformat(),
                'operator': 'officer_jones'
            }
        )
        
        result = enhancer.enhanced_volte_intercept(
            target_imsi="001010123456789",
            downgrade_to="4G",
            intercept_duration=60
        )
        
        assert result['chain_type'] == ChainType.DOWNGRADE_VOLTE.value
        assert result['warrant_id'] == "WRT-2026-00123"
        assert result['target_imsi'] == "001010123456789"
        assert 'steps' in result
    
    def test_statistics_tracking(self, enhancer):
        """Test statistics tracking"""
        enhancer.enable_le_mode(
            warrant_id="WRT-2026-00123",
            warrant_metadata={
                'valid_until': (datetime.now() + timedelta(days=30)).isoformat()
            }
        )
        
        # Execute chains
        enhancer.chain_dos_with_imsi_catch("192.168.1.100", 5, 10)
        enhancer.enhanced_volte_intercept("001010123456789", "4G", 60)
        
        stats = enhancer.get_statistics()
        
        assert stats['chains_executed'] >= 2
        assert stats['active_warrant'] == "WRT-2026-00123"
        assert stats['le_mode_enabled'] == True
        assert 'success_rate' in stats


@pytest.mark.integration
class TestLEModeIntegration:
    """Integration tests for LE mode with orchestrator"""
    
    def test_end_to_end_chain_with_evidence(self):
        """Test complete chain with evidence export"""
        # This would require full orchestrator setup
        # Placeholder for integration test
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
