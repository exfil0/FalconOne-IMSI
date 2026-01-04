"""
FalconOne Post-Quantum Cryptography Unit Tests
Tests for post_quantum.py hybrid schemes and OQS integration

Version: 1.9.6
Coverage: HybridKEMScheme, HybridSignatureScheme, OQSWrapper, QuantumAttackSimulator
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys


@pytest.fixture
def mock_logger():
    """Mock logger"""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


# =============================================================================
# OQSWrapper Tests
# =============================================================================

class TestOQSWrapper:
    """Tests for OQS library wrapper"""
    
    def test_wrapper_initialization(self, mock_logger):
        """Test OQSWrapper initializes correctly"""
        from falconone.crypto.post_quantum import OQSWrapper
        
        wrapper = OQSWrapper(mock_logger)
        
        assert wrapper is not None
        assert hasattr(wrapper, 'is_available')
        assert hasattr(wrapper, '_available_kems')
        assert hasattr(wrapper, '_available_sigs')
    
    def test_supported_kem_algorithms(self, mock_logger):
        """Test supported KEM algorithms are defined"""
        from falconone.crypto.post_quantum import OQSWrapper
        
        wrapper = OQSWrapper(mock_logger)
        
        assert 'kyber512' in wrapper.SUPPORTED_KEMS
        assert 'kyber768' in wrapper.SUPPORTED_KEMS
        assert 'kyber1024' in wrapper.SUPPORTED_KEMS
    
    def test_supported_sig_algorithms(self, mock_logger):
        """Test supported signature algorithms are defined"""
        from falconone.crypto.post_quantum import OQSWrapper
        
        wrapper = OQSWrapper(mock_logger)
        
        assert 'dilithium2' in wrapper.SUPPORTED_SIGS
        assert 'dilithium3' in wrapper.SUPPORTED_SIGS
        assert 'dilithium5' in wrapper.SUPPORTED_SIGS
    
    def test_list_available_kems(self, mock_logger):
        """Test listing available KEMs"""
        from falconone.crypto.post_quantum import OQSWrapper
        
        wrapper = OQSWrapper(mock_logger)
        kems = wrapper.list_available_kems()
        
        assert isinstance(kems, list)
    
    def test_list_available_sigs(self, mock_logger):
        """Test listing available signatures"""
        from falconone.crypto.post_quantum import OQSWrapper
        
        wrapper = OQSWrapper(mock_logger)
        sigs = wrapper.list_available_sigs()
        
        assert isinstance(sigs, list)
    
    def test_create_kem_invalid_algorithm(self, mock_logger):
        """Test creating KEM with invalid algorithm returns None"""
        from falconone.crypto.post_quantum import OQSWrapper
        
        wrapper = OQSWrapper(mock_logger)
        result = wrapper.create_kem('invalid_algorithm')
        
        assert result is None
    
    def test_create_sig_invalid_algorithm(self, mock_logger):
        """Test creating signature with invalid algorithm returns None"""
        from falconone.crypto.post_quantum import OQSWrapper
        
        wrapper = OQSWrapper(mock_logger)
        result = wrapper.create_sig('invalid_algorithm')
        
        assert result is None


# =============================================================================
# HybridKEMScheme Tests
# =============================================================================

class TestHybridKEMScheme:
    """Tests for hybrid classical + PQ key encapsulation"""
    
    def test_init_default_algorithms(self, mock_logger):
        """Test initialization with default algorithms"""
        from falconone.crypto.post_quantum import HybridKEMScheme
        
        kem = HybridKEMScheme(logger=mock_logger)
        
        assert kem.classical == 'x25519'
        assert kem.pq == 'kyber768'
        assert kem.security_bits >= 192
    
    def test_init_custom_algorithms(self, mock_logger):
        """Test initialization with custom algorithms"""
        from falconone.crypto.post_quantum import HybridKEMScheme
        
        kem = HybridKEMScheme(
            classical='ecdh_p384',
            pq='kyber1024',
            logger=mock_logger
        )
        
        assert kem.classical == 'ecdh_p384'
        assert kem.pq == 'kyber1024'
        assert kem.security_bits >= 256
    
    def test_init_invalid_classical_raises(self, mock_logger):
        """Test invalid classical algorithm raises ValueError"""
        from falconone.crypto.post_quantum import HybridKEMScheme
        
        with pytest.raises(ValueError, match="Unsupported classical"):
            HybridKEMScheme(classical='invalid', logger=mock_logger)
    
    def test_init_invalid_pq_raises(self, mock_logger):
        """Test invalid PQ algorithm raises ValueError"""
        from falconone.crypto.post_quantum import HybridKEMScheme
        
        with pytest.raises(ValueError, match="Unsupported PQ"):
            HybridKEMScheme(pq='invalid', logger=mock_logger)
    
    def test_supported_classical_algorithms(self, mock_logger):
        """Test all classical algorithms are supported"""
        from falconone.crypto.post_quantum import HybridKEMScheme
        
        assert 'x25519' in HybridKEMScheme.CLASSICAL_ALGORITHMS
        assert 'ecdh_p256' in HybridKEMScheme.CLASSICAL_ALGORITHMS
        assert 'ecdh_p384' in HybridKEMScheme.CLASSICAL_ALGORITHMS
        assert 'ecdh_p521' in HybridKEMScheme.CLASSICAL_ALGORITHMS
    
    def test_keygen_returns_hybrid_keypair(self, mock_logger):
        """Test keygen returns HybridKeyPair dataclass"""
        from falconone.crypto.post_quantum import HybridKEMScheme, HybridKeyPair
        
        kem = HybridKEMScheme(logger=mock_logger)
        keypair = kem.keygen()
        
        assert isinstance(keypair, HybridKeyPair)
        assert keypair.classical_algorithm == 'x25519'
        assert keypair.pq_algorithm == 'kyber768'
        assert len(keypair.classical_public_key) > 0
        assert len(keypair.pq_public_key) > 0
        assert len(keypair.combined_public_key) > 0
    
    def test_encapsulate_decapsulate_roundtrip(self, mock_logger):
        """Test full encapsulate/decapsulate roundtrip"""
        from falconone.crypto.post_quantum import HybridKEMScheme
        
        kem = HybridKEMScheme(logger=mock_logger)
        
        # Generate keypair
        keypair = kem.keygen()
        
        # Encapsulate (sender side)
        ciphertext, shared_secret_sender = kem.encapsulate(keypair.combined_public_key)
        
        # Decapsulate (receiver side)
        shared_secret_receiver = kem.decapsulate(ciphertext, keypair)
        
        # Both parties should derive the same shared secret
        assert shared_secret_sender == shared_secret_receiver
        assert len(shared_secret_sender) == 48  # SHA-384 output
    
    def test_encapsulate_returns_hybrid_ciphertext(self, mock_logger):
        """Test encapsulate returns HybridCiphertext"""
        from falconone.crypto.post_quantum import HybridKEMScheme, HybridCiphertext
        
        kem = HybridKEMScheme(logger=mock_logger)
        keypair = kem.keygen()
        
        ciphertext, _ = kem.encapsulate(keypair.combined_public_key)
        
        assert isinstance(ciphertext, HybridCiphertext)
        assert len(ciphertext.classical_ciphertext) > 0
        assert len(ciphertext.pq_ciphertext) > 0
        assert len(ciphertext.combined_ciphertext) > 0
    
    def test_different_keypairs_different_secrets(self, mock_logger):
        """Test different keypairs produce different shared secrets"""
        from falconone.crypto.post_quantum import HybridKEMScheme
        
        kem = HybridKEMScheme(logger=mock_logger)
        
        keypair1 = kem.keygen()
        keypair2 = kem.keygen()
        
        _, secret1 = kem.encapsulate(keypair1.combined_public_key)
        _, secret2 = kem.encapsulate(keypair2.combined_public_key)
        
        assert secret1 != secret2
    
    def test_ecdh_p256_variant(self, mock_logger):
        """Test ECDH-P256 + Kyber combination"""
        from falconone.crypto.post_quantum import HybridKEMScheme
        
        kem = HybridKEMScheme(classical='ecdh_p256', pq='kyber768', logger=mock_logger)
        keypair = kem.keygen()
        
        ciphertext, sender_secret = kem.encapsulate(keypair.combined_public_key)
        receiver_secret = kem.decapsulate(ciphertext, keypair)
        
        assert sender_secret == receiver_secret


# =============================================================================
# HybridSignatureScheme Tests
# =============================================================================

class TestHybridSignatureScheme:
    """Tests for hybrid classical + PQ digital signatures"""
    
    def test_init_default_algorithms(self, mock_logger):
        """Test initialization with default algorithms"""
        from falconone.crypto.post_quantum import HybridSignatureScheme
        
        sig = HybridSignatureScheme(logger=mock_logger)
        
        assert sig.classical == 'ed25519'
        assert sig.pq == 'dilithium3'
        assert sig.security_bits >= 192
    
    def test_init_custom_algorithms(self, mock_logger):
        """Test initialization with custom algorithms"""
        from falconone.crypto.post_quantum import HybridSignatureScheme
        
        sig = HybridSignatureScheme(
            classical='ecdsa_p384',
            pq='dilithium5',
            logger=mock_logger
        )
        
        assert sig.classical == 'ecdsa_p384'
        assert sig.pq == 'dilithium5'
    
    def test_init_invalid_classical_raises(self, mock_logger):
        """Test invalid classical algorithm raises ValueError"""
        from falconone.crypto.post_quantum import HybridSignatureScheme
        
        with pytest.raises(ValueError, match="Unsupported classical"):
            HybridSignatureScheme(classical='invalid', logger=mock_logger)
    
    def test_keygen_returns_hybrid_keypair(self, mock_logger):
        """Test keygen returns HybridKeyPair"""
        from falconone.crypto.post_quantum import HybridSignatureScheme, HybridKeyPair
        
        sig = HybridSignatureScheme(logger=mock_logger)
        keypair = sig.keygen()
        
        assert isinstance(keypair, HybridKeyPair)
        assert keypair.classical_algorithm == 'ed25519'
        assert keypair.pq_algorithm == 'dilithium3'
    
    def test_sign_returns_hybrid_signature(self, mock_logger):
        """Test sign returns HybridSignature"""
        from falconone.crypto.post_quantum import HybridSignatureScheme, HybridSignature
        
        sig = HybridSignatureScheme(logger=mock_logger)
        keypair = sig.keygen()
        
        message = b"Test message for signing"
        signature = sig.sign(message, keypair)
        
        assert isinstance(signature, HybridSignature)
        assert signature.classical_algorithm == 'ed25519'
        assert signature.pq_algorithm == 'dilithium3'
        assert len(signature.classical_signature) > 0
        assert len(signature.pq_signature) > 0
    
    def test_sign_verify_roundtrip(self, mock_logger):
        """Test full sign/verify roundtrip"""
        from falconone.crypto.post_quantum import HybridSignatureScheme
        
        sig = HybridSignatureScheme(logger=mock_logger)
        keypair = sig.keygen()
        
        message = b"Test message for signing"
        signature = sig.sign(message, keypair)
        
        is_valid = sig.verify(message, signature, keypair.combined_public_key)
        
        assert is_valid is True
    
    def test_verify_wrong_message_fails(self, mock_logger):
        """Test verification fails with wrong message"""
        from falconone.crypto.post_quantum import HybridSignatureScheme
        
        sig = HybridSignatureScheme(logger=mock_logger)
        keypair = sig.keygen()
        
        message = b"Original message"
        signature = sig.sign(message, keypair)
        
        wrong_message = b"Different message"
        is_valid = sig.verify(wrong_message, signature, keypair.combined_public_key)
        
        assert is_valid is False
    
    def test_verify_wrong_key_fails(self, mock_logger):
        """Test verification fails with wrong public key"""
        from falconone.crypto.post_quantum import HybridSignatureScheme
        
        sig = HybridSignatureScheme(logger=mock_logger)
        keypair1 = sig.keygen()
        keypair2 = sig.keygen()
        
        message = b"Test message"
        signature = sig.sign(message, keypair1)
        
        # Try to verify with different keypair's public key
        is_valid = sig.verify(message, signature, keypair2.combined_public_key)
        
        assert is_valid is False
    
    def test_ecdsa_p256_variant(self, mock_logger):
        """Test ECDSA-P256 + Dilithium combination"""
        from falconone.crypto.post_quantum import HybridSignatureScheme
        
        sig = HybridSignatureScheme(
            classical='ecdsa_p256',
            pq='dilithium3',
            logger=mock_logger
        )
        keypair = sig.keygen()
        
        message = b"Test with ECDSA"
        signature = sig.sign(message, keypair)
        is_valid = sig.verify(message, signature, keypair.combined_public_key)
        
        assert is_valid is True


# =============================================================================
# QuantumAttackSimulator Tests
# =============================================================================

class TestQuantumAttackSimulator:
    """Tests for quantum attack simulation"""
    
    def test_init(self, mock_logger):
        """Test QuantumAttackSimulator initialization"""
        from falconone.crypto.post_quantum import QuantumAttackSimulator
        
        sim = QuantumAttackSimulator(mock_logger)
        
        assert sim is not None
        assert hasattr(sim, 'is_available')
    
    def test_grovers_speedup_128bit(self, mock_logger):
        """Test Grover's speedup analysis for 128-bit keys"""
        from falconone.crypto.post_quantum import QuantumAttackSimulator
        
        sim = QuantumAttackSimulator(mock_logger)
        result = sim.simulate_grovers_speedup(128)
        
        assert result['algorithm'] == 'grovers'
        assert result['search_space_bits'] == 128
        assert result['effective_security_bits'] == 64
        assert result['speedup_factor'] > 1
        assert 'recommendation' in result
    
    def test_grovers_speedup_256bit(self, mock_logger):
        """Test Grover's speedup analysis for 256-bit keys"""
        from falconone.crypto.post_quantum import QuantumAttackSimulator
        
        sim = QuantumAttackSimulator(mock_logger)
        result = sim.simulate_grovers_speedup(256)
        
        assert result['effective_security_bits'] == 128
        assert 'Secure' in result['recommendation']
    
    def test_shors_attack_rsa(self, mock_logger):
        """Test Shor's attack analysis for RSA"""
        from falconone.crypto.post_quantum import QuantumAttackSimulator
        
        sim = QuantumAttackSimulator(mock_logger)
        result = sim.simulate_shors_attack('rsa', 2048)
        
        assert result['algorithm'] == 'shors'
        assert result['target_algorithm'] == 'rsa'
        assert result['key_bits'] == 2048
        assert result['qubits_required'] > 0
        assert result['quantum_resistant'] is False
        assert 'PQC' in result['recommendation']
    
    def test_shors_attack_ecdh(self, mock_logger):
        """Test Shor's attack analysis for ECDH"""
        from falconone.crypto.post_quantum import QuantumAttackSimulator
        
        sim = QuantumAttackSimulator(mock_logger)
        result = sim.simulate_shors_attack('ecdh', 256)
        
        assert result['target_algorithm'] == 'ecdh'
        assert result['time_complexity'] == 'polynomial'
        assert result['quantum_resistant'] is False
    
    def test_validate_hybrid_scheme(self, mock_logger):
        """Test hybrid scheme validation"""
        from falconone.crypto.post_quantum import QuantumAttackSimulator
        
        sim = QuantumAttackSimulator(mock_logger)
        result = sim.validate_hybrid_scheme('x25519', 'kyber768')
        
        assert result['hybrid_scheme'] == 'x25519+kyber768'
        assert result['classical_component']['shor_vulnerable'] is True
        assert result['pq_component']['shor_vulnerable'] is False
        assert result['hybrid_security']['defense_in_depth'] is True
        assert 'IETF' in str(result['compliant_with'])
    
    def test_qkd_simulation_unavailable_graceful(self, mock_logger):
        """Test QKD simulation handles missing qiskit gracefully"""
        from falconone.crypto.post_quantum import QuantumAttackSimulator
        
        sim = QuantumAttackSimulator(mock_logger)
        result = sim.run_qkd_simulation(num_qubits=8)
        
        # Should return a result dict (success or error)
        assert isinstance(result, dict)
        assert 'success' in result or 'error' in result or 'num_qubits' in result


# =============================================================================
# Dataclass Tests
# =============================================================================

class TestPQCDataclasses:
    """Tests for post-quantum dataclasses"""
    
    def test_hybrid_keypair_creation(self):
        """Test HybridKeyPair dataclass"""
        from falconone.crypto.post_quantum import HybridKeyPair
        
        keypair = HybridKeyPair(
            classical_algorithm='x25519',
            pq_algorithm='kyber768',
            classical_public_key=b'classical_pub',
            classical_private_key=b'classical_priv',
            pq_public_key=b'pq_pub',
            pq_private_key=b'pq_priv',
            combined_public_key=b'combined',
            security_level='192-bit hybrid'
        )
        
        assert keypair.classical_algorithm == 'x25519'
        assert keypair.pq_algorithm == 'kyber768'
        assert keypair.security_level == '192-bit hybrid'
        assert keypair.created_at is not None
    
    def test_hybrid_ciphertext_creation(self):
        """Test HybridCiphertext dataclass"""
        from falconone.crypto.post_quantum import HybridCiphertext
        
        ct = HybridCiphertext(
            classical_ciphertext=b'ct_classical',
            pq_ciphertext=b'ct_pq',
            combined_ciphertext=b'ct_combined',
            kdf_info=b'info'
        )
        
        assert ct.classical_ciphertext == b'ct_classical'
        assert ct.pq_ciphertext == b'ct_pq'
    
    def test_hybrid_signature_creation(self):
        """Test HybridSignature dataclass"""
        from falconone.crypto.post_quantum import HybridSignature
        
        sig = HybridSignature(
            classical_algorithm='ed25519',
            pq_algorithm='dilithium3',
            classical_signature=b'sig_classical',
            pq_signature=b'sig_pq',
            combined_signature=b'sig_combined',
            message_hash=b'hash'
        )
        
        assert sig.classical_algorithm == 'ed25519'
        assert sig.pq_algorithm == 'dilithium3'


# =============================================================================
# Integration Tests
# =============================================================================

class TestPQCIntegration:
    """Integration tests for PQC module"""
    
    def test_full_hybrid_key_exchange_flow(self, mock_logger):
        """Test complete hybrid key exchange between two parties"""
        from falconone.crypto.post_quantum import HybridKEMScheme
        
        # Alice's side
        alice_kem = HybridKEMScheme(logger=mock_logger)
        alice_keypair = alice_kem.keygen()
        
        # Alice sends public key to Bob
        alice_public = alice_keypair.combined_public_key
        
        # Bob's side - encapsulates to Alice's public key
        bob_kem = HybridKEMScheme(logger=mock_logger)
        ciphertext, bob_shared_secret = bob_kem.encapsulate(alice_public)
        
        # Alice decapsulates
        alice_shared_secret = alice_kem.decapsulate(ciphertext, alice_keypair)
        
        # Both should have same shared secret
        assert alice_shared_secret == bob_shared_secret
    
    def test_full_hybrid_signature_flow(self, mock_logger):
        """Test complete hybrid signature flow"""
        from falconone.crypto.post_quantum import HybridSignatureScheme
        
        # Signer generates keypair
        signer = HybridSignatureScheme(logger=mock_logger)
        keypair = signer.keygen()
        
        # Signer signs document
        document = b"Important document content"
        signature = signer.sign(document, keypair)
        
        # Verifier verifies (only needs public key)
        verifier = HybridSignatureScheme(logger=mock_logger)
        is_valid = verifier.verify(document, signature, keypair.combined_public_key)
        
        assert is_valid is True
    
    def test_oqs_fallback_to_simulator(self, mock_logger):
        """Test OQS wrapper falls back to simulator when unavailable"""
        from falconone.crypto.post_quantum import HybridKEMScheme, OQSWrapper
        
        # Create wrapper - may or may not have OQS
        wrapper = OQSWrapper(mock_logger)
        
        # Create KEM without OQS wrapper - should use simulator
        kem = HybridKEMScheme(oqs_wrapper=None, logger=mock_logger)
        keypair = kem.keygen()
        
        # Should still work with simulator
        ciphertext, secret1 = kem.encapsulate(keypair.combined_public_key)
        secret2 = kem.decapsulate(ciphertext, keypair)
        
        assert secret1 == secret2
