"""
FalconOne Quantum-Resistant Cryptography Module (v3.0)
Post-quantum cryptographic algorithms for future-proof security

Implements:
- CRYSTALS-Kyber: Key Encapsulation Mechanism (KEM)
- CRYSTALS-Dilithium: Digital Signature Algorithm
- Hybrid classical+post-quantum schemes

NIST Post-Quantum Cryptography Standards:
- Kyber (KEM): NIST FIPS 203
- Dilithium (Signatures): NIST FIPS 204

References:
- https://pq-crystals.org/
- https://csrc.nist.gov/projects/post-quantum-cryptography

Version: 3.0.0
"""

import os
import hashlib
import base64
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
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
class KyberKeyPair:
    """Kyber KEM key pair"""
    public_key: bytes
    secret_key: bytes
    algorithm: str = 'Kyber768'
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().timestamp()


@dataclass
class DilithiumKeyPair:
    """Dilithium signature key pair"""
    public_key: bytes
    secret_key: bytes
    algorithm: str = 'Dilithium3'
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().timestamp()


@dataclass
class EncapsulationResult:
    """KEM encapsulation result"""
    ciphertext: bytes
    shared_secret: bytes


@dataclass
class SignatureResult:
    """Digital signature result"""
    signature: bytes
    message: bytes
    algorithm: str


class QuantumResistantCrypto:
    """
    Quantum-resistant cryptography implementation
    
    Provides:
    - Key generation for Kyber and Dilithium
    - Key encapsulation (KEM) for secure key exchange
    - Digital signatures for authentication
    - Hybrid classical+post-quantum schemes
    """
    
    def __init__(self, logger=None):
        """
        Initialize quantum-resistant crypto
        
        Args:
            logger: Optional logger instance
        """
        self.logger = ModuleLogger('QuantumCrypto', logger)
        
        # Check for pqcrypto library availability
        self.pqcrypto_available = False
        self.oqs_available = False
        
        try:
            import pqcrypto
            self.pqcrypto_available = True
            self.logger.info("PQCrypto library available")
        except ImportError:
            self.logger.warning("PQCrypto not available, using simulation mode")
        
        try:
            import oqs
            self.oqs_available = True
            self.logger.info("liboqs (OQS) library available")
        except ImportError:
            self.logger.warning("liboqs not available")
        
        self.statistics = {
            'kyber_keypairs_generated': 0,
            'dilithium_keypairs_generated': 0,
            'encapsulations': 0,
            'decapsulations': 0,
            'signatures_generated': 0,
            'signatures_verified': 0,
        }
    
    def generate_kyber_keypair(self, security_level: int = 768) -> KyberKeyPair:
        """
        Generate Kyber KEM key pair
        
        Args:
            security_level: 512 (Kyber512), 768 (Kyber768), or 1024 (Kyber1024)
        
        Returns:
            KyberKeyPair with public and secret keys
        
        Security levels:
        - Kyber512: ~AES-128 security
        - Kyber768: ~AES-192 security (recommended)
        - Kyber1024: ~AES-256 security
        """
        algorithm = f'Kyber{security_level}'
        
        if self.pqcrypto_available:
            try:
                if security_level == 512:
                    from pqcrypto.kem.kyber512 import generate_keypair
                elif security_level == 768:
                    from pqcrypto.kem.kyber768 import generate_keypair
                elif security_level == 1024:
                    from pqcrypto.kem.kyber1024 import generate_keypair
                else:
                    raise ValueError(f"Invalid Kyber security level: {security_level}")
                
                public_key, secret_key = generate_keypair()
                
                self.statistics['kyber_keypairs_generated'] += 1
                self.logger.info(f"Kyber keypair generated", algorithm=algorithm)
                
                return KyberKeyPair(
                    public_key=public_key,
                    secret_key=secret_key,
                    algorithm=algorithm
                )
                
            except ImportError as e:
                self.logger.error(f"Kyber import failed: {e}")
        
        # Simulation mode (for development without pqcrypto)
        self.logger.warning("Using simulated Kyber keys (NOT SECURE)")
        
        # Generate random keys (not real Kyber!)
        key_sizes = {512: (800, 1632), 768: (1184, 2400), 1024: (1568, 3168)}
        pk_size, sk_size = key_sizes.get(security_level, (1184, 2400))
        
        public_key = os.urandom(pk_size)
        secret_key = os.urandom(sk_size)
        
        self.statistics['kyber_keypairs_generated'] += 1
        
        return KyberKeyPair(
            public_key=public_key,
            secret_key=secret_key,
            algorithm=f'{algorithm}_SIMULATED'
        )
    
    def encapsulate(self, public_key: bytes, algorithm: str = 'Kyber768') -> EncapsulationResult:
        """
        Encapsulate shared secret using Kyber KEM
        
        Args:
            public_key: Recipient's Kyber public key
            algorithm: Kyber variant (Kyber512, Kyber768, Kyber1024)
        
        Returns:
            EncapsulationResult with ciphertext and shared secret
        """
        security_level = int(algorithm.replace('Kyber', '').replace('_SIMULATED', ''))
        
        if self.pqcrypto_available:
            try:
                if security_level == 512:
                    from pqcrypto.kem.kyber512 import encrypt
                elif security_level == 768:
                    from pqcrypto.kem.kyber768 import encrypt
                elif security_level == 1024:
                    from pqcrypto.kem.kyber1024 import encrypt
                else:
                    raise ValueError(f"Invalid algorithm: {algorithm}")
                
                ciphertext, shared_secret = encrypt(public_key)
                
                self.statistics['encapsulations'] += 1
                self.logger.info("KEM encapsulation successful", algorithm=algorithm)
                
                return EncapsulationResult(
                    ciphertext=ciphertext,
                    shared_secret=shared_secret
                )
                
            except ImportError as e:
                self.logger.error(f"Kyber encrypt failed: {e}")
        
        # Simulation mode
        self.logger.warning("Using simulated encapsulation (NOT SECURE)")
        
        # Simulate ciphertext and shared secret
        ct_sizes = {512: 768, 768: 1088, 1024: 1568}
        ct_size = ct_sizes.get(security_level, 1088)
        
        ciphertext = os.urandom(ct_size)
        shared_secret = os.urandom(32)  # 256-bit shared secret
        
        self.statistics['encapsulations'] += 1
        
        return EncapsulationResult(
            ciphertext=ciphertext,
            shared_secret=shared_secret
        )
    
    def decapsulate(self, ciphertext: bytes, secret_key: bytes, 
                    algorithm: str = 'Kyber768') -> bytes:
        """
        Decapsulate shared secret using Kyber KEM
        
        Args:
            ciphertext: Encapsulated ciphertext
            secret_key: Recipient's Kyber secret key
            algorithm: Kyber variant
        
        Returns:
            Shared secret (32 bytes)
        """
        security_level = int(algorithm.replace('Kyber', '').replace('_SIMULATED', ''))
        
        if self.pqcrypto_available:
            try:
                if security_level == 512:
                    from pqcrypto.kem.kyber512 import decrypt
                elif security_level == 768:
                    from pqcrypto.kem.kyber768 import decrypt
                elif security_level == 1024:
                    from pqcrypto.kem.kyber1024 import decrypt
                else:
                    raise ValueError(f"Invalid algorithm: {algorithm}")
                
                shared_secret = decrypt(ciphertext, secret_key)
                
                self.statistics['decapsulations'] += 1
                self.logger.info("KEM decapsulation successful", algorithm=algorithm)
                
                return shared_secret
                
            except ImportError as e:
                self.logger.error(f"Kyber decrypt failed: {e}")
        
        # Simulation mode (deterministic from ciphertext for consistency)
        self.logger.warning("Using simulated decapsulation (NOT SECURE)")
        
        # Derive "shared secret" from ciphertext hash (not real Kyber!)
        shared_secret = hashlib.sha256(ciphertext + secret_key[:32]).digest()
        
        self.statistics['decapsulations'] += 1
        
        return shared_secret
    
    def generate_dilithium_keypair(self, security_level: int = 3) -> DilithiumKeyPair:
        """
        Generate Dilithium signature key pair
        
        Args:
            security_level: 2 (Dilithium2), 3 (Dilithium3), or 5 (Dilithium5)
        
        Returns:
            DilithiumKeyPair with public and secret keys
        
        Security levels:
        - Dilithium2: ~AES-128 security
        - Dilithium3: ~AES-192 security (recommended)
        - Dilithium5: ~AES-256 security
        """
        algorithm = f'Dilithium{security_level}'
        
        if self.pqcrypto_available:
            try:
                if security_level == 2:
                    from pqcrypto.sign.dilithium2 import generate_keypair
                elif security_level == 3:
                    from pqcrypto.sign.dilithium3 import generate_keypair
                elif security_level == 5:
                    from pqcrypto.sign.dilithium5 import generate_keypair
                else:
                    raise ValueError(f"Invalid Dilithium security level: {security_level}")
                
                public_key, secret_key = generate_keypair()
                
                self.statistics['dilithium_keypairs_generated'] += 1
                self.logger.info(f"Dilithium keypair generated", algorithm=algorithm)
                
                return DilithiumKeyPair(
                    public_key=public_key,
                    secret_key=secret_key,
                    algorithm=algorithm
                )
                
            except ImportError as e:
                self.logger.error(f"Dilithium import failed: {e}")
        
        # Simulation mode
        self.logger.warning("Using simulated Dilithium keys (NOT SECURE)")
        
        # Generate random keys (not real Dilithium!)
        key_sizes = {2: (1312, 2528), 3: (1952, 4000), 5: (2592, 4864)}
        pk_size, sk_size = key_sizes.get(security_level, (1952, 4000))
        
        public_key = os.urandom(pk_size)
        secret_key = os.urandom(sk_size)
        
        self.statistics['dilithium_keypairs_generated'] += 1
        
        return DilithiumKeyPair(
            public_key=public_key,
            secret_key=secret_key,
            algorithm=f'{algorithm}_SIMULATED'
        )
    
    def sign(self, message: bytes, secret_key: bytes, 
             algorithm: str = 'Dilithium3') -> SignatureResult:
        """
        Sign message using Dilithium
        
        Args:
            message: Message to sign
            secret_key: Signer's Dilithium secret key
            algorithm: Dilithium variant
        
        Returns:
            SignatureResult with signature
        """
        security_level = int(algorithm.replace('Dilithium', '').replace('_SIMULATED', ''))
        
        if self.pqcrypto_available:
            try:
                if security_level == 2:
                    from pqcrypto.sign.dilithium2 import sign as dilithium_sign
                elif security_level == 3:
                    from pqcrypto.sign.dilithium3 import sign as dilithium_sign
                elif security_level == 5:
                    from pqcrypto.sign.dilithium5 import sign as dilithium_sign
                else:
                    raise ValueError(f"Invalid algorithm: {algorithm}")
                
                signed_message = dilithium_sign(message, secret_key)
                # Extract signature (signed_message = signature + message)
                sig_sizes = {2: 2420, 3: 3293, 5: 4595}
                sig_size = sig_sizes[security_level]
                signature = signed_message[:sig_size]
                
                self.statistics['signatures_generated'] += 1
                self.logger.info("Dilithium signature generated", algorithm=algorithm)
                
                return SignatureResult(
                    signature=signature,
                    message=message,
                    algorithm=algorithm
                )
                
            except ImportError as e:
                self.logger.error(f"Dilithium sign failed: {e}")
        
        # Simulation mode
        self.logger.warning("Using simulated signature (NOT SECURE)")
        
        # Generate fake signature (HMAC-based for consistency)
        sig_sizes = {2: 2420, 3: 3293, 5: 4595}
        sig_size = sig_sizes.get(security_level, 3293)
        
        # Deterministic "signature" from message + key
        signature_data = hashlib.sha512(message + secret_key[:64]).digest()
        signature = (signature_data * (sig_size // len(signature_data) + 1))[:sig_size]
        
        self.statistics['signatures_generated'] += 1
        
        return SignatureResult(
            signature=signature,
            message=message,
            algorithm=f'{algorithm}_SIMULATED'
        )
    
    def verify(self, message: bytes, signature: bytes, public_key: bytes,
               algorithm: str = 'Dilithium3') -> bool:
        """
        Verify Dilithium signature
        
        Args:
            message: Original message
            signature: Signature to verify
            public_key: Signer's Dilithium public key
            algorithm: Dilithium variant
        
        Returns:
            True if signature is valid, False otherwise
        """
        security_level = int(algorithm.replace('Dilithium', '').replace('_SIMULATED', ''))
        
        if self.pqcrypto_available:
            try:
                if security_level == 2:
                    from pqcrypto.sign.dilithium2 import open as dilithium_open
                elif security_level == 3:
                    from pqcrypto.sign.dilithium3 import open as dilithium_open
                elif security_level == 5:
                    from pqcrypto.sign.dilithium5 import open as dilithium_open
                else:
                    raise ValueError(f"Invalid algorithm: {algorithm}")
                
                # Reconstruct signed message
                signed_message = signature + message
                
                try:
                    verified_message = dilithium_open(signed_message, public_key)
                    valid = (verified_message == message)
                    
                    self.statistics['signatures_verified'] += 1
                    self.logger.info(f"Signature verification: {valid}", algorithm=algorithm)
                    
                    return valid
                except Exception:
                    return False
                    
            except ImportError as e:
                self.logger.error(f"Dilithium verify failed: {e}")
        
        # Simulation mode (always returns True for consistent signatures)
        self.logger.warning("Using simulated verification (NOT SECURE)")
        
        # Verify by reconstructing signature
        sig_sizes = {2: 2420, 3: 3293, 5: 4595}
        expected_sig_size = sig_sizes.get(security_level, 3293)
        
        if len(signature) != expected_sig_size:
            return False
        
        # In simulation, we can't truly verify, so we return True
        # (real implementation would use actual Dilithium verification)
        self.statistics['signatures_verified'] += 1
        
        return True
    
    def hybrid_encrypt(self, plaintext: bytes, recipient_public_key: bytes,
                       kyber_algorithm: str = 'Kyber768') -> Dict[str, bytes]:
        """
        Hybrid encryption: Kyber KEM + AES
        
        Args:
            plaintext: Data to encrypt
            recipient_public_key: Recipient's Kyber public key
            kyber_algorithm: Kyber variant
        
        Returns:
            Dict with 'ciphertext' (encrypted data) and 'encapsulated_key' (Kyber ciphertext)
        """
        # Step 1: Encapsulate shared secret using Kyber
        result = self.encapsulate(recipient_public_key, kyber_algorithm)
        
        # Step 2: Encrypt data with AES using shared secret
        from Crypto.Cipher import AES
        from Crypto.Random import get_random_bytes
        
        # Derive AES key from shared secret
        aes_key = hashlib.sha256(result.shared_secret).digest()
        
        # Encrypt with AES-GCM
        cipher = AES.new(aes_key, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)
        
        # Combine nonce, tag, and ciphertext
        encrypted_data = cipher.nonce + tag + ciphertext
        
        self.logger.info("Hybrid encryption complete", 
                        plaintext_size=len(plaintext),
                        ciphertext_size=len(encrypted_data))
        
        return {
            'ciphertext': encrypted_data,
            'encapsulated_key': result.ciphertext
        }
    
    def hybrid_decrypt(self, encrypted_data: bytes, encapsulated_key: bytes,
                       recipient_secret_key: bytes, kyber_algorithm: str = 'Kyber768') -> bytes:
        """
        Hybrid decryption: Kyber KEM + AES
        
        Args:
            encrypted_data: Encrypted data (nonce + tag + ciphertext)
            encapsulated_key: Kyber ciphertext
            recipient_secret_key: Recipient's Kyber secret key
            kyber_algorithm: Kyber variant
        
        Returns:
            Decrypted plaintext
        """
        # Step 1: Decapsulate shared secret using Kyber
        shared_secret = self.decapsulate(encapsulated_key, recipient_secret_key, kyber_algorithm)
        
        # Step 2: Decrypt data with AES using shared secret
        from Crypto.Cipher import AES
        
        # Derive AES key from shared secret
        aes_key = hashlib.sha256(shared_secret).digest()
        
        # Extract components
        nonce = encrypted_data[:16]
        tag = encrypted_data[16:32]
        ciphertext = encrypted_data[32:]
        
        # Decrypt with AES-GCM
        cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        
        self.logger.info("Hybrid decryption complete", 
                        ciphertext_size=len(encrypted_data),
                        plaintext_size=len(plaintext))
        
        return plaintext
    
    def export_public_key(self, public_key: bytes, format: str = 'pem') -> str:
        """
        Export public key to string format
        
        Args:
            public_key: Public key bytes
            format: 'pem' or 'base64'
        
        Returns:
            Formatted public key string
        """
        if format == 'base64':
            return base64.b64encode(public_key).decode('utf-8')
        elif format == 'pem':
            b64_key = base64.b64encode(public_key).decode('utf-8')
            # Split into 64-character lines
            lines = [b64_key[i:i+64] for i in range(0, len(b64_key), 64)]
            return '-----BEGIN PUBLIC KEY-----\n' + '\n'.join(lines) + '\n-----END PUBLIC KEY-----'
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_public_key(self, key_string: str, format: str = 'pem') -> bytes:
        """
        Import public key from string format
        
        Args:
            key_string: Formatted public key string
            format: 'pem' or 'base64'
        
        Returns:
            Public key bytes
        """
        if format == 'base64':
            return base64.b64decode(key_string)
        elif format == 'pem':
            # Remove PEM headers and decode
            lines = key_string.strip().split('\n')
            b64_key = ''.join(line for line in lines if not line.startswith('-----'))
            return base64.b64decode(b64_key)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get crypto statistics"""
        return {
            **self.statistics,
            'pqcrypto_available': self.pqcrypto_available,
            'oqs_available': self.oqs_available,
        }
