"""
FalconOne GAN Payload Generator
Polymorphic payload generation using Generative Adversarial Networks
"""

from typing import Dict, List, Any, Optional
import logging
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..utils.logger import ModuleLogger


class PayloadGenerator:
    """GAN-based polymorphic payload generation"""
    
    def __init__(self, config, logger: logging.Logger):
        """Initialize payload generator"""
        self.config = config
        self.logger = ModuleLogger('PayloadGenerator', logger)
        
        self.payload_size = config.get('ai.gan.payload_size', 256)
        self.latent_dim = config.get('ai.gan.latent_dim', 100)
        
        self.generator = None
        self.discriminator = None
        self.gan = None
        
        if TF_AVAILABLE:
            self._build_gan()
        else:
            self.logger.warning("TensorFlow not available - GAN disabled")
        
        self.logger.info("Payload generator initialized")
    
    def _build_gan(self):
        """Build Generator and Discriminator networks"""
        try:
            # Generator: latent vector -> payload bytes
            self.generator = keras.Sequential([
                keras.layers.Dense(128, input_dim=self.latent_dim),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.BatchNormalization(),
                
                keras.layers.Dense(256),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.BatchNormalization(),
                
                keras.layers.Dense(512),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.BatchNormalization(),
                
                keras.layers.Dense(self.payload_size, activation='tanh')
            ])
            
            # Discriminator: payload bytes -> real/fake
            self.discriminator = keras.Sequential([
                keras.layers.Dense(512, input_dim=self.payload_size),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Dropout(0.3),
                
                keras.layers.Dense(256),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Dropout(0.3),
                
                keras.layers.Dense(128),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Dropout(0.3),
                
                keras.layers.Dense(1, activation='sigmoid')
            ])
            
            self.discriminator.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Combined GAN model
            self.discriminator.trainable = False
            gan_input = keras.layers.Input(shape=(self.latent_dim,))
            generated_payload = self.generator(gan_input)
            gan_output = self.discriminator(generated_payload)
            
            self.gan = keras.Model(gan_input, gan_output)
            self.gan.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                loss='binary_crossentropy'
            )
            
            self.logger.info("GAN model built successfully")
            
        except Exception as e:
            self.logger.error(f"GAN build failed: {e}")
    
    def generate_payload(self, num_payloads: int = 1) -> List[bytes]:
        """
        Generate polymorphic payloads
        
        Args:
            num_payloads: Number of payloads to generate
            
        Returns:
            List of payload bytes
        """
        if not TF_AVAILABLE or not self.generator:
            self.logger.warning("GAN not available - generating random payloads")
            return [np.random.bytes(self.payload_size) for _ in range(num_payloads)]
        
        try:
            # Sample from latent space
            latent_samples = np.random.normal(0, 1, size=(num_payloads, self.latent_dim))
            
            # Generate payloads
            generated = self.generator.predict(latent_samples, verbose=0)
            
            # Convert from [-1, 1] to [0, 255] bytes
            payloads = []
            for gen_payload in generated:
                payload_bytes = ((gen_payload + 1) * 127.5).astype(np.uint8).tobytes()
                payloads.append(payload_bytes)
            
            self.logger.debug(f"Generated {num_payloads} polymorphic payloads")
            
            return payloads
            
        except Exception as e:
            self.logger.error(f"Payload generation failed: {e}")
            return []
    
    def train_gan(self, real_payloads: np.ndarray, epochs: int = 1000, batch_size: int = 32):
        """
        Train GAN on real payload examples
        
        Args:
            real_payloads: Real payload samples (NxPayloadSize)
            epochs: Number of training epochs
            batch_size: Batch size
        """
        if not TF_AVAILABLE or not self.gan:
            self.logger.error("GAN not available")
            return
        
        try:
            self.logger.info(f"Training GAN for {epochs} epochs...")
            
            # Normalize to [-1, 1]
            real_payloads = (real_payloads.astype(np.float32) - 127.5) / 127.5
            
            num_samples = real_payloads.shape[0]
            
            for epoch in range(epochs):
                # Train discriminator
                idx = np.random.randint(0, num_samples, batch_size)
                real_batch = real_payloads[idx]
                
                # Generate fake payloads
                latent = np.random.normal(0, 1, size=(batch_size, self.latent_dim))
                fake_batch = self.generator.predict(latent, verbose=0)
                
                # Labels
                real_labels = np.ones((batch_size, 1))
                fake_labels = np.zeros((batch_size, 1))
                
                # Train discriminator
                d_loss_real = self.discriminator.train_on_batch(real_batch, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(fake_batch, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # Train generator
                latent = np.random.normal(0, 1, size=(batch_size, self.latent_dim))
                misleading_labels = np.ones((batch_size, 1))
                
                g_loss = self.gan.train_on_batch(latent, misleading_labels)
                
                # Log progress
                if epoch % 100 == 0:
                    self.logger.debug(f"Epoch {epoch}/{epochs} - D loss: {d_loss[0]:.4f}, G loss: {g_loss:.4f}")
            
            self.logger.info("GAN training completed")
            
        except Exception as e:
            self.logger.error(f"GAN training failed: {e}")
    
    def save_models(self, generator_path: str, discriminator_path: str):
        """Save trained models"""
        if self.generator:
            self.generator.save(generator_path)
            self.logger.info(f"Generator saved to {generator_path}")
        
        if self.discriminator:
            self.discriminator.save(discriminator_path)
            self.logger.info(f"Discriminator saved to {discriminator_path}")
    
    def load_models(self, generator_path: str, discriminator_path: str):
        """Load trained models"""
        if TF_AVAILABLE:
            self.generator = keras.models.load_model(generator_path)
            self.discriminator = keras.models.load_model(discriminator_path)
            self.logger.info("Models loaded successfully")
    
    def morph_payload(self, original_payload: bytes, mutation_rate: float = 0.1) -> bytes:
        """
        Morph existing payload to evade detection
        
        Args:
            original_payload: Original payload bytes
            mutation_rate: Fraction of bytes to mutate
            
        Returns:
            Morphed payload
        """
        try:
            payload_array = np.frombuffer(original_payload, dtype=np.uint8).copy()
            
            # Randomly mutate some bytes
            num_mutations = int(len(payload_array) * mutation_rate)
            mutation_indices = np.random.choice(len(payload_array), num_mutations, replace=False)
            
            for idx in mutation_indices:
                # XOR with random byte
                payload_array[idx] ^= np.random.randint(0, 256)
            
            return payload_array.tobytes()
            
        except Exception as e:
            self.logger.error(f"Payload morphing failed: {e}")
            return original_payload
    
    def generate(self, payload_type: str, config: Dict[str, Any] = None) -> np.ndarray:
        """
        Generate payload for specific attack type.
        
        This method provides a unified interface for generating payloads
        across different attack categories (ISAC, NTN, O-RAN, etc.).
        
        Args:
            payload_type: Type of payload to generate
                - 'isac_waveform_poison': ISAC waveform manipulation payload
                - 'isac_ai_poison': AI model poisoning samples
                - 'ntn_beam_hijack': NTN beam hijacking payload
                - 'oran_xapp_exploit': O-RAN xApp exploitation payload
                - 'quantum_attack': Quantum attack sequence
            config: Configuration dict with:
                - evasion_level: 0-1 (how evasive the payload should be)
                - target_system: Target system identifier
                - polymorphic: bool (use GAN for polymorphism)
                - payload_size: int (override default size)
                
        Returns:
            numpy array containing generated payload samples
        """
        config = config or {}
        evasion_level = config.get('evasion_level', 0.5)
        target_size = config.get('payload_size', self.payload_size)
        polymorphic = config.get('polymorphic', True)
        
        # Generate base payload using GAN if available
        if polymorphic and TF_AVAILABLE and self.generator:
            try:
                # Generate multiple candidates, select best for evasion
                num_candidates = max(1, int(10 * evasion_level))
                latent_samples = np.random.normal(0, 1, size=(num_candidates, self.latent_dim))
                
                # Add structured noise based on payload type
                if 'waveform' in payload_type:
                    # Add frequency-domain structure for waveform payloads
                    freq_component = np.sin(np.linspace(0, 4*np.pi, self.latent_dim))
                    latent_samples += freq_component * 0.3 * evasion_level
                elif 'ai_poison' in payload_type:
                    # Add adversarial perturbation structure
                    latent_samples += np.random.laplace(0, 0.1, latent_samples.shape)
                
                candidates = self.generator.predict(latent_samples, verbose=0)
                
                # Select candidate with highest variance (more evasive)
                variances = np.var(candidates, axis=1)
                best_idx = np.argmax(variances)
                payload = candidates[best_idx]
                
                # Scale to target size if needed
                if len(payload) != target_size:
                    payload = np.interp(
                        np.linspace(0, 1, target_size),
                        np.linspace(0, 1, len(payload)),
                        payload
                    )
                
                self.logger.debug(f"Generated {payload_type} payload: size={target_size}, evasion={evasion_level}")
                return payload.astype(np.float32)
                
            except Exception as e:
                self.logger.warning(f"GAN generation failed, using fallback: {e}")
        
        # Fallback: Generate structured random payload
        if 'waveform' in payload_type:
            # Generate chirp-like waveform for ISAC attacks
            t = np.linspace(0, 1, target_size)
            payload = np.sin(2 * np.pi * (10 + 50 * t) * t) * evasion_level
            payload += np.random.normal(0, 0.1, target_size)
        elif 'ai_poison' in payload_type:
            # Generate adversarial perturbation
            payload = np.random.laplace(0, 0.05 * evasion_level, target_size)
        elif 'quantum' in payload_type:
            # Generate quantum-like superposition states
            payload = np.random.choice([-1, 1], target_size) * np.random.uniform(0, 1, target_size)
        else:
            # Default: Random uniform noise
            payload = np.random.uniform(-1, 1, target_size) * evasion_level
        
        self.logger.debug(f"Generated fallback {payload_type} payload: size={target_size}")
        return payload.astype(np.float32)
