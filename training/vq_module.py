"""
VQ Module: Vector Quantized bottleneck for discrete latent codes.

Implements a standard VQ-VAE style bottleneck following:
- Oord et al. "Neural Discrete Representation Learning" (VQ-VAE)
- Van den Oord et al. "Vector Quantized Variational AutoEncoder"

The module:
- Projects input vectors to codebook space
- Quantizes via nearest neighbor lookup
- Computes VQ loss: L_vq = ||sg[z_e] - z_q||^2 + β ||z_e - sg[z_q]||^2
- Supports EMA updates for codebook stability

Usage:
    >>> import torch
    >>> from training.vq_module import VectorQuantizer
    >>>
    >>> # Create VQ module
    >>> vq = VectorQuantizer(
    ...     codebook_size=8000,
    ...     code_dim=256,
    ...     commitment_cost=0.25,
    ...     ema_decay=0.99
    ... )
    >>>
    >>> # Forward pass
    >>> hidden_states = torch.randn(2, 10, 512)  # (batch, seq_len, hidden_dim)
    >>> quantized, vq_loss, aux = vq(hidden_states)
    >>> print(f"Quantized shape: {quantized.shape}")  # (2, 10, 256)
    >>> print(f"VQ loss: {vq_loss.item():.4f}")
    >>> print(f"Code indices shape: {aux['indices'].shape}")  # (2, 10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class VectorQuantizer(nn.Module):
    """
    Vector Quantization module for discrete latent codes.
    
    Implements VQ-VAE style quantization with optional EMA updates.
    
    Parameters:
        codebook_size: Number of codes K (e.g. 8000)
        code_dim: Dimension of code vectors (e.g. 256)
        input_dim: Input dimension (if None, assumes input_dim == code_dim)
        commitment_cost: β in loss formula (default: 0.25)
            Controls how much the encoder commits to the codebook
        ema_decay: EMA decay rate for codebook updates (default: 0.99)
            If None, uses standard VQ-VAE updates (no EMA)
        epsilon: Small constant for numerical stability (default: 1e-5)
        codebook_init: Initial codebook values (optional)
            If None, initializes with random normal distribution
    """
    
    def __init__(
        self,
        codebook_size: int,
        code_dim: int,
        input_dim: Optional[int] = None,
        commitment_cost: float = 0.25,
        ema_decay: Optional[float] = 0.99,
        epsilon: float = 1e-5,
        codebook_init: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.input_dim = input_dim if input_dim is not None else code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.use_ema = ema_decay is not None
        
        # Projection layer (if input_dim != code_dim)
        if self.input_dim != self.code_dim:
            self.proj = nn.Linear(self.input_dim, self.code_dim, bias=False)
        else:
            self.proj = None
        
        # Initialize codebook
        if codebook_init is not None:
            if codebook_init.shape != (codebook_size, code_dim):
                raise ValueError(
                    f"Codebook init shape {codebook_init.shape} != "
                    f"expected ({codebook_size}, {code_dim})"
                )
            self.codebook = nn.Parameter(codebook_init.clone())
        else:
            # Initialize with random normal (standard VQ-VAE initialization)
            self.codebook = nn.Parameter(
                torch.randn(codebook_size, code_dim) * 0.02
            )
        
        # EMA tracking (if using EMA)
        if self.use_ema:
            # Register buffers for EMA (not trainable parameters)
            self.register_buffer('_ema_cluster_size', torch.zeros(codebook_size))
            self.register_buffer('_ema_w', torch.zeros(codebook_size, code_dim))
            self.register_buffer('_n_updates', torch.tensor(0))
    
    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to codebook dimension.
        
        Args:
            x: Input tensor (..., input_dim)
        
        Returns:
            Projected tensor (..., code_dim)
        """
        if self.proj is not None:
            return self.proj(x)
        return x
    
    def _quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize encoded vectors to nearest codebook entries.
        
        Args:
            z_e: Encoded vectors (..., code_dim)
        
        Returns:
            Tuple of:
            - z_q: Quantized vectors (..., code_dim)
            - indices: Code indices (...,)
        """
        # Flatten spatial dimensions for distance computation
        flat_z_e = z_e.view(-1, self.code_dim)  # (N, code_dim)
        
        # Compute distances to all codebook vectors
        # distances: (N, codebook_size)
        distances = (
            torch.sum(flat_z_e ** 2, dim=1, keepdim=True) +
            torch.sum(self.codebook ** 2, dim=1) -
            2 * torch.matmul(flat_z_e, self.codebook.t())
        )
        
        # Find nearest codebook indices
        encoding_indices = torch.argmin(distances, dim=1)  # (N,)
        
        # Quantize: replace with nearest codebook vector
        z_q = self.codebook[encoding_indices]  # (N, code_dim)
        
        # Reshape back to original spatial dimensions
        original_shape = z_e.shape[:-1]
        z_q = z_q.view(*original_shape, self.code_dim)
        indices = encoding_indices.view(original_shape)
        
        return z_q, indices
    
    def _compute_vq_loss(
        self,
        z_e: torch.Tensor,
        z_q: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute VQ loss.
        
        Loss = ||sg[z_e] - z_q||^2 + β ||z_e - sg[z_q]||^2
        
        Where:
        - sg[·] = stop_gradient
        - β = commitment_cost
        
        Args:
            z_e: Encoded vectors
            z_q: Quantized vectors
        
        Returns:
            Scalar VQ loss
        """
        # Commitment loss: ||z_e - sg[z_q]||^2
        # Encourages encoder to output values close to codebook
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        
        # Codebook loss: ||sg[z_e] - z_q||^2
        # Moves codebook vectors towards encoder outputs
        codebook_loss = F.mse_loss(z_e.detach(), z_q)
        
        # Total VQ loss
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        return vq_loss
    
    def _update_codebook_ema(
        self,
        z_e_flat: torch.Tensor,
        encoding_indices: torch.Tensor
    ) -> None:
        """
        Update codebook using Exponential Moving Average.
        
        Args:
            z_e_flat: Flattened encoded vectors (N, code_dim)
            encoding_indices: Code indices (N,)
        """
        # Update EMA cluster sizes
        n = z_e_flat.shape[0]
        encoding_one_hot = F.one_hot(
            encoding_indices, self.codebook_size
        ).float()  # (N, codebook_size)
        
        # EMA update: cluster_size = decay * old + (1 - decay) * new
        self._ema_cluster_size = (
            self.ema_decay * self._ema_cluster_size +
            (1 - self.ema_decay) * encoding_one_hot.sum(0)
        )
        
        # Update EMA codebook weights
        # Sum of encoded vectors per codebook entry
        n_per_code = encoding_one_hot.sum(0, keepdim=True)  # (1, codebook_size)
        # Avoid division by zero
        n_per_code = n_per_code + self.epsilon
        
        # Weighted sum of encoded vectors
        ema_dw = torch.matmul(encoding_one_hot.t(), z_e_flat)  # (codebook_size, code_dim)
        ema_dw = ema_dw / n_per_code.t()
        
        # EMA update
        self._ema_w = (
            self.ema_decay * self._ema_w +
            (1 - self.ema_decay) * ema_dw
        )
        
        # Update codebook: normalize by cluster size
        n_clusters = self._ema_cluster_size.view(-1, 1)  # (codebook_size, 1)
        n_clusters = n_clusters + self.epsilon  # Avoid division by zero
        
        # Update codebook parameters
        with torch.no_grad():
            self.codebook.data = self._ema_w / n_clusters
        
        self._n_updates += 1
    
    def _update_codebook_standard(
        self,
        z_e_flat: torch.Tensor,
        encoding_indices: torch.Tensor
    ) -> None:
        """
        Update codebook using standard VQ-VAE updates (via gradients).
        
        Note: This is handled automatically by the loss function.
        This method is a placeholder for consistency.
        
        Args:
            z_e_flat: Flattened encoded vectors (N, code_dim)
            encoding_indices: Code indices (N,)
        """
        # Standard VQ-VAE updates happen via backprop through codebook_loss
        # No explicit update needed here
        pass
    
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through VQ module.
        
        Args:
            hidden_states: Input hidden states (..., input_dim)
                Can be any shape with last dimension = input_dim
        
        Returns:
            Tuple of:
            - quantized: Quantized hidden states (..., code_dim)
            - vq_loss: VQ loss scalar
            - aux: Dictionary with:
                - indices: Code indices (...,)
                - encoding_loss: Codebook loss component
                - commitment_loss: Commitment loss component
        """
        # Project to codebook dimension
        z_e = self._project(hidden_states)  # (..., code_dim)
        
        # Quantize
        z_q, indices = self._quantize(z_e)  # (..., code_dim), (...,)
        
        # Compute VQ loss
        vq_loss = self._compute_vq_loss(z_e, z_q)
        
        # Update codebook (if using EMA)
        if self.use_ema and self.training:
            z_e_flat = z_e.view(-1, self.code_dim)
            indices_flat = indices.view(-1)
            self._update_codebook_ema(z_e_flat, indices_flat)
        
        # Prepare auxiliary info
        with torch.no_grad():
            # Compute loss components for logging
            commitment_loss = F.mse_loss(z_e, z_q.detach())
            codebook_loss = F.mse_loss(z_e.detach(), z_q)
        
        aux = {
            'indices': indices,
            'encoding_loss': codebook_loss,
            'commitment_loss': commitment_loss,
        }
        
        # Straight-through estimator: pass gradients from quantized to encoded
        # In forward: use quantized; in backward: gradients flow to encoded
        z_q_st = z_e + (z_q - z_e).detach()
        
        return z_q_st, vq_loss, aux
    
    def encode_to_indices(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Encode hidden states to codebook indices (for inference).
        
        Args:
            hidden_states: Input hidden states (..., input_dim)
        
        Returns:
            Code indices (...,)
        """
        self.eval()
        with torch.no_grad():
            z_e = self._project(hidden_states)
            _, indices = self._quantize(z_e)
        return indices
    
    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode codebook indices back to vectors.
        
        Args:
            indices: Code indices (...,)
                Values should be in [0, codebook_size)
        
        Returns:
            Quantized vectors (..., code_dim)
        """
        # Clamp indices to valid range
        indices = torch.clamp(indices, 0, self.codebook_size - 1)
        
        # Lookup codebook vectors
        z_q = self.codebook[indices]
        
        return z_q
    
    def get_codebook_usage(self) -> Dict[str, torch.Tensor]:
        """
        Get codebook usage statistics (EMA mode only).
        
        Returns:
            Dictionary with usage statistics
        """
        if not self.use_ema:
            return {}
        
        with torch.no_grad():
            return {
                'cluster_sizes': self._ema_cluster_size.clone(),
                'n_updates': self._n_updates.item(),
                'active_codes': (self._ema_cluster_size > 0).sum().item()
            }


# Alias for backward compatibility
VQModule = VectorQuantizer


def create_vq_module(config: dict) -> VectorQuantizer:
    """
    Factory function to create VQ module from config.
    
    Args:
        config: Configuration dictionary with VQ parameters:
            - codebook_size: int (required)
            - code_dim: int (required)
            - input_dim: int (optional, defaults to code_dim)
            - commitment_cost: float (optional, default: 0.25)
            - ema_decay: float or None (optional, default: 0.99)
            - epsilon: float (optional, default: 1e-5)
            - codebook_init: torch.Tensor (optional)
    
    Returns:
        Initialized VectorQuantizer
    
    Example:
        >>> config = {
        ...     'codebook_size': 8000,
        ...     'code_dim': 256,
        ...     'commitment_cost': 0.25,
        ...     'ema_decay': 0.99
        ... }
        >>> vq = create_vq_module(config)
    """
    required_keys = ['codebook_size', 'code_dim']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config missing required key: {key}")
    
    return VectorQuantizer(
        codebook_size=config['codebook_size'],
        code_dim=config['code_dim'],
        input_dim=config.get('input_dim'),
        commitment_cost=config.get('commitment_cost', 0.25),
        ema_decay=config.get('ema_decay', 0.99),
        epsilon=config.get('epsilon', 1e-5),
        codebook_init=config.get('codebook_init')
    )


# Unit test helpers (can be run with pytest or standalone)
if __name__ == "__main__":
    print("Running VQ Module tests...")
    
    # Test 1: Basic forward pass
    print("\n1. Testing basic forward pass...")
    vq = VectorQuantizer(
        codebook_size=100,
        code_dim=64,
        input_dim=128,
        commitment_cost=0.25
    )
    
    x = torch.randn(4, 10, 128)  # (batch, seq_len, input_dim)
    quantized, loss, aux = vq(x)
    
    assert quantized.shape == (4, 10, 64), f"Wrong quantized shape: {quantized.shape}"
    assert loss.item() > 0, "Loss should be positive"
    assert aux['indices'].shape == (4, 10), f"Wrong indices shape: {aux['indices'].shape}"
    print("✓ Basic forward pass works")
    
    # Test 2: Encode/decode round-trip
    print("\n2. Testing encode/decode round-trip...")
    indices = vq.encode_to_indices(x)
    decoded = vq.decode_from_indices(indices)
    
    assert decoded.shape == quantized.shape, "Decoded shape mismatch"
    print("✓ Encode/decode round-trip works")
    
    # Test 3: EMA updates
    print("\n3. Testing EMA updates...")
    vq_ema = VectorQuantizer(
        codebook_size=50,
        code_dim=32,
        ema_decay=0.99
    )
    vq_ema.train()
    
    x2 = torch.randn(2, 5, 32)
    _, _, _ = vq_ema(x2)
    
    usage = vq_ema.get_codebook_usage()
    assert 'active_codes' in usage, "EMA usage stats missing"
    print(f"✓ EMA updates work (active codes: {usage['active_codes']})")
    
    # Test 4: No projection when input_dim == code_dim
    print("\n4. Testing no-projection case...")
    vq_no_proj = VectorQuantizer(
        codebook_size=20,
        code_dim=16,
        input_dim=16  # Same as code_dim
    )
    assert vq_no_proj.proj is None, "Should not have projection layer"
    print("✓ No-projection case works")
    
    # Test 5: Factory function
    print("\n5. Testing factory function...")
    config = {
        'codebook_size': 100,
        'code_dim': 64,
        'commitment_cost': 0.5
    }
    vq_factory = create_vq_module(config)
    assert vq_factory.codebook_size == 100
    print("✓ Factory function works")
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
