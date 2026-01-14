"""Gated Cross-Attention layer for Flamingo-style personalization."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class GatedCrossAttentionLayer(nn.Module):
    """
    Gated cross-attention layer for injecting user context into transformer layers.

    Implements Flamingo-style gated cross-attention:
        output = x + tanh(gate) * cross_attention(x, memory)

    The gate is initialized to a negative value so tanh(gate) ≈ 0,
    allowing the model to start from the base LLM behavior and
    gradually learn to incorporate user context.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_init: float = -5.0,
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states
            num_heads: Number of attention heads
            dropout: Dropout probability
            gate_init: Initial value for learnable gate (negative = start near zero)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Cross-attention components
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer norm before cross-attention
        self.norm = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Learnable gate - initialized so tanh(gate) ≈ 0
        self.gate = nn.Parameter(torch.tensor(gate_init))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values for stable training."""
        # Xavier initialization for projections
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1 / math.sqrt(2))
            nn.init.zeros_(module.bias)

        # Output projection initialized smaller
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply gated cross-attention.

        Args:
            hidden_states: (batch, seq_len, hidden_dim) - input from transformer layer
            memory: (batch, memory_len, hidden_dim) - user context embeddings
            attention_mask: Optional mask for hidden states (not typically used)
            memory_mask: Optional mask for memory (1 = attend, 0 = ignore)

        Returns:
            Updated hidden states with cross-attention applied
        """
        batch_size, seq_len, _ = hidden_states.shape
        memory_len = memory.shape[1]

        # Pre-norm
        normed = self.norm(hidden_states)

        # Project queries from hidden states, keys/values from memory
        q = self.q_proj(normed)  # (batch, seq_len, hidden_dim)
        k = self.k_proj(memory)  # (batch, memory_len, hidden_dim)
        v = self.v_proj(memory)  # (batch, memory_len, hidden_dim)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, memory_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, memory_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch, num_heads, seq_len/memory_len, head_dim)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        # Shape: (batch, num_heads, seq_len, memory_len)

        # Apply memory mask if provided
        if memory_mask is not None:
            # memory_mask: (batch, memory_len) -> (batch, 1, 1, memory_len)
            memory_mask = memory_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(memory_mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # Shape: (batch, num_heads, seq_len, head_dim)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        # Output projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # Gated residual connection
        # tanh(gate) starts near 0, allowing gradual incorporation of cross-attention
        gate_value = torch.tanh(self.gate)
        output = hidden_states + gate_value * attn_output

        return output

    @property
    def gate_value(self) -> float:
        """Return current gate value (after tanh)."""
        with torch.no_grad():
            return torch.tanh(self.gate).item()

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads}, gate={self.gate_value:.4f}"


class CrossAttentionBlock(nn.Module):
    """
    Full cross-attention block with feed-forward network.

    Architecture:
        x -> LayerNorm -> CrossAttention -> x + gate * attn_out
          -> LayerNorm -> FFN -> x + ffn_out
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        gate_init: float = -5.0,
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states
            num_heads: Number of attention heads
            ffn_dim: Feed-forward dimension (default: 4 * hidden_dim)
            dropout: Dropout probability
            gate_init: Initial gate value
        """
        super().__init__()

        ffn_dim = ffn_dim or 4 * hidden_dim

        # Cross-attention
        self.cross_attention = GatedCrossAttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            gate_init=gate_init,
        )

        # Feed-forward network
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # FFN gate
        self.ffn_gate = nn.Parameter(torch.tensor(gate_init))

        # Initialize FFN
        self._init_ffn()

    def _init_ffn(self) -> None:
        """Initialize FFN with small output."""
        # Last layer initialized smaller
        nn.init.xavier_uniform_(self.ffn[-2].weight, gain=0.1)
        nn.init.zeros_(self.ffn[-2].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply cross-attention block.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            memory: (batch, memory_len, hidden_dim)
            memory_mask: Optional mask for memory

        Returns:
            Updated hidden states
        """
        # Cross-attention (already gated internally)
        hidden_states = self.cross_attention(
            hidden_states, memory, memory_mask=memory_mask
        )

        # Gated FFN
        ffn_out = self.ffn(self.ffn_norm(hidden_states))
        gate_value = torch.tanh(self.ffn_gate)
        hidden_states = hidden_states + gate_value * ffn_out

        return hidden_states
