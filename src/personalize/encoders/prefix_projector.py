"""Prefix projector modules for mapping user embeddings to soft prompts."""

from typing import Optional

import torch
import torch.nn as nn


class PrefixProjector(nn.Module):
    """
    Projects user embeddings to soft prefix tokens for LLM injection.

    Maps a dense user embedding (e.g., 384-dim from sentence transformer)
    to one or more soft tokens in the LLM's embedding space.

    Based on E2P (Embedding-to-Prefix) approach from the paper.
    """

    def __init__(
        self,
        user_embedding_dim: int,
        llm_embedding_dim: int,
        num_prefix_tokens: int = 1,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize prefix projector.

        Args:
            user_embedding_dim: Dimension of input user embeddings
            llm_embedding_dim: Dimension of LLM's token embeddings
            num_prefix_tokens: Number of soft prefix tokens to generate
            hidden_dim: Hidden layer dimension (default: mean of in/out dims)
            num_layers: Number of MLP layers (minimum 2 recommended)
            dropout: Dropout probability
            activation: Activation function ("gelu", "relu", "silu")
        """
        super().__init__()

        self.user_embedding_dim = user_embedding_dim
        self.llm_embedding_dim = llm_embedding_dim
        self.num_prefix_tokens = num_prefix_tokens

        # Output dimension is num_tokens * embedding_dim
        output_dim = num_prefix_tokens * llm_embedding_dim
        hidden_dim = hidden_dim or (user_embedding_dim + output_dim) // 2

        # Activation function
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }
        act_fn = activations.get(activation, nn.GELU())

        # Build MLP
        layers = []
        in_dim = user_embedding_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                act_fn,
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize final layer with small weights for stable training
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, user_embedding: torch.Tensor) -> torch.Tensor:
        """
        Project user embedding to soft prefix tokens.

        Args:
            user_embedding: User embedding of shape (batch, user_embedding_dim)
                           or (user_embedding_dim,) for single sample

        Returns:
            Soft prefix tokens of shape (batch, num_prefix_tokens, llm_embedding_dim)
            or (num_prefix_tokens, llm_embedding_dim) for single sample
        """
        single_input = user_embedding.dim() == 1
        if single_input:
            user_embedding = user_embedding.unsqueeze(0)

        # Project: (batch, user_dim) -> (batch, num_tokens * llm_dim)
        projected = self.mlp(user_embedding)

        # Reshape: (batch, num_tokens * llm_dim) -> (batch, num_tokens, llm_dim)
        prefix_tokens = projected.view(
            -1, self.num_prefix_tokens, self.llm_embedding_dim
        )

        if single_input:
            return prefix_tokens.squeeze(0)
        return prefix_tokens


class MultiLayerPrefixProjector(nn.Module):
    """
    Projects user embeddings to prefix tokens for multiple LLM layers.

    Similar to P-Tuning v2, this generates separate prefix tokens for
    each transformer layer, providing deeper personalization.
    """

    def __init__(
        self,
        user_embedding_dim: int,
        llm_embedding_dim: int,
        num_layers: int,
        num_prefix_tokens: int = 1,
        hidden_dim: Optional[int] = None,
        mlp_layers: int = 2,
        dropout: float = 0.1,
        shared_projector: bool = False,
    ):
        """
        Initialize multi-layer prefix projector.

        Args:
            user_embedding_dim: Dimension of input user embeddings
            llm_embedding_dim: Dimension of LLM's hidden states
            num_layers: Number of LLM layers to inject prefixes into
            num_prefix_tokens: Number of prefix tokens per layer
            hidden_dim: Hidden dimension for projector MLPs
            mlp_layers: Number of layers in each projector MLP
            dropout: Dropout probability
            shared_projector: If True, share one projector across all layers
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_prefix_tokens = num_prefix_tokens
        self.llm_embedding_dim = llm_embedding_dim

        if shared_projector:
            # Single projector, output is duplicated for each layer
            self.projector = PrefixProjector(
                user_embedding_dim=user_embedding_dim,
                llm_embedding_dim=llm_embedding_dim,
                num_prefix_tokens=num_prefix_tokens,
                hidden_dim=hidden_dim,
                num_layers=mlp_layers,
                dropout=dropout,
            )
            self.projectors = None
        else:
            # Separate projector for each layer
            self.projector = None
            self.projectors = nn.ModuleList([
                PrefixProjector(
                    user_embedding_dim=user_embedding_dim,
                    llm_embedding_dim=llm_embedding_dim,
                    num_prefix_tokens=num_prefix_tokens,
                    hidden_dim=hidden_dim,
                    num_layers=mlp_layers,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])

    def forward(
        self, user_embedding: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """
        Project user embedding to prefix tokens for each layer.

        Args:
            user_embedding: User embedding of shape (batch, user_embedding_dim)

        Returns:
            Dictionary mapping layer index to prefix tokens
            Each tensor has shape (batch, num_prefix_tokens, llm_embedding_dim)
        """
        if self.projector is not None:
            # Shared projector
            prefix = self.projector(user_embedding)
            return {i: prefix for i in range(self.num_layers)}
        else:
            # Per-layer projectors
            return {
                i: proj(user_embedding)
                for i, proj in enumerate(self.projectors)
            }
