"""Memory Bank for storing user context embeddings."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class MemoryBank(nn.Module):
    """
    Stores and projects user context embeddings for cross-attention.

    Encodes user history items to embeddings and projects them
    to match the LLM hidden dimension for cross-attention.
    """

    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_hidden_dim: int = 1536,
        max_memory_slots: int = 32,
        device: Optional[str] = None,
    ):
        """
        Args:
            encoder_name: Sentence transformer for encoding history
            llm_hidden_dim: Target dimension for cross-attention
            max_memory_slots: Maximum history items to store
            device: Compute device
        """
        super().__init__()

        self.encoder_name = encoder_name
        self.llm_hidden_dim = llm_hidden_dim
        self.max_memory_slots = max_memory_slots
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder (frozen)
        self._encoder = None
        self._tokenizer = None
        self._encoder_dim = None

        # Projection layer (trainable)
        self.projection = None

    def _load_encoder(self) -> None:
        """Load encoder model."""
        if self._encoder is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self._encoder = AutoModel.from_pretrained(self.encoder_name).to(self.device)
        self._encoder.eval()

        # Freeze encoder
        for param in self._encoder.parameters():
            param.requires_grad = False

        self._encoder_dim = self._encoder.config.hidden_size

        # Create projection layer
        self.projection = nn.Sequential(
            nn.Linear(self._encoder_dim, self.llm_hidden_dim),
            nn.LayerNorm(self.llm_hidden_dim),
            nn.GELU(),
            nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim),
        ).to(self.device)

        # Initialize projection
        self._init_projection()

    def _init_projection(self) -> None:
        """Initialize projection with small values."""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    @property
    def encoder_dim(self) -> int:
        """Return encoder embedding dimension."""
        self._load_encoder()
        return self._encoder_dim

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts to embeddings (frozen)."""
        self._load_encoder()

        if not texts:
            return torch.zeros(1, self._encoder_dim, device=self.device)

        # Tokenize
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        # Encode
        outputs = self._encoder(**encoded)

        # Mean pooling
        attention_mask = encoded["attention_mask"]
        hidden_states = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        return embeddings

    def encode(
        self,
        history_items: list[str],
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Encode history items to memory embeddings.

        Args:
            history_items: List of history item strings
            return_mask: Whether to return attention mask

        Returns:
            Memory tensor of shape (1, num_items, llm_hidden_dim)
            Optionally, attention mask of shape (1, num_items)
        """
        self._load_encoder()

        if not history_items:
            memory = torch.zeros(1, 1, self.llm_hidden_dim, device=self.device)
            if return_mask:
                mask = torch.zeros(1, 1, device=self.device)
                return memory, mask
            return memory

        # Limit items
        items = history_items[: self.max_memory_slots]

        # Encode with frozen encoder
        embeddings = self._encode(items)  # (num_items, encoder_dim)

        # Project to LLM dimension (trainable)
        memory = self.projection(embeddings)  # (num_items, llm_hidden_dim)

        # Add batch dimension
        memory = memory.unsqueeze(0)  # (1, num_items, llm_hidden_dim)

        if return_mask:
            mask = torch.ones(1, len(items), device=self.device)
            return memory, mask

        return memory

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return trainable parameters (projection layer only)."""
        self._load_encoder()
        return list(self.projection.parameters())

    def forward(
        self,
        history_items: list[str],
    ) -> torch.Tensor:
        """
        Forward pass - encode history items.

        Args:
            history_items: List of history strings

        Returns:
            Memory tensor (1, num_items, llm_hidden_dim)
        """
        return self.encode(history_items)

    def save(self, path: str) -> None:
        """Save projection weights."""
        self._load_encoder()
        torch.save(self.projection.state_dict(), path)

    def load_weights(self, path: str) -> None:
        """Load projection weights."""
        self._load_encoder()
        state_dict = torch.load(path, map_location=self.device)
        self.projection.load_state_dict(state_dict)

    def __repr__(self) -> str:
        return (
            f"MemoryBank(encoder={self.encoder_name}, "
            f"llm_dim={self.llm_hidden_dim}, max_slots={self.max_memory_slots})"
        )
