"""User context encoders for creating dense user embeddings."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class UserEncoder(ABC):
    """Abstract base class for user context encoders."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of output user embeddings."""
        pass

    @abstractmethod
    def encode(self, user_context: str | list[str]) -> torch.Tensor:
        """
        Encode user context into dense embedding(s).

        Args:
            user_context: Single context string or batch of context strings

        Returns:
            Tensor of shape (embedding_dim,) or (batch_size, embedding_dim)
        """
        pass


class SentenceTransformerEncoder(UserEncoder):
    """
    Encode user context using a sentence transformer model.

    Uses mean pooling over token embeddings to create a single dense vector
    representing the user's context (profile + history).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        Initialize sentence transformer encoder.

        Args:
            model_name: HuggingFace model name (sentence-transformer compatible)
            device: Device to run on
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self._embedding_dim = self.model.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _mean_pooling(
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean pooling to token embeddings."""
        token_embeddings = model_output[0]  # First element is last hidden state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @torch.no_grad()
    def encode(self, user_context: str | list[str]) -> torch.Tensor:
        """
        Encode user context into dense embedding.

        Args:
            user_context: User context text or list of texts

        Returns:
            Embeddings tensor of shape (embedding_dim,) or (batch, embedding_dim)
        """
        single_input = isinstance(user_context, str)
        if single_input:
            user_context = [user_context]

        encoded = self.tokenizer(
            user_context,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        embeddings = self._mean_pooling(outputs, encoded["attention_mask"])

        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        if single_input:
            return embeddings.squeeze(0)
        return embeddings


class LearnedUserEncoder(nn.Module, UserEncoder):
    """
    Learnable user encoder that can be trained end-to-end.

    Wraps a pretrained encoder and adds optional learnable projection layers.
    """

    def __init__(
        self,
        base_encoder: UserEncoder,
        output_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initialize learned user encoder.

        Args:
            base_encoder: Base encoder (e.g., SentenceTransformerEncoder)
            output_dim: Output embedding dimension (None = same as base)
            hidden_dim: Hidden layer dimension for MLP projection
            num_layers: Number of projection layers
            dropout: Dropout probability
        """
        super().__init__()
        self.base_encoder = base_encoder
        self._output_dim = output_dim or base_encoder.embedding_dim

        # Build projection MLP if dimensions differ or num_layers > 0
        if output_dim and output_dim != base_encoder.embedding_dim or num_layers > 1:
            layers = []
            in_dim = base_encoder.embedding_dim
            hidden = hidden_dim or (in_dim + self._output_dim) // 2

            for i in range(num_layers - 1):
                layers.extend([
                    nn.Linear(in_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                in_dim = hidden

            layers.append(nn.Linear(in_dim, self._output_dim))
            self.projection = nn.Sequential(*layers)
        else:
            self.projection = nn.Identity()

    @property
    def embedding_dim(self) -> int:
        return self._output_dim

    def encode(self, user_context: str | list[str]) -> torch.Tensor:
        """Encode and project user context."""
        base_embedding = self.base_encoder.encode(user_context)
        return self.projection(base_embedding)

    def forward(self, user_context: str | list[str]) -> torch.Tensor:
        """Forward pass (alias for encode)."""
        return self.encode(user_context)
