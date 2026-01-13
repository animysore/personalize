"""PERSOMA: Personalized Soft Prompt Adapter.

Two-stage encoder that:
1. Encodes each history item separately
2. Resamples/compresses into fixed number of soft prompt tokens

This handles variable-length user histories more effectively than
single-shot encoding.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class HistoryEncoder(nn.Module):
    """
    Encodes individual history items into embeddings.

    Uses a pretrained transformer to encode each item, then
    applies optional projection.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: Optional[int] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.eval()

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.hidden_dim = self.encoder.config.hidden_size
        self.output_dim = output_dim or self.hidden_dim

        if output_dim and output_dim != self.hidden_dim:
            self.projection = nn.Linear(self.hidden_dim, output_dim)
        else:
            self.projection = nn.Identity()

    def _mean_pooling(self, outputs, attention_mask):
        """Mean pooling over token embeddings."""
        token_embeds = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        return torch.sum(token_embeds * mask_expanded, 1) / torch.clamp(
            mask_expanded.sum(1), min=1e-9
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Encode a list of history items.

        Args:
            texts: List of history item strings

        Returns:
            Tensor of shape (num_items, output_dim)
        """
        if not texts:
            return torch.zeros(1, self.output_dim, device=self.device)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.encoder(**encoded)

        pooled = self._mean_pooling(outputs, encoded["attention_mask"])
        return self.projection(pooled)


class PerceiverResampler(nn.Module):
    """
    Perceiver-style resampler that compresses variable-length
    history embeddings into fixed number of latent tokens.

    Uses cross-attention: learnable queries attend to history embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latents: int = 4,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of input history embeddings
            output_dim: Dimension of output latent tokens
            num_latents: Number of output latent tokens (soft prompts)
            num_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            dropout: Dropout probability
        """
        super().__init__()

        self.num_latents = num_latents
        self.output_dim = output_dim

        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, output_dim) * 0.02)

        # Project input to output dim if needed
        self.input_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

        # Cross-attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    output_dim, num_heads, dropout=dropout, batch_first=True
                ),
                "norm1": nn.LayerNorm(output_dim),
                "ffn": nn.Sequential(
                    nn.Linear(output_dim, output_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(output_dim * 4, output_dim),
                    nn.Dropout(dropout),
                ),
                "norm2": nn.LayerNorm(output_dim),
            })
            for _ in range(num_layers)
        ])

    def forward(self, history_embeds: torch.Tensor) -> torch.Tensor:
        """
        Resample history embeddings to fixed latent tokens.

        Args:
            history_embeds: (num_items, input_dim) or (batch, num_items, input_dim)

        Returns:
            Latent tokens of shape (num_latents, output_dim) or (batch, num_latents, output_dim)
        """
        single_input = history_embeds.dim() == 2
        if single_input:
            history_embeds = history_embeds.unsqueeze(0)

        batch_size = history_embeds.shape[0]

        # Project history embeddings
        kv = self.input_proj(history_embeds)  # (batch, num_items, output_dim)

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_latents, output_dim)

        # Cross-attention layers
        for layer in self.layers:
            # Cross attention: latents attend to history
            attn_out, _ = layer["cross_attn"](latents, kv, kv)
            latents = layer["norm1"](latents + attn_out)

            # FFN
            ffn_out = layer["ffn"](latents)
            latents = layer["norm2"](latents + ffn_out)

        if single_input:
            return latents.squeeze(0)
        return latents


class MLPAdapter(nn.Module):
    """
    Simple MLP adapter that compresses history embeddings.

    Aggregates via mean pooling then projects to soft tokens.
    Simpler than Perceiver but less expressive.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_tokens: int = 4,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.output_dim = output_dim

        hidden = hidden_dim or input_dim * 2
        total_output = num_tokens * output_dim

        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
            ])
            in_dim = hidden

        layers.append(nn.Linear(in_dim, total_output))
        self.mlp = nn.Sequential(*layers)

    def forward(self, history_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compress history embeddings to soft tokens.

        Args:
            history_embeds: (num_items, input_dim) or (batch, num_items, input_dim)

        Returns:
            Soft tokens of shape (num_tokens, output_dim) or (batch, num_tokens, output_dim)
        """
        single_input = history_embeds.dim() == 2
        if single_input:
            history_embeds = history_embeds.unsqueeze(0)

        # Mean pool over history items
        pooled = history_embeds.mean(dim=1)  # (batch, input_dim)

        # Project to soft tokens
        out = self.mlp(pooled)  # (batch, num_tokens * output_dim)
        out = out.view(-1, self.num_tokens, self.output_dim)

        if single_input:
            return out.squeeze(0)
        return out


class PERSOMA(nn.Module):
    """
    PERSOMA: Personalized Soft Prompt Adapter.

    Two-stage architecture:
    1. HistoryEncoder: Encodes each history item to embedding
    2. Adapter (Perceiver or MLP): Compresses to fixed soft tokens

    The soft tokens are then used as prefix for the LLM.
    """

    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_hidden_dim: int = 576,
        num_soft_tokens: int = 4,
        adapter_type: str = "perceiver",  # "perceiver" or "mlp"
        num_adapter_layers: int = 2,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        """
        Args:
            encoder_name: Pretrained encoder for history items
            llm_hidden_dim: Hidden dimension of target LLM
            num_soft_tokens: Number of soft prompt tokens to generate
            adapter_type: "perceiver" for cross-attention, "mlp" for simple
            num_adapter_layers: Depth of adapter network
            dropout: Dropout probability
            device: Device to run on
        """
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_soft_tokens = num_soft_tokens

        # Stage 1: History encoder
        self.history_encoder = HistoryEncoder(
            model_name=encoder_name,
            device=self.device,
        )
        encoder_dim = self.history_encoder.output_dim

        # Stage 2: Adapter
        if adapter_type == "perceiver":
            self.adapter = PerceiverResampler(
                input_dim=encoder_dim,
                output_dim=llm_hidden_dim,
                num_latents=num_soft_tokens,
                num_layers=num_adapter_layers,
                dropout=dropout,
            )
        else:
            self.adapter = MLPAdapter(
                input_dim=encoder_dim,
                output_dim=llm_hidden_dim,
                num_tokens=num_soft_tokens,
                num_layers=num_adapter_layers,
                dropout=dropout,
            )

        self.to(self.device)

    def forward(self, history_items: list[str]) -> torch.Tensor:
        """
        Encode history and compress to soft tokens.

        Args:
            history_items: List of history item strings

        Returns:
            Soft tokens of shape (num_soft_tokens, llm_hidden_dim)
        """
        # Encode each history item
        history_embeds = self.history_encoder(history_items)  # (num_items, encoder_dim)

        # Compress to soft tokens
        soft_tokens = self.adapter(history_embeds)  # (num_soft_tokens, llm_hidden_dim)

        return soft_tokens

    def get_trainable_parameters(self):
        """Get trainable parameters (adapter only, encoder is frozen)."""
        return list(self.adapter.parameters())
