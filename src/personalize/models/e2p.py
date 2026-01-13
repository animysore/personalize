"""E2P (Embedding-to-Prefix) model for efficient personalization."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..encoders import PrefixProjector, SentenceTransformerEncoder, UserEncoder
from .base import GenerationOutput, PersonalizedLLM


class E2PLLM(PersonalizedLLM):
    """
    Embedding-to-Prefix personalized LLM.

    Compresses user context into a dense embedding, then projects it to
    soft prefix tokens that are prepended to the LLM's input embeddings.

    Key benefits over text baseline:
    - User context compressed to 1-5 tokens instead of 50-500+ text tokens
    - Minimal latency overhead
    - LLM weights remain frozen

    Architecture:
        User Context (text) -> Encoder -> User Embedding (384-dim)
        User Embedding -> Prefix Projector (MLP) -> Soft Tokens (1-5 x LLM_dim)
        Soft Tokens + Input Embeddings -> Frozen LLM -> Output
    """

    def __init__(
        self,
        model_name: str,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_prefix_tokens: int = 1,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        projector_hidden_dim: Optional[int] = None,
        projector_layers: int = 2,
        projector_dropout: float = 0.1,
    ):
        """
        Initialize E2P model.

        Args:
            model_name: HuggingFace LLM model name
            encoder_name: Sentence transformer model for encoding user context
            num_prefix_tokens: Number of soft prefix tokens to generate
            device: Device to run on
            torch_dtype: Model dtype
            projector_hidden_dim: Hidden dimension for prefix projector MLP
            projector_layers: Number of layers in prefix projector
            projector_dropout: Dropout in prefix projector
        """
        super().__init__(model_name, device, torch_dtype)

        self.encoder_name = encoder_name
        self.num_prefix_tokens = num_prefix_tokens
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_layers = projector_layers
        self.projector_dropout = projector_dropout

        self.user_encoder: UserEncoder = None
        self.prefix_projector: PrefixProjector = None

    def load(self) -> None:
        """Load all components: LLM, encoder, and projector."""
        # Load tokenizer and LLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Freeze LLM parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Load user encoder
        self.user_encoder = SentenceTransformerEncoder(
            model_name=self.encoder_name,
            device=self.device,
        )

        # Get LLM embedding dimension
        llm_embed_dim = self.model.get_input_embeddings().weight.shape[1]

        # Initialize prefix projector (keep in float32 for training stability)
        self.prefix_projector = PrefixProjector(
            user_embedding_dim=self.user_encoder.embedding_dim,
            llm_embedding_dim=llm_embed_dim,
            num_prefix_tokens=self.num_prefix_tokens,
            hidden_dim=self.projector_hidden_dim,
            num_layers=self.projector_layers,
            dropout=self.projector_dropout,
        ).to(self.device)

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get parameters that should be trained (projector only)."""
        return list(self.prefix_projector.parameters())

    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.prefix_projector.parameters())

    def encode_user(self, user_context: str) -> torch.Tensor:
        """Encode user context to embedding."""
        return self.user_encoder.encode(user_context)

    def get_prefix_embeddings(self, user_context: str) -> torch.Tensor:
        """
        Get soft prefix embeddings for user context.

        Args:
            user_context: User context text

        Returns:
            Prefix embeddings of shape (num_prefix_tokens, llm_embed_dim)
        """
        user_embedding = self.encode_user(user_context)
        return self.prefix_projector(user_embedding)

    def _prepare_inputs_with_prefix(
        self,
        prompt: str,
        user_context: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> dict:
        """
        Prepare model inputs with soft prefix prepended.

        Args:
            prompt: Input text prompt
            user_context: Optional user context to encode
            max_length: Maximum sequence length

        Returns:
            Dictionary with inputs_embeds and attention_mask
        """
        # Tokenize prompt
        max_len = max_length or (self.model.config.max_position_embeddings - self.num_prefix_tokens - 256)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(self.device)

        # Get token embeddings
        input_ids = encoded["input_ids"]
        token_embeddings = self.model.get_input_embeddings()(input_ids)

        if user_context:
            # Get prefix embeddings from user context
            prefix_embeds = self.get_prefix_embeddings(user_context)
            if prefix_embeds.dim() == 2:
                prefix_embeds = prefix_embeds.unsqueeze(0)  # Add batch dim

            # Cast prefix embeddings to match token embeddings dtype
            prefix_embeds = prefix_embeds.to(dtype=token_embeddings.dtype)

            # Concatenate prefix with token embeddings
            inputs_embeds = torch.cat([prefix_embeds, token_embeddings], dim=1)

            # Extend attention mask for prefix tokens
            prefix_attention = torch.ones(
                (1, self.num_prefix_tokens),
                dtype=encoded["attention_mask"].dtype,
                device=self.device,
            )
            attention_mask = torch.cat([prefix_attention, encoded["attention_mask"]], dim=1)
        else:
            inputs_embeds = token_embeddings
            attention_mask = encoded["attention_mask"]

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "prompt_length": input_ids.shape[1],
        }

    def generate(
        self,
        prompt: str,
        user_context: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate personalized response using soft prefix.

        Args:
            prompt: Input prompt/question
            user_context: User context text (will be encoded to prefix)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation arguments

        Returns:
            GenerationOutput with response and token counts
        """
        # Prepare inputs with prefix
        prepared = self._prepare_inputs_with_prefix(
            prompt,
            user_context,
            max_length=self.model.config.max_position_embeddings - max_new_tokens - self.num_prefix_tokens,
        )

        total_prompt_tokens = prepared["inputs_embeds"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=prepared["inputs_embeds"],
                attention_mask=prepared["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        generated_tokens = outputs.shape[1]
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return GenerationOutput(
            text=response,
            prompt_tokens=total_prompt_tokens,
            generated_tokens=generated_tokens,
            total_tokens=total_prompt_tokens + generated_tokens,
        )

    def forward_for_training(
        self,
        prompt: str,
        target: str,
        user_context: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training the prefix projector.

        Args:
            prompt: Input prompt
            target: Target completion
            user_context: User context to encode

        Returns:
            Language modeling loss
        """
        # Prepare full sequence: prompt + target
        full_text = prompt + target
        prepared = self._prepare_inputs_with_prefix(full_text, user_context)

        # Tokenize with same settings as _prepare_inputs_with_prefix
        max_len = self.model.config.max_position_embeddings - self.num_prefix_tokens - 256
        full_ids = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )["input_ids"].to(self.device)

        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )["input_ids"]
        prompt_len = prompt_ids.shape[1]

        # Create labels matching the input_embeds length
        # For prefix tokens, use -100 (ignore in loss)
        if user_context:
            prefix_labels = torch.full(
                (1, self.num_prefix_tokens),
                -100,
                dtype=full_ids.dtype,
                device=self.device,
            )
            labels = torch.cat([prefix_labels, full_ids], dim=1)
            # Mask prompt tokens (after prefix)
            labels[:, self.num_prefix_tokens : self.num_prefix_tokens + prompt_len] = -100
        else:
            labels = full_ids.clone()
            labels[:, :prompt_len] = -100

        # Verify sequence lengths match
        expected_len = prepared["inputs_embeds"].shape[1]
        if labels.shape[1] != expected_len:
            # Truncate or pad labels to match
            if labels.shape[1] > expected_len:
                labels = labels[:, :expected_len]
            else:
                pad_len = expected_len - labels.shape[1]
                labels = torch.cat([
                    labels,
                    torch.full((1, pad_len), -100, dtype=labels.dtype, device=self.device)
                ], dim=1)

        # Ensure we have at least one non-masked label token
        num_valid_labels = (labels != -100).sum().item()
        if num_valid_labels == 0:
            # No valid labels - skip this sample
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Forward pass
        outputs = self.model(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            labels=labels,
        )

        return outputs.loss

    def save_projector(self, path: str) -> None:
        """Save prefix projector weights."""
        torch.save(self.prefix_projector.state_dict(), path)

    def load_projector(self, path: str) -> None:
        """Load prefix projector weights."""
        self.prefix_projector.load_state_dict(torch.load(path, map_location=self.device))

    def get_context_stats(self, user_context: str) -> dict:
        """Get statistics about user context encoding."""
        # Count text tokens for comparison
        text_tokens = len(self.tokenizer.encode(user_context, add_special_tokens=False))
        user_embedding = self.encode_user(user_context)

        return {
            "text_tokens": text_tokens,
            "soft_tokens": self.num_prefix_tokens,
            "compression_ratio": text_tokens / self.num_prefix_tokens,
            "user_embedding_dim": user_embedding.shape[-1],
        }
