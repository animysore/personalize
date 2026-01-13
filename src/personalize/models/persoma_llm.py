"""PERSOMA-based personalized LLM."""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..encoders.persoma import PERSOMA
from .base import GenerationOutput, PersonalizedLLM


class PERSOMALLM(PersonalizedLLM):
    """
    PERSOMA-based personalized LLM.

    Uses two-stage encoding:
    1. Each history item encoded separately
    2. Perceiver/MLP compresses to N soft tokens

    Benefits over E2P:
    - Handles variable-length histories naturally
    - Per-item encoding preserves fine-grained info
    - Attention-based compression can focus on relevant items
    """

    def __init__(
        self,
        model_name: str,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_soft_tokens: int = 4,
        adapter_type: str = "perceiver",
        num_adapter_layers: int = 2,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(model_name, device, torch_dtype)

        self.encoder_name = encoder_name
        self.num_soft_tokens = num_soft_tokens
        self.adapter_type = adapter_type
        self.num_adapter_layers = num_adapter_layers

        self.persoma: PERSOMA = None

    def load(self) -> None:
        """Load LLM and PERSOMA encoder."""
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

        # Freeze LLM
        for param in self.model.parameters():
            param.requires_grad = False

        # Get LLM hidden dim
        llm_hidden_dim = self.model.get_input_embeddings().weight.shape[1]

        # Initialize PERSOMA
        self.persoma = PERSOMA(
            encoder_name=self.encoder_name,
            llm_hidden_dim=llm_hidden_dim,
            num_soft_tokens=self.num_soft_tokens,
            adapter_type=self.adapter_type,
            num_adapter_layers=self.num_adapter_layers,
            device=self.device,
        )

    def get_trainable_parameters(self):
        """Get trainable parameters (PERSOMA adapter only)."""
        return self.persoma.get_trainable_parameters()

    def get_num_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())

    def encode_history(self, history_items: list[str]) -> torch.Tensor:
        """
        Encode user history to soft tokens.

        Args:
            history_items: List of history strings

        Returns:
            Soft tokens (num_soft_tokens, llm_hidden_dim)
        """
        return self.persoma(history_items)

    def _prepare_inputs_with_prefix(
        self,
        prompt: str,
        history_items: Optional[list[str]] = None,
        max_length: Optional[int] = None,
    ) -> dict:
        """Prepare model inputs with PERSOMA soft prefix."""
        max_len = max_length or (
            self.model.config.max_position_embeddings - self.num_soft_tokens - 256
        )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(self.device)

        input_ids = encoded["input_ids"]
        token_embeds = self.model.get_input_embeddings()(input_ids)

        if history_items:
            # Get soft tokens from PERSOMA
            soft_tokens = self.encode_history(history_items)
            if soft_tokens.dim() == 2:
                soft_tokens = soft_tokens.unsqueeze(0)

            # Match dtype
            soft_tokens = soft_tokens.to(dtype=token_embeds.dtype)

            # Concatenate
            inputs_embeds = torch.cat([soft_tokens, token_embeds], dim=1)

            # Extend attention mask
            prefix_attn = torch.ones(
                (1, self.num_soft_tokens),
                dtype=encoded["attention_mask"].dtype,
                device=self.device,
            )
            attention_mask = torch.cat([prefix_attn, encoded["attention_mask"]], dim=1)
        else:
            inputs_embeds = token_embeds
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
        history_items: Optional[list[str]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate with PERSOMA soft prefix.

        Args:
            prompt: Input prompt
            user_context: User context text (will be split into history items)
            history_items: Explicit list of history items (preferred)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            do_sample: Use sampling

        Returns:
            GenerationOutput
        """
        # Parse user_context into history items if not provided explicitly
        if history_items is None and user_context:
            history_items = self._parse_context_to_items(user_context)

        prepared = self._prepare_inputs_with_prefix(
            prompt,
            history_items,
            max_length=self.model.config.max_position_embeddings - max_new_tokens - self.num_soft_tokens,
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

    def _parse_context_to_items(self, context: str) -> list[str]:
        """Parse context text into individual history items."""
        items = []
        for line in context.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                items.append(line[2:])
            elif line and not line.startswith("#"):
                items.append(line)
        return items if items else [context]

    def forward_for_training(
        self,
        prompt: str,
        target: str,
        history_items: list[str],
    ) -> torch.Tensor:
        """Forward pass for training."""
        full_text = prompt + target
        prepared = self._prepare_inputs_with_prefix(full_text, history_items)

        full_ids = self.tokenizer(full_text, return_tensors="pt")["input_ids"].to(self.device)

        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]

        # Create labels
        if history_items:
            prefix_labels = torch.full(
                (1, self.num_soft_tokens),
                -100,
                dtype=full_ids.dtype,
                device=self.device,
            )
            labels = torch.cat([prefix_labels, full_ids], dim=1)
            labels[:, self.num_soft_tokens : self.num_soft_tokens + prompt_len] = -100
        else:
            labels = full_ids.clone()
            labels[:, :prompt_len] = -100

        outputs = self.model(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            labels=labels,
        )

        return outputs.loss

    def save_adapter(self, path: str) -> None:
        """Save PERSOMA adapter weights."""
        torch.save(self.persoma.adapter.state_dict(), path)

    def load_adapter(self, path: str) -> None:
        """Load PERSOMA adapter weights."""
        self.persoma.adapter.load_state_dict(
            torch.load(path, map_location=self.device)
        )

    def get_context_stats(self, history_items: list[str]) -> dict:
        """Get encoding statistics."""
        total_text_tokens = sum(
            len(self.tokenizer.encode(item, add_special_tokens=False))
            for item in history_items
        )
        return {
            "num_history_items": len(history_items),
            "total_text_tokens": total_text_tokens,
            "soft_tokens": self.num_soft_tokens,
            "compression_ratio": total_text_tokens / self.num_soft_tokens if self.num_soft_tokens else 0,
        }
