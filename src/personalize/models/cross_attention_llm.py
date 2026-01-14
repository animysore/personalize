"""Flamingo-style Cross-Attention Personalized LLM."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from personalize.encoders import GatedCrossAttentionLayer, MemoryBank
from personalize.models.base import GenerationOutput, PersonalizedLLM


class CrossAttentionLLM(PersonalizedLLM):
    """
    Flamingo-style cross-attention personalized LLM.

    Injects gated cross-attention layers between transformer layers,
    allowing each layer to attend to user context memory.

    Key differences from E2P/PERSOMA:
    - Deeper integration (per-layer vs prefix-only)
    - Dynamic attention to user context at each layer
    - Gated mechanism for stable training
    """

    def __init__(
        self,
        model_name: str,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_attention_interval: int = 2,
        num_heads: int = 8,
        gate_init: float = -5.0,
        max_memory_slots: int = 32,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            model_name: HuggingFace LLM model
            encoder_name: Sentence transformer for history encoding
            cross_attention_interval: Add cross-attention every N layers
            num_heads: Number of attention heads in cross-attention
            gate_init: Initial gate value for stable training
            max_memory_slots: Maximum user history items
            device: Compute device
            torch_dtype: Model dtype
        """
        super().__init__(model_name, device, torch_dtype)

        self.encoder_name = encoder_name
        self.cross_attention_interval = cross_attention_interval
        self.num_heads = num_heads
        self.gate_init = gate_init
        self.max_memory_slots = max_memory_slots

        # Components (created during load)
        self.memory_bank: Optional[MemoryBank] = None
        self.cross_attention_layers: nn.ModuleDict = nn.ModuleDict()
        self._hooks = []

        # Runtime state
        self._user_memory: Optional[torch.Tensor] = None
        self._memory_mask: Optional[torch.Tensor] = None

    def load(self) -> None:
        """Load LLM and create cross-attention layers."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )

        # Freeze LLM
        for param in self.model.parameters():
            param.requires_grad = False

        # Get hidden dimension
        llm_hidden_dim = self.model.get_input_embeddings().weight.shape[1]

        # Create memory bank
        self.memory_bank = MemoryBank(
            encoder_name=self.encoder_name,
            llm_hidden_dim=llm_hidden_dim,
            max_memory_slots=self.max_memory_slots,
            device=self.device,
        )

        # Get transformer layers
        layers = self._get_transformer_layers()
        num_layers = len(layers)

        # Create cross-attention layers
        target_indices = list(range(0, num_layers, self.cross_attention_interval))

        for idx in target_indices:
            self.cross_attention_layers[str(idx)] = GatedCrossAttentionLayer(
                hidden_dim=llm_hidden_dim,
                num_heads=self.num_heads,
                gate_init=self.gate_init,
            ).to(self.device)

        # Keep cross-attention layers in float32 for training stability
        # (float16 causes NaN during optimizer updates)
        for layer in self.cross_attention_layers.values():
            layer.to(dtype=torch.float32)

        # Install hooks
        self._install_hooks()

    def _get_transformer_layers(self) -> nn.ModuleList:
        """Get transformer layers for different architectures."""
        # Llama, Qwen, Mistral
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        # GPT-2, GPT-Neo
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            return self.model.transformer.h
        # Falcon
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "blocks"
        ):
            return self.model.transformer.blocks
        # Phi
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        else:
            raise ValueError(f"Unsupported architecture: {type(self.model)}")

    def _install_hooks(self) -> None:
        """Install forward hooks on transformer layers."""
        layers = self._get_transformer_layers()

        for idx_str, xattn_layer in self.cross_attention_layers.items():
            idx = int(idx_str)

            def make_hook(layer_idx: int, xattn: GatedCrossAttentionLayer):
                def hook(module, args, output):
                    # Get hidden states from output
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output

                    # Apply cross-attention if memory is set
                    if self._user_memory is not None:
                        # Store original dtype for conversion back
                        original_dtype = hidden_states.dtype

                        # Convert to float32 for cross-attention (numerical stability)
                        hidden_states_f32 = hidden_states.to(dtype=torch.float32)
                        memory_f32 = self._user_memory.to(dtype=torch.float32)

                        hidden_states_f32 = xattn(
                            hidden_states_f32,
                            memory_f32,
                            memory_mask=self._memory_mask,
                        )

                        # Convert back to original dtype
                        hidden_states = hidden_states_f32.to(dtype=original_dtype)

                    # Return updated output
                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    return hidden_states

                return hook

            handle = layers[idx].register_forward_hook(make_hook(idx, xattn_layer))
            self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    def _set_memory(self, history_items: Optional[list[str]]) -> None:
        """Set user memory for cross-attention."""
        if history_items:
            self._user_memory, self._memory_mask = self.memory_bank.encode(
                history_items, return_mask=True
            )
        else:
            self._user_memory = None
            self._memory_mask = None

    def _clear_memory(self) -> None:
        """Clear user memory."""
        self._user_memory = None
        self._memory_mask = None

    def generate(
        self,
        prompt: str,
        user_context: Optional[str] = None,
        history_items: Optional[list[str]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate with cross-attention to user memory.

        Args:
            prompt: Input prompt
            user_context: Single context string (converted to history item)
            history_items: User history items
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample
            **kwargs: Additional generation arguments

        Returns:
            Generation output
        """
        # Handle user_context as single history item
        if history_items is None and user_context:
            history_items = [user_context]

        # Set memory for cross-attention
        self._set_memory(history_items)

        try:
            # Tokenize prompt
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)

            prompt_tokens = encoded["input_ids"].shape[1]

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else None,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs,
                )

            # Decode
            generated_ids = outputs[0][prompt_tokens:]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            return GenerationOutput(
                text=generated_text,
                prompt_tokens=prompt_tokens,
                generated_tokens=len(generated_ids),
                total_tokens=len(outputs[0]),
            )
        finally:
            # Clear memory
            self._clear_memory()

    def forward_for_training(
        self,
        prompt: str,
        target: str,
        history_items: list[str],
    ) -> torch.Tensor:
        """
        Forward pass for training cross-attention layers.

        Args:
            prompt: Input prompt
            target: Target output
            history_items: User history items

        Returns:
            Loss tensor
        """
        # Set memory
        self._set_memory(history_items)

        try:
            # Concatenate prompt and target
            full_text = prompt + target

            # Tokenize
            encoded = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)

            input_ids = encoded["input_ids"]

            # Create labels (mask prompt tokens)
            prompt_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )["input_ids"]
            prompt_len = prompt_ids.shape[1]

            labels = input_ids.clone()
            labels[:, :prompt_len] = -100  # Mask prompt

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=encoded["attention_mask"],
                labels=labels,
            )

            return outputs.loss

        finally:
            self._clear_memory()

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return trainable parameters (cross-attention + memory projection)."""
        params = []

        # Cross-attention layers
        for layer in self.cross_attention_layers.values():
            params.extend(layer.parameters())

        # Memory bank projection
        if self.memory_bank is not None:
            params.extend(self.memory_bank.get_trainable_parameters())

        return params

    def get_num_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())

    def save_adapter(self, path: str) -> None:
        """Save cross-attention and memory bank weights."""
        state = {
            "cross_attention": {
                k: v.state_dict() for k, v in self.cross_attention_layers.items()
            },
            "memory_bank": self.memory_bank.projection.state_dict(),
            "config": {
                "cross_attention_interval": self.cross_attention_interval,
                "num_heads": self.num_heads,
                "gate_init": self.gate_init,
                "max_memory_slots": self.max_memory_slots,
            },
        }
        torch.save(state, path)

    def load_adapter(self, path: str) -> None:
        """Load cross-attention and memory bank weights."""
        state = torch.load(path, map_location=self.device)

        # Load cross-attention
        for k, v in state["cross_attention"].items():
            if k in self.cross_attention_layers:
                self.cross_attention_layers[k].load_state_dict(v)

        # Load memory bank
        self.memory_bank.projection.load_state_dict(state["memory_bank"])

    def get_gate_values(self) -> dict[int, float]:
        """Return current gate values for all cross-attention layers."""
        return {
            int(k): layer.gate_value
            for k, layer in self.cross_attention_layers.items()
        }

    def __repr__(self) -> str:
        num_xattn = len(self.cross_attention_layers)
        return (
            f"CrossAttentionLLM(model={self.model_name}, "
            f"xattn_layers={num_xattn}, interval={self.cross_attention_interval})"
        )
