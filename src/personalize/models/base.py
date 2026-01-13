"""Base classes for personalized LLM generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


@dataclass
class GenerationOutput:
    """Output from personalized generation."""

    text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int


class PersonalizedLLM(ABC):
    """Abstract base class for personalized LLM generation."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or torch.float16

        self.tokenizer: PreTrainedTokenizer = None
        self.model: PreTrainedModel = None

    @abstractmethod
    def load(self) -> None:
        """Load model and tokenizer."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        user_context: Optional[str] = None,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> GenerationOutput:
        """Generate personalized response."""
        pass

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
