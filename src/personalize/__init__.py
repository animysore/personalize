"""Efficient Personalization of LLMs via Compact Context Representations."""

__version__ = "0.1.0"

from .datasets import UserContext, UserHistory, UserProfile, create_demo_dataset
from .encoders import PrefixProjector, SentenceTransformerEncoder, UserEncoder
from .models import E2PLLM, GenerationOutput, PersonalizedLLM, TextBaselineLLM

__all__ = [
    "__version__",
    "PersonalizedLLM",
    "TextBaselineLLM",
    "E2PLLM",
    "GenerationOutput",
    "UserContext",
    "UserProfile",
    "UserHistory",
    "UserEncoder",
    "SentenceTransformerEncoder",
    "PrefixProjector",
    "create_demo_dataset",
]
