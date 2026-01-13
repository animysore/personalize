"""Dataset utilities for personalization experiments."""

from typing import Iterator, Optional

from .user_context import UserContext


class PersonalizationDataset:
    """
    Simple dataset for personalization experiments.

    Each sample contains:
    - user_context: User's profile and/or history
    - prompt: The input query/instruction
    - reference: Optional reference/expected output
    """

    def __init__(self, samples: Optional[list[dict]] = None):
        self.samples = samples or []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "user_context": UserContext.from_dict(sample.get("user", {})),
            "prompt": sample.get("prompt", ""),
            "reference": sample.get("reference"),
            "metadata": sample.get("metadata", {}),
        }

    def __iter__(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]

    def add_sample(
        self,
        prompt: str,
        user: Optional[dict] = None,
        reference: Optional[str] = None,
        **metadata,
    ) -> None:
        """Add a sample to the dataset."""
        self.samples.append(
            {
                "prompt": prompt,
                "user": user or {},
                "reference": reference,
                "metadata": metadata,
            }
        )

    @classmethod
    def from_list(cls, data: list[dict]) -> "PersonalizationDataset":
        """Create dataset from list of sample dictionaries."""
        return cls(samples=data)


def create_demo_dataset() -> PersonalizationDataset:
    """Create a small demo dataset for testing and training."""
    dataset = PersonalizationDataset()

    # Movie recommendations
    dataset.add_sample(
        prompt="Recommend a movie for me.",
        user={
            "user_id": "user_001",
            "profile": {
                "preferences": {"genres": "sci-fi, thriller", "dislikes": "horror, romance"},
            },
            "history": [
                "Loved Inception",
                "Loved The Matrix",
                "Loved Interstellar",
            ],
        },
        reference="Based on your love for sci-fi, I recommend Arrival - it's a thoughtful sci-fi thriller you'll enjoy.",
    )

    dataset.add_sample(
        prompt="What movie should I watch?",
        user={
            "user_id": "user_002",
            "profile": {
                "preferences": {"genres": "comedy, animation", "mood": "lighthearted"},
            },
            "history": [
                "Enjoyed Pixar movies",
                "Liked The Grand Budapest Hotel",
            ],
        },
        reference="For a fun, lighthearted watch, try Coco or Inside Out - both are heartwarming animated films.",
    )

    # Writing style
    dataset.add_sample(
        prompt="Write a greeting for my website.",
        user={
            "user_id": "user_003",
            "profile": {
                "preferences": {"style": "casual and friendly", "tone": "warm"},
            },
        },
        reference="Hey there! Welcome to our little corner of the internet. We're so glad you stopped by!",
    )

    dataset.add_sample(
        prompt="Write a greeting for my website.",
        user={
            "user_id": "user_004",
            "profile": {
                "preferences": {"style": "professional", "tone": "formal"},
            },
        },
        reference="Welcome to our platform. We are committed to providing you with exceptional service and solutions.",
    )

    # Technical explanations
    dataset.add_sample(
        prompt="Explain variables.",
        user={
            "user_id": "user_005",
            "profile": {
                "attributes": {"level": "beginner", "background": "no coding experience"},
            },
        },
        reference="Variables are like labeled boxes where you store information. You give the box a name, put something in it, and can look at or change what's inside later.",
    )

    dataset.add_sample(
        prompt="Explain variables.",
        user={
            "user_id": "user_006",
            "profile": {
                "attributes": {"level": "experienced", "background": "senior developer"},
            },
        },
        reference="Variables are named memory locations that hold references to objects. In Python, they're dynamically typed and use reference semantics.",
    )

    # --- New Samples ---

    dataset.add_sample(
        prompt="Recommend a movie for me.",
        user={
            "user_id": "user_007",
            "profile": {
                "preferences": {"genres": "historical drama, biography", "dislikes": "fantasy"},
            },
            "history": ["Loved The King's Speech", "Liked Schindler's List"],
        },
        reference="Given your interest in historical dramas, you must watch Oppenheimer - it's a gripping biographical masterpiece.",
    )

    dataset.add_sample(
        prompt="What should I watch tonight?",
        user={
            "user_id": "user_008",
            "profile": {
                "preferences": {"genres": "action, adventure", "pace": "fast"},
            },
            "history": ["Loved Mad Max: Fury Road", "Liked John Wick"],
        },
        reference="If you want high-octane action, Mission: Impossible - Fallout is unmatched in its intensity and stunts.",
    )

    dataset.add_sample(
        prompt="Write a greeting for my website.",
        user={
            "user_id": "user_009",
            "profile": {
                "preferences": {"style": "poetic and flowery", "tone": "elegant"},
            },
        },
        reference="Greetings, traveler. May your journey through our digital garden be filled with wonder and discovery.",
    )

    dataset.add_sample(
        prompt="Write a greeting.",
        user={
            "user_id": "user_010",
            "profile": {
                "preferences": {"style": "concise and direct", "tone": "efficient"},
            },
        },
        reference="Welcome. Browse our services below.",
    )

    dataset.add_sample(
        prompt="Explain neural networks.",
        user={
            "user_id": "user_011",
            "profile": {
                "attributes": {"level": "expert", "background": "data scientist"},
            },
            "history": ["Reads research papers", "Implements transformers"],
        },
        reference="Neural networks are function approximators composed of stacked affine transformations followed by non-linearities, optimized via backpropagation.",
    )

    dataset.add_sample(
        prompt="Explain neural networks.",
        user={
            "user_id": "user_012",
            "profile": {
                "attributes": {"level": "child", "age": "5"},
            },
        },
        reference="A neural network is like a giant brain made of math that learns to recognize pictures of cats by looking at thousands of them!",
    )

    return dataset
