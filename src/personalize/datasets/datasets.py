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
    """Create a small demo dataset for testing."""
    dataset = PersonalizationDataset()

    # Movie recommendation example
    dataset.add_sample(
        prompt="Recommend a movie for me to watch tonight.",
        user={
            "user_id": "user_001",
            "profile": {
                "name": "Alex",
                "preferences": {
                    "favorite_genres": "sci-fi, thriller",
                    "dislikes": "horror, romance",
                    "preferred_length": "under 2 hours",
                },
            },
            "history": [
                {"content": "Watched and loved Inception", "type": "watch"},
                {"content": "Watched and loved The Matrix", "type": "watch"},
                {"content": "Didn't finish The Notebook", "type": "watch"},
                {"content": "Rated Interstellar 5 stars", "type": "rating"},
            ],
        },
        reference="Based on your love for sci-fi thrillers like Inception and The Matrix, "
        "I'd recommend Arrival or Ex Machina.",
    )

    # Writing style example
    dataset.add_sample(
        prompt="Write a short product description for wireless earbuds.",
        user={
            "user_id": "user_002",
            "profile": {
                "preferences": {
                    "writing_style": "casual and friendly",
                    "tone": "enthusiastic",
                    "avoid": "technical jargon",
                },
            },
            "history": [
                {"content": "Previous writing used lots of exclamation marks", "type": "style"},
                {"content": "Prefers short sentences", "type": "style"},
            ],
        },
    )

    # Coding assistant example
    dataset.add_sample(
        prompt="Explain how to sort a list in Python.",
        user={
            "user_id": "user_003",
            "profile": {
                "attributes": {
                    "experience_level": "beginner",
                    "background": "coming from JavaScript",
                },
                "preferences": {
                    "explanation_style": "with examples",
                    "detail_level": "thorough",
                },
            },
        },
    )

    # No context example (baseline comparison)
    dataset.add_sample(
        prompt="What's the capital of France?",
        user={"user_id": "user_004"},
    )

    return dataset
