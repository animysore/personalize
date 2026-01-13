"""LaMP (Language Model Personalization) benchmark data loader.

LaMP tasks:
- LaMP-1: Personalized Citation Identification
- LaMP-2: Personalized Movie Tagging
- LaMP-3: Personalized Product Rating
- LaMP-4: Personalized News Headline Generation
- LaMP-5: Personalized Scholarly Title Generation
- LaMP-6: Personalized Email Subject Generation
- LaMP-7: Personalized Tweet Paraphrasing

Reference: https://lamp-benchmark.github.io/
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import urllib.request
import gzip
import os


LAMP_BASE_URL = "https://ciir.cs.umass.edu/downloads/LaMP"

LAMP_TASKS = {
    "LaMP-1": {
        "name": "Citation Identification",
        "type": "classification",
        "description": "Identify relevant citations based on user's publication history",
    },
    "LaMP-2": {
        "name": "Movie Tagging",
        "type": "classification",
        "description": "Generate tags for movies based on user's tagging history",
    },
    "LaMP-3": {
        "name": "Product Rating",
        "type": "classification",
        "description": "Predict product ratings based on user's review history",
    },
    "LaMP-4": {
        "name": "News Headline Generation",
        "type": "generation",
        "description": "Generate news headlines in user's writing style",
    },
    "LaMP-5": {
        "name": "Scholarly Title Generation",
        "type": "generation",
        "description": "Generate paper titles based on researcher's previous work",
    },
    "LaMP-6": {
        "name": "Email Subject Generation",
        "type": "generation",
        "description": "Generate email subjects matching user's style",
    },
    "LaMP-7": {
        "name": "Tweet Paraphrasing",
        "type": "generation",
        "description": "Paraphrase tweets in user's voice",
    },
}


@dataclass
class LaMPSample:
    """A single sample from LaMP dataset."""

    id: str
    input: str
    output: Optional[str]
    profile: list[dict]  # User's history items
    task: str


class LaMPDataset:
    """LaMP benchmark dataset loader."""

    def __init__(
        self,
        task: str,
        split: str = "train",
        data_dir: str = "./data/lamp",
        max_profile_items: int = 10,
        separation: str = "user",  # "user" or "time"
    ):
        """
        Initialize LaMP dataset.

        Args:
            task: Task name (e.g., "LaMP-2", "LaMP-4")
            split: Dataset split ("train", "dev", "test")
            data_dir: Directory to store/load data
            max_profile_items: Maximum profile items to include
            separation: Separation type ("user" or "time")
        """
        self.task = task
        self.split = split
        self.data_dir = Path(data_dir)
        self.max_profile_items = max_profile_items
        self.separation = separation

        self.task_info = LAMP_TASKS.get(task, {})
        self.samples: list[LaMPSample] = []

        self._load_data()

    def _get_file_urls(self) -> tuple[str, str]:
        """Get URLs for questions and outputs files."""
        # Convert task name: LaMP-2 -> LaMP_2
        task_underscore = self.task.replace("-", "_")

        if self.split == "dev":
            split_name = "dev"
        elif self.split == "test":
            split_name = "test"
        else:
            split_name = "train"

        # New URL structure (as of 2024):
        # User-based: https://ciir.cs.umass.edu/downloads/LaMP/LaMP_2/new/train/train_questions.json
        # Time-based: https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_2/new/train/train_questions.json
        if self.separation == "time":
            base = f"{LAMP_BASE_URL}/time/{task_underscore}/new/{split_name}"
        else:
            base = f"{LAMP_BASE_URL}/{task_underscore}/new/{split_name}"

        questions_url = f"{base}/{split_name}_questions.json"
        outputs_url = f"{base}/{split_name}_outputs.json" if split_name != "test" else None

        return questions_url, outputs_url

    def _download_file(self, url: str, local_path: Path) -> bool:
        """Download a file if it doesn't exist."""
        if local_path.exists():
            return True

        local_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {url}...")
        try:
            urllib.request.urlretrieve(url, local_path)
            return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            # Try gzipped version
            try:
                gz_url = url + ".gz"
                gz_path = str(local_path) + ".gz"
                urllib.request.urlretrieve(gz_url, gz_path)
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(local_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                os.remove(gz_path)
                return True
            except Exception as e2:
                print(f"Also failed with .gz: {e2}")
                return False

    def _load_data(self) -> None:
        """Load data from files."""
        questions_url, outputs_url = self._get_file_urls()

        # Local paths
        sep = "user" if self.separation == "user" else "time"
        local_dir = self.data_dir / self.task / sep / self.split
        questions_path = local_dir / f"{self.split}_questions.json"
        outputs_path = local_dir / f"{self.split}_outputs.json" if outputs_url else None

        # Download if needed
        if not self._download_file(questions_url, questions_path):
            raise RuntimeError(f"Could not download {questions_url}")

        if outputs_url and outputs_path:
            self._download_file(outputs_url, outputs_path)

        # Load questions
        with open(questions_path) as f:
            questions_data = json.load(f)

        # Load outputs if available
        outputs_map = {}
        if outputs_path and outputs_path.exists():
            with open(outputs_path) as f:
                outputs_data = json.load(f)
                for item in outputs_data.get("golds", []):
                    outputs_map[item["id"]] = item["output"]

        # Build samples
        for item in questions_data:
            sample_id = item["id"]
            input_text = item["input"]
            profile = item.get("profile", [])[:self.max_profile_items]
            output = outputs_map.get(sample_id)

            self.samples.append(LaMPSample(
                id=sample_id,
                input=input_text,
                output=output,
                profile=profile,
                task=self.task,
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> LaMPSample:
        return self.samples[idx]

    def __iter__(self) -> Iterator[LaMPSample]:
        return iter(self.samples)

    def get_profile_text(self, sample: LaMPSample, max_items: Optional[int] = None) -> str:
        """
        Convert profile to text representation.

        Args:
            sample: LaMP sample
            max_items: Maximum items to include (None = use dataset default)

        Returns:
            Formatted profile text
        """
        items = sample.profile[:max_items] if max_items else sample.profile

        if not items:
            return ""

        lines = ["## User Profile/History"]
        for i, item in enumerate(items):
            # Format depends on task
            if self.task == "LaMP-2":  # Movie tagging
                text = item.get("text", "")
                tag = item.get("tag", "")
                lines.append(f"- Movie: {text[:100]}... → Tag: {tag}")
            elif self.task == "LaMP-3":  # Product rating
                text = item.get("text", "")
                score = item.get("score", "")
                lines.append(f"- Review: {text[:80]}... → Rating: {score}")
            elif self.task == "LaMP-4":  # News headlines
                text = item.get("text", "")
                title = item.get("title", "")
                lines.append(f"- Article: {text[:60]}... → Headline: {title}")
            elif self.task == "LaMP-5":  # Scholarly titles
                abstract = item.get("abstract", "")
                title = item.get("title", "")
                lines.append(f"- Abstract: {abstract[:60]}... → Title: {title}")
            elif self.task == "LaMP-6":  # Email subjects
                text = item.get("text", "")
                title = item.get("title", "")
                lines.append(f"- Email: {text[:60]}... → Subject: {title}")
            elif self.task == "LaMP-7":  # Tweet paraphrasing
                text = item.get("text", "")
                lines.append(f"- Tweet: {text[:100]}")
            else:
                # Generic format
                lines.append(f"- {str(item)[:100]}")

        return "\n".join(lines)

    def get_prompt(self, sample: LaMPSample) -> str:
        """Get the task prompt for a sample."""
        return sample.input

    def format_for_training(
        self,
        sample: LaMPSample,
        max_profile_items: Optional[int] = None,
    ) -> dict:
        """
        Format sample for training.

        Returns:
            Dict with 'prompt', 'context', 'target', 'profile_items'
        """
        profile_text = self.get_profile_text(sample, max_profile_items)
        profile_items = [
            self._profile_item_to_text(item)
            for item in sample.profile[:max_profile_items or self.max_profile_items]
        ]

        return {
            "prompt": sample.input,
            "context": profile_text,
            "target": sample.output,
            "profile_items": profile_items,
            "id": sample.id,
        }

    def _profile_item_to_text(self, item: dict) -> str:
        """Convert a single profile item to text."""
        if self.task == "LaMP-2":
            return f"Movie: {item.get('text', '')[:100]} Tag: {item.get('tag', '')}"
        elif self.task == "LaMP-3":
            return f"Review: {item.get('text', '')[:100]} Rating: {item.get('score', '')}"
        elif self.task == "LaMP-4":
            return f"Article: {item.get('text', '')[:80]} Headline: {item.get('title', '')}"
        elif self.task == "LaMP-5":
            return f"Abstract: {item.get('abstract', '')[:80]} Title: {item.get('title', '')}"
        elif self.task == "LaMP-6":
            return f"Email: {item.get('text', '')[:80]} Subject: {item.get('title', '')}"
        elif self.task == "LaMP-7":
            return f"Tweet: {item.get('text', '')[:100]}"
        else:
            return str(item)[:150]


def download_lamp_task(task: str, data_dir: str = "./data/lamp") -> None:
    """Download all splits for a LaMP task."""
    print(f"Downloading {task}...")
    for split in ["train", "dev"]:
        try:
            dataset = LaMPDataset(task, split=split, data_dir=data_dir)
            print(f"  {split}: {len(dataset)} samples")
        except Exception as e:
            print(f"  {split}: Failed - {e}")


if __name__ == "__main__":
    # Test download
    download_lamp_task("LaMP-2")
