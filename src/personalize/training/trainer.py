"""Training utilities for E2P models."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..data import PersonalizationDataset
from ..models import E2PLLM


@dataclass
class TrainingConfig:
    """Configuration for E2P training."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    log_every: int = 10
    eval_every: int = 100
    save_path: Optional[str] = None


class E2PTrainer:
    """
    Trainer for E2P prefix projector.

    Only trains the prefix projector while keeping the LLM and
    user encoder frozen.
    """

    def __init__(
        self,
        model: E2PLLM,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: E2P model to train
            config: Training configuration
        """
        self.model = model
        self.config = config or TrainingConfig()

        # Setup optimizer for projector parameters only
        self.optimizer = AdamW(
            model.get_trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.global_step = 0
        self.train_losses = []

    def train_step(
        self,
        prompt: str,
        target: str,
        user_context: Optional[str] = None,
    ) -> float:
        """
        Single training step.

        Args:
            prompt: Input prompt
            target: Target completion
            user_context: User context to encode

        Returns:
            Loss value
        """
        self.model.prefix_projector.train()

        self.optimizer.zero_grad()

        loss = self.model.forward_for_training(prompt, target, user_context)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.get_trainable_parameters(),
            self.config.max_grad_norm,
        )

        self.optimizer.step()
        self.global_step += 1

        return loss.item()

    def train(
        self,
        train_dataset: PersonalizationDataset,
        eval_dataset: Optional[PersonalizationDataset] = None,
    ) -> dict:
        """
        Train the prefix projector.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset

        Returns:
            Training statistics
        """
        num_steps = len(train_dataset) * self.config.num_epochs

        # Setup scheduler
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_steps,
            eta_min=self.config.learning_rate * 0.1,
        )

        print(f"Training E2P projector for {self.config.num_epochs} epochs")
        print(f"Trainable parameters: {self.model.get_num_trainable_parameters():,}")
        print(f"Total steps: {num_steps}")

        for epoch in range(self.config.num_epochs):
            epoch_losses = []

            progress = tqdm(
                train_dataset,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            )

            for sample in progress:
                user_context = sample["user_context"]
                context_text = user_context.to_text() if user_context.profile or user_context.history else None

                # Skip samples without context for E2P training
                if not context_text:
                    continue

                prompt = sample["prompt"]
                target = sample.get("reference") or ""

                if not target:
                    continue

                loss = self.train_step(prompt, target, context_text)
                epoch_losses.append(loss)
                self.train_losses.append(loss)

                scheduler.step()

                # Logging
                if self.global_step % self.config.log_every == 0:
                    avg_loss = sum(epoch_losses[-self.config.log_every:]) / min(
                        len(epoch_losses), self.config.log_every
                    )
                    progress.set_postfix({"loss": f"{avg_loss:.4f}"})

                # Evaluation
                if eval_dataset and self.global_step % self.config.eval_every == 0:
                    eval_loss = self.evaluate(eval_dataset)
                    print(f"\nStep {self.global_step}: Eval loss = {eval_loss:.4f}")

            print(f"Epoch {epoch + 1} average loss: {sum(epoch_losses) / len(epoch_losses):.4f}")

        # Save final model
        if self.config.save_path:
            self.model.save_projector(self.config.save_path)
            print(f"Saved projector to {self.config.save_path}")

        return {
            "train_losses": self.train_losses,
            "final_loss": self.train_losses[-1] if self.train_losses else None,
            "total_steps": self.global_step,
        }

    @torch.no_grad()
    def evaluate(self, dataset: PersonalizationDataset) -> float:
        """
        Evaluate on a dataset.

        Args:
            dataset: Evaluation dataset

        Returns:
            Average loss
        """
        self.model.prefix_projector.eval()
        losses = []

        for sample in dataset:
            user_context = sample["user_context"]
            context_text = user_context.to_text() if user_context.profile or user_context.history else None

            if not context_text:
                continue

            prompt = sample["prompt"]
            target = sample.get("reference") or ""

            if not target:
                continue

            loss = self.model.forward_for_training(prompt, target, context_text)
            losses.append(loss.item())

        return sum(losses) / len(losses) if losses else 0.0
