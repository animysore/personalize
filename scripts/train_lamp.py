#!/usr/bin/env python3
"""Train E2P on LaMP benchmark tasks."""

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from personalize.datasets import LaMPDataset, LAMP_TASKS
from personalize.models import E2PLLM


def main():
    parser = argparse.ArgumentParser(description="Train E2P on LaMP")
    parser.add_argument("--task", type=str, default="LaMP-2", choices=list(LAMP_TASKS.keys()))
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--num-prefix-tokens", type=int, default=1)
    parser.add_argument("--max-profile-items", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=1000, help="Max training samples")
    parser.add_argument("--eval-samples", type=int, default=100, help="Samples for evaluation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="./data/lamp")
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = f"e2p_{args.task.lower()}_{args.num_prefix_tokens}tok.pt"

    print("=" * 70)
    print(f"Training E2P on {args.task}: {LAMP_TASKS[args.task]['name']}")
    print("=" * 70)
    print(f"Task type: {LAMP_TASKS[args.task]['type']}")
    print(f"Description: {LAMP_TASKS[args.task]['description']}")

    # Load datasets
    print(f"\nLoading {args.task} dataset...")
    train_data = LaMPDataset(
        args.task,
        split="train",
        data_dir=args.data_dir,
        max_profile_items=args.max_profile_items,
    )
    print(f"Train samples: {len(train_data)}")

    dev_data = LaMPDataset(
        args.task,
        split="dev",
        data_dir=args.data_dir,
        max_profile_items=args.max_profile_items,
    )
    print(f"Dev samples: {len(dev_data)}")

    # Load model
    print(f"\nLoading model: {args.model}")
    e2p = E2PLLM(
        model_name=args.model,
        encoder_name=args.encoder,
        num_prefix_tokens=args.num_prefix_tokens,
        device=args.device,
    )
    e2p.load()
    print(f"Trainable parameters: {e2p.get_num_trainable_parameters():,}")

    # Setup optimizer
    optimizer = AdamW(e2p.get_trainable_parameters(), lr=args.lr)

    # Training
    print(f"\nTraining for {args.epochs} epochs on {min(len(train_data), args.max_samples)} samples...")
    print("-" * 70)

    e2p.prefix_projector.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_samples = 0

        # Limit samples per epoch
        samples = list(train_data)[:args.max_samples]

        progress = tqdm(samples, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for sample in progress:
            if sample.output is None:
                continue

            # Get formatted data
            formatted = train_data.format_for_training(sample)
            context_text = formatted["context"]
            prompt = formatted["prompt"]
            target = formatted["target"]

            if not context_text or not target:
                continue

            optimizer.zero_grad()

            try:
                loss = e2p.forward_for_training(prompt, target, context_text)

                if loss.item() == 0.0 or torch.isnan(loss):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(e2p.get_trainable_parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_samples += 1

                if num_samples % 50 == 0:
                    progress.set_postfix({"loss": f"{epoch_loss / num_samples:.4f}"})

            except Exception as ex:
                # Skip problematic samples
                pass
            finally:
                # Clear CUDA cache to prevent memory fragmentation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_loss = epoch_loss / num_samples if num_samples > 0 else 0
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f} ({num_samples} samples)")

        # Evaluation
        if (epoch + 1) % 1 == 0:
            eval_loss = evaluate(e2p, dev_data, args.eval_samples)
            print(f"  Dev loss: {eval_loss:.4f}")

    print("-" * 70)
    print("Training complete!")

    # Save
    print(f"\nSaving to {args.save_path}...")
    e2p.save_projector(args.save_path)

    # Test generation
    print("\n" + "=" * 70)
    print("Sample Outputs")
    print("=" * 70)

    e2p.prefix_projector.eval()

    for i, sample in enumerate(dev_data):
        if i >= 3:
            break
        if sample.output is None:
            continue

        formatted = dev_data.format_for_training(sample)
        context_text = formatted["context"]

        print(f"\n--- Sample {i + 1} ---")
        print(f"Prompt: {sample.input[:100]}...")
        print(f"Context: {context_text[:150]}...")
        print(f"Target: {sample.output[:100]}...")

        output = e2p.generate(
            sample.input,
            user_context=context_text,
            max_new_tokens=50,
            temperature=0.7,
        )
        print(f"E2P Output: {output.text[:100]}...")


@torch.no_grad()
def evaluate(model: E2PLLM, dataset: LaMPDataset, max_samples: int) -> float:
    """Evaluate model on dataset."""
    model.prefix_projector.eval()
    total_loss = 0.0
    num_samples = 0

    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        if sample.output is None:
            continue

        formatted = dataset.format_for_training(sample)
        context_text = formatted["context"]
        prompt = formatted["prompt"]
        target = formatted["target"]

        if not context_text or not target:
            continue

        try:
            loss = model.forward_for_training(prompt, target, context_text)
            total_loss += loss.item()
            num_samples += 1
        except:
            continue

    model.prefix_projector.train()
    return total_loss / num_samples if num_samples > 0 else 0.0


if __name__ == "__main__":
    main()
