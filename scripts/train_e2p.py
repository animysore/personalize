#!/usr/bin/env python3
"""Train E2P projector on demo data."""

import argparse

import torch
from torch.optim import AdamW

from personalize.datasets import PersonalizationDataset, UserContext, create_demo_dataset
from personalize.models import E2PLLM


def main():
    parser = argparse.ArgumentParser(description="Train E2P projector")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--num-prefix-tokens", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-path", type=str, default="e2p_projector.pt", help="Path to save trained projector")
    args = parser.parse_args()

    print("=" * 60)
    print("E2P Projector Training")
    print("=" * 60)

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

    # Create dataset - using the expanded demo dataset
    dataset = create_demo_dataset()
    print(f"Training samples: {len(dataset)}")

    # Setup optimizer
    optimizer = AdamW(e2p.get_trainable_parameters(), lr=args.lr)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)

    e2p.prefix_projector.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_samples = 0

        for sample in dataset:
            user_context = sample["user_context"]
            context_text = user_context.to_text()

            if not context_text or not sample["reference"]:
                continue

            prompt = sample["prompt"]
            target = sample["reference"]

            optimizer.zero_grad()

            loss = e2p.forward_for_training(prompt, target, context_text)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(e2p.get_trainable_parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_samples += 1

        avg_loss = epoch_loss / num_samples if num_samples > 0 else 0
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f}")

    print("-" * 60)
    print("Training complete!\n")
    
    # Save model
    print(f"Saving projector to {args.save_path}...")
    e2p.save_projector(args.save_path)
    print("Saved.")

    # Test generation
    print("=" * 60)
    print("Testing trained model")
    print("=" * 60)

    e2p.prefix_projector.eval()

    test_cases = [
        {
            "prompt": "Recommend a movie for me.",
            "context": {
                "profile": {"preferences": {"genres": "sci-fi", "dislikes": "romance"}},
                "history": ["Loved Blade Runner", "Loved Ex Machina"],
            },
        },
        {
            "prompt": "Write a greeting.",
            "context": {
                "profile": {"preferences": {"style": "casual", "tone": "friendly"}},
            },
        },
        {
            "prompt": "Explain loops.",
            "context": {
                "profile": {"attributes": {"level": "beginner"}},
            },
        },
    ]

    for i, tc in enumerate(test_cases):
        print(f"\n--- Test {i + 1} ---")

        user_ctx = UserContext.from_dict({"user_id": f"test_{i}", **tc["context"]})
        context_text = user_ctx.to_text()

        print(f"Context: {context_text[:100]}...")
        print(f"Prompt: {tc['prompt']}")

        output = e2p.generate(
            tc["prompt"],
            user_context=context_text,
            max_new_tokens=100,
            temperature=0.7,
        )
        print(f"Response: {output.text[:200]}")


if __name__ == "__main__":
    main()
