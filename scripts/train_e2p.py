#!/usr/bin/env python3
"""Train E2P projector on demo data."""

import argparse

import torch
from torch.optim import AdamW

from personalize.data import PersonalizationDataset
from personalize.models import E2PLLM


def create_training_dataset() -> PersonalizationDataset:
    """Create a small training dataset with reference outputs."""
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


def main():
    parser = argparse.ArgumentParser(description="Train E2P projector")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--num-prefix-tokens", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
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

    # Create dataset
    dataset = create_training_dataset()
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

    from personalize.data import UserContext

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
