#!/usr/bin/env python3
"""
Final optimized test for complete answers
"""

from language_diffusion_model_actual import LanguageDiffusionModel, DiffusionConfig


def test_complete_answer():
    print("Testing Complete Answer Generation...")

    # Optimized config for complete answers
    config = DiffusionConfig(
        model_name="gpt2",
        max_length=15,
        num_diffusion_steps=8,
        temperature=0.4,  # Very focused generation
        top_p=0.6,  # Even more focused
    )

    # Initialize model
    model = LanguageDiffusionModel(config)

    # Test with completion-style prompts
    test_cases = [
        ("The capital of Japan is", 3),
        ("Japan's capital:", 2),
        ("Tokyo is the capital of", 4),
    ]

    for prompt, length in test_cases:
        print(f"\nPrompt: '{prompt}'")
        print("Generating...")

        response = model.generate_response(
            prompt, target_length=length, num_steps=6, verbose=True
        )

        print(f"Complete: {prompt} {response}")
        print("=" * 60)


if __name__ == "__main__":
    test_complete_answer()
