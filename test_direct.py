#!/usr/bin/env python3
"""
Enhanced test with better prompting for direct answers
"""

from language_diffusion_model_actual import LanguageDiffusionModel, DiffusionConfig


def test_direct_answer():
    print("Testing Language Diffusion Model with Direct Answer Prompting...")

    # Create config
    config = DiffusionConfig(
        model_name="gpt2",
        max_length=10,
        num_diffusion_steps=6,
        temperature=0.5,  # Lower temperature for more focused answers
        top_p=0.7,
    )

    # Initialize model
    model = LanguageDiffusionModel(config)

    # Test with different prompt formats
    questions = [
        "The capital of Japan is",
        "Japan's capital city is",
        "What is the capital of Japan? Answer:",
    ]

    for question in questions:
        print(f"\nPrompt: {question}")
        response = model.generate_response(
            question, target_length=2, num_steps=5, verbose=False
        )
        print(f"Answer: {response}")
        print("-" * 50)


if __name__ == "__main__":
    test_direct_answer()
