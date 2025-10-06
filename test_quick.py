#!/usr/bin/env python3
"""
Quick test of the language diffusion model
"""

from language_diffusion_model_actual import LanguageDiffusionModel, DiffusionConfig


def test_capital_question():
    print("Testing Language Diffusion Model...")

    # Create config with optimal settings for short answers
    config = DiffusionConfig(
        model_name="gpt2",
        max_length=10,
        num_diffusion_steps=6,
        temperature=0.6,
        top_p=0.75,
    )

    # Initialize model
    model = LanguageDiffusionModel(config)

    # Test the capital question
    question = "which is the capital of japan?"
    print(f"\nQuestion: {question}")

    # Generate response with short target length
    response = model.generate_response(
        question, target_length=3, num_steps=6, verbose=True
    )

    print(f"\nFinal Answer: {response}")
    return response


if __name__ == "__main__":
    test_capital_question()
