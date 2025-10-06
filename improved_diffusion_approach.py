#!/usr/bin/env python3
"""
Improved Language Diffusion Model using proper bidirectional architecture
"""

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM
from typing import List
import numpy as np


class TrueDiffusionModel:
    """
    True Language Diffusion Model using BERT's bidirectional attention
    This is more appropriate for diffusion than autoregressive models
    """

    def __init__(self, model_name="bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use BERT which is designed for masked language modeling
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Loaded BERT model for true diffusion approach")

    def diffusion_step(self, prompt: str, masked_sequence: List[str]) -> torch.Tensor:
        """
        True diffusion step using BERT's bidirectional attention
        This can see the entire context, not just previous tokens
        """
        # Create input with [MASK] tokens that BERT understands
        full_text = prompt + " " + " ".join(masked_sequence)

        # Tokenize
        inputs = self.tokenizer(
            full_text, return_tensors="pt", max_length=512, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        return logits

    def generate_diffusion(self, prompt: str, target_length: int = 5, steps: int = 6):
        """
        Generate using true diffusion process with bidirectional attention
        """
        # Initialize with BERT's [MASK] tokens
        sequence = ["[MASK]"] * target_length

        print(f"Prompt: {prompt}")
        print(f"Initial: {' '.join(sequence)}")

        for step in range(steps):
            t = 1.0 - (step / steps)

            # Get predictions for all positions simultaneously
            logits = self.diffusion_step(prompt, sequence)

            # Find [MASK] positions in the tokenized sequence
            full_text = prompt + " " + " ".join(sequence)
            tokens = self.tokenizer.tokenize(full_text)

            # Probabilistically unmask tokens based on diffusion schedule
            num_to_unmask = max(1, int((1 - t) * target_length))

            masked_positions = [
                i for i, token in enumerate(sequence) if token == "[MASK]"
            ]
            if masked_positions and len(masked_positions) >= num_to_unmask:
                # Randomly select positions to unmask
                positions_to_unmask = np.random.choice(
                    masked_positions,
                    size=min(num_to_unmask, len(masked_positions)),
                    replace=False,
                )

                for pos in positions_to_unmask:
                    # Get prediction for this position
                    # This is simplified - in practice you'd need to map tokens correctly
                    full_input = self.tokenizer(full_text, return_tensors="pt")
                    mask_indices = (
                        full_input["input_ids"] == self.tokenizer.mask_token_id
                    ).nonzero()

                    if len(mask_indices) > 0:
                        mask_idx = mask_indices[0][1]  # Take first mask for simplicity
                        token_logits = logits[0, mask_idx, :]

                        # Sample token
                        probs = F.softmax(token_logits / 0.7, dim=-1)
                        token_id = torch.multinomial(probs, 1).item()
                        token = self.tokenizer.decode([token_id]).strip()

                        if token and token not in ["[CLS]", "[SEP]", "[PAD]"]:
                            sequence[pos] = token

            print(f"Step {step+1}: {' '.join(sequence)}")

        return sequence


# Test the improved approach
if __name__ == "__main__":
    model = TrueDiffusionModel()
    result = model.generate_diffusion(
        "The capital of Japan is", target_length=3, steps=5
    )
    print(f"Final: {' '.join(result)}")
