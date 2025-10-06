"""
Simulated Reverse Process of LLaDA (Large Language Diffusion Algorithm)
Author: Code Bala Simulation Example
"""

import random
import numpy as np
import time


# -----------------------------
# Mock diffusion model p_theta
# -----------------------------
def p_theta(prompt, masked_sequence):
    """
    Mock probability model that predicts next tokens for masked positions.
    In a real LLaDA, this would be a transformer producing token probabilities.
    """
    # We'll simulate predictions based on simple prompt-aware templates
    base_answer = {
        "what is computer?": "A computer is an electronic device that processes data.",
        "what is ai?": "Artificial intelligence is the simulation of human intelligence in machines.",
        "what is machine learning?": "Machine learning is a subset of AI that learns from data.",
    }
    text = prompt.lower().strip()
    target = base_answer.get(
        text, "This is a generic simulated answer for demonstration."
    )
    return target.split()


# -----------------------------
# Reverse Diffusion Process
# -----------------------------
def llada_reverse_process(prompt, L=12, N=10):
    """
    Simulates Algorithm 4 — Reverse Process of LLaDA.
    L = answer length (tokens)
    N = number of reverse diffusion steps
    """
    # Step 1: fully masked start
    r_t = ["[MASK]"] * L
    tokens_pred = p_theta(prompt, r_t)  # model output (mock)
    tokens_pred = (tokens_pred + ["<EOS>"] * L)[:L]  # pad to length L

    print(f"\nPrompt: {prompt}")
    print(f"Target tokens (for simulation): {tokens_pred}\n")

    # Reverse process loop
    for step, t in enumerate(np.linspace(1, 0, N), 1):
        s = max(t - 1 / N, 0)
        r_0 = r_t.copy()

        for i in range(L):
            if r_t[i] != "[MASK]":
                r_0[i] = r_t[i]  # keep existing token
            else:
                # probability to stay masked
                if random.random() < (s / t if t != 0 else 0):
                    r_0[i] = "[MASK]"
                else:
                    # unmask using greedy sample from p_theta
                    r_0[i] = tokens_pred[i]

        # show intermediate results
        print(f"Step {step:02d} (t={t:.2f}) : {' '.join(r_0)}")

        r_t = r_0.copy()
        time.sleep(0.3)

    print("\nFinal generated answer:")
    print(" ".join(r_t))
    return r_t


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    prompt = input("Enter your question (e.g., 'What is computer?') : ").strip()
    llada_reverse_process(prompt, L=12, N=12)
