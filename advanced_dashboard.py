#!/usr/bin/env python3
"""
Advanced Language Model Comparison Dashboard
Compares Autoregressive (GPT-2) vs Diffusion (BERT) models side by side
"""

import streamlit as st
import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Import our models
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertForMaskedLM, BertTokenizer
import psutil
import gc


class PerformanceMetrics:
    """Track and calculate performance metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.memory_before = 0
        self.memory_after = 0
        self.gpu_memory_before = 0
        self.gpu_memory_after = 0
        self.tokens_generated = 0
        self.steps_taken = 0

    def start_tracking(self):
        self.start_time = time.time()
        self.memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        if torch.cuda.is_available():
            self.gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB

    def end_tracking(self, tokens_generated: int, steps_taken: int = 1):
        self.end_time = time.time()
        self.memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        if torch.cuda.is_available():
            self.gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        self.tokens_generated = tokens_generated
        self.steps_taken = steps_taken

    def get_metrics(self) -> Dict:
        total_time = (
            self.end_time - self.start_time if self.start_time and self.end_time else 0
        )
        return {
            "total_time": total_time,
            "time_per_token": total_time / max(self.tokens_generated, 1),
            "time_per_step": total_time / max(self.steps_taken, 1),
            "tokens_per_second": self.tokens_generated / max(total_time, 0.001),
            "memory_used": self.memory_after - self.memory_before,
            "gpu_memory_used": self.gpu_memory_after - self.gpu_memory_before,
            "peak_memory": self.memory_after,
            "tokens_generated": self.tokens_generated,
            "steps_taken": self.steps_taken,
        }


class AutoregressiveModel:
    """GPT-2 Autoregressive Model"""

    def __init__(self, model_name="gpt2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        track_steps: bool = True,
    ) -> Tuple[str, List[str], PerformanceMetrics]:
        """Generate text autoregressively"""
        metrics = PerformanceMetrics()
        metrics.start_tracking()

        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        original_length = inputs.shape[1]

        generated_tokens = []
        step_outputs = [prompt]

        with torch.no_grad():
            current_inputs = inputs

            for step in range(max_length - original_length):
                # Get next token logits
                outputs = self.model(current_inputs)
                logits = outputs.logits[0, -1, :] / temperature

                # Apply top-p sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    0, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                # Add to sequence
                current_inputs = torch.cat(
                    [current_inputs, next_token.unsqueeze(0)], dim=1
                )

                token_text = self.tokenizer.decode([next_token.item()]).strip()
                generated_tokens.append(token_text)

                if track_steps:
                    current_text = self.tokenizer.decode(current_inputs[0])
                    step_outputs.append(current_text)

                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        final_text = self.tokenizer.decode(current_inputs[0])
        metrics.end_tracking(len(generated_tokens), len(generated_tokens))

        return final_text, step_outputs, metrics


class DiffusionModel:
    """BERT-based Diffusion Model"""

    def __init__(self, model_name="bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        target_length: int = 10,
        num_steps: int = 8,
        temperature: float = 0.7,
        track_steps: bool = True,
    ) -> Tuple[str, List[str], PerformanceMetrics]:
        """Generate text using diffusion process"""
        metrics = PerformanceMetrics()
        metrics.start_tracking()

        # Initialize with [MASK] tokens
        sequence = ["[MASK]"] * target_length
        step_outputs = [f"{prompt} {' '.join(sequence)}"]

        with torch.no_grad():
            for step in range(num_steps):
                t = 1.0 - (step / num_steps)

                # Create full input
                full_text = prompt + " " + " ".join(sequence)

                # Tokenize
                inputs = self.tokenizer(
                    full_text, return_tensors="pt", max_length=512, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get predictions
                outputs = self.model(**inputs)
                logits = outputs.logits / temperature

                # Find mask positions
                input_ids = inputs["input_ids"][0]
                mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(
                    as_tuple=True
                )[0]

                if len(mask_positions) == 0:
                    break

                # Probabilistically unmask tokens
                num_to_unmask = max(
                    1, int((1 - t) * len([s for s in sequence if s == "[MASK]"]))
                )
                masked_indices = [
                    i for i, token in enumerate(sequence) if token == "[MASK]"
                ]

                if masked_indices and len(masked_indices) >= num_to_unmask:
                    indices_to_unmask = np.random.choice(
                        masked_indices,
                        size=min(num_to_unmask, len(masked_indices)),
                        replace=False,
                    )

                    for seq_idx, mask_pos in zip(
                        indices_to_unmask, mask_positions[: len(indices_to_unmask)]
                    ):
                        # Get prediction for this position
                        token_logits = logits[0, mask_pos, :]
                        probs = F.softmax(token_logits, dim=-1)
                        token_id = torch.multinomial(probs, 1).item()
                        token = self.tokenizer.decode([token_id]).strip()

                        # Filter out special tokens
                        if (
                            token
                            and token not in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]
                            and len(token) > 0
                        ):
                            sequence[seq_idx] = token

                if track_steps:
                    current_text = f"{prompt} {' '.join(sequence)}"
                    step_outputs.append(current_text)

        final_text = (
            f"{prompt} {' '.join([token for token in sequence if token != '[MASK]'])}"
        )
        metrics.end_tracking(target_length, num_steps)

        return final_text, step_outputs, metrics


def create_comparison_dashboard():
    """Main Streamlit dashboard"""

    st.set_page_config(
        page_title="Language Model Comparison Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🤖 Autoregressive vs Diffusion Language Models Dashboard")
    st.markdown("---")

    # Sidebar configuration
    st.sidebar.header("⚙️ Model Configuration")

    prompt = st.sidebar.text_area(
        "Enter your prompt:", value="The capital of Japan is", height=100
    )

    # Autoregressive settings
    st.sidebar.subheader("🔄 Autoregressive (GPT-2) Settings")
    ar_max_length = st.sidebar.slider("Max Length", 10, 100, 30)
    ar_temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7, step=0.1)
    ar_top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.9, step=0.1)

    # Diffusion settings
    st.sidebar.subheader("🌊 Diffusion (BERT) Settings")
    diff_target_length = st.sidebar.slider("Target Length", 3, 20, 8)
    diff_steps = st.sidebar.slider("Diffusion Steps", 3, 15, 8)
    diff_temperature = st.sidebar.slider(
        "Diffusion Temperature", 0.1, 2.0, 0.7, step=0.1
    )

    # Initialize models (with caching)
    @st.cache_resource
    def load_models():
        with st.spinner("Loading models..."):
            ar_model = AutoregressiveModel()
            diff_model = DiffusionModel()
        return ar_model, diff_model

    ar_model, diff_model = load_models()

    if st.button("🚀 Generate and Compare", type="primary"):
        # Create columns for side-by-side comparison
        col1, col2 = st.columns(2)

        with col1:
            st.header("🔄 Autoregressive Model (GPT-2)")
            with st.spinner("Generating with GPT-2..."):
                ar_result, ar_steps, ar_metrics = ar_model.generate(
                    prompt, ar_max_length, ar_temperature, ar_top_p
                )

            st.subheader("📝 Generated Text")
            st.text_area("Output", ar_result, height=150, key="ar_output")

            st.subheader("⚡ Performance Metrics")
            ar_metrics_dict = ar_metrics.get_metrics()
            metrics_df = pd.DataFrame([ar_metrics_dict]).T
            metrics_df.columns = ["Value"]
            st.dataframe(metrics_df)

        with col2:
            st.header("🌊 Diffusion Model (BERT)")
            with st.spinner("Generating with BERT Diffusion..."):
                diff_result, diff_steps, diff_metrics = diff_model.generate(
                    prompt, diff_target_length, diff_steps, diff_temperature
                )

            st.subheader("📝 Generated Text")
            st.text_area("Output", diff_result, height=150, key="diff_output")

            st.subheader("⚡ Performance Metrics")
            diff_metrics_dict = diff_metrics.get_metrics()
            metrics_df = pd.DataFrame([diff_metrics_dict]).T
            metrics_df.columns = ["Value"]
            st.dataframe(metrics_df)

        # Comparison Charts
        st.markdown("---")
        st.header("📊 Detailed Comparison & Analytics")

        # Metrics comparison
        comparison_data = {
            "Metric": [
                "Total Time (s)",
                "Time per Token (s)",
                "Tokens/Second",
                "Memory Used (MB)",
                "Steps Taken",
            ],
            "Autoregressive": [
                ar_metrics_dict["total_time"],
                ar_metrics_dict["time_per_token"],
                ar_metrics_dict["tokens_per_second"],
                ar_metrics_dict["memory_used"],
                ar_metrics_dict["steps_taken"],
            ],
            "Diffusion": [
                diff_metrics_dict["total_time"],
                diff_metrics_dict["time_per_token"],
                diff_metrics_dict["tokens_per_second"],
                diff_metrics_dict["memory_used"],
                diff_metrics_dict["steps_taken"],
            ],
        }

        comparison_df = pd.DataFrame(comparison_data)

        # Create comparison charts
        col1, col2 = st.columns(2)

        with col1:
            # Time comparison
            fig_time = px.bar(
                comparison_df,
                x="Metric",
                y=["Autoregressive", "Diffusion"],
                title="⏱️ Performance Comparison",
                barmode="group",
            )
            fig_time.update_layout(height=400)
            st.plotly_chart(fig_time, use_container_width=True)

        with col2:
            # Generation speed
            speed_data = pd.DataFrame(
                {
                    "Model": ["Autoregressive", "Diffusion"],
                    "Tokens/Second": [
                        ar_metrics_dict["tokens_per_second"],
                        diff_metrics_dict["tokens_per_second"],
                    ],
                    "Total Time": [
                        ar_metrics_dict["total_time"],
                        diff_metrics_dict["total_time"],
                    ],
                }
            )

            fig_speed = px.scatter(
                speed_data,
                x="Total Time",
                y="Tokens/Second",
                color="Model",
                size=[20, 20],
                title="🚀 Speed vs Time Trade-off",
            )
            fig_speed.update_layout(height=400)
            st.plotly_chart(fig_speed, use_container_width=True)

        # Step-by-step generation comparison
        st.subheader("🔍 Step-by-Step Generation Process")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Autoregressive Steps:**")
            for i, step in enumerate(ar_steps[:10]):  # Show first 10 steps
                st.text(f"Step {i}: {step[:100]}...")

        with col2:
            st.write("**Diffusion Steps:**")
            for i, step in enumerate(diff_steps):
                st.text(f"Step {i}: {step[:100]}...")

        # Detailed analysis
        st.markdown("---")
        st.header("🔬 Detailed Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Time Difference",
                f"{abs(ar_metrics_dict['total_time'] - diff_metrics_dict['total_time']):.3f}s",
                f"{'AR faster' if ar_metrics_dict['total_time'] < diff_metrics_dict['total_time'] else 'Diffusion faster'}",
            )

        with col2:
            st.metric(
                "Memory Efficiency",
                f"{abs(ar_metrics_dict['memory_used'] - diff_metrics_dict['memory_used']):.1f}MB",
                f"{'AR efficient' if ar_metrics_dict['memory_used'] < diff_metrics_dict['memory_used'] else 'Diffusion efficient'}",
            )

        with col3:
            st.metric(
                "Generation Speed",
                f"{max(ar_metrics_dict['tokens_per_second'], diff_metrics_dict['tokens_per_second']):.2f} tok/s",
                f"{'AR faster' if ar_metrics_dict['tokens_per_second'] > diff_metrics_dict['tokens_per_second'] else 'Diffusion faster'}",
            )

        # Key differences explanation
        st.markdown("---")
        st.header("📚 Key Differences Explained")

        differences_data = {
            "Aspect": [
                "Generation Method",
                "Attention Type",
                "Parallelization",
                "Best Use Cases",
                "Training Objective",
                "Context Handling",
            ],
            "Autoregressive (GPT-2)": [
                "Sequential (left-to-right)",
                "Causal (can't see future)",
                "Limited (one token at a time)",
                "Text completion, dialogue",
                "Next token prediction",
                "Previous tokens only",
            ],
            "Diffusion (BERT)": [
                "Parallel (all positions)",
                "Bidirectional (full context)",
                "High (all positions together)",
                "Controllable generation, editing",
                "Masked token prediction",
                "Full sequence context",
            ],
        }

        diff_df = pd.DataFrame(differences_data)
        st.table(diff_df)


if __name__ == "__main__":
    create_comparison_dashboard()
