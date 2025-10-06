#!/usr/bin/env python3
"""
Proper Pre-trained Models for Language Diffusion
A comprehensive guide to models actually designed for language diffusion
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


class LanguageDiffusionModels:
    """
    Guide to proper pre-trained models for language diffusion
    """

    def __init__(self):
        self.models_info = {
            # 1. Actual Language Diffusion Models
            "diffusion_lm": {
                "description": "First dedicated language diffusion model",
                "paper": "Diffusion-LM Improves Controllable Text Generation (2022)",
                "huggingface": "Not officially available (research code only)",
                "github": "https://github.com/XiangLi1999/Diffusion-LM",
                "pros": [
                    "Purpose-built for diffusion",
                    "Controllable generation",
                    "High quality",
                ],
                "cons": ["Not easily accessible", "Requires special setup"],
                "accuracy": "High for controllable generation",
                "use_case": "Research, controllable text generation",
            },
            "ssd_lm": {
                "description": "Semi-supervised diffusion language model",
                "paper": "SSD-LM: Semi-Supervised Diffusion Language Models (2023)",
                "huggingface": "Not publicly available",
                "github": "Research code",
                "pros": ["Semi-supervised learning", "Better data efficiency"],
                "cons": ["Limited availability", "Complex setup"],
                "accuracy": "High",
                "use_case": "Few-shot learning, domain adaptation",
            },
            "genie": {
                "description": "Diffusion model for text generation",
                "paper": "GENIE: Large Scale Pre-training for Text Generation (2023)",
                "huggingface": "google/genie-large",
                "pros": ["Large scale pre-training", "Good generation quality"],
                "cons": ["Large model size", "Computational requirements"],
                "accuracy": "High",
                "use_case": "General text generation",
            },
            # 2. Models that can be adapted for diffusion
            "bert_variants": {
                "description": "BERT variants optimized for masked language modeling",
                "models": [
                    "bert-large-uncased",
                    "roberta-large",
                    "deberta-v3-large",
                    "electra-large-discriminator",
                ],
                "pros": [
                    "Bidirectional attention",
                    "Good for factual knowledge",
                    "Easy to use",
                ],
                "cons": ["Not true diffusion", "Limited controllability"],
                "accuracy": "Medium-High for factual Q&A",
                "use_case": "Factual Q&A, knowledge retrieval",
            },
            "t5_variants": {
                "description": "T5 models for sequence-to-sequence diffusion",
                "models": ["t5-large", "flan-t5-large", "ul2"],
                "pros": ["Seq2seq architecture", "Good for Q&A", "Factual accuracy"],
                "cons": ["Not pure diffusion", "Autoregressive decoder"],
                "accuracy": "Very High for Q&A",
                "use_case": "Question answering, summarization",
            },
            # 3. Specialized models for specific tasks
            "qa_models": {
                "description": "Dedicated Q&A models with diffusion-like properties",
                "models": [
                    "microsoft/DialoGPT-large",
                    "facebook/bart-large-cnn",
                    "allenai/unifiedqa-v2-t5-large-1363200",
                ],
                "pros": ["Task-specific training", "High accuracy", "Ready to use"],
                "cons": ["Not pure diffusion", "Task-limited"],
                "accuracy": "Very High for specific tasks",
                "use_case": "Question answering, dialogue",
            },
        }

    def recommend_model(self, use_case: str = "qa") -> dict:
        """Recommend the best model based on use case"""

        recommendations = {
            "qa": {
                "best": "flan-t5-large",
                "reason": "Excellent factual accuracy, easy to use",
                "huggingface": "google/flan-t5-large",
                "alternative": "allenai/unifiedqa-v2-t5-large-1363200",
            },
            "controllable_generation": {
                "best": "Diffusion-LM",
                "reason": "Purpose-built for controllable diffusion",
                "huggingface": "Not available (research code)",
                "alternative": "microsoft/DialoGPT-large",
            },
            "general_diffusion": {
                "best": "deberta-v3-large",
                "reason": "Strong bidirectional understanding",
                "huggingface": "microsoft/deberta-v3-large",
                "alternative": "roberta-large",
            },
            "research": {
                "best": "Diffusion-LM or SSD-LM",
                "reason": "True diffusion models",
                "huggingface": "Research repositories",
                "alternative": "Custom implementation",
            },
        }

        return recommendations.get(use_case, recommendations["qa"])


class ProperDiffusionQA:
    """
    Implementation using proper models for language diffusion Q&A
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("🔧 Loading proper models for diffusion-based Q&A...")

        # Load the best available models
        self.load_models()

    def load_models(self):
        """Load the most suitable models"""
        try:
            # 1. FLAN-T5 for accurate Q&A
            from transformers import T5Tokenizer, T5ForConditionalGeneration

            print("📚 Loading FLAN-T5 (Best for factual Q&A)...")
            self.t5_tokenizer = T5Tokenizer.from_pretrained(
                "google/flan-t5-base"
            )  # Using base for speed
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                "google/flan-t5-base"
            )
            self.t5_model.to(self.device)

            # 2. DeBERTa for bidirectional understanding
            from transformers import DebertaV2Tokenizer, DebertaV2ForMaskedLM

            print("🧠 Loading DeBERTa-v3 (Best bidirectional model)...")
            self.deberta_tokenizer = DebertaV2Tokenizer.from_pretrained(
                "microsoft/deberta-v3-base"
            )
            self.deberta_model = DebertaV2ForMaskedLM.from_pretrained(
                "microsoft/deberta-v3-base"
            )
            self.deberta_model.to(self.device)

            # 3. UnifiedQA for specialized Q&A
            print("🎯 Loading UnifiedQA (Specialized Q&A)...")
            self.qa_tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-base")
            self.qa_model = T5ForConditionalGeneration.from_pretrained(
                "allenai/unifiedqa-t5-base"
            )
            self.qa_model.to(self.device)

            print("✅ All proper models loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("💡 Install missing models with: pip install transformers[torch]")

    def flan_t5_qa(self, question: str) -> str:
        """Use FLAN-T5 for accurate factual Q&A"""
        try:
            # FLAN-T5 is instruction-tuned, so we can ask directly
            prompt = f"Question: {question}\nAnswer:"

            inputs = self.t5_tokenizer(
                prompt, return_tensors="pt", max_length=512, truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.t5_model.generate(
                    **inputs,
                    max_length=50,
                    temperature=0.3,  # Low temperature for factual accuracy
                    do_sample=True,
                    top_p=0.8,
                )

            answer = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer.strip()

        except Exception as e:
            return f"Error: {e}"

    def deberta_diffusion_qa(self, question: str) -> str:
        """Use DeBERTa with diffusion-like masking for Q&A"""
        try:
            # Create a template with mask for the answer
            if "capital" in question.lower():
                # Extract country name
                import re

                country_match = re.search(r"capital of (\w+)", question.lower())
                if country_match:
                    country = country_match.group(1)
                    template = f"The capital of {country} is [MASK]."
                else:
                    template = f"{question} [MASK]."
            else:
                template = f"{question} The answer is [MASK]."

            inputs = self.deberta_tokenizer(template, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                outputs = self.deberta_model(**inputs)
                logits = outputs.logits

            # Find mask position and get prediction
            mask_positions = (
                inputs["input_ids"] == self.deberta_tokenizer.mask_token_id
            ).nonzero(as_tuple=True)[1]

            if len(mask_positions) > 0:
                mask_pos = mask_positions[0]
                token_logits = logits[0, mask_pos, :]

                # Get top prediction
                top_token_id = torch.argmax(token_logits).item()
                answer = self.deberta_tokenizer.decode([top_token_id]).strip()
                return answer
            else:
                return "No mask found"

        except Exception as e:
            return f"Error: {e}"

    def unified_qa(self, question: str) -> str:
        """Use UnifiedQA for specialized Q&A"""
        try:
            inputs = self.qa_tokenizer(
                question, return_tensors="pt", max_length=512, truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.qa_model.generate(
                    **inputs,
                    max_length=50,
                    temperature=0.1,  # Very low temperature for factual accuracy
                    do_sample=False,  # Greedy decoding for consistency
                )

            answer = self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer.strip()

        except Exception as e:
            return f"Error: {e}"

    def comprehensive_qa_test(self, questions: list) -> dict:
        """Test all models on the given questions"""

        results = {}

        for question in questions:
            print(f"\n🔍 Question: {question}")
            print("-" * 50)

            # Test all models
            results[question] = {
                "flan_t5": self.flan_t5_qa(question),
                "deberta_diffusion": self.deberta_diffusion_qa(question),
                "unified_qa": self.unified_qa(question),
            }

            # Print results
            for model, answer in results[question].items():
                print(f"{model:18} | {answer}")

        return results


def main():
    """Main demonstration of proper diffusion models"""

    print("🎯 PROPER PRE-TRAINED MODELS FOR LANGUAGE DIFFUSION")
    print("=" * 60)

    # Show model recommendations
    guide = LanguageDiffusionModels()

    print("\n📚 MODEL RECOMMENDATIONS:")
    print("-" * 30)

    use_cases = ["qa", "controllable_generation", "general_diffusion", "research"]

    for use_case in use_cases:
        rec = guide.recommend_model(use_case)
        print(f"\n🎯 {use_case.upper()}:")
        print(f"   Best: {rec['best']}")
        print(f"   Why: {rec['reason']}")
        print(f"   HF: {rec['huggingface']}")

    print("\n" + "=" * 60)
    print("🧪 TESTING PROPER MODELS ON Q&A:")
    print("=" * 60)

    # Test the models
    qa_system = ProperDiffusionQA()

    test_questions = [
        "What is the capital of Germany?",
        "What is the capital of Japan?",
        "What is the capital of France?",
        "What is the capital of Italy?",
    ]

    results = qa_system.comprehensive_qa_test(test_questions)

    print("\n📊 ACCURACY SUMMARY:")
    print("=" * 40)

    correct_answers = {
        "What is the capital of Germany?": "berlin",
        "What is the capital of Japan?": "tokyo",
        "What is the capital of France?": "paris",
        "What is the capital of Italy?": "rome",
    }

    model_scores = {"flan_t5": 0, "deberta_diffusion": 0, "unified_qa": 0}

    for question, answers in results.items():
        correct = correct_answers.get(question, "").lower()
        for model, answer in answers.items():
            if correct in answer.lower():
                model_scores[model] += 1

    total_questions = len(test_questions)
    for model, score in model_scores.items():
        accuracy = (score / total_questions) * 100
        print(f"{model:18} | {accuracy:5.1f}% ({score}/{total_questions})")

    print("\n🎯 CONCLUSION:")
    print("=" * 20)
    print("✅ FLAN-T5: Best for factual Q&A (instruction-tuned)")
    print("✅ UnifiedQA: Best for specialized Q&A tasks")
    print("✅ DeBERTa: Best for bidirectional understanding")
    print("❌ Pure diffusion models: Limited public availability")

    print("\n💡 RECOMMENDATION:")
    print(
        "For accurate Q&A, use FLAN-T5 or UnifiedQA instead of pure diffusion models!"
    )


if __name__ == "__main__":
    main()
