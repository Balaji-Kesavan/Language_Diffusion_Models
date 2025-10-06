#!/usr/bin/env python3
"""
Quick test of the dashboard components
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_dashboard import AutoregressiveModel, DiffusionModel, PerformanceMetrics


def test_models():
    """Test both models quickly"""

    print("🧪 Testing Language Model Comparison Components")
    print("=" * 50)

    # Test prompt
    prompt = "The capital of Japan is"

    print(f"📝 Prompt: '{prompt}'")
    print("\n🔄 Testing Autoregressive Model (GPT-2)...")

    try:
        # Test autoregressive model
        ar_model = AutoregressiveModel()
        ar_result, ar_steps, ar_metrics = ar_model.generate(
            prompt, max_length=25, temperature=0.7, top_p=0.9
        )
        ar_metrics_dict = ar_metrics.get_metrics()

        print(f"✅ Autoregressive Result: {ar_result}")
        print(f"⏱️  Time: {ar_metrics_dict['total_time']:.3f}s")
        print(f"🚀 Speed: {ar_metrics_dict['tokens_per_second']:.2f} tokens/sec")
        print(f"📊 Memory: {ar_metrics_dict['memory_used']:.1f} MB")

    except Exception as e:
        print(f"❌ Autoregressive model error: {e}")
        return False

    print("\n🌊 Testing Diffusion Model (BERT)...")

    try:
        # Test diffusion model
        diff_model = DiffusionModel()
        diff_result, diff_steps, diff_metrics = diff_model.generate(
            prompt, target_length=5, num_steps=6, temperature=0.7
        )
        diff_metrics_dict = diff_metrics.get_metrics()

        print(f"✅ Diffusion Result: {diff_result}")
        print(f"⏱️  Time: {diff_metrics_dict['total_time']:.3f}s")
        print(f"🚀 Speed: {diff_metrics_dict['tokens_per_second']:.2f} tokens/sec")
        print(f"📊 Memory: {diff_metrics_dict['memory_used']:.1f} MB")

    except Exception as e:
        print(f"❌ Diffusion model error: {e}")
        return False

    print("\n📊 Comparison Summary:")
    print(
        f"⚡ Speed Winner: {'Autoregressive' if ar_metrics_dict['tokens_per_second'] > diff_metrics_dict['tokens_per_second'] else 'Diffusion'}"
    )
    print(
        f"⏱️  Time Winner: {'Autoregressive' if ar_metrics_dict['total_time'] < diff_metrics_dict['total_time'] else 'Diffusion'}"
    )
    print(
        f"💾 Memory Winner: {'Autoregressive' if ar_metrics_dict['memory_used'] < diff_metrics_dict['memory_used'] else 'Diffusion'}"
    )

    print("\n✅ All components working! Dashboard ready to launch.")
    return True


if __name__ == "__main__":
    if test_models():
        print("\n🚀 To launch the full dashboard, run:")
        print("   python launch_dashboard.py")
        print("\n📱 Or directly with streamlit:")
        print("   streamlit run advanced_dashboard.py")
