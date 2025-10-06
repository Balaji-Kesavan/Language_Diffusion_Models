#!/usr/bin/env python3
"""
Simple launcher for the Language Model Comparison Dashboard
"""

import subprocess
import sys
import os


def launch_dashboard():
    """Launch the Streamlit dashboard"""

    print("🚀 Starting Language Model Comparison Dashboard...")
    print("📊 This will open in your web browser automatically")
    print("⏱️  Loading models may take a minute...")
    print("🔄 Dashboard will show Autoregressive vs Diffusion model comparison")
    print("-" * 60)

    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(current_dir, "advanced_dashboard.py")

    # Launch streamlit
    try:
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            dashboard_path,
            "--server.port",
            "8501",
            "--server.address",
            "localhost",
            "--browser.gatherUsageStats",
            "false",
        ]

        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("\n💡 Try running manually with:")
        print(f"   streamlit run {dashboard_path}")


if __name__ == "__main__":
    launch_dashboard()
