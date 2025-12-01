# freeze_env.py
"""
Utility script to lock the currently installed environment.
Creates: requirements.lock.txt
"""

import os
import subprocess
import sys
from datetime import datetime

ESSENTIALS = [
    "docling",
    "openai",
    "faiss-cpu",
    "plotly",
    "matplotlib",
    "pandas",
    "PyYAML",
    "rich",
    "numpy",
]
import faiss
def main():
    print("🔍 Checking installed packages...\n")
    missing = []
    for pkg in ESSENTIALS:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg} not installed.")
            missing.append(pkg)

    if missing:
        print("\n⚠️ Missing packages detected:")
        for m in missing:
            print(f"  → try: pip install {m}")
        sys.exit(1)

    # Generate lock file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lock_path = "requirements.lock.txt"

    print(f"\n📦 Generating {lock_path} ...")
    result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(f"# Locked environment snapshot\n# Generated: {timestamp}\n\n")
        f.write(result.stdout)

    print(f"\n✅ Environment frozen successfully → {lock_path}")
    print("You can reinstall this exact environment anytime with:")
    print("    pip install -r requirements.lock.txt")

if __name__ == "__main__":
    main()
