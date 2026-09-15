"""Convenience runner for the upgraded Phase-3 implementation."""
import subprocess
import sys

subprocess.run([sys.executable, "train_masac.py", "--episodes", "40", "--max-tasks", "15000", "--save", "outputs/masac.pt"], check=True)
subprocess.run([sys.executable, "evaluate_masac.py", "--max-tasks", "10000", "--model", "outputs/masac.pt"], check=True)
