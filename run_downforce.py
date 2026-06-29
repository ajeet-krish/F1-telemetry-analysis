"""
Runner script for downforce analysis.

Generates all downforce analysis visuals and saves to docs/assets/images/.
Idempotent -- safe to run multiple times.

Usage:
    uv run python run_downforce.py
"""

from src.analysis.downforce import run_all

if __name__ == "__main__":
    run_all()
