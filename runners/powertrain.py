"""Runner for powertrain + aero analysis.

Usage:
    uv run python -m runners.powertrain
"""

from src.analysis.powertrain import run_all

if __name__ == "__main__":
    run_all()
