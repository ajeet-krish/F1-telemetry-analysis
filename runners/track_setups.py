"""Runner for track setup analysis.

Usage:
    uv run python -m runners.track_setups
"""

from src.analysis.track_setups import run_all

if __name__ == "__main__":
    run_all()
