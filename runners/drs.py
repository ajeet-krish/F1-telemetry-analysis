"""Runner for DRS and active aero analysis.

Usage:
    uv run python -m runners.drs
"""

from src.analysis.drs import run_all

if __name__ == "__main__":
    run_all()
