"""Run all analysis modules sequentially.

Usage:
    uv run python -m runners.all
"""

from src.analysis import downforce, ride_height, drs, track_setups, cornering, strategy

MODULES = [
    ("Downforce", downforce),
    ("Ride Height", ride_height),
    ("DRS & Active Aero", drs),
    ("Track Setups", track_setups),
    ("Cornering", cornering),
    ("Strategy", strategy),
]


def run_all():
    for name, mod in MODULES:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        mod.run_all()


if __name__ == "__main__":
    run_all()
