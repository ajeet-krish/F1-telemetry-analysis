"""Run all analysis modules + interactive + CFD export sequentially.

Usage:
    uv run python -m runners.all                           # full pipeline
    uv run python -m runners.all --cfd                     # include CFD sweep export
    uv run python -m runners.all --skip-interactive        # skip Plotly JSON generation
"""

import sys
from src.analysis import downforce, ride_height, drs, track_setups, cornering, strategy, powertrain
from src.cfd.venturi import export_all_visuals
from runners import interactive

MODULES = [
    ("Downforce", downforce),
    ("Ride Height", ride_height),
    ("DRS & Active Aero", drs),
    ("Track Setups", track_setups),
    ("Cornering", cornering),
    ("Strategy", strategy),
    ("Powertrain & Aero", powertrain),
]


def run_all():
    skip_interactive = "--skip-interactive" in sys.argv
    run_cfd = "--cfd" in sys.argv

    for name, mod in MODULES:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        mod.run_all()

    if not skip_interactive:
        print(f"\n{'=' * 60}")
        print("  Interactive Assets")
        print(f"{'=' * 60}")
        interactive.run_all()

    if run_cfd:
        print(f"\n{'=' * 60}")
        print("  CFD PyVista Export")
        print(f"{'=' * 60}")
        export_all_visuals()

    print("\nAll done.")


if __name__ == "__main__":
    run_all()
