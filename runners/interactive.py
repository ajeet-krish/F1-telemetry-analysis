"""Runner for interactive visualization asset generation.

Generates Plotly JSON assets for the Synchronized Telemetry Explorer
and 3D Performance Envelope visualizations.

Usage:
    uv run python -m runners.interactive
"""

import numpy as np
from src.core.telemetry import TelemetryLoader
from src.viz.interactive import (
    get_interactive_track_map,
    get_telemetry_traces,
    get_performance_envelope_3d,
)

ASSET_DIR = "docs/assets/data"


def run_all():
    print("Generating interactive visualization assets...")

    loader = TelemetryLoader(2024, "Monaco", "R")
    lap = loader.fastest_lap("VER")
    tel = lap.get_telemetry()

    track_json = get_interactive_track_map(tel)
    with open(f"{ASSET_DIR}/track_map.json", "w") as f:
        f.write(track_json)
    print("  Saved track_map.json")

    traces_json = get_telemetry_traces(tel)
    with open(f"{ASSET_DIR}/telemetry_traces.json", "w") as f:
        f.write(traces_json)
    print("  Saved telemetry_traces.json")

    envelope_data = {}
    for d in ["VER", "LEC", "HAM"]:
        try:
            loader = TelemetryLoader(2024, "Monaco", "R")
            lap = loader.fastest_lap(d)
            tel = lap.get_telemetry()
            envelope_data[f"{d} - Monaco"] = tel
        except Exception as e:
            print(f"  Skipping {d}: {e}")

    envelope_json = get_performance_envelope_3d(envelope_data)
    with open(f"{ASSET_DIR}/performance_envelope.json", "w") as f:
        f.write(envelope_json)
    print("  Saved performance_envelope.json")

    print("Done.")


if __name__ == "__main__":
    run_all()
