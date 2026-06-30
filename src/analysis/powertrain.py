"""
Powertrain and aero interaction analysis for F1 aerodynamics.

Maps real telemetry to visualise the boundary between drag-limited and
power-limited regimes using v^2 (proportional to aero drag) vs RPM.

Generates:
  - v^2 vs RPM scatter colored by gear (Monaco and Monza)
  - Drag-limited vs power-limited regime markers
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.core.style import (
    set_f1_style,
    MERCEDES_TEAL,
    MERCEDES_DARK,
    MERCEDES_CARD,
    MERCEDES_GRAY,
    MERCEDES_WHITE,
    MERCEDES_RED,
)
from src.core.telemetry import TelemetryLoader

set_f1_style()

ASSET_DIR = Path("docs/assets/images")
ASSET_DIR.mkdir(parents=True, exist_ok=True)

GEAR_COLORS = {
    1: "#E94560",
    2: "#F97316",
    3: "#FFB347",
    4: "#4ECDC4",
    5: "#00D2BE",
    6: "#3B82F6",
    7: "#A855F7",
    8: "#22C55E",
}


def _load_speed_trace(circuit: str, year: int = 2024, driver: str = "VER"):
    """Load telemetry for a given circuit and return speed trace DataFrame."""
    loader = TelemetryLoader(year, circuit, "R")
    return loader.lap_telemetry(driver)


def v_squared_vs_rpm(telemetry, circuit_name: str, speed_limit: float = 30, save: bool = True):
    """Scatter plot of v^2 (drag proxy) vs RPM, colored by gear.

    v^2 is proportional to aerodynamic drag and downforce. This plot reveals
    where the car is drag-limited (high v^2, RPM struggles to rise) vs
    power-limited (RPM rises freely, limited by gearing/traction).

    Low gears (1-3): traction-limited, RPM climbs freely but v^2 capped.
    Mid gears (4-5): transition zone, both RPM and v^2 increase.
    High gears (6-8): drag-limited, v^2 is high and RPM flattens.
    """
    tel = telemetry.copy()
    speed_ms = tel["Speed"] / 3.6
    mask = speed_ms > speed_limit / 3.6
    tel = tel[mask].copy()

    v_sq = tel["Speed"].values ** 2
    rpm = tel["RPM"].values
    gear_col = "nGear" if "nGear" in tel.columns else "Gear"
    gears = tel[gear_col].values

    fig, ax = plt.subplots(figsize=(12, 7))

    for gear in sorted(set(g for g in gears if g >= 1)):
        gmask = gears == gear
        if gmask.sum() < 5:
            continue
        color = GEAR_COLORS.get(int(gear), MERCEDES_GRAY)
        label = f"Gear {int(gear)}" if gear == int(gear) else f"Gear {gear:.0f}"
        ax.scatter(
            v_sq[gmask], rpm[gmask],
            c=color, s=12, alpha=0.5, edgecolors="none",
            label=label,
        )

    ax.set_xlabel("v$^2$ (km$^2$/h$^2$)")
    ax.set_ylabel("Engine RPM")
    ax.set_title(f"v$^2$ vs RPM -- {circuit_name} VER Fastest Lap")
    ax.legend(
        framealpha=0.9, fontsize=8, loc="upper left",
        ncol=2, title="Gear",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    v_max = v_sq.max()
    rpm_max = rpm.max()
    ax.annotate(
        "Drag-limited\n(high v$^2$, RPM capped)",
        xy=(v_max * 0.85, rpm_max * 0.55),
        fontsize=9, color=MERCEDES_RED, ha="center",
        style="italic", alpha=0.8,
    )
    ax.annotate(
        "Power/traction-limited\n(RPM rises freely)",
        xy=(v_max * 0.2, rpm_max * 0.85),
        fontsize=9, color=MERCEDES_TEAL, ha="center",
        style="italic", alpha=0.8,
    )

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    fig.tight_layout()

    if save:
        fname = f"powertrain_vsq_vs_rpm_{circuit_name.lower().replace(' ', '_')}.png"
        path = ASSET_DIR / fname
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def run_all(save: bool = True):
    """Generate all powertrain analysis visuals."""
    print("Generating powertrain analysis visuals...")

    print("  Loading Monza telemetry...")
    try:
        monza_tel = _load_speed_trace("Monza")
        v_squared_vs_rpm(monza_tel, "Monza", save=save)
    except Exception as e:
        print(f"  WARNING: Monza failed ({e})")

    print("  Loading Monaco telemetry...")
    try:
        monaco_tel = _load_speed_trace("Monaco")
        v_squared_vs_rpm(monaco_tel, "Monaco", save=save)
    except Exception as e:
        print(f"  WARNING: Monaco failed ({e})")

    print("Done. Files saved to docs/assets/images/")


if __name__ == "__main__":
    run_all()
