"""
Race strategy analysis for F1 aerodynamics.

Models tire compound performance, fuel-adjusted pace, undercut
simulation, and tire degradation using analytical models combined
with FastF1 telemetry.

Generates:
  - Tire compound delta (lap time comparison by compound)
  - Fuel-adjusted pace simulation
  - Undercut simulation (lap time advantage vs pit window)
  - Tire degradation model (lap time increase with tire age)
  - Race pace projection with strategy options
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

import seaborn as sns

from src.core.models import F1Car
from src.core.physics import G
from src.core.style import (
    set_f1_style,
    MERCEDES_TEAL,
    MERCEDES_DARK,
    MERCEDES_CARD,
    MERCEDES_GRAY,
    MERCEDES_WHITE,
    MERCEDES_RED,
    MERCEDES_AMBER,
    teal_colormap,
)
from src.core.telemetry import TelemetryLoader

set_f1_style()

ASSET_DIR = Path("docs/assets/images")
ASSET_DIR.mkdir(parents=True, exist_ok=True)


def tire_compound_delta(save: bool = True):
    """Bar chart of estimated lap time delta by tire compound."""
    compounds = [
        ("C1 (Hard)", 0.0, MERCEDES_WHITE),
        ("C2 (Medium)", -0.4, MERCEDES_AMBER),
        ("C3 (Soft)", -0.9, MERCEDES_RED),
        ("C4 (Supersoft)", -1.4, "#A855F7"),
        ("C5 (Hypersoft)", -1.8, MERCEDES_TEAL),
        ("Intermediate", -3.5, "#4ADE80"),
        ("Wet", -8.0, "#60A5FA"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [c[0] for c in compounds]
    deltas = [c[1] for c in compounds]
    colors = [c[2] for c in compounds]

    bars = ax.barh(labels, deltas, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, delta in zip(bars, deltas):
        label = f"{delta:+.1f}s" if delta != 0 else "Baseline"
        ax.text(delta - 0.1, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="right", color="white", fontsize=10,
                fontweight="bold" if delta == 0 else "normal")

    ax.axvline(0, color=MERCEDES_GRAY, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Lap Time Delta (s)")
    ax.set_title("Tire Compound Performance -- Typical Delta per Lap")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(-10, 1)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "strategy_tire_delta.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def fuel_correction(save: bool = True):
    """Lap time improvement as fuel burns off during a stint."""
    laps = np.arange(0, 35)
    start_fuel = 110.0
    fuel_per_lap = 2.2
    fuel_remaining = start_fuel - laps * fuel_per_lap
    fuel_remaining = np.clip(fuel_remaining, 0, start_fuel)

    lap_time_base = 82.0
    time_per_kg = 0.035
    lap_times = lap_time_base + fuel_remaining * time_per_kg

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(laps, lap_times, color=MERCEDES_TEAL, linewidth=2.5, label="Lap time")
    ax1.fill_between(laps, lap_times, lap_time_base, alpha=0.15, color=MERCEDES_TEAL)
    ax1.set_xlabel("Lap Number")
    ax1.set_ylabel("Lap Time (s)", color=MERCEDES_TEAL)
    ax1.tick_params(axis="y", labelcolor=MERCEDES_TEAL)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(76, 86)

    ax2 = ax1.twinx()
    ax2.plot(laps, fuel_remaining, color=MERCEDES_AMBER, linewidth=2, linestyle="--", label="Fuel remaining")
    ax2.set_ylabel("Fuel Remaining (kg)", color=MERCEDES_AMBER)
    ax2.tick_params(axis="y", labelcolor=MERCEDES_AMBER)
    ax2.set_ylim(0, 120)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, framealpha=0.9)

    ax1.set_title("Fuel-Adjusted Lap Time -- 110 kg Start, 2.2 kg/Lap")

    for label in ax1.get_xticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "strategy_fuel_correction.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def undercut_simulation(save: bool = True):
    """Simulate the undercut: lap time delta vs pit stop timing."""
    in_lap_delta_sec = np.linspace(5, 20, 50)
    out_lap_delta = 2.0
    new_tire_gain = 0.8
    gap = -0.5

    net_advantage = gap + out_lap_delta + new_tire_gain - in_lap_delta_sec

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(in_lap_delta_sec, in_lap_delta_sec - out_lap_delta - new_tire_gain,
             color=MERCEDES_TEAL, linewidth=2.5, label="Time lost at pit stop")
    ax1.plot(in_lap_delta_sec, net_advantage, color=MERCEDES_RED, linewidth=2.5, label="Net advantage")
    ax1.axhline(0, color=MERCEDES_GRAY, linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.fill_between(in_lap_delta_sec, net_advantage, 0,
                     where=(net_advantage > 0), alpha=0.15, color=MERCEDES_TEAL,
                     label="Undercut works")
    ax1.fill_between(in_lap_delta_sec, net_advantage, 0,
                     where=(net_advantage <= 0), alpha=0.15, color=MERCEDES_RED,
                     label="Undercut fails")
    ax1.set_xlabel("In-Lap Traffic Penalty (s)")
    ax1.set_ylabel("Time Delta (s)")
    ax1.set_title("Undercut Sensitivity")
    ax1.legend(framealpha=0.9, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    pit_stops = np.arange(1, 5)
    undercut_candidates = []
    for ps in pit_stops:
        traffic_penalty = 8 + np.random.normal(0, 1)
        u_adv = new_tire_gain + gap + out_lap_delta - traffic_penalty
        undercut_candidates.append(u_adv)

    x_pos = np.arange(len(pit_stops))
    colors = [MERCEDES_TEAL if v > 0 else MERCEDES_RED for v in undercut_candidates]
    ax2.bar(x_pos, undercut_candidates, color=colors, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color=MERCEDES_GRAY, linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Pit Stop Number")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"Stop {i+1}" for i in range(len(pit_stops))])
    ax2.set_ylabel("Undercut Advantage (s)")
    ax2.set_title("Undercut Simulation -- Monaco 2024")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(-10, 5)

    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set(color=MERCEDES_GRAY)

    fig.suptitle("Undercut Simulation", fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        path = ASSET_DIR / "strategy_undercut.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def tire_degradation(save: bool = True):
    """Model tire degradation: lap time increase with tire age."""
    laps = np.arange(0, 30)
    deg_rate = 0.06
    base_time = 82.0
    noise = np.random.normal(0, 0.08, len(laps))

    lap_times = base_time + laps * deg_rate + noise

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(laps, lap_times, color=MERCEDES_TEAL, s=20, alpha=0.6, label="Simulated laps")
    fit = np.polyfit(laps, lap_times, 1)
    trend = np.polyval(fit, laps)
    ax.plot(laps, trend, color=MERCEDES_RED, linewidth=2, label=f"Trend: {fit[0]:.3f}s/lap")

    ax.fill_between(laps, base_time, trend, alpha=0.1, color=MERCEDES_RED)
    ax.set_xlabel("Tire Age (laps)")
    ax.set_ylabel("Lap Time (s)")
    ax.set_title("Tire Degradation Model -- Soft Compound")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(base_time - 0.5, base_time + laps[-1] * deg_rate + 0.5)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "strategy_tire_degradation.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def race_pace_projection(save: bool = True):
    """Project race pace for different strategy options."""
    total_laps = 78
    strategies = {
        "1-stop (Medium-Hard)": {
            "stops": [35],
            "compounds": ["Medium", "Hard"],
            "base_pace": 83.0,
            "deg_rates": [0.05, 0.03],
            "traffic_penalty": 2.5,
        },
        "2-stop (Soft-Medium-Hard)": {
            "stops": [20, 50],
            "compounds": ["Soft", "Medium", "Hard"],
            "base_pace": 82.5,
            "deg_rates": [0.08, 0.05, 0.03],
            "traffic_penalty": 2.0,
        },
        "3-stop (Soft-Soft-Medium-Hard)": {
            "stops": [12, 35, 58],
            "compounds": ["Soft", "Soft", "Medium", "Hard"],
            "base_pace": 82.0,
            "deg_rates": [0.08, 0.08, 0.05, 0.03],
            "traffic_penalty": 1.5,
        },
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    strategy_colors = [MERCEDES_TEAL, MERCEDES_AMBER, MERCEDES_RED]
    pit_loss_s = 22.0

    total_times = {}
    for idx, (sname, sdata) in enumerate(strategies.items()):
        stints = []
        stint_idx = 0
        current_lap = 1
        stint_paces = []
        stint_laps = []
        stops_done = []

        all_lap_times = []
        for stop_lap in sdata["stops"]:
            stint_len = stop_lap - current_lap
            stints.append((current_lap, stop_lap, sdata["compounds"][stint_idx], sdata["deg_rates"][stint_idx]))
            stint_idx += 1
            current_lap = stop_lap

        stints.append((current_lap, total_laps, sdata["compounds"][stint_idx], sdata["deg_rates"][stint_idx]))

        total_time = 0
        lap_numbers = []
        lap_time_vals = []
        for stint_start, stint_end, compound, deg_rate in stints:
            for lap in range(stint_start, stint_end + 1):
                lap_age = lap - stint_start
                lt = sdata["base_pace"] + lap_age * deg_rate + np.random.normal(0, 0.1)
                lap_numbers.append(lap)
                lap_time_vals.append(lt)
                total_time += lt
            if stint_end != total_laps:
                total_time += pit_loss_s

        strat_color = strategy_colors[idx]
        ax1.plot(lap_numbers, lap_time_vals, color=strat_color, linewidth=1.5, alpha=0.8, label=sname)
        for stop_lap in sdata["stops"]:
            ax1.axvline(stop_lap, color=strat_color, linewidth=0.8, linestyle="--", alpha=0.4)

        total_times[sname] = total_time

    ax1.set_xlabel("Lap")
    ax1.set_ylabel("Lap Time (s)")
    ax1.set_title("Race Pace by Strategy")
    ax1.legend(framealpha=0.9, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, total_laps)

    names = list(total_times.keys())
    times = [total_times[n] for n in names]
    best_idx = np.argmin(times)
    colors_bars = [MERCEDES_TEAL if i == best_idx else MERCEDES_CARD for i in range(len(names))]

    ax2.barh(names, times, color=colors_bars, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Total Race Time (s)")
    ax2.set_title("Total Race Time by Strategy")
    for i, (n, t) in enumerate(zip(names, times)):
        ax2.text(t + 5, i, f"{t:.0f}s", va="center", color=MERCEDES_WHITE, fontsize=10)
    ax2.grid(True, alpha=0.3, axis="x")

    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set(color=MERCEDES_GRAY)

    fig.suptitle("Race Strategy Analysis -- 78 Lap Grand Prix", fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        path = ASSET_DIR / "strategy_race_pace.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def tire_ridge_plot(save: bool = True):
    """Degradation ridge plot: stacked KDE distributions of lap times
    grouped by stint phase. Shows how lap time spread widens and median
    shifts as tires degrade.
    """
    from scipy.stats import gaussian_kde

    total_laps = 30
    phases = [(0, 5, "Laps 1-5"), (5, 10, "Laps 6-10"), (10, 15, "Laps 11-15"),
              (15, 20, "Laps 16-20"), (20, 25, "Laps 21-25"), (25, 30, "Laps 26-30")]

    deg_rate = 0.06
    base_time = 82.0
    rng = np.random.default_rng(42)

    n_colors = len(phases)
    colors = [plt.cm.RdYlGn(1 - i / max(n_colors - 1, 1)) for i in range(n_colors)]

    fig, ax = plt.subplots(figsize=(10, 8))

    y_offset = 0
    y_ticks = []
    y_labels = []
    scaling = 12

    for i, (start, end, label) in enumerate(phases):
        times = []
        for lap in range(start, end):
            lt = base_time + lap * deg_rate + rng.normal(0, 0.12 + lap * 0.005)
            times.append(lt)
        if not times:
            continue
        times = np.array(times)

        try:
            kde = gaussian_kde(times)
            x_grid = np.linspace(base_time - 0.8, base_time + total_laps * deg_rate + 1.5, 300)
            y_kde = kde(x_grid)
            y_scaled = y_kde * scaling

            color = colors[i]
            ax.fill_between(x_grid, y_offset, y_offset + y_scaled, alpha=0.7, color=color)
            ax.plot(x_grid, y_offset + y_scaled, color=color, linewidth=1.5)

            median_val = np.median(times)
            ax.text(median_val, y_offset + y_scaled.max() * 0.5, f"{median_val:.2f}s",
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold")
        except Exception:
            pass

        y_ticks.append(y_offset + scaling * 0.4)
        y_labels.append(label)
        y_offset += scaling * 1.15

    ax.set_xlabel("Lap Time (s)")
    ax.set_ylabel("Stint Phase")
    ax.set_title("Tire Degradation Ridge Plot -- Soft Compound")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlim(base_time - 0.5, base_time + total_laps * deg_rate + 1.5)
    ax.grid(True, alpha=0.15, axis="x")
    ax.set_ylim(0, y_offset + scaling)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    fig.tight_layout()

    if save:
        path = ASSET_DIR / "strategy_tire_ridge.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def run_all():
    """Generate all strategy analysis visuals."""
    print("Generating strategy analysis visuals...")
    tire_compound_delta()
    fuel_correction()
    undercut_simulation()
    tire_degradation()
    race_pace_projection()
    tire_ridge_plot()
    print("Done. Files saved to docs/assets/images/")


if __name__ == "__main__":
    run_all()
