"""
Track-specific aero setup analysis for F1 aerodynamics.

Compares high-downforce (Monaco) vs low-downforce (Monza) setups
using FastF1 telemetry data.

Generates:
  - Speed-on-track heatmap for both circuits
  - Gear distribution comparison
  - Sector speed analysis
  - DRS zone comparison
  - Aero setup parameter difference (modeled)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import pandas as pd

from src.core.models import F1Car
from src.core.physics import kmh_to_ms
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

ASSET_DIR = Path("docs/assets/images/track_setups")
ASSET_DIR.mkdir(parents=True, exist_ok=True)


def _load_track_data(year: int = 2024):
    """Load telemetry for Monaco (high DF) and Monza (low DF)."""
    monaco = TelemetryLoader(year, "Monaco", "R")
    monza = TelemetryLoader(year, "Monza", "R")
    mco_tel = monaco.lap_telemetry("VER")
    mza_tel = monza.lap_telemetry("VER")
    return monaco, monza, mco_tel, mza_tel


def speed_on_track_map(save: bool = True):
    """Speed-colored track maps for 4 circuits in a 2x2 layout."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.text(0.02, 0.98, "Telemetry-based", transform=fig.transFigure, fontsize=9, color="#00D2BE", alpha=0.7, va="top")

    tracks = [
        ("Monaco", 2024, "VER"),
        ("Monza", 2024, "VER"),
        ("Bahrain", 2024, "VER"),
        ("Silverstone", 2024, "VER"),
    ]

    cmap = plt.cm.plasma
    all_speeds = []

    track_data = []
    for name, year, driver in tracks:
        try:
            loader = TelemetryLoader(year, name, "R")
            lap = loader.fastest_lap(driver)
            tel = lap.get_telemetry()
            all_speeds.extend(tel["Speed"].values)
            track_data.append((name, tel["X"], tel["Y"], tel["Speed"]))
        except Exception as e:
            print(f"    Skipping {name}: {e}")
            track_data.append((name, None, None, None))

    vmin = min(all_speeds) if all_speeds else 0
    vmax = max(all_speeds) if all_speeds else 300
    norm = Normalize(vmin=vmin, vmax=vmax)

    for idx, (name, x, y, speed) in enumerate(track_data):
        ax = axes[idx // 2, idx % 2]
        if x is None:
            ax.text(0.5, 0.5, f"{name}\n(Data unavailable)", ha="center", va="center",
                    color=MERCEDES_GRAY, transform=ax.transAxes)
            ax.set_facecolor(MERCEDES_DARK)
            continue

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3, alpha=0.9)
        lc.set_array(speed.values)
        ax.add_collection(lc)

        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(f"{name} - VER Fastest Lap", color=MERCEDES_WHITE)
        ax.set_facecolor(MERCEDES_DARK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.02])
    cbar = fig.colorbar(lc, cax=cbar_ax, orientation="horizontal", label="Speed (km/h)")
    cbar.ax.xaxis.set_tick_params(color=MERCEDES_GRAY)
    plt.setp(plt.getp(cbar.ax, "xticklabels"), color=MERCEDES_GRAY)

    fig.suptitle("Speed on Track: Circuit Comparison", fontsize=16, y=0.98, color=MERCEDES_WHITE)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.95, wspace=0.05, hspace=0.15)

    if save:
        path = ASSET_DIR / "speed_on_track.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def gear_distribution(save: bool = True):
    """Gear usage histogram comparing Monaco vs Monza."""
    tracks = [
        ("Monaco", 2024, MERCEDES_TEAL),
        ("Monza", 2024, MERCEDES_RED),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.text(0.02, 0.98, "Telemetry-based", transform=fig.transFigure, fontsize=9, color="#00D2BE", alpha=0.7, va="top")
    bar_width = 0.35
    x = np.arange(1, 9)

    for i, (name, year, color) in enumerate(tracks):
        loader = TelemetryLoader(year, name, "R")
        tel = loader.lap_telemetry("VER")
        gear_col = "nGear" if "nGear" in tel.columns else "Gear"
        if gear_col not in tel.columns:
            continue
        gear_counts = tel[gear_col].value_counts().sort_index()
        gear_pct = gear_counts / gear_counts.sum() * 100
        all_gears = pd.Series(0.0, index=range(1, 9))
        for g, pct in gear_pct.items():
            if int(g) in all_gears.index:
                all_gears[int(g)] = pct
        offset = (i - 0.5) * bar_width
        ax.bar(x + offset, all_gears.values, bar_width, label=name, color=color, alpha=0.85)

    ax.set_xlabel("Gear")
    ax.set_ylabel("Usage (%)")
    ax.set_title("Gear Distribution: Monaco vs Monza")
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in x])
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "gear_distribution.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def sector_speed_comparison(save: bool = True):
    """Sector-by-sector speed comparison between circuits."""
    tracks = [
        ("Monaco", 2024, MERCEDES_TEAL, "o-"),
        ("Monza", 2024, MERCEDES_RED, "s--"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.text(0.02, 0.98, "Telemetry-based", transform=fig.transFigure, fontsize=9, color="#00D2BE", alpha=0.7, va="top")

    all_sectors = {}
    for name, year, color, marker in tracks:
        loader = TelemetryLoader(year, name, "R")
        lap = loader.fastest_lap("VER")
        tel = lap.get_telemetry()

        distance = tel["Distance"]
        speed = tel["Speed"]
        total_dist = distance.max()

        n_sectors = 5
        sector_boundaries = np.linspace(0, total_dist, n_sectors + 1)
        sector_speeds = []
        for i in range(n_sectors):
            mask = (distance >= sector_boundaries[i]) & (distance < sector_boundaries[i + 1])
            sector_speeds.append(speed[mask].mean())

        sector_labels = [f"S{i+1}" for i in range(n_sectors)]
        ax.plot(sector_labels, sector_speeds, marker, color=color, linewidth=2.5,
                markersize=8, label=name)
        all_sectors[name] = sector_speeds

    ax.set_xlabel("Sector")
    ax.set_ylabel("Average Speed (km/h)")
    ax.set_title("Sector Speed Profile: Monaco vs Monza")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "sector_speeds.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def aero_setup_comparison(save: bool = True):
    """Model aero setup differences between Monaco and Monza.

    Monaco: high downforce (steeper wing angles, lower ride height)
    Monza: low drag (shallower wing angles, higher ride height)
    """
    car = F1Car()
    speeds_kmh = np.linspace(60, 340, 80)
    speeds_ms = kmh_to_ms(speeds_kmh)

    monaco_df = []
    monaco_drag = []
    monza_df = []
    monza_drag = []

    for v in speeds_ms:
        mco = car.component_breakdown(v, front_alpha=-8.0, rear_alpha=-17.0, ride_height=0.035)
        mza = car.component_breakdown(v, front_alpha=-3.0, rear_alpha=-8.0, ride_height=0.065)
        monaco_df.append(mco["downforce"]["total"] / 1000)
        monaco_drag.append(mco["drag"]["total"] / 1000)
        monza_df.append(mza["downforce"]["total"] / 1000)
        monza_drag.append(mza["drag"]["total"] / 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(speeds_kmh, monaco_df, color=MERCEDES_TEAL, linewidth=2.5, label="Monaco (High DF)")
    ax1.plot(speeds_kmh, monza_df, color=MERCEDES_RED, linewidth=2.5, linestyle="--", label="Monza (Low DF)")
    ax1.fill_between(speeds_kmh, monaco_df, monza_df, alpha=0.15, color=MERCEDES_TEAL)
    ax1.set_xlabel("Speed (km/h)")
    ax1.set_ylabel("Downforce (kN)")
    ax1.set_title("Downforce: Monaco vs Monza Setup")
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(speeds_kmh, monaco_drag, color=MERCEDES_TEAL, linewidth=2.5, label="Monaco (High DF)")
    ax2.plot(speeds_kmh, monza_drag, color=MERCEDES_RED, linewidth=2.5, linestyle="--", label="Monza (Low DF)")
    ax2.fill_between(speeds_kmh, monaco_drag, monza_drag, alpha=0.15, color=MERCEDES_RED)
    ax2.set_xlabel("Speed (km/h)")
    ax2.set_ylabel("Drag (kN)")
    ax2.set_title("Drag: Monaco vs Monza Setup")
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set(color=MERCEDES_GRAY)

    fig.suptitle("Aero Setup: Monaco vs Monza", fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        path = ASSET_DIR / "aero_comparison.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def speed_profile_comparison(save: bool = True):
    """Normalized speed profile overlay for Monaco vs Monza."""
    tracks = [
        ("Monaco", 2024, MERCEDES_TEAL),
        ("Monza", 2024, MERCEDES_RED),
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.text(0.02, 0.98, "Telemetry-based", transform=fig.transFigure, fontsize=9, color="#00D2BE", alpha=0.7, va="top")

    for name, year, color in tracks:
        loader = TelemetryLoader(year, name, "R")
        tel = loader.lap_telemetry("VER")
        distance = tel["Distance"]
        speed = tel["Speed"]
        dist_norm = distance / distance.max() * 100
        ax.plot(dist_norm, speed, color=color, linewidth=1.5, alpha=0.8, label=name)

    ax.set_xlabel("Lap Progress (%)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Normalized Speed Profile: Monaco vs Monza")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "speed_profile.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def driver_time_delta_map(save: bool = True):
    """Track map colored by time delta between two drivers.

    Segments are red where Driver A is gaining time, blue where
    Driver B is gaining. Shows where aero setup differences matter --
    advantage on straights (low drag) vs in corners (high downforce).
    """
    year = 2024
    circuit = "Bahrain"
    driver_a = "VER"
    driver_b = "LEC"

    loader = TelemetryLoader(year, circuit, "R")
    tel_a = loader.lap_telemetry(driver_a)
    tel_b = loader.lap_telemetry(driver_b)

    distance_a = tel_a["Distance"].values
    distance_b = tel_b["Distance"].values
    speed_a = tel_a["Speed"].values
    speed_b = tel_b["Speed"].values
    x = tel_a["X"].values
    y = tel_a["Y"].values

    if "SessionTime" in tel_a.columns:
        st = tel_a["SessionTime"].values
        st_sec = st.astype(np.float64)
        if st_sec.max() > 1e6:
            st_sec = st_sec * 1e-9
        dt = np.diff(st_sec)
        dt = np.where(dt <= 0, 0.01, dt)
        dt = np.concatenate([[dt[0] + 1e-10], dt + 1e-10])
    else:
        dt = np.full(len(x), 0.01)

    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    speed_computed = np.sqrt(vx ** 2 + vy ** 2)

    SPEED_FLOOR = 40
    mask = speed_computed > SPEED_FLOOR / 3.6

    common_dist = np.linspace(0, min(distance_a.max(), distance_b.max()), 5000)
    speed_a_interp = np.interp(common_dist, distance_a, speed_a)
    speed_b_interp = np.interp(common_dist, distance_b, speed_b)

    time_a = np.where(speed_a_interp > 1, common_dist / speed_a_interp * 3.6, 0)
    time_b = np.where(speed_b_interp > 1, common_dist / speed_b_interp * 3.6, 0)
    time_delta = np.cumsum(time_a - time_b)

    delta_normalized = time_delta - time_delta.min()
    delta_normalized = delta_normalized / (delta_normalized.max() or 1) * 2 - 1

    x_interp = np.interp(common_dist, distance_a, x)
    y_interp = np.interp(common_dist, distance_a, y)

    points = np.array([x_interp, y_interp]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(10, 10))

    norm = plt.Normalize(-1, 1)
    cmap = plt.cm.RdYlBu_r
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3, alpha=0.9)
    lc.set_array(delta_normalized[:-1])
    ax.add_collection(lc)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(
        f"Time Delta: {driver_a} vs {driver_b} -- {circuit} {year}",
        color=MERCEDES_WHITE,
    )
    ax.set_facecolor(MERCEDES_DARK)
    ax.tick_params(colors=MERCEDES_GRAY)
    for spine in ax.spines.values():
        spine.set_color(MERCEDES_CARD)

    cbar = fig.colorbar(lc, ax=ax, label=f"Time Delta ({driver_a} faster / {driver_b} faster)", shrink=0.7)
    cbar.ax.yaxis.set_tick_params(color=MERCEDES_GRAY)
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color=MERCEDES_GRAY)
    cbar.outline.set_edgecolor(MERCEDES_GRAY)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels([f"{driver_b} faster", "Even", f"{driver_a} faster"])

    fig.tight_layout()

    if save:
        path = ASSET_DIR / "time_delta_map.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def run_all():
    """Generate all track setup analysis visuals."""
    print("Generating track setup analysis visuals...")
    speed_on_track_map()
    gear_distribution()
    sector_speed_comparison()
    aero_setup_comparison()
    speed_profile_comparison()
    driver_time_delta_map()
    print("Done. Files saved to docs/assets/images/track_setups/")


if __name__ == "__main__":
    run_all()
