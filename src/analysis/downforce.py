"""
Downforce analysis module for F1 aerodynamics.

Generates:
  - Component breakdown stacked area chart (front wing, rear wing, floor, body)
  - L/D ratio vs speed
  - Drag polar (C_D vs C_L)
  - Speed vs downforce scatter with component contribution
  - Aero balance (front vs rear) across speed range
"""

from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.core.models import F1Car
from src.core.physics import kmh_to_ms, ms_to_kmh
from src.core.style import set_f1_style, MERCEDES_TEAL, MERCEDES_DARK, MERCEDES_GRAY, MERCEDES_CARD

set_f1_style()

ASSET_DIR = Path("docs/assets/images")
ASSET_DIR.mkdir(parents=True, exist_ok=True)


def component_breakdown_chart(car: F1Car, save: bool = True):
    """Stacked area chart: component downforce percentage vs speed."""
    speeds_kmh = np.linspace(60, 340, 50)
    speeds_ms = kmh_to_ms(speeds_kmh)

    fw_pct = []
    rw_pct = []
    fl_pct = []
    body_pct = []

    for v in speeds_ms:
        comp = car.component_breakdown(v)
        cd = comp["downforce"]
        fw_pct.append(cd["front_wing_pct"])
        rw_pct.append(cd["rear_wing_pct"])
        fl_pct.append(cd["floor_pct"])
        body_pct.append(100 - cd["front_wing_pct"] - cd["rear_wing_pct"] - cd["floor_pct"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(
        speeds_kmh,
        fw_pct,
        rw_pct,
        fl_pct,
        body_pct,
        labels=["Front Wing", "Rear Wing", "Floor", "Body"],
        colors=["#00D2BE", "#E94560", "#FFB347", "#4ECDC4"],
        alpha=0.85,
    )
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Downforce Contribution (%)")
    ax.set_title("Downforce Component Breakdown vs Speed")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(60, 340)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "downforce_component_breakdown.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def ld_ratio_curve(car: F1Car, save: bool = True):
    """L/D ratio vs speed for different aero configurations."""
    speeds_kmh = np.linspace(60, 340, 80)
    speeds_ms = kmh_to_ms(speeds_kmh)

    ld_normal = [car.ld_ratio(v) for v in speeds_ms]
    ld_drs = [car.ld_ratio(v, drs_open=True) for v in speeds_ms]
    ld_low_rh = [car.ld_ratio(v, ride_height=0.025) for v in speeds_ms]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(speeds_kmh, ld_normal, color="#00D2BE", linewidth=2.5, label="Normal")
    ax.plot(speeds_kmh, ld_drs, color="#E94560", linewidth=2, linestyle="--", label="DRS Open")
    ax.plot(speeds_kmh, ld_low_rh, color="#FFB347", linewidth=2, linestyle=":", label="Low Ride Height (25mm)")
    ax.axhline(0, color=MERCEDES_GRAY, linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("L/D Ratio")
    ax.set_title("Lift-to-Drag Ratio vs Speed")
    ax.legend(framealpha=0.9)
    ax.set_xlim(60, 340)
    ax.grid(True, alpha=0.3)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "downforce_ld_ratio.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def drag_polar(car: F1Car, save: bool = True):
    """Drag polar (C_D vs C_L) across operating range."""
    speeds_ms = kmh_to_ms(np.linspace(60, 340, 50))

    cl_values = []
    cd_values = []
    colors = []
    for v in speeds_ms:
        breakdown = car.component_breakdown(v)
        total_df = breakdown["downforce"]["total"]
        total_drag = breakdown["drag"]["total"]
        q = 0.5 * 1.225 * v**2
        cl = abs(total_df) / (q * car.cfg.frontal_area) if q > 0 else 0
        cd = total_drag / (q * car.cfg.frontal_area) if q > 0 else 0
        cl_values.append(cl)
        cd_values.append(cd)
        colors.append(v)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        cd_values,
        cl_values,
        c=speeds_ms,  # color by speed (m/s)
        cmap="viridis",
        s=40,
        alpha=0.8,
        edgecolors="white",
        linewidth=0.3,
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Speed (m/s)")
    cbar.ax.yaxis.set_tick_params(color=MERCEDES_GRAY)
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color=MERCEDES_GRAY)
    cbar.outline.set_edgecolor(MERCEDES_GRAY)

    # Quadratic fit for drag polar trend
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeffs = np.polyfit(cd_values, cl_values, 2)
    cd_fit = np.linspace(min(cd_values), max(cd_values), 100)
    cl_fit = np.polyval(coeffs, cd_fit)
    ax.plot(cd_fit, cl_fit, "--", color="#00D2BE", alpha=0.5, label="Quadratic fit")

    ax.set_xlabel("Drag Coefficient C$_D$")
    ax.set_ylabel("Lift Coefficient |C$_L$|")
    ax.set_title("Drag Polar -- C$_D$ vs |C$_L$|")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "downforce_drag_polar.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def speed_vs_downforce(car: F1Car, save: bool = True):
    """Speed vs downforce with component contributions."""
    speeds_kmh = np.linspace(60, 340, 50)
    speeds_ms = kmh_to_ms(speeds_kmh)

    df_fw = []
    df_rw = []
    df_floor = []
    df_total = []
    for v in speeds_ms:
        comp = car.component_breakdown(v)
        cd = comp["downforce"]
        df_fw.append(cd["front_wing"])
        df_rw.append(cd["rear_wing"])
        df_floor.append(cd["floor"])
        df_total.append(cd["total"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(speeds_kmh, df_fw, alpha=0.3, color="#00D2BE", label="Front Wing")
    ax.fill_between(speeds_kmh, df_rw, alpha=0.3, color="#E94560", label="Rear Wing")
    ax.fill_between(speeds_kmh, df_floor, alpha=0.3, color="#FFB347", label="Floor")
    ax.plot(speeds_kmh, df_total, color="white", linewidth=2.5, label="Total")

    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Downforce (N)")
    ax.set_title("Downforce vs Speed by Component")
    ax.legend(framealpha=0.9)
    ax.set_xlim(60, 340)
    ax.grid(True, alpha=0.3)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "downforce_vs_speed.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def aero_balance_chart(car: F1Car, save: bool = True):
    """Front aero balance percentage across speed range."""
    speeds_kmh = np.linspace(60, 340, 50)
    speeds_ms = kmh_to_ms(speeds_kmh)

    front_pct = []
    for v in speeds_ms:
        comp = car.component_breakdown(v)
        cd = comp["downforce"]
        pct = cd["front_wing"] / cd["total"] * 100
        front_pct.append(pct)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(speeds_kmh, front_pct, color="#00D2BE", linewidth=2.5)
    ax.axhline(40, color=MERCEDES_GRAY, linestyle="--", alpha=0.5, label="Typical front aero balance (40%)")
    ax.fill_between(speeds_kmh, front_pct, alpha=0.15, color="#00D2BE")

    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Front Aero Balance (%)")
    ax.set_title("Aero Balance Shift vs Speed")
    ax.legend(framealpha=0.9)
    ax.set_xlim(60, 340)
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "downforce_aero_balance.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def run_all():
    """Generate all downforce analysis visuals."""
    print("Generating downforce analysis visuals...")
    car = F1Car()

    component_breakdown_chart(car)
    ld_ratio_curve(car)
    drag_polar(car)
    speed_vs_downforce(car)
    aero_balance_chart(car)

    print("Done. Files saved to docs/assets/images/")


if __name__ == "__main__":
    run_all()
