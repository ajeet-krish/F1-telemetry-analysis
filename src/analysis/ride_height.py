"""
Ride height sensitivity and porpoising analysis for F1 aerodynamics.

Generates:
  - Contour map: C_L across (ride height, velocity) space
  - Downforce vs ride height curves at fixed speeds
  - Aero balance shift with front/rear ride height
  - Porpoising stability map (dC_L/dh gradient analysis)
  - Venturi stall visualization
"""

import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.core.models import F1Car
from src.core.physics import kmh_to_ms, ms_to_kmh, dynamic_pressure
from src.core.style import (
    set_f1_style,
    MERCEDES_TEAL,
    MERCEDES_DARK,
    MERCEDES_CARD,
    MERCEDES_GRAY,
    MERCEDES_WHITE,
    MERCEDES_RED,
    teal_colormap,
)

set_f1_style()

ASSET_DIR = Path("docs/assets/images/ride_height")
ASSET_DIR.mkdir(parents=True, exist_ok=True)


def cl_with_venturi_stall(ride_height_m: float, cl_base: float = -0.5, A: float = 0.05, B: float = 0.001) -> float:
    """
    Floor lift coefficient with venturi stall at very low ride heights.

    Below a critical height (~20mm), the diffuser experiences flow separation,
    causing a sudden loss of downforce. This is the porpoising trigger mechanism.
    """
    h = max(ride_height_m, 0.005)
    ground_mult = 1.0 + A / h + B / (h**2)
    cl_attached = cl_base * ground_mult

    # Venturi stall model: sigmoid drop below h_stall
    h_stall = 0.020
    sigma = 0.004
    if ride_height_m < h_stall:
        stall_factor = 1.0 / (1.0 + np.exp((h_stall - ride_height_m) / sigma))
        stalled_cl = cl_base * 1.2
        return cl_attached * stall_factor + stalled_cl * (1 - stall_factor)
    return cl_attached


def downforce_vs_ride_height(car: F1Car, save: bool = True):
    """Downforce vs ride height curves at multiple fixed speeds."""
    ride_heights_mm = np.linspace(10, 100, 200)
    ride_heights_m = ride_heights_mm / 1000.0
    speeds_kmh = [100, 180, 260, 320]
    colors = ["#4ECDC4", "#00D2BE", "#FFB347", "#E94560"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for speed_kmh, color in zip(speeds_kmh, colors):
        v = kmh_to_ms(speed_kmh)
        df = []
        for h in ride_heights_m:
            cl_val = cl_with_venturi_stall(h)
            q = dynamic_pressure(v)
            df.append(abs(cl_val) * q * car.floor.cfg.area)
        ax.plot(ride_heights_mm, df, color=color, linewidth=2.5, label=f"{speed_kmh} km/h")

    ax.axvspan(10, 20, alpha=0.08, color=MERCEDES_RED, label="Porpoising risk zone")
    ax.axvline(20, color=MERCEDES_RED, linestyle=":", alpha=0.5)

    ax.set_xlabel("Ride Height (mm)")
    ax.set_ylabel("Floor Downforce (N)")
    ax.set_title("Floor Downforce vs Ride Height at Fixed Speeds")
    ax.legend(framealpha=0.9)
    ax.set_xlim(10, 100)
    ax.grid(True, alpha=0.3)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "downforce_curves.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def cl_contour_map(save: bool = True):
    """Contour map of C_L magnitude across (ride height, velocity) space."""
    ride_heights_mm = np.linspace(10, 100, 100)
    speeds_kmh = np.linspace(60, 340, 100)
    RH, V = np.meshgrid(ride_heights_mm, speeds_kmh)

    CL = np.zeros_like(RH)
    for i in range(len(speeds_kmh)):
        for j in range(len(ride_heights_mm)):
            CL[i, j] = abs(cl_with_venturi_stall(RH[i, j] / 1000.0))

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = teal_colormap()
    contour = ax.contourf(RH, V, CL, levels=25, cmap=cmap)
    cbar = plt.colorbar(contour, ax=ax, label="|C$_L$| (magnitude)")
    cbar.ax.yaxis.set_tick_params(color=MERCEDES_GRAY)
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color=MERCEDES_GRAY)
    cbar.outline.set_edgecolor(MERCEDES_GRAY)

    contours = ax.contour(RH, V, CL, levels=8, colors="white", linewidths=0.6, alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f", colors=MERCEDES_GRAY)

    ax.axvline(20, color=MERCEDES_RED, linestyle="--", alpha=0.6, label="Venturi stall threshold")
    ax.set_xlabel("Ride Height (mm)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Floor Lift Coefficient |C$_L$| -- Ground Effect Map")
    ax.legend(framealpha=0.9)
    ax.set_xlim(10, 100)
    ax.set_ylim(60, 340)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "cl_contour.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def porpoising_stability_map(save: bool = True):
    """Gradient d|C_L|/dh map showing porpoising instability regions.

    Positive gradient means downforce decreases as ride height drops (stable).
    Negative gradient means downforce increases as ride height drops (unstable
    -- can trigger porpoising oscillation).
    Actually in porpoising: when d(Downforce)/dh is very negative (strong
    downforce gain as h decreases), the car is sucked down aggressively.
    If this is followed by stall (sudden loss), the cycle begins.
    """
    ride_heights_mm = np.linspace(10, 100, 200)
    speeds_kmh = np.linspace(60, 340, 200)
    RH, V = np.meshgrid(ride_heights_mm, speeds_kmh)

    dCL_dh = np.zeros_like(RH)
    dh = 0.0002
    for i in range(len(speeds_kmh)):
        for j in range(len(ride_heights_mm)):
            h = RH[i, j] / 1000.0
            cl_plus = abs(cl_with_venturi_stall(h + dh))
            cl_minus = abs(cl_with_venturi_stall(h - dh))
            dCL_dh[i, j] = (cl_plus - cl_minus) / (2 * dh)

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.RdYlGn_r
    levels = np.linspace(-20, 5, 26)
    contour = ax.contourf(RH, V, dCL_dh, levels=levels, cmap=cmap, extend="both")
    cbar = plt.colorbar(contour, ax=ax, label="d|C$_L$|/dh (1/m)")
    cbar.ax.yaxis.set_tick_params(color=MERCEDES_GRAY)
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color=MERCEDES_GRAY)
    cbar.outline.set_edgecolor(MERCEDES_GRAY)

    contour_lines = ax.contour(RH, V, dCL_dh, levels=[0], colors="white", linewidths=1.5, alpha=0.8)

    ax.axvline(20, color=MERCEDES_RED, linestyle="--", alpha=0.6, label="Stall threshold")
    ax.set_xlabel("Ride Height (mm)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Porpoising Stability -- d|C$_L$|/dh Gradient Map")
    ax.legend(framealpha=0.9)
    ax.set_xlim(10, 100)
    ax.set_ylim(60, 340)

    ax.text(0.02, 0.03, "Unstable (strong suck-down)", transform=ax.transAxes,
            color=MERCEDES_RED, fontsize=9, alpha=0.8)
    ax.text(0.7, 0.95, "Stable", transform=ax.transAxes,
            color="green", fontsize=9, alpha=0.8)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "porpoising_stability.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def aero_balance_ride_height(car: F1Car, save: bool = True):
    """Front aero balance as a function of both front and rear ride height."""
    ride_heights_mm = np.linspace(15, 80, 50)
    speeds_kmh = [120, 200, 280]
    colors = ["#00D2BE", "#FFB347", "#E94560"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for speed_kmh, color in zip(speeds_kmh, colors):
        v = kmh_to_ms(speed_kmh)
        front_pct = []
        rear_pct = []
        for h_mm in ride_heights_mm:
            h = h_mm / 1000.0
            comp = car.component_breakdown(v, ride_height=h)
            front_pct.append(comp["downforce"]["front_wing_pct"])
            rear_pct.append(comp["downforce"]["rear_wing_pct"])
        ax.plot(ride_heights_mm, front_pct, color=color, linewidth=2.5, label=f"{speed_kmh} km/h - Front")
        ax.plot(ride_heights_mm, rear_pct, color=color, linewidth=1.5, linestyle="--", alpha=0.6)

    ax.set_xlabel("Ride Height (mm)")
    ax.set_ylabel("Aero Balance Contribution (%)")
    ax.set_title("Aero Balance Shift with Ride Height")
    ax.legend(framealpha=0.9)
    ax.set_xlim(15, 80)
    ax.grid(True, alpha=0.3)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "aero_balance.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def ride_height_sensitivity(car: F1Car, save: bool = True):
    """dF/dh showing how sensitive total downforce is to ride height changes."""
    ride_heights_mm = np.linspace(15, 80, 100)
    ride_heights_m = ride_heights_mm / 1000.0
    speeds_kmh = [120, 200, 280]
    colors = ["#00D2BE", "#FFB347", "#E94560"]
    dh = 0.0005

    fig, ax = plt.subplots(figsize=(12, 6))

    for speed_kmh, color in zip(speeds_kmh, colors):
        v = kmh_to_ms(speed_kmh)
        sensitivity = []
        for h in ride_heights_m:
            df_plus = car.floor.downforce(v, h + dh)
            df_minus = car.floor.downforce(v, max(h - dh, 0.005))
            sens = (df_plus - df_minus) / (2 * dh) / 1000.0
            sensitivity.append(sens)
        ax.plot(ride_heights_mm, sensitivity, color=color, linewidth=2.5, label=f"{speed_kmh} km/h")

    ax.axhline(0, color=MERCEDES_GRAY, linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Ride Height (mm)")
    ax.set_ylabel("dF/dh (kN/m)")
    ax.set_title("Ride Height Sensitivity -- Downforce Gradient")
    ax.legend(framealpha=0.9)
    ax.set_xlim(15, 80)
    ax.grid(True, alpha=0.3)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "sensitivity.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def run_all():
    """Generate all ride height analysis visuals."""
    print("Generating ride height analysis visuals...")
    car = F1Car()

    downforce_vs_ride_height(car)
    cl_contour_map()
    porpoising_stability_map()
    aero_balance_ride_height(car)
    ride_height_sensitivity(car)

    print("Done. Files saved to docs/assets/images/ride_height/")


if __name__ == "__main__":
    run_all()
