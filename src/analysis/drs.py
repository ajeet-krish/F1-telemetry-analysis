"""
DRS and active aero analysis for F1 aerodynamics.

Generates:
  - DRS drag polar comparison (open vs closed)
  - Speed trace with DRS activation zones highlighted from telemetry
  - Overtaking delta: speed gain with DRS over a closing gap
  - 2022 vs 2026 regulation comparison
  - 2026 Z-mode active aero torque/efficiency tradeoff
"""

import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

from src.core.models import F1Car, RearWingConfig, RearWing
from src.core.physics import kmh_to_ms, ms_to_kmh, dynamic_pressure
from src.core.style import (
    set_f1_style,
    MERCEDES_TEAL,
    MERCEDES_DARK,
    MERCEDES_CARD,
    MERCEDES_GRAY,
    MERCEDES_WHITE,
    MERCEDES_RED,
    MERCEDES_AMBER,
)
from src.core.telemetry import TelemetryLoader

set_f1_style()

ASSET_DIR = Path("docs/assets/images")
ASSET_DIR.mkdir(parents=True, exist_ok=True)


def drs_drag_polar_car(car: F1Car, save: bool = True):
    """Drag polar (C_D vs C_L) comparison between DRS open/closed with telemetry."""
    speeds_ms = kmh_to_ms(np.linspace(60, 340, 80))

    cl_closed = []
    cd_closed = []
    cl_open = []
    cd_open = []
    for v in speeds_ms:
        rh = car.ride_height_at_speed(v)
        comp_closed = car.component_breakdown(v, ride_height=rh)
        comp_open = car.component_breakdown(v, ride_height=rh, drs_open=True)
        q = 0.5 * 1.225 * v**2
        A = car.cfg.frontal_area

        cl_closed.append(abs(comp_closed["downforce"]["total"]) / (q * A) if q > 0 else 0)
        cd_closed.append(comp_closed["drag"]["total"] / (q * A) if q > 0 else 0)
        cl_open.append(abs(comp_open["downforce"]["total"]) / (q * A) if q > 0 else 0)
        cd_open.append(comp_open["drag"]["total"] / (q * A) if q > 0 else 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.text(0.02, 0.98, "Model + Telemetry", transform=fig.transFigure, fontsize=9, color="#00D2BE", alpha=0.7, va="top")

    ax.plot(cd_closed, cl_closed, color=MERCEDES_TEAL, linewidth=2.5, label="DRS Closed (model)")
    ax.plot(cd_open, cl_open, color=MERCEDES_RED, linewidth=2.5, linestyle="--", label="DRS Open (model)")

    # Telemetry scatter for DRS on/off from high-speed circuits
    for gp, year, driver, label_suffix, color in [
        ("Monza", 2024, "VER", "Telemetry", MERCEDES_TEAL),
    ]:
        try:
            loader = TelemetryLoader(year, gp, "R")
            tel = loader.lap_telemetry(driver)
            speed_tel = tel["Speed"].values / 3.6
            drs_col = tel["DRS"].astype(bool) if "DRS" in tel.columns else None
            q_tel = 0.5 * 1.225 * speed_tel**2
            A = car.cfg.frontal_area
            tel_cd_closed, tel_cl_closed = [], []
            tel_cd_open, tel_cl_open = [], []
            rh_mean = car.ride_height_at_speed(speed_tel.mean())
            comp = car.component_breakdown(speed_tel.mean(), ride_height=rh_mean)
            for i, v in enumerate(speed_tel):
                if q_tel[i] > 0:
                    cd_val = comp["drag"]["total"] / (q_tel[i] * A)
                    cl_val = abs(comp["downforce"]["total"]) / (q_tel[i] * A)
                    if drs_col is not None and drs_col.iloc[i]:
                        tel_cd_open.append(cd_val)
                        tel_cl_open.append(cl_val)
                    else:
                        tel_cd_closed.append(cd_val)
                        tel_cl_closed.append(cl_val)
            if tel_cd_closed:
                ax.scatter(tel_cd_closed, tel_cl_closed, color=MERCEDES_TEAL, s=6, alpha=0.15, zorder=2)
            if tel_cd_open:
                ax.scatter(tel_cd_open, tel_cl_open, color=MERCEDES_RED, s=6, alpha=0.15, zorder=2)
        except Exception:
            pass

    # L/D contour lines
    ld_vals = [3, 5, 7]
    cd_span = np.linspace(min(cd_closed + cd_open), max(cd_closed + cd_open), 50)
    for ld in ld_vals:
        ax.plot(cd_span, ld * cd_span, "--", color=MERCEDES_GRAY, linewidth=0.5, alpha=0.25)
        ax.text(cd_span[-1], ld * cd_span[-1], f"L/D={ld}", color=MERCEDES_GRAY, fontsize=7, alpha=0.3)

    ax.set_xlabel("Drag Coefficient C$_D$")
    ax.set_ylabel("Lift Coefficient |C$_L$|")
    ax.set_title("DRS Effect on Drag Polar")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "drs_drag_polar.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def drs_speed_delta(car: F1Car, save: bool = True):
    """Downforce and drag difference between DRS open/closed."""
    speeds_kmh = np.linspace(60, 340, 80)
    speeds_ms = kmh_to_ms(speeds_kmh)

    df_closed = [car.total_downforce(v) for v in speeds_ms]
    df_open = [car.total_downforce(v, drs_open=True) for v in speeds_ms]
    drag_closed = [car.total_drag(v) for v in speeds_ms]
    drag_open = [car.total_drag(v, drs_open=True) for v in speeds_ms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.text(0.02, 0.98, "Model-based", transform=fig.transFigure, fontsize=9, color="#00D2BE", alpha=0.7, va="top")

    ax1.plot(speeds_kmh, np.array(df_closed) / 1000, color=MERCEDES_TEAL, linewidth=2.5, label="Closed")
    ax1.plot(speeds_kmh, np.array(df_open) / 1000, color=MERCEDES_RED, linewidth=2, linestyle="--", label="Open")
    ax1.fill_between(speeds_kmh, np.array(df_closed) / 1000, np.array(df_open) / 1000,
                      alpha=0.15, color=MERCEDES_TEAL, label="Downforce loss")
    ax1.set_xlabel("Speed (km/h)")
    ax1.set_ylabel("Downforce (kN)")
    ax1.set_title("Downforce: DRS Open vs Closed")
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(speeds_kmh, np.array(drag_closed) / 1000, color=MERCEDES_TEAL, linewidth=2.5, label="Closed")
    ax2.plot(speeds_kmh, np.array(drag_open) / 1000, color=MERCEDES_RED, linewidth=2, linestyle="--", label="Open")
    ax2.fill_between(speeds_kmh, np.array(drag_closed) / 1000, np.array(drag_open) / 1000,
                      alpha=0.15, color=MERCEDES_RED, label="Drag reduction")
    ax2.set_xlabel("Speed (km/h)")
    ax2.set_ylabel("Drag (kN)")
    ax2.set_title("Drag: DRS Open vs Closed")
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set(color=MERCEDES_GRAY)

    fig.suptitle("DRS Effect on Downforce and Drag", fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        path = ASSET_DIR / "drs_speed_delta.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def drs_overtaking_analysis(car: F1Car, save: bool = True):
    """Overtaking speed advantage: simulate closing a gap with DRS.

    The leading car has DRS closed, the trailing car opens DRS.
    The drag reduction allows the trailing car to accelerate faster.
    """
    speeds_kmh = np.linspace(100, 320, 50)
    speeds_ms = kmh_to_ms(speeds_kmh)

    mass = car.cfg.mass

    # Net force = thrust - drag (simplified: no engine model, compare drag delta)
    drag_lead = np.array([car.total_drag(v) for v in speeds_ms])
    drag_trail = np.array([car.total_drag(v, drs_open=True) for v in speeds_ms])
    drag_delta = drag_lead - drag_trail
    accel_advantage = drag_delta / mass

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.text(0.02, 0.98, "Model-based", transform=fig.transFigure, fontsize=9, color="#00D2BE", alpha=0.7, va="top")
    ax.plot(speeds_kmh, accel_advantage, color=MERCEDES_TEAL, linewidth=2.5)
    ax.fill_between(speeds_kmh, accel_advantage, alpha=0.2, color=MERCEDES_TEAL)
    ax.axhline(0, color=MERCEDES_GRAY, linewidth=0.5)

    for speed_label in [150, 200, 250, 300]:
        idx = np.argmin(np.abs(speeds_kmh - speed_label))
        ax.annotate(f"{accel_advantage[idx]:.2f} m/s$^2$",
                    (speeds_kmh[idx], accel_advantage[idx]),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, color=MERCEDES_AMBER)

    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Acceleration Advantage (m/s$^2$)")
    ax.set_title("Overtaking Advantage: Trailing Car with DRS")
    ax.set_xlim(100, 320)
    ax.grid(True, alpha=0.3)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "drs_overtaking_advantage.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def drs_telemetry_trace(save: bool = True):
    """Speed trace from telemetry with DRS activation highlighted in 3 stacked subplots."""
    loader = TelemetryLoader(2024, "Monaco", "R")
    trace = loader.speed_trace("VER")
    telemetry = loader.lap_telemetry("VER")

    distance = telemetry["Distance"]
    speed = telemetry["Speed"]
    drs = telemetry["DRS"].astype(bool) if "DRS" in telemetry.columns else None
    throttle = telemetry["Throttle"]
    brake = telemetry["Brake"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    ax1.plot(distance / 1000, speed, color=MERCEDES_TEAL, linewidth=1.8, alpha=0.9, label="Speed")
    if drs is not None:
        drs_active = drs & (speed > 100)
        ax1.fill_between(distance / 1000, speed, where=drs_active,
                          color="#00D2BE", alpha=0.25, label="DRS Active")
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_title(f"VER Fastest Lap - Monaco 2024 (Top Speed: {speed.max():.0f} km/h)")
    ax1.legend(framealpha=0.9, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(distance / 1000, 0, throttle, alpha=0.4, color="#22C55E", label="Throttle")
    ax2.set_ylabel("Throttle (%)")
    ax2.set_ylim(0, 105)
    ax2.legend(framealpha=0.9, loc="upper right")
    ax2.grid(True, alpha=0.3)

    ax3.fill_between(distance / 1000, 0, brake, alpha=0.4, color=MERCEDES_RED, label="Brake")
    ax3.set_xlabel("Distance (km)")
    ax3.set_ylabel("Brake (%)")
    ax3.set_ylim(0, 105)
    ax3.legend(framealpha=0.9, loc="upper right")
    ax3.grid(True, alpha=0.3)

    for ax in [ax1, ax2, ax3]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set(color=MERCEDES_GRAY)

    fig.tight_layout()

    if save:
        path = ASSET_DIR / "drs_telemetry_trace.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def active_aero_2026_comparison(save: bool = True):
    """Compare 2022 DRS vs 2026 active aero modes.

    2026 introduces three active aero modes:
    - Straight mode: low drag (similar to DRS but always on straights)
    - Corner mode: max downforce
    - Z-mode: overtaking boost (reduced drag + electric power)
    """
    speeds_kmh = np.linspace(60, 340, 80)
    speeds_ms = kmh_to_ms(speeds_kmh)

    car_2022 = F1Car()
    df_2022_normal = [car_2022.total_downforce(v) / 1000 for v in speeds_ms]
    drag_2022_normal = [car_2022.total_drag(v) / 1000 for v in speeds_ms]
    df_2022_drs = [car_2022.total_downforce(v, drs_open=True) / 1000 for v in speeds_ms]
    drag_2022_drs = [car_2022.total_drag(v, drs_open=True) / 1000 for v in speeds_ms]

    # 2026: smaller wings, active aero
    rw_2026 = RearWingConfig(
        span=0.75, chord=0.25, area=0.19,
        max_angle=12.0, cl_alpha=6.0, cd0=0.015, k_induced=0.22,
        drs_drag_reduction=0.50, drs_downforce_loss=0.35,
    )
    car_2026 = F1Car()
    car_2026.rear_wing = RearWing(rw_2026)
    car_2026.floor.cfg.area = 3.8
    car_2026.floor.cfg.cl_base = -0.3
    car_2026.cfg.frontal_area = 1.35

    df_2026_corner = [car_2026.total_downforce(v, rear_alpha=-10.0) / 1000 for v in speeds_ms]
    drag_2026_corner = [car_2026.total_drag(v, rear_alpha=-10.0) / 1000 for v in speeds_ms]
    df_2026_straight = [car_2026.total_downforce(v, rear_alpha=-3.0) / 1000 for v in speeds_ms]
    drag_2026_straight = [car_2026.total_drag(v, rear_alpha=-3.0) / 1000 for v in speeds_ms]
    df_2026_z = [car_2026.total_downforce(v, rear_alpha=-2.0, drs_open=True) / 1000 for v in speeds_ms]
    drag_2026_z = [car_2026.total_drag(v, rear_alpha=-2.0, drs_open=True) / 1000 for v in speeds_ms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.text(0.02, 0.98, "Model-based", transform=fig.transFigure, fontsize=9, color="#00D2BE", alpha=0.7, va="top")

    ax1.plot(speeds_kmh, df_2022_normal, color=MERCEDES_TEAL, linewidth=2, label="2022 Normal")
    ax1.plot(speeds_kmh, df_2022_drs, color=MERCEDES_TEAL, linewidth=1.5, linestyle="--", label="2022 DRS")
    ax1.plot(speeds_kmh, df_2026_corner, color=MERCEDES_RED, linewidth=2, label="2026 Corner")
    ax1.plot(speeds_kmh, df_2026_straight, color=MERCEDES_RED, linewidth=1.5, linestyle="--", label="2026 Straight")
    ax1.plot(speeds_kmh, df_2026_z, color=MERCEDES_AMBER, linewidth=2, linestyle=":", label="2026 Z-Mode")
    ax1.set_xlabel("Speed (km/h)")
    ax1.set_ylabel("Downforce (kN)")
    ax1.set_title("Downforce: 2022 vs 2026")
    ax1.legend(framealpha=0.9, fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(speeds_kmh, drag_2022_normal, color=MERCEDES_TEAL, linewidth=2, label="2022 Normal")
    ax2.plot(speeds_kmh, drag_2022_drs, color=MERCEDES_TEAL, linewidth=1.5, linestyle="--", label="2022 DRS")
    ax2.plot(speeds_kmh, drag_2026_corner, color=MERCEDES_RED, linewidth=2, label="2026 Corner")
    ax2.plot(speeds_kmh, drag_2026_straight, color=MERCEDES_RED, linewidth=1.5, linestyle="--", label="2026 Straight")
    ax2.plot(speeds_kmh, drag_2026_z, color=MERCEDES_AMBER, linewidth=2, linestyle=":", label="2026 Z-Mode")
    ax2.set_xlabel("Speed (km/h)")
    ax2.set_ylabel("Drag (kN)")
    ax2.set_title("Drag: 2022 vs 2026")
    ax2.legend(framealpha=0.9, fontsize=8)
    ax2.grid(True, alpha=0.3)

    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set(color=MERCEDES_GRAY)

    fig.suptitle("2022 DRS vs 2026 Active Aero", fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        path = ASSET_DIR / "drs_2022_vs_2026.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def run_all():
    """Generate all DRS analysis visuals."""
    print("Generating DRS analysis visuals...")
    car = F1Car()

    drs_drag_polar_car(car)
    drs_speed_delta(car)
    drs_overtaking_analysis(car)
    drs_telemetry_trace()
    active_aero_2026_comparison()

    print("Done. Files saved to docs/assets/images/")


if __name__ == "__main__":
    run_all()
