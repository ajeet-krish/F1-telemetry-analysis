"""
Cornering performance analysis for F1 aerodynamics.

Maps downforce to lateral grip, generates G-g diagrams from telemetry,
estimates corner radius from speed and lateral acceleration.

Generates:
  - G-g diagram (lateral vs longitudinal acceleration)
  - Corner radius estimation from telemetry
  - Downforce contribution to cornering speed envelope
  - Grip margin chart (mechanical + aero grip vs speed)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import warnings
import seaborn as sns

from src.core.models import F1Car
from src.core.physics import kmh_to_ms, ms_to_kmh, dynamic_pressure, G
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

ASSET_DIR = Path("docs/assets/images/cornering")
ASSET_DIR.mkdir(parents=True, exist_ok=True)


def gg_diagram(save: bool = True):
    """G-g diagram from telemetry: lateral vs longitudinal acceleration."""
    loader = TelemetryLoader(2024, "Monaco", "R")
    tel = loader.lap_telemetry("VER")

    speed_ms = tel["Speed"].values / 3.6
    throttle = tel["Throttle"].values / 100
    brake = tel["Brake"].values.astype(bool)
    x = tel["X"].values
    y = tel["Y"].values

    SPEED_THRESHOLD = 30
    mask = speed_ms > SPEED_THRESHOLD
    speed_ms = speed_ms[mask]
    throttle = throttle[mask]
    brake = brake[mask]
    x = x[mask]
    y = y[mask]

    # Longitudinal acceleration from speed trace
    if "SessionTime" in tel.columns:
        st_all = tel["SessionTime"].values
        st_sec_all = st_all.astype(np.float64)
        if st_sec_all.max() > 1e6:
            st_sec_all = st_sec_all * 1e-9
        st = st_sec_all[mask]
        dt_long = np.diff(st)
        dt_long = np.where(dt_long <= 0, 0.01, dt_long)
        dt_long = np.concatenate([[dt_long[0]], dt_long])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ax = np.gradient(speed_ms, dt_long) / G
    else:
        ax = np.zeros_like(speed_ms)

    # Lateral acceleration from trajectory curvature
    curvature = np.zeros_like(speed_ms)
    for i in range(1, len(x) - 1):
        dx1, dy1 = x[i] - x[i-1], y[i] - y[i-1]
        dx2, dy2 = x[i+1] - x[i], y[i+1] - y[i]
        cross = dx1 * dy2 - dy1 * dx2
        denom = np.sqrt((dx1**2 + dy1**2) * (dx2**2 + dy2**2))
        if denom > 1e-6:
            curvature[i] = 2 * cross / denom
    ay = curvature * speed_ms**2 / G

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax = np.clip(np.nan_to_num(ax, nan=0), -5, 5)
        ay = np.clip(np.nan_to_num(ay, nan=0), -5, 5)
        speed_clip = np.clip(speed_ms, 30, 350)

    fig, ax_plot = plt.subplots(figsize=(9, 9))
    fig.text(0.02, 0.98, "Telemetry-based", transform=fig.transFigure, fontsize=9, color="#00D2BE", alpha=0.7, va="top")
    scatter = ax_plot.scatter(
        ax, ay, c=speed_clip, cmap="plasma", s=8, alpha=0.6, edgecolors="none"
    )
    cbar = plt.colorbar(scatter, ax=ax_plot, label="Speed (km/h)")
    cbar.ax.yaxis.set_tick_params(color=MERCEDES_GRAY)
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color=MERCEDES_GRAY)
    cbar.outline.set_edgecolor(MERCEDES_GRAY)

    theta = np.linspace(0, 2 * np.pi, 100)
    for g_level in [1, 2, 3]:
        circle = plt.Circle((0, 0), g_level, fill=False, color=MERCEDES_GRAY,
                            linewidth=0.5, linestyle="--", alpha=0.3)
        ax_plot.add_patch(circle)
        ax_plot.text(g_level * 0.7, g_level * 0.7, f"{g_level}g",
                     color=MERCEDES_GRAY, fontsize=8, alpha=0.4)

    ax_plot.set_xlabel("Longitudinal Acceleration (g)")
    ax_plot.set_ylabel("Lateral Acceleration (g)")
    ax_plot.set_title("G-G Diagram -- VER Monaco 2024 Fastest Lap")
    ax_plot.set_aspect("equal")
    ax_plot.set_xlim(-5, 5)
    ax_plot.set_ylim(-5, 5)
    ax_plot.axhline(0, color=MERCEDES_GRAY, linewidth=0.5, alpha=0.3)
    ax_plot.axvline(0, color=MERCEDES_GRAY, linewidth=0.5, alpha=0.3)
    ax_plot.grid(True, alpha=0.2)

    for label in ax_plot.get_xticklabels() + ax_plot.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "gg_diagram.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def speed_vs_corner_radius(save: bool = True):
    """Corner radius estimated from speed and trajectory curvature."""
    loader = TelemetryLoader(2024, "Monaco", "R")
    tel = loader.lap_telemetry("VER")

    speed_ms = tel["Speed"].values / 3.6
    x = tel["X"].values
    y = tel["Y"].values

    # Compute curvature from 3-point path approximation
    curvature = np.zeros_like(speed_ms)
    for i in range(1, len(x) - 1):
        dx1, dy1 = x[i] - x[i-1], y[i] - y[i-1]
        dx2, dy2 = x[i+1] - x[i], y[i+1] - y[i]
        cross = dx1 * dy2 - dy1 * dx2
        denom = np.sqrt((dx1**2 + dy1**2) * (dx2**2 + dy2**2))
        if denom > 1e-6:
            curvature[i] = 2 * cross / denom
    lateral_accel = np.abs(curvature * speed_ms**2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lateral_accel = np.clip(np.nan_to_num(lateral_accel, nan=0.0), 0, 50)

    # Radius = v^2 / a_lat
    SPEED_FLOOR = 10
    mask = (speed_ms > SPEED_FLOOR) & (lateral_accel > 0.5)
    speed_vals = speed_ms[mask]
    lat_accel_vals = lateral_accel[mask]

    radius = speed_vals**2 / np.clip(lat_accel_vals, 0.1, None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        radius = np.clip(radius, 5, 500)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.text(0.02, 0.98, "Telemetry-based", transform=fig.transFigure, fontsize=9, color="#00D2BE", alpha=0.7, va="top")
    scatter = ax.scatter(
        speed_vals * 3.6, radius, c=lat_accel_vals, cmap="viridis",
        s=15, alpha=0.6, edgecolors="none"
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Lateral Acceleration (m/s$^2$)")
    cbar.ax.yaxis.set_tick_params(color=MERCEDES_GRAY)
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color=MERCEDES_GRAY)
    cbar.outline.set_edgecolor(MERCEDES_GRAY)

    for r_const in [50, 100, 200]:
        v_range = np.linspace(30, 280, 50)
        lat_g = speed_vals.mean()
        ax.plot(v_range, np.full_like(v_range, r_const), "--", color=MERCEDES_GRAY,
                linewidth=0.5, alpha=0.3)
        ax.text(270, r_const + 5, f"R={r_const}m", color=MERCEDES_GRAY,
                fontsize=8, alpha=0.4, ha="right")

    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Corner Radius (m)")
    ax.set_title("Corner Radius vs Speed -- VER Monaco 2024")
    ax.set_xlim(0, 280)
    ax.set_ylim(0, 300)
    ax.grid(True, alpha=0.3)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "speed_radius.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def downforce_grip_envelope(save: bool = True):
    """Grip envelope showing mechanical + aero contribution to cornering speed."""
    car = F1Car()
    speeds_kmh = np.linspace(40, 340, 80)
    speeds_ms = kmh_to_ms(speeds_kmh)

    mass = car.cfg.mass
    mu = 1.2
    cl_total = []
    for v in speeds_ms:
        comp = car.component_breakdown(v)
        q = dynamic_pressure(v)
        cl = abs(comp["downforce"]["total"]) / (q * car.cfg.frontal_area) if q > 0 else 0
        cl_total.append(cl)
    cl_total = np.array(cl_total)

    df = np.array([car.total_downforce(v) for v in speeds_ms])
    grip_mech = np.full_like(df, mu * mass * G)
    grip_aero = mu * df
    grip_total = grip_mech + grip_aero

    corner_speed_mech = np.sqrt(grip_mech * 50 / mass)
    corner_speed_total = np.sqrt(grip_total * 50 / mass)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.text(0.02, 0.98, "Model-based", transform=fig.transFigure, fontsize=9, color="#00D2BE", alpha=0.7, va="top")

    ax1.fill_between(speeds_kmh, 0, grip_mech / 1000, alpha=0.3, color=MERCEDES_TEAL, label="Mechanical grip")
    ax1.fill_between(speeds_kmh, grip_mech / 1000, grip_total / 1000, alpha=0.3, color=MERCEDES_RED, label="Aero contribution")
    ax1.plot(speeds_kmh, grip_total / 1000, color=MERCEDES_WHITE, linewidth=2, label="Total grip")
    ax1.set_xlabel("Speed (km/h)")
    ax1.set_ylabel("Available Lateral Force (kN)")
    ax1.set_title("Grip Envelope: Mechanical + Aero")
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(speeds_kmh, ms_to_kmh(corner_speed_mech), color=MERCEDES_TEAL, linewidth=2, label="Mechanical only")
    ax2.plot(speeds_kmh, ms_to_kmh(corner_speed_total), color=MERCEDES_RED, linewidth=2, label="With aero")
    ax2.fill_between(speeds_kmh, ms_to_kmh(corner_speed_mech), ms_to_kmh(corner_speed_total),
                      alpha=0.15, color=MERCEDES_RED)
    ax2.set_xlabel("Speed (km/h)")
    ax2.set_ylabel("Corner Speed (km/h) for R=50m")
    ax2.set_title("Cornering Speed: Aero Contribution")
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set(color=MERCEDES_GRAY)

    fig.suptitle("Downforce Contribution to Cornering Performance", fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        path = ASSET_DIR / "grip_envelope.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def corner_classification(save: bool = True):
    """Classify corners by speed and lateral g from telemetry."""
    loader = TelemetryLoader(2024, "Monaco", "R")
    tel = loader.lap_telemetry("VER")

    speed_ms = tel["Speed"].values / 3.6
    x = tel["X"].values
    y = tel["Y"].values

    if "SessionTime" in tel.columns:
        st = tel["SessionTime"].values
        st_sec = st.astype(np.float64)
        if st_sec.max() > 1e6:
            st_sec = st_sec * 1e-9
        dt = np.diff(st_sec)
        dt = np.where(dt <= 0, 0.01, dt)
        dt = np.concatenate([[dt[0] + 1e-10], dt + 1e-10])
    else:
        dt = np.full(len(x), 0.01)

    # Compute lateral acceleration from yaw rate: ay = v^2 / R
    # R = path curvature from coordinate trajectory
    d = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    d = np.concatenate([[d[0]], d])
    curvature = np.zeros_like(speed_ms)
    for i in range(1, len(x) - 1):
        dx1, dy1 = x[i] - x[i-1], y[i] - y[i-1]
        dx2, dy2 = x[i+1] - x[i], y[i+1] - y[i]
        cross = dx1 * dy2 - dy1 * dx2
        denom = np.sqrt((dx1**2 + dy1**2) * (dx2**2 + dy2**2))
        if denom > 1e-6:
            curvature[i] = 2 * cross / denom
    lateral_accel = np.abs(curvature * speed_ms**2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lateral_accel = np.clip(np.nan_to_num(lateral_accel, nan=0.0), 0, 50)

    SPEED_FLOOR = 8
    G_FLOOR = 0.3
    mask = (speed_ms > SPEED_FLOOR) & (lateral_accel > G_FLOOR)

    corner_speeds = speed_ms[mask]
    corner_g = lateral_accel[mask]

    corners = {
        "Hairpin (<80 km/h)": (corner_speeds[corner_speeds < 22.2], corner_g[corner_speeds < 22.2]),
        "Slow (80-130 km/h)": (corner_speeds[(corner_speeds >= 22.2) & (corner_speeds < 36.1)],
                               corner_g[(corner_speeds >= 22.2) & (corner_speeds < 36.1)]),
        "Medium (130-180 km/h)": (corner_speeds[(corner_speeds >= 36.1) & (corner_speeds < 50.0)],
                                  corner_g[(corner_speeds >= 36.1) & (corner_speeds < 50.0)]),
        "Fast (>180 km/h)": (corner_speeds[corner_speeds >= 50.0], corner_g[corner_speeds >= 50.0]),
    }

    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.text(0.02, 0.98, "Telemetry-based", transform=fig.transFigure, fontsize=9, color="#00D2BE", alpha=0.7, va="top")
    colors = [MERCEDES_TEAL, MERCEDES_AMBER, MERCEDES_RED, "#A855F7"]
    x_positions = []

    for i, (label, (c_speeds, c_g)) in enumerate(corners.items()):
        if len(c_speeds) == 0:
            x_positions.append(i + 1)
            continue
        x_pos = i + 1
        x_positions.append(x_pos)
        ax.scatter(np.full_like(c_speeds, x_pos) + np.random.normal(0, 0.08, len(c_speeds)),
                   c_g, color=colors[i], alpha=0.4, s=15)
        avg_g = np.mean(c_g)
        ax.bar(x_pos, avg_g, 0.5, color=colors[i], alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.text(x_pos, avg_g + 0.1, f"{avg_g:.2f}g", ha="center", color=MERCEDES_WHITE, fontsize=10)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([k.split(" (")[0] for k in corners.keys()])
    ax.set_ylabel("Lateral Acceleration (g)")
    ax.set_title("Corner Classification by Lateral g -- Monaco 2024")
    ax.set_ylim(0, 4.5)
    ax.grid(True, alpha=0.3, axis="y")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "classification.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def driver_kde_comparison(save: bool = True):
    """Bivariate KDE comparison of driving style between two drivers.

    Speed vs Throttle and Speed vs Brake contour density plots reveal
    the statistical 'fingerprint' of each driver's technique.
    Dense regions show where a driver spends most of their time.
    """
    year = 2024
    circuit = "Monaco"
    driver_a = "VER"
    driver_b = "LEC"

    loader = TelemetryLoader(year, circuit, "R")

    def _safe_telemetry(driver):
        try:
            return loader.lap_telemetry(driver)
        except Exception:
            return None

    tel_a = _safe_telemetry(driver_a)
    tel_b = _safe_telemetry(driver_b)
    if tel_a is None or tel_b is None:
        print(f"  WARNING: Telemetry unavailable for {driver_a} or {driver_b}, skipping KDE")
        return None

    SPEED_MIN = 40
    speed_a = tel_a["Speed"].values
    speed_b = tel_b["Speed"].values
    throttle_a = tel_a["Throttle"].values
    throttle_b = tel_b["Throttle"].values

    brake_col = next((c for c in ["Brake", "brake", "BrakeStatus"] if c in tel_a.columns), None)
    if brake_col and brake_col in tel_a.columns:
        brake_a = tel_a[brake_col].values.astype(bool)
        brake_b = tel_b[brake_col].values.astype(bool)
    else:
        brake_a = np.zeros_like(speed_a, dtype=bool)
        brake_b = np.zeros_like(speed_b, dtype=bool)

    mask_a = speed_a > SPEED_MIN
    mask_b = speed_b > SPEED_MIN

    if mask_a.sum() < 50 or mask_b.sum() < 50:
        print(f"  WARNING: Insufficient data points for KDE, skipping")
        return None

    if brake_a.sum() < 10 or brake_b.sum() < 10:
        print(f"  WARNING: Insufficient brake events for KDE, skipping brake panel")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.text(0.02, 0.98, "Telemetry-based", transform=fig.transFigure, fontsize=9, color="#00D2BE", alpha=0.7, va="top")

    custom_style = {
        "axes.facecolor": MERCEDES_DARK,
        "axes.edgecolor": MERCEDES_CARD,
        "axes.labelcolor": MERCEDES_GRAY,
        "text.color": MERCEDES_GRAY,
        "grid.color": MERCEDES_CARD,
        "grid.alpha": 0.3,
    }
    with sns.axes_style("whitegrid", rc=custom_style):
        sns.kdeplot(
            x=speed_a[mask_a], y=throttle_a[mask_a],
            ax=ax1, fill=True, alpha=0.35,
            cmap="Blues", thresh=0.05,
        )
        sns.kdeplot(
            x=speed_b[mask_b], y=throttle_b[mask_b],
            ax=ax1, fill=True, alpha=0.35,
            cmap="Oranges", thresh=0.05,
        )
        ax1.legend(handles=[
            Patch(color="#1f77b4", alpha=0.5, label=driver_a),
            Patch(color="#ff7f0e", alpha=0.5, label=driver_b),
        ], framealpha=0.9)

        if brake_a.sum() >= 10 and brake_b.sum() >= 10:
            sns.kdeplot(
                x=speed_a[mask_a & brake_a],
                ax=ax2, fill=True, alpha=0.35,
                color="#00D2BE", thresh=0.05,
                bw_adjust=0.5,
            )
            sns.kdeplot(
                x=speed_b[mask_b & brake_b],
                ax=ax2, fill=True, alpha=0.35,
                color="#F97316", thresh=0.05,
                bw_adjust=0.5,
            )
            ax2.legend(handles=[
                Patch(color="#00D2BE", alpha=0.5, label=driver_a),
                Patch(color="#F97316", alpha=0.5, label=driver_b),
            ], framealpha=0.9)

    ax1.set_xlabel("Speed (km/h)")
    ax1.set_ylabel("Throttle (%)")
    ax1.set_title(f"Speed vs Throttle Density -- {driver_a} vs {driver_b}")

    ax2.set_xlabel("Speed at Brake Application (km/h)")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Brake Application Speed Profile -- {driver_a} vs {driver_b}")

    for ax in [ax1, ax2]:
        ax.tick_params(colors=MERCEDES_GRAY)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set(color=MERCEDES_GRAY)
        ax.set_facecolor(MERCEDES_DARK)

    fig.suptitle(
        f"Driver Style Comparison -- {circuit} {year} Fastest Lap",
        fontsize=14, y=1.02, color=MERCEDES_WHITE,
    )
    fig.tight_layout()

    if save:
        path = ASSET_DIR / "driver_kde.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def run_all():
    """Generate all cornering analysis visuals."""
    print("Generating cornering analysis visuals...")
    gg_diagram()
    speed_vs_corner_radius()
    downforce_grip_envelope()
    corner_classification()
    driver_kde_comparison()
    print("Done. Files saved to docs/assets/images/cornering/")


if __name__ == "__main__":
    run_all()
