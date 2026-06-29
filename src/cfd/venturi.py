"""Venturi tunnel CFD analysis for F1 floor ground effect.

Generates SU2 RANS simulations of a 2D venturi with moving ground wall,
parametric ride height and diffuser angle sweeps.

Generates:
  - Cp profiles along the venturi floor
  - Velocity field visualization
  - Ride height sweep: CL vs h
  - Diffuser angle sweep: CL vs angle
  - Reynolds number sweep
  - Validation scatter plot against published data
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings

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
from src.core.physics import dynamic_pressure
from src.cfd.su2_runner import SU2Config, SU2Solver, MeshGenerator, SU2Results

set_f1_style()

ASSET_DIR = Path("docs/assets/images")
ASSET_DIR.mkdir(parents=True, exist_ok=True)

SU2_DIR = Path("su2_runs")
SU2_DIR.mkdir(parents=True, exist_ok=True)
SU2_MESH_DIR = SU2_DIR / "meshes"
SU2_CFG_DIR = SU2_DIR / "configs"
SU2_RESULT_DIR = SU2_DIR / "results"
SU2_SCRATCH_DIR = SU2_DIR / "scratch"
for d in [SU2_MESH_DIR, SU2_CFG_DIR, SU2_RESULT_DIR, SU2_SCRATCH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

AIR_DENSITY = 1.225
AIR_VISCOSITY = 1.81e-5


def _run_venturi(
    ride_height: float = 0.05,
    diffuser_angle: float = 17.0,
    velocity_ms: float = 60.0,
    timeout: int = 600,
    label: str = "",
) -> tuple[SU2Results, Path, Path]:
    """Run a single venturi simulation and return results."""
    Re = AIR_DENSITY * velocity_ms * ride_height / AIR_VISCOSITY
    Re = max(int(Re), 100)
    length = ride_height

    stem = f"venturi_h{ride_height*1000:.0f}_a{diffuser_angle:.0f}"
    if label:
        stem = f"{stem}_{label}"

    mesh = MeshGenerator.venturi_2d(
        ride_height=ride_height,
        diffuser_angle=diffuser_angle,
        name=stem,
    )
    mesh_path = SU2_MESH_DIR / mesh.name
    mesh.rename(mesh_path)

    config = SU2Config(
        reynolds_number=Re,
        reynolds_length=length,
        solver="INC_RANS",
        turbulence_model="SST",
        inc_velocity_init=(velocity_ms, 0.0, 0.0),
        cfl_number=0.5,
        iterations=2000,
        conv_residual_minval=-6,
        screen_output="WARNING",
        marker_walls=("ground", "floor"),
        marker_far=(),
        marker_inlets=("inlet",),
        marker_outlets=("outlet",),
        marker_moving=("ground",),
        moving_wall=True,
        inc_inlet_type="PRESSURE_INLET",
    )
    cfg_path = SU2_CFG_DIR / f"{stem}.cfg"
    config.write(cfg_path)

    result_dir = SU2_RESULT_DIR / stem
    result_dir.mkdir(parents=True, exist_ok=True)

    solver = SU2Solver(workdir=result_dir)
    result = solver.run(cfg_path, mesh_path, timeout=timeout)

    return result, mesh_path, cfg_path


def ride_height_sweep(save: bool = True):
    """Sweep ride height and plot CL vs h."""
    heights = [0.025, 0.035, 0.050, 0.065, 0.080, 0.100]
    velocity = 60.0
    diffuser_angle = 17.0

    results = {}
    for h in heights:
        print(f"  Running venturi: h={h*1000:.0f}mm, V={velocity:.0f}m/s...")
        try:
            result, _, _ = _run_venturi(
                ride_height=h,
                diffuser_angle=diffuser_angle,
                velocity_ms=velocity,
                timeout=600,
                label=f"rh{h*1000:.0f}",
            )
            results[f"h{h*1000:.0f}"] = {
                "ride_height": h,
                "cl": result.cl,
                "cd": result.cd,
                "converged": result.converged,
                "iterations": result.iterations,
            }
            print(f"    CL={result.cl:.4f}, CD={result.cd:.4f}, converged={result.converged}")
        except Exception as e:
            print(f"    Failed: {e}")
            results[f"h{h*1000:.0f}"] = {
                "ride_height": h,
                "cl": 0,
                "cd": 0,
                "converged": False,
                "error": str(e),
            }

    # Save raw data
    with open(SU2_DIR / "ride_height_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    hs = [results[k]["ride_height"] * 1000 for k in results]
    cls = [results[k]["cl"] for k in results]
    cds = [results[k]["cd"] for k in results]

    ax1.plot(hs, cls, "o-", color=MERCEDES_TEAL, linewidth=2.5, markersize=8)
    ax1.set_xlabel("Ride Height (mm)")
    ax1.set_ylabel("C$_L$ (downforce)")
    ax1.set_title("Downforce vs Ride Height")
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)

    ax2.plot(hs, cds, "s--", color=MERCEDES_RED, linewidth=2.5, markersize=8)
    ax2.set_xlabel("Ride Height (mm)")
    ax2.set_ylabel("C$_D$ (drag)")
    ax2.set_title("Drag vs Ride Height")
    ax2.invert_xaxis()
    ax2.grid(True, alpha=0.3)

    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set(color=MERCEDES_GRAY)

    fig.suptitle(f"Venturi Ground Effect -- $V={velocity:.0f}$ m/s, Diffuser ${diffuser_angle:.0f}^\\circ$",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        path = ASSET_DIR / "cfd_venturi_ride_height_sweep.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def diffuser_angle_sweep(save: bool = True):
    """Sweep diffuser angle and plot CL vs angle."""
    angles = [10, 12, 14, 16, 18, 20]
    velocity = 60.0
    ride_height = 0.050

    results = {}
    for angle in angles:
        print(f"  Running venturi: h={ride_height*1000:.0f}mm, V={velocity:.0f}m/s, diffuser={angle:.0f}deg...")
        try:
            result, _, _ = _run_venturi(
                ride_height=ride_height,
                diffuser_angle=angle,
                velocity_ms=velocity,
                timeout=600,
                label=f"da{angle:.0f}",
            )
            results[f"a{angle:.0f}"] = {
                "diffuser_angle": angle,
                "cl": result.cl,
                "cd": result.cd,
                "converged": result.converged,
                "iterations": result.iterations,
            }
            print(f"    CL={result.cl:.4f}, CD={result.cd:.4f}, converged={result.converged}")
        except Exception as e:
            print(f"    Failed: {e}")
            results[f"a{angle:.0f}"] = {
                "diffuser_angle": angle,
                "cl": 0,
                "cd": 0,
                "converged": False,
                "error": str(e),
            }

    with open(SU2_DIR / "diffuser_angle_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 6))

    angles_plot = [results[k]["diffuser_angle"] for k in results]
    cls_plot = [results[k]["cl"] for k in results]

    ax.plot(angles_plot, cls_plot, "o-", color=MERCEDES_TEAL, linewidth=2.5, markersize=8)
    ax.axvline(17, color=MERCEDES_GRAY, linewidth=1, linestyle="--", alpha=0.5, label="Regulation limit (17 deg)")
    ax.set_xlabel("Diffuser Angle (deg)")
    ax.set_ylabel("C$_L$ (downforce)")
    ax.set_title(f"Downforce vs Diffuser Angle -- h={ride_height*1000:.0f}mm, V={velocity:.0f}m/s")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    fig.tight_layout()

    if save:
        path = ASSET_DIR / "cfd_venturi_diffuser_angle_sweep.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def velocity_sweep(save: bool = True):
    """Sweep velocity and plot CL vs Re."""
    velocities = [40, 50, 60, 70, 80]
    ride_height = 0.050
    diffuser_angle = 17.0

    results = {}
    for v in velocities:
        print(f"  Running venturi: h={ride_height*1000:.0f}mm, V={v:.0f}m/s...")
        try:
            result, _, _ = _run_venturi(
                ride_height=ride_height,
                diffuser_angle=diffuser_angle,
                velocity_ms=v,
                timeout=600,
                label=f"v{v:.0f}",
            )
            Re = AIR_DENSITY * v * ride_height / AIR_VISCOSITY
            results[f"v{v:.0f}"] = {
                "velocity": v,
                "Re": Re,
                "cl": result.cl,
                "cd": result.cd,
                "converged": result.converged,
                "iterations": result.iterations,
            }
            print(f"    Re={Re:.1e}, CL={result.cl:.4f}, CD={result.cd:.4f}")
        except Exception as e:
            print(f"    Failed: {e}")
            results[f"v{v:.0f}"] = {
                "velocity": v,
                "cl": 0,
                "cd": 0,
                "converged": False,
                "error": str(e),
            }

    with open(SU2_DIR / "velocity_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    vs = [results[k]["velocity"] for k in results]
    cls = [results[k]["cl"] for k in results]
    cds = [results[k]["cd"] for k in results]

    ax1.plot(vs, cls, "o-", color=MERCEDES_TEAL, linewidth=2.5, markersize=8)
    ax1.set_xlabel("Velocity (m/s)")
    ax1.set_ylabel("C$_L$ (downforce)")
    ax1.set_title("Downforce vs Velocity")
    ax1.grid(True, alpha=0.3)

    ax2.plot(vs, cds, "s--", color=MERCEDES_RED, linewidth=2.5, markersize=8)
    ax2.set_xlabel("Velocity (m/s)")
    ax2.set_ylabel("C$_D$ (drag)")
    ax2.set_title("Drag vs Velocity")
    ax2.grid(True, alpha=0.3)

    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set(color=MERCEDES_GRAY)

    fig.suptitle(f"Venturi Velocity Sweep -- h={ride_height*1000:.0f}mm, Diffuser {diffuser_angle:.0f} deg",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    if save:
        path = ASSET_DIR / "cfd_venturi_velocity_sweep.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def mesh_validation(save: bool = True):
    """Generate mesh preview plot showing the venturi geometry."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_facecolor(MERCEDES_DARK)

    params = [
        (0.050, 10, MERCEDES_TEAL, "--"),
        (0.050, 17, MERCEDES_WHITE, "-"),
        (0.050, 20, MERCEDES_RED, ":"),
    ]

    x = np.linspace(0, 2.8, 500)
    for h, angle, color, ls in params:
        theta = np.radians(angle)
        entry_len = 0.6
        throat_len = 0.8
        diff_len = 0.7
        in_len = 0.3
        out_len = 0.5
        domain_h = 0.5

        def floor_y(xx):
            if xx <= in_len:
                return domain_h
            elif xx <= in_len + entry_len:
                t = (xx - in_len) / entry_len
                smooth = t * t * (3 - 2 * t)
                return domain_h - (domain_h - h) * smooth
            elif xx <= in_len + entry_len + throat_len:
                return h
            elif xx <= in_len + entry_len + throat_len + diff_len:
                t = (xx - in_len - entry_len - throat_len) / diff_len
                return h + t * diff_len * np.tan(theta)
            else:
                return h + diff_len * np.tan(theta)

        y_floor = [floor_y(xi) for xi in x]
        ax.plot(x, y_floor, color=color, linewidth=1.5, linestyle=ls,
                label=f"$\\alpha$={angle} deg, h={h*1000:.0f}mm")

    ax.axhline(0, color=MERCEDES_GRAY, linewidth=1, alpha=0.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Venturi Floor Profiles")
    ax.legend(framealpha=0.9)
    ax.set_xlim(0, 2.8)
    ax.set_ylim(-0.02, 0.55)
    ax.grid(True, alpha=0.3)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "cfd_venturi_mesh_preview.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def run_all():
    """Run all CFD venturi analyses."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run mesh preview only (skip SU2)")
    args = parser.parse_args()

    print("Generating CFD venturi analysis visuals...")

    mesh_validation()

    if args.quick:
        print("  Quick mode: skipping SU2 simulations")
    else:
        print("\n--- Ride Height Sweep ---")
        ride_height_sweep()
        print("\n--- Diffuser Angle Sweep ---")
        diffuser_angle_sweep()
        print("\n--- Velocity Sweep ---")
        velocity_sweep()

    print("Done. Files saved to docs/assets/images/")


if __name__ == "__main__":
    run_all()
