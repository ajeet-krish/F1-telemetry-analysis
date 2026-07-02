"""Front wing CFD analysis for F1 3-element inverted wing in ground effect.

Generates SU2 RANS simulations of a 3-element inverted front wing with
parametric sweeps over AoA, ride height, slot gap, and active aero deployment.

Sweeps:
  Reference:  AoA=0deg, h=50mm, sg=50mm
  AoA:        -20, -16, -12, -8, -4, 0, 4 deg
  Ride height: 10, 20, 35, 50, 65, 80 mm
  Slot gap:    10, 15, 20, 25 mm
  Active aero: flap deploy +0, +5, +10, +15, +20 deg
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.core.style import set_f1_style, MERCEDES_TEAL, MERCEDES_RED
from src.cfd.airfoil import front_wing_3_element
from src.cfd.su2_runner import MeshGenerator, SU2Config, SU2Solver, SU2Results

set_f1_style()

ASSET_DIR = Path("docs/assets/images/cfd/front_wing")
ASSET_DIR.mkdir(parents=True, exist_ok=True)

SU2_DIR = Path("su2_runs")
SU2_MESH_DIR = SU2_DIR / "meshes"
SU2_CFG_DIR = SU2_DIR / "configs"
SU2_RESULT_DIR = SU2_DIR / "results" / "front_wing"
for d in [SU2_MESH_DIR, SU2_CFG_DIR, SU2_RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

AIR_DENSITY = 1.225
AIR_VISCOSITY = 1.81e-5

# Sweep definition matching AGENTS.md
# Using negative AoA convention (nose-down = TE above LE)
SWEEP_PARAMS = {
    "reference": {"aoa_main": 0, "rh": 0.050, "sg": 0.050, "flap_deploy": 0},
    "aoa_main": {"values": [-20, -16, -12, -8, -4, 0, 4],
            "fixed": {"rh": 0.050, "sg": 0.050, "flap_deploy": 0}},
    "rh": {"values": [0.010, 0.020, 0.035, 0.050, 0.065, 0.080],
                    "fixed": {"aoa_main": 0, "sg": 0.050, "flap_deploy": 0}},
    "sg": {"values": [0.010, 0.015, 0.020, 0.025],
                 "fixed": {"aoa_main": 0, "rh": 0.050, "flap_deploy": 0}},
    "flap_deploy": {"values": [0, 5, 10, 15, 20],
                    "fixed": {"aoa_main": 0, "rh": 0.050, "sg": 0.050}},
}


def case_stem(sweep: str, value) -> str:
    """Generate a unique stem for each simulation case."""
    if sweep == "reference":
        return "fw_reference"
    return f"fw_{sweep}_{value}"


def run_case(
    sweep: str,
    value,
    rh: float = 0.050,
    aoa_main: float = 0.0,
    sg: float = 0.050,
    flap_deploy: float = 0.0,
    velocity_ms: float = 60.0,
    timeout: int = 600,
    iterations: int = 4000,
    bl_layers: int = 30,
) -> tuple[SU2Results, Path, Path]:
    """Run a single front wing simulation case."""
    stem = case_stem(sweep, value)
    elements = front_wing_3_element(
        aoa_main=aoa_main, slot_gap=sg, flap_deploy=flap_deploy,
    )
    mesh = MeshGenerator.airfoil_cgrid_2d(
        elements, ride_height=rh, name=stem,
        max_size_body=0.01, max_size_far=0.15,
        bl_n_layers=bl_layers,
    )
    mesh_path = SU2_MESH_DIR / mesh.name
    mesh.rename(mesh_path)

    chord_ref = elements[0]["chord"]
    config = SU2Config.for_airfoil(
        aoa=aoa_main, vel=velocity_ms, chord=chord_ref, ride_height=rh, euler=True,
    )
    config.screen_output = "WARNING"

    cfg_path = SU2_CFG_DIR / f"{stem}.cfg"
    config.write(cfg_path)

    result_dir = SU2_RESULT_DIR / stem
    result_dir.mkdir(parents=True, exist_ok=True)

    solver = SU2Solver(workdir=result_dir)
    result = solver.run(cfg_path, mesh_path, timeout=timeout)

    return result, mesh_path, cfg_path


def run_sweep(
    sweep: str,
    save_plot: bool = True,
    timeout: int = 600,
) -> dict:
    """Run all cases in a sweep and generate comparison plots."""
    params = SWEEP_PARAMS[sweep]
    if sweep == "reference":
        params_run = [params]
    else:
        params_run = [{**params["fixed"], sweep: v} for v in params["values"]]

    results = {}
    for p in params_run:
        if sweep == "reference":
            val = "ref"
        else:
            val = p[sweep]
        stem = case_stem(sweep, val)
        print(f"  Running {stem}...")

        try:
            result, _, _ = run_case(
                sweep=sweep, value=val,
                aoa_main=p.get("aoa_main", SWEEP_PARAMS["reference"]["aoa_main"]),
                rh=p.get("rh", SWEEP_PARAMS["reference"]["rh"]),
                sg=p.get("sg", SWEEP_PARAMS["reference"]["sg"]),
                flap_deploy=p.get("flap_deploy", 0),
                timeout=timeout,
            )
            results[stem] = {
                "cl": result.cl,
                "cd": result.cd,
                "converged": result.converged,
                "iterations": result.iterations,
            }
            print(f"    CL={result.cl:.4f}, CD={result.cd:.4f}, converged={result.converged}")
        except Exception as e:
            print(f"    Failed: {e}")
            results[stem] = {"cl": 0, "cd": 0, "converged": False, "error": str(e)}

    # Save sweep data
    sweep_file = SU2_DIR / f"front_wing_{sweep}.json"
    with open(sweep_file, "w") as f:
        json.dump(results, f, indent=2)

    if save_plot and sweep != "reference":
        _plot_sweep(sweep, results)

    return results


def _plot_sweep(sweep: str, results: dict):
    """Plot sweep results: CL, CD, L/D vs sweep parameter."""
    params = SWEEP_PARAMS[sweep]
    values = params["values"]

    x_label_map = {
        "aoa_main": "Main AoA (deg)",
        "rh": "Ride Height (mm)",
        "sg": "Slot Gap (mm)",
        "flap_deploy": "Flap Deploy (deg)",
    }
    x_scale_map = {
        "aoa_main": 1.0,
        "rh": 1000.0,
        "sg": 1000.0,
        "flap_deploy": 1.0,
    }

    x_vals = [v * x_scale_map[sweep] for v in values if f"fw_{sweep.replace('ride_height','h').replace('slot_gap','sg').replace('active_aero','f')}{v}" in results]

    # Actually build keys correctly
    stems = [case_stem(sweep, v) for v in values]
    x_vals = [v * x_scale_map[sweep] for v in values]

    cl_vals = [results[s].get("cl", 0) for s in stems if s in results]
    cd_vals = [results[s].get("cd", 0) for s in stems if s in results]
    ld_vals = [abs(cl/cd) if cd != 0 else 0 for cl, cd in zip(cl_vals, cd_vals)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(x_vals[:len(cl_vals)], cl_vals, "o-", color=MERCEDES_TEAL, linewidth=2.5, markersize=8)
    ax.set_xlabel(x_label_map[sweep])
    ax.set_ylabel("C$_L$ (downforce)")
    ax.set_title("Downforce")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x_vals[:len(cd_vals)], cd_vals, "s--", color=MERCEDES_RED, linewidth=2.5, markersize=8)
    ax.set_xlabel(x_label_map[sweep])
    ax.set_ylabel("C$_D$ (drag)")
    ax.set_title("Drag")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(x_vals[:len(ld_vals)], ld_vals, "D-.", color="#FFB347", linewidth=2.5, markersize=8)
    ax.set_xlabel(x_label_map[sweep])
    ax.set_ylabel("L/D")
    ax.set_title("Aerodynamic Efficiency")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Front Wing {sweep.replace('_',' ').title()} Sweep", fontsize=14, y=1.02)
    fig.tight_layout()
    path = ASSET_DIR / f"{sweep}_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def run_reference():
    """Run reference case only."""
    return run_sweep("reference")


def run_aoa_sweep():
    """Run AoA sweep (7 cases)."""
    return run_sweep("aoa_main")


def run_ride_height_sweep():
    """Run ride height sweep (6 cases)."""
    return run_sweep("rh")


def run_slot_gap_sweep():
    """Run slot gap sweep (4 cases)."""
    return run_sweep("sg")


def run_active_aero_sweep():
    """Run active aero sweep (5 cases)."""
    return run_sweep("flap_deploy")


def run_all():
    """Run all sweeps sequentially."""
    print("=" * 50)
    print("Front Wing CFD: Running all sweeps")
    print("=" * 50)

    print("\n--- Reference ---")
    run_reference()

    for sweep in ["aoa_main", "rh", "sg", "flap_deploy"]:
        print(f"\n--- {sweep.replace('_',' ').title()} Sweep ---")
        run_sweep(sweep)
