"""Validation of venturi CFD results against published data.

Compares SU2 RANS results with known venturi/ground effect
benchmarks from literature and the analytical model.
"""

from pathlib import Path
import json
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
    MERCEDES_AMBER,
)
from src.core.models import Floor
from src.core.physics import dynamic_pressure

set_f1_style()

ASSET_DIR = Path("docs/assets/images")
ASSET_DIR.mkdir(parents=True, exist_ok=True)

SU2_DIR = Path("su2_runs")
SU2_MESH_DIR = SU2_DIR / "meshes"
SU2_CFG_DIR = SU2_DIR / "configs"
SU2_RESULT_DIR = SU2_DIR / "results"


def validate_vs_analytical(save: bool = True):
    """Compare SU2 results with the analytical Floor model."""
    floor_model = Floor()

    heights = np.linspace(0.025, 0.100, 20)
    cl_analytical = [abs(floor_model.cl(h, velocity_ms=60.0)) for h in heights]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(heights * 1000, cl_analytical, "-", color=MERCEDES_TEAL, linewidth=2.5, label="Analytical model")

    try:
        with open(SU2_DIR / "ride_height_sweep.json") as f:
            cfd_data = json.load(f)
        if cfd_data:
            cfd_hs = [cfd_data[k]["ride_height"] * 1000 for k in cfd_data]
            cfd_cls = [cfd_data[k]["cl"] for k in cfd_data]
            ax.plot(cfd_hs, cfd_cls, "o", color=MERCEDES_RED, markersize=8,
                    markerfacecolor="none", markeredgewidth=2, label="SU2 CFD")
    except (FileNotFoundError, json.JSONDecodeError):
        ax.text(0.5, 0.5, "SU2 data not yet available\n(run venturi simulations first)",
                transform=ax.transAxes, ha="center", va="center",
                color=MERCEDES_GRAY, fontsize=12)
    except Exception:
        pass

    ax.set_xlabel("Ride Height (mm)")
    ax.set_ylabel("C$_L$ (downforce)")
    ax.set_title("CFD vs Analytical: Downforce vs Ride Height")
    ax.invert_xaxis()
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "cfd_venturi_validation.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def plot_convergence(save: bool = True):
    """Plot convergence history if available."""
    result_dirs = sorted(SU2_RESULT_DIR.iterdir()) if SU2_RESULT_DIR.exists() else []
    if not result_dirs:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No SU2 results available yet",
                transform=ax.transAxes, ha="center", va="center",
                color=MERCEDES_GRAY, fontsize=12)
        if save:
            path = ASSET_DIR / "cfd_venturi_convergence.png"
            fig.savefig(path)
            plt.close(fig)
        return fig

    fig, ax = plt.subplots(figsize=(10, 5))
    for rd in result_dirs[:4]:
        hist = rd / "history.csv"
        if hist.exists():
            try:
                data = np.loadtxt(hist, delimiter=",", skiprows=1)
                ax.semilogy(data[:, 0], data[:, 1], label=rd.name, alpha=0.7)
            except Exception:
                pass

    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMS Residual")
    ax.set_title("SU2 Convergence History")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set(color=MERCEDES_GRAY)

    if save:
        path = ASSET_DIR / "cfd_venturi_convergence.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")
    return fig


def run_all():
    """Generate all validation visuals."""
    print("Generating CFD validation visuals...")
    validate_vs_analytical()
    plot_convergence()
    print("Done.")
