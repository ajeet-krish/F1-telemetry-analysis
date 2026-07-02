"""Runner for CFD front wing analysis.

Usage:
    uv run python -m runners.cfd_front_wing                # all sweeps
    uv run python -m runners.cfd_front_wing --quick        # mesh preview only
    uv run python -m runners.cfd_front_wing --export       # reference case
    uv run python -m runners.cfd_front_wing --export-all   # all sweeps
"""

import sys
from pathlib import Path
import numpy as np

from src.cfd.airfoil import front_wing_3_element, preview_front_wing
from src.cfd.su2_runner import MeshGenerator
from src.cfd.wing import run_reference, run_all
from src.core.style import set_f1_style


def run_mesh_preview():
    """Generate mesh validation/preview plots without running SU2."""
    print("Generating mesh preview (no SU2 simulations)...")
    out_dir = Path("docs/assets/images/cfd/front_wing")
    out_dir.mkdir(parents=True, exist_ok=True)

    elements = front_wing_3_element(aoa_main=-6.0)
    preview_front_wing(out_dir)

    print("Generating reference mesh...")
    mesh_file = MeshGenerator.airfoil_cgrid_2d(
        elements, ride_height=0.05, name="fw_reference",
        max_size_body=0.05, max_size_far=0.5, bl_n_layers=0,
    )
    mesh_size_kb = mesh_file.stat().st_size / 1024
    print(f"  Mesh: {mesh_file} ({mesh_size_kb:.0f} KB)")

    main = elements[0]["coords"]
    mid = elements[1]["coords"]
    flap = elements[2]["coords"]
    print(f"  Main: y=[{main[:,1].min():.4f}, {main[:,1].max():.4f}]")
    print(f"  Mid:  y=[{mid[:,1].min():.4f}, {mid[:,1].max():.4f}]")
    print(f"  Flap: y=[{flap[:,1].min():.4f}, {flap[:,1].max():.4f}]")
    print("Done.")


if __name__ == "__main__":
    if "--quick" in sys.argv:
        run_mesh_preview()
    elif "--export-all" in sys.argv:
        run_all()
        from src.cfd.pyvista_viz import export_fw_all
        export_fw_all()
    elif "--export" in sys.argv:
        run_reference()
        from src.cfd.pyvista_viz import export_fw_reference
        export_fw_reference()
    else:
        run_all()
        from src.cfd.pyvista_viz import export_fw_all
        export_fw_all()
