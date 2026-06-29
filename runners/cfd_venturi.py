"""Runner for CFD venturi analysis.

Usage:
    uv run python -m runners.cfd_venturi            # full simulation
    uv run python -m runners.cfd_venturi --quick    # mesh preview only
"""

import sys

from src.cfd.venturi import run_all
from src.cfd.validate import run_all as run_validate
from src.cfd.su2_runner import MeshGenerator


def run_mesh_preview():
    """Generate the mesh validation/preview plots without running SU2."""
    print("Generating mesh preview (no SU2 simulations)...")
    from src.cfd.venturi import mesh_validation
    mesh_validation()
    print("Done.")


if __name__ == "__main__":
    if "--quick" in sys.argv:
        run_mesh_preview()
    else:
        run_all()
        run_validate()
