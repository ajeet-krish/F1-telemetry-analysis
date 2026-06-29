"""SU2 simulation runner -- config generation, mesh creation, solver invocation, result parsing, PyVista viz."""

from __future__ import annotations

import dataclasses
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

SU2_BIN = Path("/Users/ajeet/SU2_CFD/bin")
SU2_CFD = SU2_BIN / "SU2_CFD"
SU2_DEF = SU2_BIN / "SU2_DEF"
SU2_GEO = SU2_BIN / "SU2_GEO"


# ── Config generation ──

@dataclasses.dataclass
class SU2Config:
    """Parameters for an SU2 .cfg file."""

    reynolds_number: float = 40_000
    reynolds_length: float = 0.6
    mach_number: float = 0.059
    angle_of_attack: float = 0.0
    solver: str = "INC_RANS"
    turbulence_model: str = "SST"
    kind_trans_model: str = ""
    inc_density_model: str = "CONSTANT"
    inc_energy_eq: str = "NO"
    inc_velocity_init: tuple = (1.0, 0.0, 0.0)
    inc_density_init: float = 1.2886
    cfl_number: float = 1.0
    conv_residual_minval: float = -8
    conv_startiter: float = 10
    conv_field: str = "DRAG"
    iterations: int = 3000
    screen_output: str = "WARNING"
    ref_area: Optional[float] = None
    rotation_rate: float = 0.0
    wall_roughness: float = 0.0
    muscl_flow: str = "NO"
    conv_numerical_method_flow: str = "FDS"
    # Marker names (override for custom meshes)
    marker_walls: tuple = ("wall",)
    marker_far: tuple = ("farfield",)
    marker_inlets: tuple = ()
    marker_outlets: tuple = ()
    marker_moving: tuple = ("wall",)
    moving_wall: bool = False
    inc_inlet_type: str = "VELOCITY_INLET"  # VELOCITY_INLET or PRESSURE_INLET
    inc_outlet_type: str = "PRESSURE_OUTLET"  # PRESSURE_OUTLET for incompressible
    time_domain: bool = False
    time_marching: str = "NO"
    time_step: float = 0.0
    max_time: float = 0.0
    time_iter: int = 1
    inner_iter: int = 15
    output_wrt_freq: int = 1

    @classmethod
    def from_re(cls, Re: float, length: float = 0.6, incompressible: bool = True) -> "SU2Config":
        if incompressible:
            return cls(reynolds_number=Re, reynolds_length=length, solver="INC_RANS")
        return cls(reynolds_number=Re, reynolds_length=length, solver="RANS",
                    mach_number=0.059 * (Re / 200_000))

    def write(self, path: Path) -> None:
        is_inc = self.solver.startswith("INC_")
        lines = [
            f"% ------- CONFIG FILE (auto-generated) --------",
            f"SOLVER= {self.solver}",
            f"{('KIND_TURB_MODEL= ' + self.turbulence_model) if ('RANS' in self.solver or 'rans' in self.solver) else ('KIND_TURB_MODEL= NONE' if self.turbulence_model == 'NONE' else '% No turbulence model (laminar)')}",
            f"{f'KIND_TRANS_MODEL= {self.kind_trans_model}' if self.kind_trans_model else '% No transition model'}",
            f"MATH_PROBLEM= DIRECT",
            f"RESTART_SOL= NO",
            f"SYSTEM_MEASUREMENTS= SI",
            f"",
        ]
        if is_inc:
            mu_nd = 1.0 / self.reynolds_number
            lines += [
                f"% ---------------- INCOMPRESSIBLE FLOW CONDITION DEFINITION ------------",
                f"INC_DENSITY_MODEL= {self.inc_density_model}",
                f"INC_ENERGY_EQUATION= {self.inc_energy_eq}",
                f"INC_DENSITY_INIT= {self.inc_density_init}",
                f"INC_VELOCITY_INIT= ( {self.inc_velocity_init[0]}, {self.inc_velocity_init[1]}, {self.inc_velocity_init[2]} )",
                f"INC_NONDIM= INITIAL_VALUES",
                f"REYNOLDS_NUMBER= {int(self.reynolds_number)}",
                f"REYNOLDS_LENGTH= {self.reynolds_length}",
                f"VISCOSITY_MODEL= CONSTANT_VISCOSITY",
                f"MU_CONSTANT= {mu_nd:.6e}",
                f"",
            ]
        else:
            lines += [
                f"% -------------------- COMPRESSIBLE FREE-STREAM DEFINITION -----------",
                f"MACH_NUMBER= {self.mach_number:.6f}",
                f"AOA= {self.angle_of_attack:.2f}",
                f"REYNOLDS_NUMBER= {int(self.reynolds_number)}",
                f"REYNOLDS_LENGTH= {self.reynolds_length}",
                f"FREESTREAM_TEMPERATURE= 288.15",
                f"",
            ]
        lines += [
            f"% ------------------------ BOUNDARY CONDITIONS -------------------------",
            f"MARKER_HEATFLUX= ( {' , '.join(f'{m}, 0.0' for m in self.marker_walls)} )",
            f"{f'WALL_ROUGHNESS= ( {self.marker_walls[0]}, {self.wall_roughness:.6e} )' if self.wall_roughness > 0 else '% No wall roughness'}",
            f"MARKER_MONITORING= ( {self.marker_walls[0]} )",
            f"MARKER_FAR= ( {', '.join(self.marker_far)} )" if self.marker_far else "% No farfield",
            f"{'INC_INLET_TYPE= ' + ' '.join([self.inc_inlet_type] * len(self.marker_inlets)) if self.marker_inlets else '% No inlet'}",
            f"MARKER_INLET= ( {self.marker_inlets[0]}, 60.0, 1.0, 0.0, 0.0, 1.225 )" if self.marker_inlets and self.inc_inlet_type == "VELOCITY_INLET" else (
            f"MARKER_INLET= ( {self.marker_inlets[0]}, 288.15, 101325.0, 1.0, 0.0, 0.0 )" if self.marker_inlets else "% No inlet"),
            f"{f'INC_OUTLET_TYPE= {self.inc_outlet_type}' if self.marker_outlets else '% No outlet'}",
            f"{f'MARKER_OUTLET= ( {self.marker_outlets[0]}, 0.0 )' if self.marker_outlets else '% No outlet'}",
            f"{f'SURFACE_MOVEMENT= MOVING_WALL' if self.moving_wall else '% No moving wall'}",
            f"{f'MARKER_MOVING= ( {self.marker_moving[0]} )' if self.moving_wall else ''}",
            f"{f'SURFACE_TRANSLATION_RATE= 1.0 0.0 0.0' if self.moving_wall else ''}",
            f"{f'SURFACE_MOTION_ORIGIN= 0.0 0.0 0.0' if self.moving_wall else ''}",
            f"{f'SURFACE_ROTATION_RATE= 0.0 0.0 {self.rotation_rate}' if abs(self.rotation_rate) > 1e-10 else ''}",
            f"",
            f"% ------------------------ NUMERICAL METHOD DEFINITION -------------------",
            f"CONV_NUM_METHOD_FLOW= {self.conv_numerical_method_flow}",
            f"MUSCL_FLOW= {self.muscl_flow}",
            f"SLOPE_LIMITER_FLOW= VENKATAKRISHNAN",
            f"TIME_DISCRE_FLOW= EULER_IMPLICIT",
            f"{f'CONV_NUM_METHOD_TURB= SCALAR_UPWIND' if 'RANS' in self.solver or 'rans' in self.solver else ''}",
            f"{f'MUSCL_TURB= NO' if 'RANS' in self.solver or 'rans' in self.solver else ''}",
            f"",
            f"% ------------------------- CONVERGENCE PARAMETERS -----------------------",
            f"{'' if self.time_domain else f'ITER= {self.iterations}'}",
            f"CFL_NUMBER= {self.cfl_number}",
            f"CFL_ADAPT= NO",
            f"CONV_FIELD= {self.conv_field}",
            f"CONV_RESIDUAL_MINVAL= {self.conv_residual_minval}",
            f"CONV_STARTITER= {self.conv_startiter}",
            f"CONV_CAUCHY_ELEMS= 100",
            f"CONV_CAUCHY_EPS= 1E-10",
            f"",
            f"% ------------------------- TIME DOMAIN (UNSTEADY) ------------------------",
            f"{f'TIME_DOMAIN= YES' if self.time_domain else '% No time domain' }",
            f"{f'TIME_MARCHING= {self.time_marching}' if self.time_domain else ''}",
            f"{f'TIME_STEP= {self.time_step}' if self.time_domain else ''}",
            f"{f'MAX_TIME= {self.max_time}' if self.time_domain else ''}",
            f"{f'TIME_ITER= {self.time_iter}' if self.time_domain else ''}",
            f"{f'INNER_ITER= {self.inner_iter}' if self.time_domain else ''}",
            f"{f'OUTPUT_WRT_FREQ= {self.output_wrt_freq}' if self.time_domain else ''}",
            f"",
            f"% ------------------------- LINEAR SOLVER --------------------------------",
            f"LINEAR_SOLVER= FGMRES",
            f"LINEAR_SOLVER_PREC= ILU",
            f"LINEAR_SOLVER_ERROR= 1E-6",
            f"LINEAR_SOLVER_ITER= 10",
            f"",
            f"% ------------------------- REFERENCE VALUE DEFINITION -------------------",
            f"{f'REF_AREA= {self.ref_area}' if self.ref_area else '%% REF_AREA not set (assuming 2D)'}",
            f"",
            f"% ------------------------- OUTPUT ---------------------------------------",
            f"SCREEN_OUTPUT= ({self.screen_output})",
            f"HISTORY_OUTPUT= ( TIME_ITER, INNER_ITER, RMS_RES, AERO_COEFF {f', CUR_TIME' if self.time_domain else ''})",
            f"TABULAR_FORMAT= CSV",
            f"OUTPUT_FILES= (RESTART, PARAVIEW)",
            f"",
            f"% ------------------------- MULTIGRID -------------------------------------",
            f"MGLEVEL= 0",
            f"",
        ]
        path.write_text("\n".join(lines))


# ── Mesh generation ──

class MeshGenerator:
    """gmsh-based mesh generation for SU2 cases."""

    @staticmethod
    def venturi_2d(
        ride_height: float = 0.05,
        diffuser_angle: float = 17.0,
        throat_length: float = 0.8,
        entry_length: float = 0.6,
        diffuser_length: float = 0.7,
        domain_height: float = 0.5,
        domain_inlet: float = 0.3,
        domain_outlet: float = 0.5,
        n_x: int = 200,
        n_y: int = 80,
        n_wall: int = 15,
        name: str = "venturi",
    ) -> Path:
        """Create a 2D structured venturi mesh with moving ground.

        The venturi represents an F1 floor with:
        - Flat ground at y=0 (moving wall)
        - Profiled floor on top with converging, throat, and diffuser sections

        Args:
            ride_height: Minimum gap at throat (m)
            diffuser_angle: Diffuser ramp angle (degrees)
            throat_length: Length of constant-height throat section (m)
            entry_length: Length of converging entry section (m)
            diffuser_length: Length of diffuser ramp section (m)
            domain_height: Height of domain above ground at inlet (m)
            domain_inlet: Length of inlet section before venturi (m)
            domain_outlet: Length of outlet section after diffuser (m)
            n_x: Number of cells along domain
            n_y: Number of cells across height
            n_wall: Number of wall-normal cells in boundary layer
            name: Mesh file name (without extension)
        """
        import gmsh

        theta = math.radians(diffuser_angle)
        L_total = domain_inlet + entry_length + throat_length + diffuser_length + domain_outlet

        x_inlet = 0.0
        x_entry_start = domain_inlet
        x_throat_start = x_entry_start + entry_length
        x_diff_start = x_throat_start + throat_length
        x_outlet_start = x_diff_start + diffuser_length
        x_outlet = L_total

        gmsh.initialize()
        gmsh.model.add(name)

        # Venturi floor profile function: y_floor(x)
        def floor_y(x):
            if x <= x_entry_start:
                return domain_height
            elif x <= x_throat_start:
                t = (x - x_entry_start) / entry_length
                smooth = t * t * (3 - 2 * t)
                return domain_height - (domain_height - ride_height) * smooth
            elif x <= x_diff_start:
                return ride_height
            elif x <= x_outlet_start:
                t = (x - x_diff_start) / diffuser_length
                return ride_height + t * diffuser_length * math.tan(theta)
            else:
                return ride_height + diffuser_length * math.tan(theta)

        floor_pts = []
        n_floor = max(int(n_x * (L_total - domain_inlet - domain_outlet) / L_total), 10)
        for i in range(n_floor + 1):
            t = i / n_floor
            x = x_entry_start + t * (x_outlet_start - x_entry_start)
            floor_pts.append((x, floor_y(x)))

        n_in = max(int(n_x * domain_inlet / L_total), 4)
        n_out = max(int(n_x * domain_outlet / L_total), 4)
        n_total = n_in + n_floor + n_out

        # Build domain points
        # Bottom (ground) points
        bottom_pts = []
        for i in range(n_total + 1):
            t = i / n_total
            x = x_inlet + t * (x_outlet - x_inlet)
            bottom_pts.append((x, 0.0))

        # Top boundary points
        top_pts = []
        for i in range(n_total + 1):
            t = i / n_total
            x = x_inlet + t * (x_outlet - x_inlet)
            if x <= x_entry_start:
                y = domain_height
            elif x <= x_outlet_start:
                y = floor_y(x)
            else:
                y = floor_y(x_outlet_start)
            top_pts.append((x, y))

        # Create gmsh points
        gmsh_pts_bottom = []
        for i, (x, y) in enumerate(bottom_pts):
            pt = gmsh.model.geo.addPoint(x, y, 0)
            gmsh_pts_bottom.append(pt)

        gmsh_pts_top = []
        for i, (x, y) in enumerate(top_pts):
            pt = gmsh.model.geo.addPoint(x, y, 0)
            gmsh_pts_top.append(pt)

        # Lines along bottom (ground)
        bottom_lines = []
        for i in range(n_total):
            line = gmsh.model.geo.addLine(gmsh_pts_bottom[i], gmsh_pts_bottom[i + 1])
            bottom_lines.append(line)

        # Lines along top (floor profile)
        top_lines = []
        for i in range(n_total):
            line = gmsh.model.geo.addLine(gmsh_pts_top[i], gmsh_pts_top[i + 1])
            top_lines.append(line)

        # Vertical lines (inlet and outlet)
        inlet_line = gmsh.model.geo.addLine(gmsh_pts_bottom[0], gmsh_pts_top[0])
        outlet_line = gmsh.model.geo.addLine(gmsh_pts_bottom[-1], gmsh_pts_top[-1])

        gmsh.model.geo.synchronize()

        # Interior vertical lines for structured mesh
        interior_lines = []
        for i in range(1, n_total):
            vline = gmsh.model.geo.addLine(gmsh_pts_bottom[i], gmsh_pts_top[i])
            interior_lines.append(vline)

        gmsh.model.geo.synchronize()

        # Create surfaces: each quadrilateral cell = 4 lines
        surface_tags = []
        for i in range(n_total):
            loop_curves = [
                bottom_lines[i],          # bottom edge
                interior_lines[i] if i < n_total - 1 else outlet_line,  # right edge
                -top_lines[i],            # top edge (reversed)
                -interior_lines[i - 1] if i > 0 else -inlet_line,  # left edge
            ]
            # Remove None entries
            loop_curves = [c for c in loop_curves if c is not None]
            cl = gmsh.model.geo.addCurveLoop(loop_curves)
            surf = gmsh.model.geo.addPlaneSurface([cl])
            surface_tags.append(surf)

        gmsh.model.geo.synchronize()

        # Transfinite meshing for structured quads
        for line in bottom_lines + top_lines + [inlet_line, outlet_line] + interior_lines:
            gmsh.model.mesh.setTransfiniteCurve(line, 2)

        # Set number of divisions along y direction
        for vline in [inlet_line, outlet_line]:
            gmsh.model.mesh.setTransfiniteCurve(vline, n_y + 1)
        for vline in interior_lines:
            gmsh.model.mesh.setTransfiniteCurve(vline, n_y + 1)

        # Set number of divisions along x direction
        for bline in bottom_lines:
            gmsh.model.mesh.setTransfiniteCurve(bline, 2)
        for tline in top_lines:
            gmsh.model.mesh.setTransfiniteCurve(tline, 2)

        for surf in surface_tags:
            gmsh.model.mesh.setTransfiniteSurface(surf)
            gmsh.model.mesh.setRecombine(2, surf)

        gmsh.model.mesh.generate(2)

        # Physical groups
        # Ground (moving wall)
        ground_tag = gmsh.model.addPhysicalGroup(1, bottom_lines)
        gmsh.model.setPhysicalName(1, ground_tag, "ground")

        # Floor (wall)
        floor_tag = gmsh.model.addPhysicalGroup(1, top_lines)
        gmsh.model.setPhysicalName(1, floor_tag, "floor")

        # Inlet
        inlet_tag = gmsh.model.addPhysicalGroup(1, [inlet_line])
        gmsh.model.setPhysicalName(1, inlet_tag, "inlet")

        # Outlet
        outlet_tag = gmsh.model.addPhysicalGroup(1, [outlet_line])
        gmsh.model.setPhysicalName(1, outlet_tag, "outlet")

        # Fluid
        fluid_tag = gmsh.model.addPhysicalGroup(2, surface_tags)
        gmsh.model.setPhysicalName(2, fluid_tag, "fluid")

        gmsh.model.mesh.createTopology()

        out = Path(f"{name}.su2")
        gmsh.write(str(out))
        gmsh.finalize()
        return out


# ── Solver interface ──

@dataclasses.dataclass
class SU2Results:
    """Parsed results from a SU2_CFD run."""
    cd: float = 0.0
    cl: float = 0.0
    cmz: float = 0.0
    converged: bool = False
    iterations: int = 0
    history: list[dict] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class SU2Solver:
    """Invoke SU2_CFD and collect results."""

    def __init__(self, su2_cfd: Path = SU2_CFD, workdir: Optional[Path] = None):
        self.su2_cfd = su2_cfd
        self.workdir = workdir or Path(tempfile.mkdtemp())
        self.workdir.mkdir(parents=True, exist_ok=True)

    def run(self, config: Path, mesh: Path, timeout: int = 600) -> SU2Results:
        """Run SU2_CFD with the given config and mesh files."""
        cfg_local = self.workdir / config.name
        mesh_local = self.workdir / mesh.name
        cfg_local.write_text(config.read_text())
        mesh_local.write_text(mesh.read_text())

        cfg_text = cfg_local.read_text()
        if "MESH_FILENAME" not in cfg_text:
            cfg_text += f"\nMESH_FILENAME= {mesh_local.name}\n"
            cfg_local.write_text(cfg_text)

        cmd = [str(self.su2_cfd), cfg_local.name]

        proc = subprocess.run(
            cmd,
            cwd=self.workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return self._parse_results(proc, self.workdir)

    def _parse_results(self, proc: subprocess.CompletedProcess, workdir: Path) -> SU2Results:
        results = SU2Results()
        stdout = proc.stdout
        stderr = proc.stderr

        if proc.returncode != 0:
            print(f"  [SU2 exited with code {proc.returncode}]")
            for line in (stderr or "").strip().split("\n")[-10:]:
                if line.strip():
                    print(f"  STDERR: {line.strip()}")
            return results

        results.converged = ("Convergence reached" in stdout or
                             "convergence" in stdout.lower())

        col_map = {
            "ITER": "ITER", "Inner_Iter": "ITER", "INNER_ITER": "ITER",
            "DRAG": "DRAG", "CD": "DRAG", "cd": "DRAG",
            "LIFT": "LIFT", "CL": "LIFT", "cl": "LIFT",
            "MOMENT": "MOMENT", "CMZ": "MOMENT", "Cmz": "MOMENT",
        }

        hist_file = workdir / "history.csv"
        if hist_file.exists():
            lines = hist_file.read_text().strip().split("\n")
            if len(lines) > 1:
                header = [h.strip().strip('"') for h in lines[0].split(",")]
                for line in lines[1:]:
                    vals = line.split(",")
                    if len(vals) == len(header):
                        entry_orig = dict(zip(header, vals))
                        entry = {col_map.get(k, k): v for k, v in entry_orig.items()}
                        results.history.append(entry)
                        try:
                            results.iterations = int(float(entry.get("ITER", 0)))
                            results.cd = float(entry.get("DRAG", results.cd))
                            results.cl = float(entry.get("LIFT", results.cl))
                            results.cmz = float(entry.get("MOMENT", results.cmz))
                        except (ValueError, TypeError):
                            pass

        return results


# ── PyVista visualization helpers ──

def load_solution(mesh_path: Path, solution_dir: Optional[Path] = None):
    """Load SU2 mesh and solution into PyVista mesh objects."""
    import pyvista as pv
    reader = pv.get_reader(str(mesh_path))
    mesh = reader.read()
    if solution_dir:
        for f in sorted(solution_dir.glob("*.vtu")):
            sol = pv.read(str(f))
            mesh.point_data.update(sol.point_data)
    return mesh


def plot_venturi_cp(mesh_path: Path, solution_path: Optional[Path] = None, save: Optional[Path] = None):
    """Plot Cp along the venturi floor."""
    import pyvista as pv
    import matplotlib.pyplot as plt

    mesh = load_solution(mesh_path, solution_path)
    surf = mesh.separate_cells().extract_surface()
    if "Pressure" not in surf.point_data:
        print("No Pressure field in solution")
        return

    p = surf.point_data["Pressure"]
    rho = 1.225
    v_ref = 40.0
    q_ref = 0.5 * rho * v_ref**2
    cp = p / q_ref

    pts = surf.points
    x = pts[:, 0]
    y = pts[:, 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, cp, c=cp, cmap="coolwarm", s=5, alpha=0.6)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Cp")
    ax.set_title("Pressure Coefficient along Venturi Floor")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    if save:
        fig.savefig(save)
        plt.close(fig)
    return fig


def plot_velocity_field(mesh_path: Path, solution_path: Optional[Path] = None, save: Optional[Path] = None):
    """Plot velocity magnitude contours in the venturi."""
    import pyvista as pv
    import matplotlib.pyplot as plt

    mesh = load_solution(mesh_path, solution_path)
    if "Velocity" not in mesh.point_data:
        print("No Velocity field in solution")
        return

    vel = np.linalg.norm(mesh.point_data["Velocity"], axis=1)
    pts = mesh.points

    fig, ax = plt.subplots(figsize=(12, 4))
    scatter = ax.scatter(pts[:, 0], pts[:, 1], c=vel, cmap="plasma", s=3, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label="Velocity (m/s)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Velocity Magnitude in Venturi")
    ax.set_aspect("equal")

    if save:
        fig.savefig(save)
        plt.close(fig)
    return fig
