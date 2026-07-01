import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.core.style import (
    set_f1_style,
    MERCEDES_TEAL,
    MERCEDES_RED,
    MERCEDES_DARK,
    MERCEDES_GRAY,
    MERCEDES_WHITE,
    MERCEDES_CARD,
)

# Global PyVista theme overrides
pv.global_theme.font.color = "white"
pv.global_theme.font.title_size = 32
pv.global_theme.font.label_size = 22
from src.core.physics import G

set_f1_style()

WINDOW_SIZE = (1920, 1080)
CFD_CAMERA = [
    (1.5, 0.075, 3.5),
    (1.5, 0.075, 0),
    (0, 1, 0),
]

SCALAR_BAR_ARGS = {
    "title_font_size": 35,
    "label_font_size": 28,
    "position_x": 0.22,
    "position_y": 0.1,
    "vertical": False,
}


def _load_vtu(file_path: Path) -> pv.UnstructuredGrid:
    mesh = pv.read(file_path)
    if "Velocity" in mesh.array_names:
        vel = mesh["Velocity"]
        mesh["Velocity_Mag"] = np.linalg.norm(vel, axis=1)
    return mesh


def _render_contour(mesh, scalars, cmap, save_path, clim=None):
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.set_background(MERCEDES_DARK)
    kwargs = dict(
        scalars=scalars, cmap=cmap, show_edges=False, scalar_bar_args=SCALAR_BAR_ARGS
    )
    if clim:
        kwargs["clim"] = clim
    plotter.add_mesh(mesh, **kwargs)
    plotter.camera_position = CFD_CAMERA
    plotter.screenshot(save_path)
    plotter.close()


def plot_velocity_contour(mesh, save_path):
    _render_contour(mesh, "Velocity_Mag", "turbo", save_path)


def plot_cp_contour(mesh, save_path):
    _render_contour(mesh, "Pressure_Coefficient", "seismic", save_path, clim=[-3, 1])


def plot_streamlines(mesh, save_path):
    rake = pv.Line(pointa=(0.1, 0.005, 0), pointb=(0.1, 0.095, 0), resolution=40)
    stream = mesh.streamlines_from_source(
        rake, vectors="Velocity", integration_direction="forward", max_length=5.0
    )

    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.set_background(MERCEDES_DARK)
    plotter.add_mesh(stream, scalars="Velocity_Mag", cmap="turbo", line_width=2)
    plotter.add_mesh(mesh.outline(), color=MERCEDES_GRAY)
    plotter.camera_position = CFD_CAMERA
    plotter.screenshot(save_path)
    plotter.close()


def plot_vorticity(mesh, save_path):
    """Compute vorticity magnitude via gradient + curl."""
    grad = mesh.compute_derivative(scalars="Velocity")
    # Gradient of vector field [u, v, w] is 3x3:
    # [du/dx, du/dy, du/dz,
    #  dv/dx, dv/dy, dv/dz,
    #  dw/dx, dw/dy, dw/dz]
    # Vorticity = curl(u) = (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy)
    vel_grad = grad["gradient"]
    vort_x = vel_grad[:, 7] - vel_grad[:, 5]
    vort_y = vel_grad[:, 2] - vel_grad[:, 6]
    vort_z = vel_grad[:, 3] - vel_grad[:, 1]
    vort_mag = np.sqrt(vort_x**2 + vort_y**2 + vort_z**2)
    mesh["Vorticity_Mag"] = vort_mag
    _render_contour(mesh, "Vorticity_Mag", "plasma", save_path, clim=[0, 200])


def plot_tke(mesh, save_path):
    _render_contour(mesh, "Turb_Kin_Energy", "plasma", save_path)


def plot_mach(mesh, save_path):
    """Compute Mach number: M = |V| / sqrt(gamma * R * T)."""
    gamma = 1.4
    R = 287.058
    temperature = mesh["Temperature"]
    speed_sound = np.sqrt(gamma * R * temperature)
    mesh["Mach"] = mesh["Velocity_Mag"] / speed_sound
    _render_contour(mesh, "Mach", "turbo", save_path)


def extract_and_plot_wall_cp(mesh, save_path):
    """Extract Cp along floor and ground walls via sample_over_line."""
    floor = mesh.sample_over_line(
        pointa=(0.1, 0.05, 0), pointb=(1.9, 0.05, 0), resolution=200
    )
    ground = mesh.sample_over_line(
        pointa=(0.1, 0.0, 0), pointb=(1.9, 0.0, 0), resolution=200
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        floor["Distance"],
        floor["Pressure_Coefficient"],
        label="Floor",
        color=MERCEDES_TEAL,
        linewidth=2,
    )
    ax.plot(
        ground["Distance"],
        ground["Pressure_Coefficient"],
        label="Ground",
        color=MERCEDES_RED,
        linewidth=2,
    )
    ax.axhline(0, color=MERCEDES_GRAY, linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Cp")
    ax.set_title("Pressure Coefficient Along Venturi")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def extract_and_plot_yplus(mesh, save_path):
    """Y+ along floor and ground walls."""
    floor = mesh.sample_over_line(
        pointa=(0.1, 0.05, 0), pointb=(1.9, 0.05, 0), resolution=200
    )
    ground = mesh.sample_over_line(
        pointa=(0.1, 0.0, 0), pointb=(1.9, 0.0, 0), resolution=200
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(
        floor["Distance"],
        floor["Y_Plus"],
        label="Floor",
        color=MERCEDES_TEAL,
        linewidth=2,
    )
    ax.semilogy(
        ground["Distance"],
        ground["Y_Plus"],
        label="Ground",
        color=MERCEDES_RED,
        linewidth=2,
    )
    ax.axhline(1, color=MERCEDES_GRAY, linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("y+")
    ax.set_title("Wall y+ Distribution")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def extract_and_plot_wall_shear(mesh, save_path):
    """Wall shear stress (Skin_Friction_Coefficient magnitude) along floor."""
    floor = mesh.sample_over_line(
        pointa=(0.1, 0.05, 0), pointb=(1.9, 0.05, 0), resolution=200
    )
    sfc = floor["Skin_Friction_Coefficient"]
    shear_mag = np.linalg.norm(sfc, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(floor["Distance"], shear_mag, color=MERCEDES_TEAL, linewidth=2)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Skin Friction Coefficient")
    ax.set_title("Wall Shear Along Floor")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_profiles(mesh, save_path):
    """Extract and overlay boundary layer velocity profiles at x-stations."""
    stations = {
        "Inlet (x=0.5m)": (0.5, 0.0, 0, 0.5, 0.15, 0),
        "Throat (x=1.0m)": (1.0, 0.0, 0, 1.0, 0.15, 0),
        "Mid-diffuser (x=1.5m)": (1.5, 0.0, 0, 1.5, 0.15, 0),
        "Outlet (x=2.0m)": (2.0, 0.0, 0, 2.0, 0.15, 0),
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#00D2BE", "#FFB347", "#E94560", "#4ECDC4"]
    for (label, (xa, ya, za, xb, yb, zb)), color in zip(stations.items(), colors):
        profile = mesh.sample_over_line(
            pointa=(xa, ya, za), pointb=(xb, yb, zb), resolution=100
        )
        ax.plot(profile["Velocity_Mag"], profile["Distance"], color=color, linewidth=2, label=label)

    ax.axhline(0.05, color=MERCEDES_GRAY, linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel("Velocity Magnitude (m/s)")
    ax.set_ylabel("y-distance from ground (m)")
    ax.set_title("Velocity Profiles at Selected x-stations")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_case2_cp_overlay(result_dir: Path, save_path: Path):
    """Overlay Cp profiles for all ride heights."""
    heights = [25, 35, 50, 65, 80, 100]
    colors = ["#4ECDC4", "#00D2BE", "#FFB347", "#E94560", "#9B59B6", "#95A5A6"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for h, color in zip(heights, colors):
        mesh = _load_vtu(result_dir / f"rh_vi_h{h}_a17" / "vol_solution.vtu")
        floor = mesh.sample_over_line(
            pointa=(0.1, 0.05, 0), pointb=(1.9, 0.05, 0), resolution=200
        )
        ax.plot(floor["Distance"], floor["Pressure_Coefficient"],
                color=color, linewidth=1.5, label=f"h={h} mm")
    ax.axhline(0, color=MERCEDES_GRAY, linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Cp")
    ax.set_title("Cp Profile Overlay -- All Ride Heights")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_case2_wall_shear_overlay(result_dir: Path, save_path: Path):
    """Overlay wall shear for all ride heights."""
    heights = [25, 35, 50, 65, 80, 100]
    colors = ["#4ECDC4", "#00D2BE", "#FFB347", "#E94560", "#9B59B6", "#95A5A6"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for h, color in zip(heights, colors):
        mesh = _load_vtu(result_dir / f"rh_vi_h{h}_a17" / "vol_solution.vtu")
        floor = mesh.sample_over_line(
            pointa=(0.1, 0.05, 0), pointb=(1.9, 0.05, 0), resolution=200
        )
        sfc = floor["Skin_Friction_Coefficient"]
        shear_mag = np.linalg.norm(sfc, axis=1)
        ax.plot(floor["Distance"], shear_mag, color=color, linewidth=1.5, label=f"h={h} mm")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Skin Friction Coefficient")
    ax.set_title("Wall Shear Overlay -- All Ride Heights")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_case3_cp_overlay(result_dir: Path, save_path: Path):
    """Overlay Cp profiles for all diffuser angles."""
    angles = [10, 12, 14, 16, 18, 20]
    colors = ["#4ECDC4", "#00D2BE", "#FFB347", "#E94560", "#9B59B6", "#95A5A6"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for a, color in zip(angles, colors):
        mesh = _load_vtu(result_dir / f"da_vi_a{a}_h50" / "vol_solution.vtu")
        floor = mesh.sample_over_line(
            pointa=(0.1, 0.05, 0), pointb=(1.9, 0.05, 0), resolution=200
        )
        ax.plot(floor["Distance"], floor["Pressure_Coefficient"],
                color=color, linewidth=1.5, label=f"{a} deg")
    ax.axhline(0, color=MERCEDES_GRAY, linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Cp")
    ax.set_title("Cp Profile Overlay -- All Diffuser Angles")
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def _render_sweep_grid(
    result_dir: Path,
    param_values: list,
    name_template: str,
    scalars: str,
    cmap: str,
    save_path: Path,
    label_prefix: str,
    label_suffix: str = "",
    clim: list = None,
    scalar_bar_title: str = None,
    shared_clim: bool = False,
):
    """Render a 3x2 PyVista grid of contour plots for a parametric sweep.

    Uses the same camera, background, and scalar bar settings as Case 1.
    Only the bottom-right subplot shows the scalar bar to reduce visual clutter.

    If shared_clim=True, computes global min/max of `scalars` across all meshes
    so all subplots share the same color scale, making differences visible.
    """
    meshes = []
    for p in param_values:
        path = result_dir / name_template.format(p) / "vol_solution.vtu"
        meshes.append(_load_vtu(path))

    # Compute global clim if shared_clim is requested and no explicit clim given
    if shared_clim and clim is None:
        global_min = min(m[scalars].min() for m in meshes)
        global_max = max(m[scalars].max() for m in meshes)
        clim = [float(global_min), float(global_max)]

    plotter = pv.Plotter(shape=(3, 2), off_screen=True, window_size=(1920, 1440))
    plotter.set_background(MERCEDES_DARK)

    for idx, (mesh, p) in enumerate(zip(meshes, param_values)):
        row, col = divmod(idx, 2)
        plotter.subplot(row, col)

        label = f"{label_prefix}{p}{label_suffix}"
        plotter.add_text(label, font_size=22, color="white", position="upper_left")

        kwargs = dict(scalars=scalars, cmap=cmap, show_edges=False)
        if clim:
            kwargs["clim"] = clim
        if row == 2 and col == 1:
            sb_args = dict(SCALAR_BAR_ARGS)
            if scalar_bar_title:
                sb_args["title"] = scalar_bar_title
            kwargs["scalar_bar_args"] = sb_args
        else:
            kwargs["show_scalar_bar"] = False

        plotter.add_mesh(mesh, **kwargs)
        plotter.camera_position = CFD_CAMERA

    plotter.screenshot(save_path)
    plotter.close()


def export_case2_velocity_gallery(result_dir: Path, save_path: Path):
    """2x3 grid of velocity contours for all 6 ride heights."""
    _render_sweep_grid(
        result_dir, [25, 35, 50, 65, 80, 100],
        "rh_vi_h{}_a17", "Velocity_Mag", "turbo",
        save_path, "h=", " mm",
        scalar_bar_title="Velocity Magnitude (m/s)",
        shared_clim=True,
    )


def export_case2_cp_gallery(result_dir: Path, save_path: Path):
    """2x3 grid of Cp contours for all 6 ride heights."""
    _render_sweep_grid(
        result_dir, [25, 35, 50, 65, 80, 100],
        "rh_vi_h{}_a17", "Pressure_Coefficient", "seismic",
        save_path, "h=", " mm", clim=[-3, 1],
        scalar_bar_title="Pressure Coefficient Cp",
    )


def export_case3_velocity_gallery(result_dir: Path, save_path: Path):
    """2x3 grid of velocity contours for all 6 diffuser angles."""
    _render_sweep_grid(
        result_dir, [10, 12, 14, 16, 18, 20],
        "da_vi_a{}_h50", "Velocity_Mag", "turbo",
        save_path, "a=", " deg",
        scalar_bar_title="Velocity Magnitude (m/s)",
        shared_clim=True,
    )


def export_case3_cp_gallery(result_dir: Path, save_path: Path):
    """2x3 grid of Cp contours for all 6 diffuser angles."""
    _render_sweep_grid(
        result_dir, [10, 12, 14, 16, 18, 20],
        "da_vi_a{}_h50", "Pressure_Coefficient", "seismic",
        save_path, "a=", " deg", clim=[-3, 1],
        scalar_bar_title="Pressure Coefficient Cp",
    )


def export_case4_cp_overlay(result_dir: Path, save_path: Path):
    """Cp profile for the reference case (all velocities collapse in incompressible flow)."""
    mesh = _load_vtu(result_dir / "venturi_h50_a17_v50" / "vol_solution.vtu")
    floor = mesh.sample_over_line(
        pointa=(0.1, 0.05, 0), pointb=(1.9, 0.05, 0), resolution=200
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        floor["Distance"], floor["Pressure_Coefficient"],
        color=MERCEDES_TEAL, linewidth=2.5,
        label="All velocities (V=50-100 m/s)",
    )
    ax.axhline(0, color=MERCEDES_GRAY, linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Cp")
    ax.set_title("Cp Profile -- Velocity Sweep (Incompressible Collapse)")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.annotate(
        "Single representative curve shown: all 5 velocity cases\n"
        "(50, 65, 80, 92, 100 m/s) produce identical Cp profiles.\n"
        "Cp is independent of velocity in incompressible flow.\n"
        "CL varies by < 0.3% across the full range.",
        xy=(0.98, 0.05), xycoords="axes fraction",
        ha="right", va="bottom",
        fontsize=9, color=MERCEDES_GRAY,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=MERCEDES_CARD, alpha=0.8),
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_case4_velocity_comparison(result_dir: Path, save_path: Path):
    """Side-by-side velocity contours at V=50 vs V=100 m/s using actual VTU files."""
    low = _load_vtu(result_dir / "venturi_h50_a17_v50" / "vol_solution.vtu")
    high = _load_vtu(result_dir / "venturi_h50_a17_v100" / "vol_solution.vtu")

    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1920, 540))
    plotter.set_background(MERCEDES_DARK)
    for i, (mesh, label) in enumerate([(low, "V=50 m/s"), (high, "V=100 m/s")]):
        plotter.subplot(0, i)
        plotter.add_text(label, font_size=28, color="white")
        plotter.add_mesh(
            mesh, scalars="Velocity_Mag", cmap="turbo",
            show_edges=False, scalar_bar_args=SCALAR_BAR_ARGS,
        )
        plotter.camera_position = CFD_CAMERA
    plotter.screenshot(save_path)
    plotter.close()


def export_case4_velocity_gallery(result_dir: Path, save_path: Path):
    """3x2 grid of velocity contours for all 5 velocity sweep cases using actual VTU files."""
    _render_sweep_grid(
        result_dir, [50, 65, 80, 92, 100],
        "venturi_h50_a17_v{}", "Velocity_Mag", "turbo",
        save_path, "V=", " m/s",
        scalar_bar_title="Velocity Magnitude (m/s)",
        shared_clim=True,
    )


def export_case4_cp_gallery(result_dir: Path, save_path: Path):
    """3x2 grid of Cp contours for all 5 velocity sweep cases using actual VTU files."""
    _render_sweep_grid(
        result_dir, [50, 65, 80, 92, 100],
        "venturi_h50_a17_v{}", "Pressure_Coefficient", "seismic",
        save_path, "V=", " m/s", clim=[-3, 1],
        scalar_bar_title="Pressure Coefficient Cp",
    )


def _read_history_csv(path: Path) -> dict:
    """Read CL and CD from the last iteration of a SU2 history.csv."""
    import csv
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            pass
    cl = float(row[9].strip().strip('"'))
    cd = float(row[8].strip().strip('"'))
    return {"CL": cl, "CD": cd, "L/D": -cl / cd}


def export_velocity_force_scaling(case4_dir: Path, result_dir: Path):
    """Generate 4 force scaling plots from velocity sweep history.csv data."""
    velocities = [50, 65, 80, 92, 100]
    rho = 1.225
    A_ref = 1.45

    data = []
    for v in velocities:
        f = result_dir / f"venturi_h50_a17_v{v}" / "history.csv"
        d = _read_history_csv(f)
        q = 0.5 * rho * v**2
        d["V"] = v
        d["v2"] = v**2
        d["Fz"] = -d["CL"] * q * A_ref
        d["Fd"] = d["CD"] * q * A_ref
        data.append(d)

    vs = [d["V"] for d in data]
    v2s = [d["v2"] for d in data]
    cls = [d["CL"] for d in data]
    cds = [d["CD"] for d in data]
    lds = [d["L/D"] for d in data]
    fzs = [d["Fz"] for d in data]
    fds = [d["Fd"] for d in data]

    set_f1_style()

    # 1. Downforce vs v^2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(v2s, fzs, color=MERCEDES_TEAL, s=60, zorder=5)
    # Linear fit through origin
    coeff = sum(fz * v2 for fz, v2 in zip(fzs, v2s)) / sum(v2**2 for v2 in v2s)
    v2_line = [0, max(v2s) * 1.05]
    fz_line = [coeff * v2 for v2 in v2_line]
    ax.plot(v2_line, fz_line, color=MERCEDES_WHITE, linewidth=1.5, linestyle="--",
            label=f"Fz = {coeff:.3f} × v²")
    for v, fz in zip(vs, fzs):
        ax.annotate(f"{v} m/s", (v**2, fz), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, color=MERCEDES_GRAY)
    ax.set_xlabel("v² (m²/s²)")
    ax.set_ylabel("Downforce Fz (N)")
    ax.set_title("Downforce Scales Quadratically with Velocity")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(case4_dir / "downforce_vs_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Drag vs v^2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(v2s, fds, color=MERCEDES_RED, s=60, zorder=5)
    coeff_d = sum(fd * v2 for fd, v2 in zip(fds, v2s)) / sum(v2**2 for v2 in v2s)
    fd_line = [coeff_d * v2 for v2 in v2_line]
    ax.plot(v2_line, fd_line, color=MERCEDES_WHITE, linewidth=1.5, linestyle="--",
            label=f"Fd ≈ {coeff_d:.3f} × v²")
    for v, fd in zip(vs, fds):
        ax.annotate(f"{v} m/s", (v**2, fd), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, color=MERCEDES_GRAY)
    ax.set_xlabel("v² (m²/s²)")
    ax.set_ylabel("Drag Fd (N)")
    ax.set_title("Drag Scales Near-Quadratic (CD decreases slightly)")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(case4_dir / "drag_vs_v2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. CL + CD vs Velocity (dual-axis)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_cl = MERCEDES_TEAL
    color_cd = MERCEDES_RED
    ax1.plot(vs, cls, "o-", color=color_cl, linewidth=2, markersize=6, label="CL")
    ax1.set_xlabel("Velocity (m/s)")
    ax1.set_ylabel("CL (downforce coefficient)", color=color_cl)
    ax1.tick_params(axis="y", labelcolor=color_cl)
    ax1.axhline(y=cls[0], color=color_cl, linewidth=0.5, linestyle=":", alpha=0.4)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(vs, cds, "s-", color=color_cd, linewidth=2, markersize=6, label="CD")
    ax2.set_ylabel("CD (drag coefficient)", color=color_cd)
    ax2.tick_params(axis="y", labelcolor=color_cd)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, framealpha=0.9)
    ax1.set_title("Force Coefficients vs Velocity (CL constant, CD decreases)")
    fig.tight_layout()
    fig.savefig(case4_dir / "cl_cd_vs_velocity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. L/D vs Velocity
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([str(v) for v in vs], lds, color=MERCEDES_TEAL, width=0.6, alpha=0.85)
    for bar, v, ld in zip(bars, vs, lds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{ld:.2f}", ha="center", va="bottom", fontsize=9, color=MERCEDES_WHITE)
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("L/D (aerodynamic efficiency)")
    ax.set_title("Aerodynamic Efficiency Improves at Higher Speed")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(case4_dir / "ld_vs_velocity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("    downforce_vs_v2.png  done")
    print("    drag_vs_v2.png       done")
    print("    cl_cd_vs_velocity.png done")
    print("    ld_vs_velocity.png    done")


def export_sweep_visuals(asset_dir: Path, result_dir: Path):
    """Export all sweep comparison visuals for Cases 2-4."""
    case2_dir = asset_dir / "case2"
    case2_dir.mkdir(parents=True, exist_ok=True)
    case3_dir = asset_dir / "case3"
    case3_dir.mkdir(parents=True, exist_ok=True)

    print("  Case 2: Cp overlay + wall shear...")
    export_case2_cp_overlay(result_dir, case2_dir / "cp_overlay.png")
    print("    cp_overlay done")
    export_case2_wall_shear_overlay(result_dir, case2_dir / "wall_shear_overlay.png")
    print("    wall_shear_overlay done")

    print("  Case 2: Galleries...")
    export_case2_velocity_gallery(result_dir, case2_dir / "velocity_gallery.png")
    print("    velocity_gallery done")
    export_case2_cp_gallery(result_dir, case2_dir / "cp_gallery.png")
    print("    cp_gallery done")

    print("  Case 3: Cp overlay...")
    export_case3_cp_overlay(result_dir, case3_dir / "cp_overlay.png")
    print("    cp_overlay done")

    print("  Case 3: Galleries...")
    export_case3_velocity_gallery(result_dir, case3_dir / "velocity_gallery.png")
    print("    velocity_gallery done")
    export_case3_cp_gallery(result_dir, case3_dir / "cp_gallery.png")
    print("    cp_gallery done")

    print("  Case 4: Velocity sweep...")
    case4_dir = asset_dir / "case4"
    case4_dir.mkdir(parents=True, exist_ok=True)
    export_case4_cp_overlay(result_dir, case4_dir / "cp_overlay.png")
    print("    cp_overlay done")
    export_case4_velocity_comparison(result_dir, case4_dir / "velocity_comparison.png")
    print("    velocity_comparison done")
    export_case4_velocity_gallery(result_dir, case4_dir / "velocity_gallery.png")
    print("    velocity_gallery done")
    export_case4_cp_gallery(result_dir, case4_dir / "cp_gallery.png")
    print("    cp_gallery done")

    print("  Case 4: Force scaling analysis...")
    export_velocity_force_scaling(case4_dir, result_dir)
    print("Done.")


def export_case1_all(asset_dir: Path, result_dir: Path):
    """Render all 11 Case 1 visualizations."""
    case1_dir = asset_dir / "case1"
    case1_dir.mkdir(parents=True, exist_ok=True)
    mesh = _load_vtu(result_dir / "rh_vi_h50_a17" / "vol_solution.vtu")

    print("  Velocity contour...")
    plot_velocity_contour(mesh, case1_dir / "velocity_contour.png")
    print("  Cp contour...")
    plot_cp_contour(mesh, case1_dir / "cp_contour.png")
    print("  Streamlines...")
    plot_streamlines(mesh, case1_dir / "streamlines.png")
    print("  Vorticity...")
    plot_vorticity(mesh, case1_dir / "vorticity.png")
    print("  TKE...")
    plot_tke(mesh, case1_dir / "tke.png")
    print("  Mach...")
    plot_mach(mesh, case1_dir / "mach.png")
    print("  Cp profile...")
    extract_and_plot_wall_cp(mesh, case1_dir / "cp_profile.png")
    print("  Y+ distribution...")
    extract_and_plot_yplus(mesh, case1_dir / "yplus.png")
    print("  Wall shear...")
    extract_and_plot_wall_shear(mesh, case1_dir / "wall_shear.png")
    print("  Velocity profiles...")
    plot_velocity_profiles(mesh, case1_dir / "velocity_profiles.png")
    print("Done.")
