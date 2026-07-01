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
    _render_contour(mesh, "Pressure_Coefficient", "RdBu_r", save_path, clim=[-3, 1])


def plot_streamlines(mesh, save_path):
    rake = pv.Line(pointa=(0.1, 0.0, 0), pointb=(0.1, 0.15, 0), resolution=20)
    stream = mesh.streamlines_from_source(
        rake, vectors="Velocity", integration_direction="forward", max_length=2.0
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
    _render_contour(mesh, "Vorticity_Mag", "plasma", save_path)


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


def export_case2_velocity_comparison(result_dir: Path, save_path: Path):
    """Side-by-side velocity contours at low vs high ride height."""
    low = _load_vtu(result_dir / "rh_vi_h35_a17" / "vol_solution.vtu")
    high = _load_vtu(result_dir / "rh_vi_h80_a17" / "vol_solution.vtu")

    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1920, 540))
    plotter.set_background(MERCEDES_DARK)
    for i, (mesh, label) in enumerate([(low, "h=35 mm"), (high, "h=80 mm")]):
        plotter.subplot(0, i)
        plotter.add_text(label, font_size=28, color="white")
        plotter.add_mesh(mesh, scalars="Velocity_Mag", cmap="turbo",
                         show_edges=False, scalar_bar_args=SCALAR_BAR_ARGS)
        plotter.camera_position = CFD_CAMERA
    plotter.screenshot(save_path)
    plotter.close()


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


def export_case2_streamlines_comparison(result_dir: Path, save_path: Path):
    """Side-by-side streamlines at low vs high ride height."""
    low = _load_vtu(result_dir / "rh_vi_h35_a17" / "vol_solution.vtu")
    high = _load_vtu(result_dir / "rh_vi_h80_a17" / "vol_solution.vtu")
    rake = pv.Line(pointa=(0.1, 0.0, 0), pointb=(0.1, 0.15, 0), resolution=20)

    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1920, 540))
    plotter.set_background(MERCEDES_DARK)
    for i, (mesh, label) in enumerate([(low, "h=35 mm"), (high, "h=80 mm")]):
        plotter.subplot(0, i)
        plotter.add_text(label, font_size=28, color="white")
        stream = mesh.streamlines_from_source(
            rake, vectors="Velocity", integration_direction="forward", max_length=2.0
        )
        plotter.add_mesh(stream, scalars="Velocity_Mag", cmap="turbo", line_width=2)
        plotter.add_mesh(mesh.outline(), color=MERCEDES_GRAY)
        plotter.camera_position = CFD_CAMERA
    plotter.screenshot(save_path)
    plotter.close()


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


def export_case3_streamlines_comparison(result_dir: Path, save_path: Path):
    """Side-by-side streamlines at 10 vs 20 deg diffuser angle."""
    low = _load_vtu(result_dir / "da_vi_a10_h50" / "vol_solution.vtu")
    high = _load_vtu(result_dir / "da_vi_a20_h50" / "vol_solution.vtu")
    rake = pv.Line(pointa=(0.1, 0.0, 0), pointb=(0.1, 0.15, 0), resolution=20)

    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1920, 540))
    plotter.set_background(MERCEDES_DARK)
    for i, (mesh, label) in enumerate([(low, "10 deg"), (high, "20 deg")]):
        plotter.subplot(0, i)
        plotter.add_text(label, font_size=28, color="white")
        stream = mesh.streamlines_from_source(
            rake, vectors="Velocity", integration_direction="forward", max_length=2.0
        )
        plotter.add_mesh(stream, scalars="Velocity_Mag", cmap="turbo", line_width=2)
        plotter.add_mesh(mesh.outline(), color=MERCEDES_GRAY)
        plotter.camera_position = CFD_CAMERA
    plotter.screenshot(save_path)
    plotter.close()


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


def export_case3_velocity_separation(result_dir: Path, save_path: Path):
    """Velocity contour at high diffuser angle showing separation."""
    attached = _load_vtu(result_dir / "da_vi_a10_h50" / "vol_solution.vtu")
    separated = _load_vtu(result_dir / "da_vi_a20_h50" / "vol_solution.vtu")

    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1920, 540))
    plotter.set_background(MERCEDES_DARK)
    for i, (mesh, label) in enumerate([(attached, "10 deg (attached)"), (separated, "20 deg (separated)")]):
        plotter.subplot(0, i)
        plotter.add_text(label, font_size=28, color="white")
        plotter.add_mesh(mesh, scalars="Velocity_Mag", cmap="turbo",
                         show_edges=False, scalar_bar_args=SCALAR_BAR_ARGS)
        plotter.camera_position = CFD_CAMERA
    plotter.screenshot(save_path)
    plotter.close()


def export_sweep_visuals(asset_dir: Path, result_dir: Path):
    """Export all sweep comparison visuals for Cases 2-4."""
    print("  Case 2: Ride height sweep...")
    export_case2_velocity_comparison(result_dir, asset_dir / "case2_velocity_comparison.png")
    print("    velocity_comparison done")
    export_case2_cp_overlay(result_dir, asset_dir / "case2_cp_overlay.png")
    print("    cp_overlay done")
    export_case2_streamlines_comparison(result_dir, asset_dir / "case2_streamlines_comparison.png")
    print("    streamlines_comparison done")
    export_case2_wall_shear_overlay(result_dir, asset_dir / "case2_wall_shear_overlay.png")
    print("    wall_shear_overlay done")

    print("  Case 3: Diffuser angle sweep...")
    export_case3_streamlines_comparison(result_dir, asset_dir / "case3_streamlines_comparison.png")
    print("    streamlines_comparison done")
    export_case3_cp_overlay(result_dir, asset_dir / "case3_cp_overlay.png")
    print("    cp_overlay done")
    export_case3_velocity_separation(result_dir, asset_dir / "case3_velocity_separation.png")
    print("    velocity_separation done")

    print("  Case 4: Velocity sweep -- no VTU files available, skipping PyVista export")
    print("Done.")


def export_case1_all(asset_dir: Path, result_dir: Path):
    """Render all 11 Case 1 visualizations."""
    mesh = _load_vtu(result_dir / "rh_vi_h50_a17" / "vol_solution.vtu")

    print("  Velocity contour...")
    plot_velocity_contour(mesh, asset_dir / "velocity_contour.png")
    print("  Cp contour...")
    plot_cp_contour(mesh, asset_dir / "cp_contour.png")
    print("  Streamlines...")
    plot_streamlines(mesh, asset_dir / "streamlines.png")
    print("  Vorticity...")
    plot_vorticity(mesh, asset_dir / "vorticity.png")
    print("  TKE...")
    plot_tke(mesh, asset_dir / "tke.png")
    print("  Mach...")
    plot_mach(mesh, asset_dir / "mach.png")
    print("  Cp profile...")
    extract_and_plot_wall_cp(mesh, asset_dir / "cp_profile.png")
    print("  Y+ distribution...")
    extract_and_plot_yplus(mesh, asset_dir / "yplus.png")
    print("  Wall shear...")
    extract_and_plot_wall_shear(mesh, asset_dir / "wall_shear.png")
    print("  Velocity profiles...")
    plot_velocity_profiles(mesh, asset_dir / "velocity_profiles.png")
    print("Done.")
