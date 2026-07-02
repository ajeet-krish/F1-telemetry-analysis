"""NACA airfoil geometry generation and multi-element front wing configuration."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.core.style import (
    set_f1_style,
    MERCEDES_TEAL,
    MERCEDES_RED,
    MERCEDES_WHITE,
    MERCEDES_DARK,
    MERCEDES_GRAY,
    MERCEDES_CARD,
)


def naca_4digit(digits: str, n_points: int = 200, half_cosine: bool = True) -> np.ndarray:
    """Generate NACA 4-digit airfoil coordinates.

    Args:
        digits: 4-digit string, e.g. "6412" -> 6% camber, 40% max camber pos, 12% thickness
        n_points: points per surface (total return = 2 * n_points)
        half_cosine: cosine spacing clusters points at LE and TE

    Returns:
        (2*n_points, 2) array of (x, y) coordinates, upper surface then lower surface,
        both running from LE (x=0) to TE (x=1).
    """
    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    t = int(digits[2:]) / 100.0

    if half_cosine:
        beta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1.0 - np.cos(beta))
    else:
        x = np.linspace(0, 1, n_points)

    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    yc = np.where(
        x < p,
        m / p**2 * (2 * p * x - x**2),
        m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2),
    )
    dyc = np.where(
        x < p,
        2 * m / p**2 * (p - x),
        2 * m / (1 - p) ** 2 * (p - x),
    )
    theta = np.arctan(dyc)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    upper = np.column_stack([xu, yu])
    lower = np.column_stack([xl[::-1], yl[::-1]])
    return np.vstack([upper, lower])


def invert_airfoil(coords: np.ndarray) -> np.ndarray:
    """Flip airfoil vertically so positive camber faces downward (downforce)."""
    result = coords.copy()
    result[:, 1] = -result[:, 1]
    return result


def rotate_airfoil(coords: np.ndarray, angle_deg: float, pivot: tuple = (0.25, 0)) -> np.ndarray:
    """Rotate airfoil around a pivot point.

    Args:
        coords: (N, 2) airfoil coordinates (unit chord, x=0..1)
        angle_deg: rotation angle in degrees (positive = nose up)
        pivot: (x, y) pivot point, default quarter-chord

    Returns:
        Rotated coordinates.
    """
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    dx = coords[:, 0] - pivot[0]
    dy = coords[:, 1] - pivot[1]
    result = coords.copy()
    result[:, 0] = pivot[0] + c * dx - s * dy
    result[:, 1] = pivot[1] + s * dx + c * dy
    return result


def translate_airfoil(coords: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate airfoil by (dx, dy)."""
    return coords + np.array([dx, dy])


def scale_airfoil(coords: np.ndarray, chord: float) -> np.ndarray:
    """Scale airfoil to given chord length (assumes unit chord input)."""
    return coords * chord


def position_element_by_le(
    coords: np.ndarray,
    chord: float,
    angle_deg: float,
    le_x: float = 0.0,
    le_y: float = 0.0,
) -> np.ndarray:
    """Position an airfoil element by specifying its leading-edge location and AoA.

    Input coords should be unit-chord, unrotated, with LE at (0, 0) and TE at (1, 0).
    Scales to chord, rotates around LE, then translates LE to (le_x, le_y).

    Args:
        coords: unit-chord airfoil coordinates (N, 2)
        chord: desired chord length
        angle_deg: angle of attack (positive = nose up)
        le_x: x-coordinate of leading edge
        le_y: y-coordinate of leading edge

    Returns:
        Positioned (N, 2) coordinates.
    """
    result = coords * chord
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    rotated = result.copy()
    rotated[:, 0] = c * result[:, 0] + s * result[:, 1]
    rotated[:, 1] = -s * result[:, 0] + c * result[:, 1]
    rotated[:, 0] += le_x
    rotated[:, 1] += le_y
    return rotated


def position_element(
    coords: np.ndarray,
    chord: float,
    angle_deg: float,
    te_x: float = 0.0,
    te_y: float = 0.0,
) -> np.ndarray:
    """Position an airfoil element by specifying its trailing-edge location and AoA.

    The input coords should be unit-chord, unrotated, with TE at (1, 0) and LE at (0, 0).
    The function scales to chord, rotates around TE, then translates TE to (te_x, te_y).

    Args:
        coords: unit-chord airfoil coordinates (N, 2)
        chord: desired chord length
        angle_deg: angle of attack (positive = nose up)
        te_x: x-coordinate of trailing edge
        te_y: y-coordinate of trailing edge

    Returns:
        Positioned (N, 2) coordinates.
    """
    result = coords * chord
    result[:, 0] -= chord
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    rotated = result.copy()
    rotated[:, 0] = c * result[:, 0] + s * result[:, 1]
    rotated[:, 1] = -s * result[:, 0] + c * result[:, 1]
    rotated[:, 0] += te_x
    rotated[:, 1] += te_y
    return rotated


def front_wing_3_element(
    main_profile: str = "6412",
    mid_profile: str = "4410",
    flap_profile: str = "2412",
    chord_main: float = 1.0,
    chord_mid: float = 0.6,
    chord_flap: float = 0.4,
    slot_gap: float = 0.050,
    overlap: float = 0.15,
    aoa_main: float = 0.0,
    aoa_mid_rel: float = 10.0,
    aoa_flap_rel: float = 20.0,
    flap_deploy: float = 0.0,
    n_points: int = 200,
) -> list:
    """Generate a 3-element inverted front wing configuration.

    All elements are inverted (camber faces downward for downforce generation).
    AoA is defined as positive = nose up (LE above TE).
    To create an upward cascade where TE points toward the next element,
    elements have negative AoA (nose down = TE above LE).
    Each downstream element is MORE nose-down (higher TE) than the previous.

    Args:
        main_profile: NACA 4-digit for main element
        mid_profile: NACA 4-digit for middle element
        flap_profile: NACA 4-digit for flap element
        chord_main: main element chord (reference length)
        chord_mid: middle element chord
        chord_flap: flap element chord
        slot_gap: minimum vertical gap between elements (m)
        overlap: fraction of upstream chord overlapped by downstream element
        aoa_main: main element angle of attack (degrees, negative = nose down)
        aoa_mid_rel: mid element angle relative to main (deg more nose-down)
        aoa_flap_rel: flap element angle relative to main (deg more nose-down)
        flap_deploy: active aero deployment (degrees added to flap AoA)
        n_points: points per surface

    Returns:
        list of dicts with keys: 'name', 'coords', 'chord', 'aoa', 'profile'
    """
    # Generate profiles
    main_coords = invert_airfoil(naca_4digit(main_profile, n_points))
    mid_coords = invert_airfoil(naca_4digit(mid_profile, n_points))
    flap_coords = invert_airfoil(naca_4digit(flap_profile, n_points))

    # Element angles (negative = nose down = TE above LE for upward cascade)
    aoa_mid = aoa_main - aoa_mid_rel
    aoa_flap = aoa_main - aoa_flap_rel + flap_deploy

    # Main element: position TE at (chord_main, 0) so LE ends up near x=0
    main_pos = position_element(main_coords, chord_main, aoa_main, chord_main, 0.0)

    # Mid element: LE at (1 - overlap) * chord_main, above main's highest point + slot_gap
    mid_le_x = (1.0 - overlap) * chord_main
    mid_le_y = main_pos[:, 1].max() + slot_gap
    mid_pos = position_element_by_le(mid_coords, chord_mid, aoa_mid, mid_le_x, mid_le_y)

    # Flap element: LE at mid_LE_x + (1 - overlap) * mid_chord, above mid's highest point + slot_gap
    flap_le_x = mid_le_x + (1.0 - overlap) * chord_mid
    flap_le_y = mid_pos[:, 1].max() + slot_gap
    flap_pos = position_element_by_le(flap_coords, chord_flap, aoa_flap, flap_le_x, flap_le_y)

    return [
        {"name": "Main", "coords": main_pos, "chord": chord_main, "aoa": aoa_main, "profile": main_profile},
        {"name": "Mid", "coords": mid_pos, "chord": chord_mid, "aoa": aoa_mid, "profile": mid_profile},
        {"name": "Flap", "coords": flap_pos, "chord": chord_flap, "aoa": aoa_flap, "profile": flap_profile},
    ]


def get_element_bounds(elements: list) -> tuple:
    """Get (xmin, xmax, ymin, ymax) bounding box of all elements."""
    all_pts = np.vstack([e["coords"] for e in elements])
    return (
        all_pts[:, 0].min(),
        all_pts[:, 0].max(),
        all_pts[:, 1].min(),
        all_pts[:, 1].max(),
    )


def plot_airfoil(elements: list, save_path: Path = None, title: str = ""):
    """Plot airfoil element configuration.

    Args:
        elements: list of dicts from front_wing_3_element()
        save_path: optional path to save PNG
        title: optional plot title
    """
    set_f1_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    colors = [MERCEDES_TEAL, MERCEDES_RED, "#FFB347"]
    for elem, color in zip(elements, colors):
        coords = elem["coords"]
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=1.5, label=elem["name"])
        if elem["name"] == "Main":
            ax.plot(coords[0, 0], coords[0, 1], "o", color=color, markersize=4)
        midpoint = len(coords) // 2
        angle_rad = np.radians(elem["aoa"])
        ax.annotate(
            f'{elem["name"]}: AoA={elem["aoa"]:.1f} deg',
            xy=(coords[midpoint, 0], coords[midpoint, 1]),
            textcoords="offset points",
            xytext=(5, 10),
            fontsize=9,
            color=color,
        )

    ax.axhline(0, color=MERCEDES_GRAY, linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or "Three-Element Inverted Front Wing Configuration")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def preview_front_wing(save_dir: Path = None):
    """Generate preview plots of the front wing configuration for documentation."""
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Reference configuration
    ref = front_wing_3_element()
    if save_dir:
        plot_airfoil(ref, save_dir / "reference_config.png",
                     "Reference Configuration: AoA=0 deg, h=50mm, sg=25mm")

    # AoA sweep visual
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = axes.flatten()
    colors = [MERCEDES_TEAL, MERCEDES_RED, "#FFB347"]
    for idx, aoa in enumerate([-4, 0, 4, 8, 12, 16, 20]):
        elems = front_wing_3_element(aoa_main=aoa)
        ax = axes[idx]
        for elem, color in zip(elems, colors):
            coords = elem["coords"]
            ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=1.0)
        ax.set_title(f"AoA={aoa} deg", fontsize=10)
        ax.set_xlim(-0.2, 1.8)
        ax.set_ylim(-0.15, 0.05)
        ax.axhline(0, color=MERCEDES_GRAY, linewidth=0.3, alpha=0.3)
        ax.set_aspect("equal")
        ax.axis("off")
    axes[7].axis("off")
    fig.suptitle("AoA Sweep: Element Positions at Different Angles of Attack",
                 fontsize=14, color=MERCEDES_WHITE)
    fig.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "aoa_sweep_preview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if save_dir:
        print(f"Preview images saved to {save_dir}/")
