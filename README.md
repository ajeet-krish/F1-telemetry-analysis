# F1 Aerodynamics Analysis

An interactive analysis of Formula 1 car aerodynamics using real FastF1 telemetry data, analytical models from first principles, and SU2 CFD validation with PyVista visualization. Delivered as a standalone Mercedes-themed HTML portfolio site with interactive Plotly visualizations.

[Live Site](https://ajeet-krish.github.io/F1_Telemetry_Dashboard)

## Overview

This project connects theoretical aerodynamics to real-world motorsport engineering. By combining FastF1 telemetry data with analytical aerodynamic models and SU2 RANS CFD simulations, it demonstrates how fundamental fluid dynamics principles govern the performance of a modern Formula 1 car.

The site features three tiers of content:

- **Interactive Telemetry Explorer**: Plotly-powered synchronized track maps with hover cross-referencing between the circuit layout and speed/throttle/brake/gear traces, plus a 3D grip envelope (Lat G vs Long G vs Speed) with driver comparison dropdowns.
- **Static Analysis Pages**: Deep-dive analytical content with matplotlib/seaborn visualizations covering downforce breakdowns, ride height sensitivity, DRS mechanics, cornering performance, strategy, and powertrain regimes.
- **CFD Validation Page**: SU2 RANS SST simulation of the floor venturi tunnel with PyVista-generated flow field visualizations (velocity, Cp, vorticity contours; streamlines; wall profile extractions) and parametric ride height/diffuser angle/velocity sweeps.

### Analysis Pages

| Page | Topic | Method |
|------|-------|--------|
| Theory | F1 aerodynamics fundamentals | First principles with KaTeX + analytical plots |
| Downforce | Component breakdown, L/D polars, speed correlation | Analytical models + telemetry |
| Ride Height | Ground effect, porpoising, aero balance | Analytical + telemetry correlation |
| DRS & Active Aero | DRS mechanics, overtaking, 2026 regulations | Analytical + telemetry |
| Track Setups | Circuit-dependent aero, synchronized telemetry explorer | FastF1 + Plotly interactive |
| Cornering | Downforce to lateral grip, driver KDE comparison | Analytical + telemetry g-g + seaborn |
| Strategy | Tire degradation, fuel-adjusted pace, ridge plots | FastF1 lap/position data + simulated |
| Powertrain & Aero | v^2 vs RPM, drag-limited vs power-limited | FastF1 telemetry scatter plots |
| CFD Venturi | 2D venturi tunnel ground effect simulation | SU2 RANS SST + PyVista visualization |

## Architecture

```
src/
  core/               # Foundation layer
    models.py           Analytical aero models (FrontWing, RearWing, Floor, F1Car)
    telemetry.py        FastF1 wrapper, session loading, track maps
    physics.py          ISA atmosphere, Re/Mach calculators, utilities
    style.py            Mercedes-themed matplotlib style configuration
  analysis/           # Post-processing and engineering analysis
    downforce.py        Component breakdown, L/D curves, aero maps
    ride_height.py      Ground effect sensitivity, porpoising, aero balance
    drs.py              DRS effectiveness, 2026 active aero modes
    track_setups.py     Circuit-dependent aero, speed-on-track maps, time delta maps
    cornering.py        Downforce to lateral grip mapping, driver KDE comparison
    strategy.py         Tire degradation, fuel-adjusted pace, degradation ridge plots
    powertrain.py       v^2 vs RPM, drag-limited vs power-limited regimes
  viz/                # Interactive Plotly visualizations
    interactive.py      G-force computation, synchronized track maps, 3D envelope
  cfd/                # SU2 CFD integration
    su2_runner.py       SU2Config, MeshGenerator, SU2Solver
    venturi.py          2D venturi tunnel with moving wall, parametric sweeps
    pyvista_viz.py      PyVista rendering: contours, streamlines, wall profiles, sweep overlays
    validate.py         Validation against published venturi/floor data
runners/              # Entry-point scripts (invoked via python -m)
    interactive.py      uv run python -m runners.interactive       # Plotly JSON assets
    downforce.py        uv run python -m runners.downforce
    ride_height.py      uv run python -m runners.ride_height
    drs.py              uv run python -m runners.drs
    track_setups.py     uv run python -m runners.track_setups
    cornering.py        uv run python -m runners.cornering
    strategy.py         uv run python -m runners.strategy
    powertrain.py       uv run python -m runners.powertrain
    cfd_venturi.py      uv run python -m runners.cfd_venturi
    all.py              Run all analyses + interactive + CFD export
    build_site.py       Sidebar sync for HTML pages
docs/                 # Standalone HTML portfolio site (GitHub Pages root)
  *.html                index, theory, downforce, ride_height, drs_active_aero,
                        track_setups, cornering, strategy, powertrain, cfd_venturi,
                        implementation
  css/style.css         Mercedes-inspired dark theme
  assets/images/        60+ matplotlib + PyVista PNG plots by section:
    downforce/           Component breakdown, L/D, drag polar, aero balance
    ride_height/         Ground effect curves, CL contour, porpoising
    drs/                 Drag polar, speed delta, overtaking, telemetry trace
    track_setups/        Speed on track, gear distribution, sector speeds
    cornering/           G-g diagram, corner radius, grip envelope, driver KDE
    strategy/            Tire delta, fuel correction, undercut, degradation
    powertrain/          v^2 vs RPM scatter
    cfd/                 PyVista renders + sweep overlays (20+ images)
    paraview_plots/      Paraview VTK screenshots (archive)
  assets/data/          Plotly JSON assets (track_map, telemetry_traces, performance_envelope)
```

## Interactive Visualizations

Two key interactive Plotly widgets are embedded in the Track Setups page:

### Synchronized Telemetry Explorer
- **Track map**: Circuit layout colored by speed (Turbo colormap) with gear shift badges overlaid at segment midpoints
- **Telemetry traces**: Speed, throttle, brake, and gear traces plotted against distance
- **Bidirectional hover sync**: Hovering on the map highlights the corresponding point in the telemetry traces, and vice versa

### 3D Performance Envelope
- **Grip envelope**: 3D scatter (Lateral G, Longitudinal G, Speed) computed from real telemetry curvature and speed gradients
- **Driver comparison**: Dropdown to toggle between VER, LEC, HAM
- **Physics explanation**: The "bowl" shape emerges because downforce scales as v^2 -- the envelope widens at higher speeds

## Getting Started

```bash
# Install dependencies
uv sync

# Run a single analysis
uv run python -m runners.downforce
uv run python -m runners.ride_height

# Regenerate interactive Plotly assets
uv run python -m runners.interactive

# Run all telemetry/analytical analyses + interactive assets
uv run python -m runners.all

# Run all + CFD PyVista export
uv run python -m runners.all --cfd

# Run CFD venturi simulation (requires SU2 v8.4)
uv run python -m runners.cfd_venturi              # full SU2 sweeps (~30 min)
uv run python -m runners.cfd_venturi --quick      # mesh preview only
uv run python -m runners.cfd_venturi --export     # PyVista Case 1 only
uv run python -m runners.cfd_venturi --export-all # PyVista Case 1 + sweep overlays

# Sync nav bar across all HTML pages
uv run python -m runners.build_site

# Preview the site locally
uv run python -m http.server -d docs 8000
```

## Requirements

- Python 3.14+
- `uv` package manager
- FastF1 (telemetry data)
- Plotly (interactive visualizations)
- PyVista (CFD visualization, VTK rendering)
- SU2 v8.4 "Harrier" (for CFD venturi simulation, optional)
- Gmsh Python SDK (mesh generation, optional)

All Python dependencies are in `pyproject.toml` and installed via `uv sync`.

## SU2 CFD Runs

The `su2_runs/` directory is organized as:

```
su2_runs/
  configs/        -- Generated .cfg files for each case
  meshes/         -- Structured quad .su2 mesh files
  results/        -- Per-case result dirs (history.csv, vol_solution.vtu)
  scratch/        -- Ad-hoc test runs
  *.json          -- Sweep data for plotting
```

Available result directories (12 VTU solution files):
- Ride height sweep: rh_vi_h{25,35,50,65,80,100}_a17
- Diffuser angle sweep: da_vi_a{10,12,14,16,18,20}_h50

PyVista extraction pipeline generates:
- Flow field: velocity contours, Cp contours, streamlines, vorticity, TKE, Mach
- Wall profiles: Cp along floor/ground, y+ distribution, wall shear
- Velocity profiles: boundary layer profiles at 4 x-stations
- Sweep overlays: side-by-side comparisons, Cp/shear overlays for all ride heights and diffuser angles

## Project Status

| Phase | Status |
|-------|--------|
| Core infrastructure | Done |
| Downforce analysis | Done |
| Ride height | Done |
| DRS & Active aero | Done |
| Track setups | Done |
| Cornering | Done |
| Strategy | Done |
| Powertrain & Aero | Done |
| Interactive Plotly viz | Done (track map sync + 3D envelope) |
| CFD venturi | Done (12 VTU solutions, PyVista extraction pipeline, sweep overlays) |
| Theory page merge | Done (analytical plots integrated) |
| Site polish | Done |

Remaining: 2 placeholders on CFD page for velocity sweep (needs SU2 re-run at multiple velocities with VTU export).

## References

- Katz, J. (2006). "Aerodynamics of Race Cars"
- Dominy, R.G. (1994). "The aerodynamic development of Formula One cars"
- Zhang et al. (2006). "Automobile aerodynamics"
- RaceTech CFD Analysis (2025). "2026 F1 Car Aerodynamics"
- Newey, A. (2017). "How to Build a Car"

---

*Built for engineering education and portfolio demonstration. All aerodynamic coefficients are estimates calibrated against published literature, not proprietary team data.*
