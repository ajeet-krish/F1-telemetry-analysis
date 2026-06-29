# F1 Aerodynamics Analysis

An engineering analysis of Formula 1 car aerodynamics using real telemetry data, analytical models from first principles, and SU2 CFD validation. Built as a modular Python pipeline with a standalone Mercedes-themed HTML portfolio site.

[Live Site](https://ajeet-krish.github.io/F1_Telemetry_Dashboard) (coming soon)

## Overview

This project connects theoretical aerodynamics to real-world motorsport engineering. By combining FastF1 telemetry data with analytical aerodynamic models, it demonstrates how fundamental fluid dynamics principles govern the performance of a modern Formula 1 car.

Rather than treating aerodynamics as a black box, this project derives downforce, drag, and aerodynamic efficiency from first principles, calibrates analytical models against published CFD literature, and validates predictions against actual on-track telemetry.

### Analysis Pages

| Page | Topic | Method |
|------|-------|--------|
| Theory | F1 aerodynamics fundamentals | First principles with KaTeX |
| Downforce | Component breakdown, L/D polars, speed correlation | Analytical models + telemetry |
| Ride Height | Ground effect, porpoising, aero balance | Analytical + telemetry correlation |
| DRS & Active Aero | DRS mechanics, overtaking, 2026 regulations | Analytical + telemetry |
| Track Setups | Circuit-dependent aero (Monaco vs Monza) | FastF1 track speed maps |
| Cornering | Downforce to lateral grip mapping | Analytical + telemetry g-g |
| Strategy | Tire degradation, fuel-adjusted pace | FastF1 lap/position data |
| CFD Venturi | 2D venturi tunnel ground effect simulation | SU2 RANS SST |

### Engineering Relevance

**Mechanical & Aerospace Engineering:**
- Fluid Dynamics: Bernoulli, boundary layer theory, ground effect, wake dynamics
- Aerodynamic Design: Multi-element airfoils, drag polars, L/D optimization, component interaction
- Vehicle Dynamics: Aero balance, ride height sensitivity, porpoising, platform control
- Data Analysis: High-frequency time-series processing, statistical correlation, model validation

**Python for Engineering Analysis:**
- NumPy/SciPy: Numerical computation and analytical model implementation
- Matplotlib: Publication-quality engineering visualizations (polar plots, contour maps, track heatmaps)
- Pandas: Telemetry data processing and statistical analysis
- FastF1: Direct integration with official F1 timing and telemetry APIs
- SU2: High-fidelity CFD validation for venturi tunnel ground effect

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
    track_setups.py     Circuit-dependent aero, speed-on-track maps
    cornering.py        Downforce to lateral grip mapping
    strategy.py         Tire degradation, fuel-adjusted pace
  cfd/                # SU2 CFD integration
    su2_runner.py       SU2Config, MeshGenerator, SU2Solver, PyVista viz
    venturi.py          2D venturi tunnel with moving wall, diffuser angle sweep
    validate.py         Validation against published venturi/floor data
runners/              # Entry-point scripts (invoked via python -m)
    downforce.py        uv run python -m runners.downforce
    ride_height.py      uv run python -m runners.ride_height
    drs.py              uv run python -m runners.drs
    track_setups.py     uv run python -m runners.track_setups
    cornering.py        uv run python -m runners.cornering
    strategy.py         uv run python -m runners.strategy
    cfd_venturi.py      uv run python -m runners.cfd_venturi
    all.py              Run all analyses sequentially
    build_site.py       Nav bar sync for HTML pages
docs/                 # Standalone HTML portfolio site (GitHub Pages root)
  index.html, theory.html, downforce.html, ride_height.html,
  drs_active_aero.html, track_setups.html, cornering.html,
  strategy.html, cfd_venturi.html, implementation.html
  css/style.css         Mercedes-inspired dark theme
  assets/images/        31 generated plots and visualizations
```

## Getting Started

```bash
# Install dependencies
uv sync

# Run a single analysis
uv run python -m runners.downforce
uv run python -m runners.ride_height

# Run all telemetry/analytical analyses (skips CFD)
uv run python -m runners.all

# Run CFD venturi simulation (requires SU2 v8.4)
uv run python -m runners.cfd_venturi           # full SU2 sweeps (~30 min)
uv run python -m runners.cfd_venturi --quick   # mesh preview only

# Sync nav bar across all HTML pages
uv run python -m runners.build_site

# Preview the site locally
uv run python -m http.server -d docs 8000
```

## Requirements

- Python 3.14+
- `uv` package manager
- FastF1 (telemetry data)
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
  ride_height_sweep.json  -- Sweep data for plotting
```

Run sweeps:
```bash
uv run python -m runners.cfd_venturi
```

Extract field visualizations (velocity contours, Cp profiles, streamlines)
from `vol_solution.vtu` in Paraview. Placeholder slots are in the CFD
Venturi HTML page for 22 planned images.

## Project Status

| Phase | Status |
|-------|--------|
| 0 - Core infrastructure | Done |
| 1 - Downforce analysis | Done |
| 2 - Ride height | Done |
| 3 - DRS & Active aero | Done |
| 4 - Track setups | Done |
| 5 - Cornering | Done |
| 6 - Strategy | Done |
| 7 - CFD venturi | In progress (runner + 1 sweep done; Paraview images pending) |
| 8 - Site polish | Done |

## References

- Katz, J. (2006). "Aerodynamics of Race Cars"
- Dominy, R.G. (1994). "The aerodynamic development of Formula One cars"
- Zhang et al. (2006). "Automobile aerodynamics"
- RaceTech CFD Analysis (2025). "2026 F1 Car Aerodynamics"
- Newey, A. (2017). "How to Build a Car"

---

*Built for engineering education and portfolio demonstration. All aerodynamic coefficients are estimates calibrated against published literature, not proprietary team data.*
