# F1 Aerodynamics Analysis

An engineering analysis of Formula 1 car aerodynamics using real telemetry data, analytical models from first principles, and light SU2 CFD validation. Built as a modular Python pipeline with a standalone Mercedes-themed HTML portfolio site.

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
| CFD Venturi | 2D venturi tunnel ground effect simulation | SU2 RANS |

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
    regulations_2026.py Active aero modes, Z-mode, battery tradeoffs
  cfd/                # SU2 CFD integration
    su2_runner.py       SU2 config, mesh generation, solver wrapper
    venturi.py          2D venturi tunnel with moving wall, diffuser angle sweep
    validate.py         Validation against published data
docs/                 # Standalone HTML portfolio site (GitHub Pages root)
  index.html, theory.html, downforce.html, ride_height.html,
  drs_active_aero.html, track_setups.html, cornering.html,
  strategy.html, cfd_venturi.html, implementation.html
  css/style.css         Mercedes-inspired dark theme
  assets/images/        Generated plots and visualizations
run_*.py              # One runner script per analysis module
run_all.py            # Orchestrator to regenerate everything
su2_runs/             # SU2 config files, meshes, results
```

## Getting Started

```bash
# Install dependencies
uv sync

# Regenerate all analysis outputs
uv run python run_all.py

# Or run individual analyses
uv run python run_downforce.py
uv run python run_ride_height.py

# Preview the site locally
uv run python -m http.server -d docs 8000
```

## Requirements

- Python 3.14+
- `uv` package manager
- FastF1 (telemetry data)
- SU2 v8.4 "Harrier" (for CFD venturi simulation)
- Gmsh Python SDK (mesh generation)

All Python dependencies are in `pyproject.toml` and installed via `uv sync`.

## Project Status

| Phase | Status |
|-------|--------|
| 0 - Core infrastructure | Done |
| 1 - Downforce analysis | In progress |
| 2 - Ride height | Planned |
| 3 - DRS & Active aero | Planned |
| 4 - Track setups | Planned |
| 5 - Cornering | Planned |
| 6 - Strategy | Planned |
| 7 - CFD venturi | Planned |
| 8 - Site polish | Planned |

## References

- Katz, J. (2006). "Aerodynamics of Race Cars"
- Dominy, R.G. (1994). "The aerodynamic development of Formula One cars"
- Zhang et al. (2006). "Automobile aerodynamics"
- RaceTech CFD Analysis (2025). "2026 F1 Car Aerodynamics"
- Newey, A. (2017). "How to Build a Car"

---

*Built for engineering education and portfolio demonstration. All aerodynamic coefficients are estimates calibrated against published literature, not proprietary team data.*
