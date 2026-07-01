# Project Context - F1 Aerodynamics Analysis

## Style Rules

- **No em dashes in any file in this project.** No `---`, no `&mdash;`, no `&ndash;`, no literal Unicode em dash (U+2014). This applies to all Markdown, Python, HTML, CSS, and configuration files.
- **No double dashes `--` in any file in this project.** Do not use `--` as a clause separator. Instead, use proper punctuation: commas (,), periods (.), colons (:), semicolons (;), or parentheses (()).

## Project Goal

Build a modular Python pipeline for F1 aerodynamics analysis that connects first-principles analytical models to real FastF1 telemetry data. Output a standalone Mercedes-themed HTML portfolio site on GitHub Pages with interactive Plotly visualizations (synchronized track maps, 3D grip envelopes). SU2 CFD validation + PyVista visualization pipeline for the floor venturi tunnel ground effect simulation.

## Analysis Pages

| Page | Topic | Module | Runner |
|------|-------|--------|--------|
| Theory | F1 aero fundamentals + analytical plots | -- | -- |
| Downforce | Component breakdown, L/D polars | src/analysis/downforce.py | runners.downforce |
| Ride Height | Ground effect, porpoising | src/analysis/ride_height.py | runners.ride_height |
| DRS & Active Aero | DRS, 2026 modes | src/analysis/drs.py | runners.drs |
| Track Setups | Monaco vs Monza speed maps, driver time delta, synchronized telemetry explorer | src/analysis/track_setups.py | runners.track_setups |
| Cornering | Downforce to lateral g, driver KDE comparison | src/analysis/cornering.py | runners.cornering |
| Strategy | Tire/fuel strategy, degradation ridge plot | src/analysis/strategy.py | runners.strategy |
| Powertrain & Aero | v^2 vs RPM, drag-limited vs power-limited | src/analysis/powertrain.py | runners.powertrain |
| CFD Venturi | SU2 2D venturi ground effect + PyVista viz | src/cfd/venturi.py + pyvista_viz.py | runners.cfd_venturi |

## Architecture

```
src/
  core/
    models.py           F1Car, FrontWing, RearWing, Floor (dataclass-based analytical)
    telemetry.py        FastF1 wrapper, session load, lap filter, track map viz
    physics.py          ISA atmosphere, Re/Mach/dynamic pressure utilities
    style.py            Mercedes-themed matplotlib style (dark bg, teal accents)
  analysis/
    downforce.py        Component breakdown, L/D curves, speed correlation plots
    ride_height.py      Ground effect sweep, aero balance, porpoising envelopes
    drs.py              DRS drag reduction, overtaking delta, 2022 vs 2026 modes
    track_setups.py     Telemetry pull, sector analysis, speed-on-track heatmaps
    cornering.py        G-g diagram, corner radius mapping, downforce contribution
    strategy.py         Tire delta, fuel correction, undercut simulation
    powertrain.py       v^2 vs RPM scatter, drag-limited vs power-limited regimes
  viz/                  # Interactive Plotly visualizations
    interactive.py      G-force computation from curvature + speed gradient,
                        synchronized track map + telemetry traces via plotly.js,
                        3D performance envelope with driver dropdown
  cfd/
    su2_runner.py       SU2Config, SU2Solver, MeshGenerator
    venturi.py          2D venturi tunnel with moving wall, parametric sweeps, export orchestration
    pyvista_viz.py      PyVista rendering: contours, streamlines, wall profiles, sweep overlays
    validate.py         Validation against published venturi/floor data
runners/
    downforce.py        python -m runners.downforce
    ride_height.py      python -m runners.ride_height
    drs.py              python -m runners.drs
    track_setups.py     python -m runners.track_setups
    cornering.py        python -m runners.cornering
    strategy.py         python -m runners.strategy
    powertrain.py       python -m runners.powertrain
    cfd_venturi.py      python -m runners.cfd_venturi (--quick | --export | --export-all)
    interactive.py      python -m runners.interactive       # Plotly JSON asset generation
    all.py              Run all analyses + interactive + CFD export
    build_site.py       Sidebar sync for HTML pages
docs/
  html pages            Standalone, KaTeX math, Prism.js code, Mercedes theme
  css/style.css         Dark bg #1a1a1a, teal accent #00D2BE, white text
  assets/images/        Per-section subdirectories (60+ PNGs):
    downforce/           Component breakdown, L/D, drag polar, aero balance
    ride_height/         Ground effect curves, CL contour, porpoising
    drs/                 Drag polar, speed delta, overtaking, telemetry trace
    track_setups/        Speed on track, gear distribution, sector speeds
    cornering/           G-g diagram, corner radius, grip envelope, driver KDE
    strategy/            Tire delta, fuel correction, undercut, degradation
    powertrain/          v^2 vs RPM scatter
    cfd/                 PyVista: velocity contour, Cp, streamlines, vorticity, TKE,
                         Mach, wall profiles, velocity profiles, sweep overlays
    paraview_plots/      Paraview VTK screenshots (archive)
  assets/data/          Plotly JSON files for interactive visualizations
                        (track_map.json, telemetry_traces.json, performance_envelope.json)
```

## Key Patterns

- **Standalone HTML:** No build step. Pages in docs/ are self-contained. Nav sync via runners.build_site.
- **Mercedes theme:** BG #1a1a1a, accent #00D2BE teal, secondary #d4d4d4, cards #2a2a2a.
- **Matplotlib style:** Use src/core/style.py for consistent dark theme on all plots.
- **Runner scripts:** One runner per analysis module in `runners/`. Invoke with `uv run python -m runners.<name>`. Output images to docs/assets/images/.
- **Component breakdown:** Use F1Car.component_breakdown() from models.py for downforce/drag distributions.
- **FastF1 compatibility:** fastf1>=3.8.0 works with numpy 2.x (np.NaN issue fixed upstream).
- **Interactive Plotly viz:** Generate JSON via src/viz/interactive.py functions, save to docs/assets/data/, load in HTML via Plotly.js CDN + fetch calls. Bi-directional hover sync uses Plotly.Fx.hover with a 200ms re-entrant guard.
- **G-force computation:** Lateral G from 3-point curvature on (X,Y) trajectory; Longitudinal G from np.gradient on speed vs time. Formula: ay = k * v^2 / g, ax = dv/dt / g.
- **Dual-container layout:** Track map (top, 600px) and telemetry traces (bottom, 400px) are separate Plotly containers to avoid axis-type conflicts (spatial vs distance-series axes).
- **PyVista rendering:** Use `_render_contour()` for single-mesh field contours, matplotlib for wall profile overlays. Side-by-side comparisons use `plotter.shape=(1,2)`. Always use `CFD_CAMERA` and `SCALAR_BAR_ARGS` for consistency.

## Commands

```bash
uv run python -m runners.all                           # run all telemetry/analytical + interactive
uv run python -m runners.all --cfd                     # include CFD PyVista export
uv run python -m runners.all --skip-interactive        # skip Plotly JSON generation
uv run python -m runners.interactive                   # regenerate Plotly JSON assets only
uv run python -m runners.downforce                     # run one analysis
uv run python -m runners.cfd_venturi                   # run CFD venturi (needs SU2)
uv run python -m runners.cfd_venturi --quick            # mesh preview only, skip SU2
uv run python -m runners.cfd_venturi --export           # PyVista Case 1 only
uv run python -m runners.cfd_venturi --export-all       # PyVista Case 1 + sweep overlays
uv run python -m runners.powertrain                    # run powertrain + aero analysis
uv run python -m runners.build_site                    # sync sidebar across HTML pages
uv run python -m runners.build_site --check             # dry run
uv run python -m http.server -d docs 8000              # preview site
```

## FastF1 Critical Knowledge

- **Compatibility:** fastf1>=3.8.0 works with numpy 2.x (no pin needed).
- **Session loading:** `session = fastf1.get_session(year, gp, 'R')` then `session.load()`
- **Telemetry data:** `lap.get_telemetry()` returns DataFrame with Speed, RPM, nGear (not Gear), Throttle, Brake, DRS, X, Y, Z, SessionTime, Distance.
- **Track maps:** Use `track = session.get_track_status()` and FastF1 plotting utilities.
- **Caching:** FastF1 caches API responses by default in `~/.cache/fastf1/`.
- **SessionTime** values are timedelta objects; convert via `total_seconds()` then `astype(np.float64)` before `np.diff()`.
- **dt from np.diff** is one element shorter than source; pad before passing to `np.gradient()`.

## Visuals Generated per Analysis

| Module | Visuals |
|--------|---------|
| downforce.py | Stacked area (component %), L/D vs speed, drag polar, speed-downforce scatter, aero balance |
| ride_height.py | DF vs height curves, CL contour map, porpoising stability, balance shift curves, ride height sensitivity dF/dh |
| drs.py | DRS drag polar, speed delta traces, overtaking advantage, telemetry trace with DRS zones, 2022 vs 2026 comparison |
| track_setups.py | Speed-on-track heatmap, gear distribution, sector speeds, aero setup comparison, speed profile |
| cornering.py | G-g diagram, speed-radius scatter, grip envelope, corner classification, driver KDE comparison |
| strategy.py | Tire delta bar chart, fuel correction, undercut simulation, tire degradation, race pace projection, degradation ridge plot |
| powertrain.py | v^2 vs RPM scatter (Monaco + Monza), drag-limited vs power-limited regime markers |
| cfd/venturi.py | Mesh preview, ride height/diffuser/velocity sweep (CL, CD plots) |
| cfd/pyvista_viz.py | Case 1: velocity/Cp/vorticity/TKE/Mach contours, streamlines, Cp/y+/shear profiles, velocity profiles. Case 2: ride height sweep (4 side-by-side/overlay comparisons). Case 3: diffuser angle sweep (3 comparisons). |
| interactive.py | Synchronized track map (speed-colored + gear badges), telemetry traces (speed/throttle/brake/gear), 3D performance envelope (driver dropdown) |

## PyVista Rendering Pipeline

- **Global settings:** `pv.global_theme.font.color = 'white'`, `pv.global_theme.font.title_size = 32`, `pv.global_theme.font.label_size = 22`
- **Window:** `WINDOW_SIZE = (1920, 1080)`
- **Camera position:** `CFD_CAMERA = [(1.5, 0.075, 3.5), (1.5, 0.075, 0), (0, 1, 0)]`
- **Scalar bar args:** `SCALAR_BAR_ARGS = {"title_font_size": 35, "label_font_size": 28, "position_x": 0.22, "position_y": 0.1, "vertical": False}`
- **Core helper:** `_render_contour(mesh, scalars, cmap, save_path, clim=None)` -- handles off-screen rendering, background, camera, and screenshot.
- **Field computation:** `compute_derivative()` output array named `'gradient'` (not `'gradient_of_Velocity'`). Vorticity computed manually as curl of velocity gradient.
- **Wall extraction:** `mesh.sample_over_line(pointa, pointb, resolution=N)` for Cp, y+, wall shear. Matplotlib overlay via `extract_and_plot_*` functions.
- **Side-by-side:** Use `pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1920, 540))` with `plotter.subplot(0, i)`.
- **Result directory naming:** Ride height: `rh_vi_h{mm}_a{deg}`, Diffuser angle: `da_vi_a{deg}_h{mm}`. No VTU files for velocity sweep.

## SU2 v8.4 Knowledge (Harrier, Rosetta x86_64)

- **Binary:** `/Users/ajeet/SU2_CFD/bin/SU2_CFD`
- **Venturi mesh markers:** MeshGenerator.venturi_2d creates physical groups: `ground` (moving), `floor` (static), `inlet`, `outlet`, `fluid`.
- **Config for incompressible venturi:**
  - `SOLVER= INC_RANS, KIND_TURB_MODEL= SST`
  - `INC_NONDIM= INITIAL_VALUES` (mu = 1/Re)
  - `INC_VELOCITY_INIT= (V_inf, 0.0, 0.0)` -- set to physical velocity
  - `INC_INLET_TYPE= PRESSURE_INLET` (requires 6 values per marker)
  - `MARKER_INLET= ( inlet, 288.15, 101325.0, 1.0, 0.0, 0.0 )`
  - `INC_OUTLET_TYPE= PRESSURE_OUTLET`
  - `MARKER_OUTLET= ( outlet, 0.0 )`
- **Moving wall:**
  - `SURFACE_MOVEMENT= MOVING_WALL`
  - `MARKER_MOVING= ( ground )`
  - `SURFACE_TRANSLATION_RATE= 1.0 0.0 0.0` (nondim, matches INC_VELOCITY_INIT)
- **SU2Config fields for marker names:** `marker_walls`, `marker_far`, `marker_inlets`, `marker_outlets`, `marker_moving`, `moving_wall`, `inc_inlet_type`, `inc_outlet_type`.
- **Mesh export:** Must call `gmsh.model.mesh.createTopology()` before `gmsh.write()`.
- **Stepping:** `MUSCL_FLOW= NO` for RANS robustness, `CONV_NUM_METHOD_FLOW= FDS`.
- **Reference area:** 2D defaults to 1.0; CL values may need scaling calibration (trends are consistent).
- **Solver version:** v8.4.0 "Harrier" (does not accept `--version` flag).

## SU2 Run Directory Layout

```
su2_runs/
  configs/              -- Generated .cfg files
  meshes/               -- Structured quad .su2 files via gmsh
  results/<case>/       -- Per-case: history.csv, restart.dat, vol_solution.vtu
    history.csv           Columns: Iter, rms[P], rms[U], rms[V], DRAG, LIFT, ...
    vol_solution.vtu      VTK unstructured grid for PyVista
  scratch/              -- Ad-hoc test runs
  *.json                -- Sweep data (ride_height_sweep.json, diffuser_angle_sweep.json, velocity_sweep.json)
```

Available VTU files (12 total):
- Ride height: `rh_vi_h{25,35,50,65,80,100}_a17` (reference case is `rh_vi_h50_a17`)
- Diffuser angle: `da_vi_a{10,12,14,16,18,20}_h50`
- Velocity sweep: JSON only (no VTU files for non-reference velocities)

## CFD Venturi HTML Page

Page at `docs/cfd_venturi.html` structured by simulation case:
- Case 1 (Reference): 11 visualizations (10 generated, 1 placeholder for velocity_profiles)
- Case 2 (Ride height sweep): 4 comparison visuals (all generated via PyVista)
- Case 3 (Diffuser angle sweep): 3 comparison visuals (all generated via PyVista)
- Case 4 (Velocity sweep): 2 placeholders (no VTU files for non-reference velocities)

Regenerate all CFD visuals: `uv run python -m runners.cfd_venturi --export-all`
