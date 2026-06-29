# Graph Report - .  (2026-06-29)

## Corpus Check
- 69 files · ~196,586 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 317 nodes · 502 edges · 30 communities (20 shown, 10 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 20 edges (avg confidence: 0.61)
- Token cost: 64,982 input · 6,239 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Cornering & Downforce Analysis|Cornering & Downforce Analysis]]
- [[_COMMUNITY_SU2 CFD Suite|SU2 CFD Suite]]
- [[_COMMUNITY_Core Aero Models|Core Aero Models]]
- [[_COMMUNITY_Cornering Visualization|Cornering Visualization]]
- [[_COMMUNITY_Ride Height Analysis|Ride Height Analysis]]
- [[_COMMUNITY_Race Strategy|Race Strategy]]
- [[_COMMUNITY_Documentation & Site Charts|Documentation & Site Charts]]
- [[_COMMUNITY_Physics Utilities|Physics Utilities]]
- [[_COMMUNITY_CFD Validation|CFD Validation]]
- [[_COMMUNITY_Site Build Tools|Site Build Tools]]
- [[_COMMUNITY_Downforce Charts|Downforce Charts]]
- [[_COMMUNITY_DRS & Active Aero Charts|DRS & Active Aero Charts]]
- [[_COMMUNITY_Cornering Performance Charts|Cornering Performance Charts]]
- [[_COMMUNITY_Ride Height & Ground Effect Charts|Ride Height & Ground Effect Charts]]
- [[_COMMUNITY_Strategy Charts|Strategy Charts]]
- [[_COMMUNITY_Aero Setup Comparison Charts|Aero Setup Comparison Charts]]
- [[_COMMUNITY_Track Speed Maps|Track Speed Maps]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]

## God Nodes (most connected - your core abstractions)
1. `TelemetryLoader` - 25 edges
2. `F1Car` - 22 edges
3. `kmh_to_ms()` - 20 edges
4. `RearWing` - 11 edges
5. `run_all()` - 10 edges
6. `run_all()` - 10 edges
7. `run_all()` - 10 edges
8. `dynamic_pressure()` - 10 edges
9. `Floor` - 10 edges
10. `set_f1_style()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `F1 Aerodynamics Analysis README` --references--> `Analytical Aero Models`  [EXTRACTED]
  README.md → src/core/models.py
- `F1 Aerodynamics Analysis README` --references--> `SU2 CFD Runner`  [EXTRACTED]
  README.md → src/cfd/su2_runner.py
- `F1 Aerodynamics Analysis README` --references--> `FastF1 Telemetry Wrapper`  [EXTRACTED]
  README.md → src/core/telemetry.py
- `CFD Venturi Tunnel Analysis` --references--> `SU2 CFD Runner`  [EXTRACTED]
  docs/cfd_venturi.html → src/cfd/su2_runner.py
- `Code Architecture` --references--> `Analytical Aero Models`  [EXTRACTED]
  docs/implementation.html → src/core/models.py

## Import Cycles
- None detected.

## Communities (30 total, 10 thin omitted)

### Community 0 - "Cornering & Downforce Analysis"
Cohesion: 0.05
Nodes (60): downforce_grip_envelope(), Cornering performance analysis for F1 aerodynamics.  Maps downforce to lateral g, Grip envelope showing mechanical + aero contribution to cornering speed., aero_balance_chart(), component_breakdown_chart(), drag_polar(), ld_ratio_curve(), Downforce analysis module for F1 aerodynamics.  Generates:   - Component breakdo (+52 more)

### Community 1 - "SU2 CFD Suite"
Cohesion: 0.08
Nodes (37): load_solution(), MeshGenerator, plot_velocity_field(), plot_venturi_cp(), SU2 simulation runner -- config generation, mesh creation, solver invocation, re, gmsh-based mesh generation for SU2 cases., Create a 2D structured venturi mesh with moving ground.          The venturi rep, Parameters for an SU2 .cfg file. (+29 more)

### Community 2 - "Core Aero Models"
Cohesion: 0.07
Nodes (28): CarConfig, dynamic_pressure(), Floor, FloorConfig, force_from_coefficient(), FrontWing, FrontWingConfig, Analytical aerodynamic models for F1 car components.  These models use first-pri (+20 more)

### Community 3 - "Cornering Visualization"
Cohesion: 0.07
Nodes (25): corner_classification(), gg_diagram(), Corner radius estimated from speed and lateral acceleration., Classify corners by speed and lateral g from telemetry., Generate all cornering analysis visuals., G-g diagram from telemetry: lateral vs longitudinal acceleration., run_all(), speed_vs_corner_radius() (+17 more)

### Community 4 - "Ride Height Analysis"
Cohesion: 0.13
Nodes (20): aero_balance_ride_height(), cl_contour_map(), cl_with_venturi_stall(), downforce_vs_ride_height(), porpoising_stability_map(), Ride height sensitivity and porpoising analysis for F1 aerodynamics.  Generates:, Gradient d|C_L|/dh map showing porpoising instability regions.      Positive gra, Front aero balance as a function of both front and rear ride height. (+12 more)

### Community 5 - "Race Strategy"
Cohesion: 0.17
Nodes (14): fuel_correction(), race_pace_projection(), Race strategy analysis for F1 aerodynamics.  Models tire compound performance, f, Simulate the undercut: lap time delta vs pit stop timing., Model tire degradation: lap time increase with tire age., Project race pace for different strategy options., Generate all strategy analysis visuals., Bar chart of estimated lap time delta by tire compound. (+6 more)

### Community 6 - "Documentation & Site Charts"
Cohesion: 0.20
Nodes (11): CFD Venturi Tunnel Analysis, Venturi Floor Profiles Diagram, CFD Ride Height Sweep Chart, Code Architecture, Analytical Aero Models, Physics Utilities, F1 Aerodynamics Analysis README, Mercedes-themed Matplotlib Style (+3 more)

### Community 7 - "Physics Utilities"
Cohesion: 0.22
Nodes (3): air_density_isa(), Utility functions for F1 aerodynamics calculations.  ISA atmosphere, Reynolds/Ma, ISA density at altitude (simplified).

### Community 8 - "CFD Validation"
Cohesion: 0.32
Nodes (7): plot_convergence(), Validation of venturi CFD results against published data.  Compares SU2 RANS res, Generate all validation visuals., Compare SU2 results with the analytical Floor model., Plot convergence history if available., run_all(), validate_vs_analytical()

### Community 9 - "Site Build Tools"
Cohesion: 0.38
Nodes (6): get_nav_block(), Path, Nav bar sync utility for standalone HTML pages.  Propagates nav bar HTML from te, Set the 'active' class on the correct nav link., set_active_class(), sync_nav()

### Community 10 - "Downforce Charts"
Cohesion: 0.33
Nodes (6): Aero Balance Shift Chart, Downforce Component Breakdown Chart, Drag Polar Chart, Downforce Analysis, L/D Ratio Chart, Downforce vs Speed Chart

### Community 11 - "DRS & Active Aero Charts"
Cohesion: 0.33
Nodes (6): 2022 vs 2026 Regulation Comparison Chart, DRS & Active Aero Analysis, DRS Effect on Drag Polar Chart, Overtaking Advantage Chart, DRS Downforce and Drag Delta Chart, Telemetry Trace with DRS Zones Chart

### Community 12 - "Cornering Performance Charts"
Cohesion: 0.40
Nodes (5): Corner Classification Chart, G-G Diagram Chart, Grip Envelope Chart, Cornering Performance Analysis, Corner Radius vs Speed Chart

### Community 13 - "Ride Height & Ground Effect Charts"
Cohesion: 0.40
Nodes (5): Aero Balance Shift with Ride Height Chart, Ground Effect Map Contour, Downforce vs Ride Height Curves Chart, Ride Height & Ground Effect Analysis, Porpoising Stability Gradient Map

### Community 14 - "Strategy Charts"
Cohesion: 0.67
Nodes (3): Race Strategy Analysis: Race Pace and Total Time Chart, Tire Degradation Model Chart, Tire Compound Performance Delta Chart

## Knowledge Gaps
- **41 isolated node(s):** `Path`, `CompletedProcess`, `Session`, `Lap`, `Ride Height Sensitivity -- Downforce Gradient Chart` (+36 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `F1Car` connect `Cornering & Downforce Analysis` to `Core Aero Models`, `Ride Height Analysis`, `Race Strategy`?**
  _High betweenness centrality (0.169) - this node is a cross-community bridge._
- **Why does `set_f1_style()` connect `Cornering & Downforce Analysis` to `CFD Validation`, `SU2 CFD Suite`, `Ride Height Analysis`, `Race Strategy`?**
  _High betweenness centrality (0.150) - this node is a cross-community bridge._
- **Why does `TelemetryLoader` connect `Cornering Visualization` to `Cornering & Downforce Analysis`, `Race Strategy`?**
  _High betweenness centrality (0.136) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `F1Car` (e.g. with `F1Car` and `F1Car`) actually correct?**
  _`F1Car` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Run all analysis modules sequentially.  Usage:     uv run python -m runners.all`, `Path`, `Nav bar sync utility for standalone HTML pages.  Propagates nav bar HTML from te` to the rest of the system?**
  _156 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Cornering & Downforce Analysis` be split into smaller, more focused modules?**
  _Cohesion score 0.05311871227364185 - nodes in this community are weakly interconnected._
- **Should `SU2 CFD Suite` be split into smaller, more focused modules?**
  _Cohesion score 0.07585568917668825 - nodes in this community are weakly interconnected._