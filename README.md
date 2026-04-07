# F1 Aerodynamics Analysis

An engineering analysis of Formula 1 car aerodynamics using real telemetry data, analytical models, and first-principles physics. Built with Python and Jupyter Notebooks.

## Overview
This project aims to connect theoretical aerodynamics and real-world motorsport engineering. By combining FastF1 telemetry data with analytical aerodynamic models, it demonstrates how fundamental fluid dynamics principles govern the performance of a modern Formula 1 car.
Rather than treating aerodynamics as a black box, this project derives downforce, drag, and aerodynamic efficiency from first principles, calibrates analytical models against published CFD literature, and validates predictions against actual on-track telemetry.

## Engineering Relevance
### Mechanical & Aerospace Engineering Applications
This project serves as a practical demonstration of core engineering concepts applied to a high-performance system:
- **Fluid Dynamics**: Bernoulli's principle, boundary layer theory, ground effect, and wake dynamics
- **Aerodynamic Design**: Multi-element airfoils, drag polars, lift-to-drag optimization, and component interaction
- **Vehicle Dynamics**: Aero balance, ride height sensitivity, porpoising, and platform control
- **Data Analysis**: High-frequency time-series processing, statistical correlation, and model validation
### Python for Engineering Analysis
The project showcases modern Python as an engineering tool:
- **NumPy/SciPy**: Numerical computation and analytical model implementation
- **Matplotlib**: Publication-quality engineering visualizations (polar plots, contour maps, waterfall charts)
- **Pandas**: Telemetry data processing and statistical analysis
- **FastF1**: Direct integration with official F1 timing and telemetry APIs

## Understanding Theory Using Real-World Data
### Real-World Validation
Every model is tested against actual telemetry:
- **FastF1 API**: Real lap data from 2022-2026 F1 seasons
- **Telemetry Correlation**: Speed traces, DRS activation, throttle/brake inputs
- **Track-Specific Analysis**: Circuit-dependent aero performance variations

## Planned Features
- **Analytical Aero Models**: Front wing, rear wing, floor, and complete car models
- **Downforce Analysis**: Component breakdown, speed correlation, track section mapping
- **Drag & Power Analysis**: Drag polar visualization, power requirements, DRS effectiveness
- **Ride Height Sensitivity**: Ground effect modeling, porpoising analysis, aero balance shifts
- **Interactive Visualizations**: Polar plots, contour maps, waterfall charts, 3D surfaces
- **CFD Integration**: OpenFOAM setup for front wing flow visualization
- **Active Aero Modeling**: 2026 regulations with active straight mode simulation
- **Driver Comparison**: Head-to-head aero setup analysis
- **Season Trends**: Aero efficiency evolution across circuits and regulations
- **Web Dashboard**: Interactive visualization layer (returning from initial prototype)
---
*Built for engineering education and portfolio demonstration. All aerodynamic coefficients are estimates calibrated against published literature, not proprietary team data.*