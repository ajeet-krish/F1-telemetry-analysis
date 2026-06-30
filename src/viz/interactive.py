import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
import warnings

G = 9.80665


def _compute_g_forces(tel: pd.DataFrame) -> pd.DataFrame:
    """Compute longitudinal and lateral G-forces from telemetry."""
    speed_ms = tel["Speed"].values / 3.6
    x = tel["X"].values
    y = tel["Y"].values

    # Longitudinal acceleration from speed trace
    if "SessionTime" in tel.columns:
        st_all = tel["SessionTime"].values
        st_sec_all = st_all.astype(np.float64)
        if st_sec_all.max() > 1e6:
            st_sec_all = st_sec_all * 1e-9
        st = st_sec_all
        dt = np.diff(st)
        dt = np.where(dt <= 0, 0.01, dt)
        dt = np.concatenate([[dt[0]], dt])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ax = np.gradient(speed_ms, dt) / G
    else:
        ax = np.zeros_like(speed_ms)

    # Lateral acceleration from curvature
    curvature = np.zeros_like(speed_ms)
    for i in range(1, len(x) - 1):
        dx1, dy1 = x[i] - x[i - 1], y[i] - y[i - 1]
        dx2, dy2 = x[i + 1] - x[i], y[i + 1] - y[i]
        cross = dx1 * dy2 - dy1 * dx2
        denom = np.sqrt((dx1**2 + dy1**2) * (dx2**2 + dy2**2))
        if denom > 1e-6:
            curvature[i] = 2 * cross / denom
    ay = curvature * speed_ms**2 / G

    ax = np.clip(np.nan_to_num(ax, nan=0), -5, 5)
    ay = np.clip(np.nan_to_num(ay, nan=0), -5, 5)

    tel = tel.copy()
    tel["LongitudinalG"] = ax
    tel["LateralG"] = ay
    return tel


def get_interactive_track_map(lap_telemetry: pd.DataFrame):
    """
    Interactive track map with speed coloring and gear shift labels.
    Returns JSON string for Plotly.js.
    """
    tel = lap_telemetry.copy()
    speed = tel["Speed"].values
    gear_col = "nGear" if "nGear" in tel.columns else ("Gear" if "Gear" in tel.columns else None)
    distance = tel["Distance"].values

    # Track colored by speed
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=tel["X"], y=tel["Y"],
        mode="markers",
        marker=dict(
            size=5,
            color=speed,
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(title="Speed<br>(km/h)", x=1.02, len=0.6)
        ),
        hovertemplate="Speed: %{marker.color:.0f} km/h<extra></extra>",
        name="Speed",
    ))

    # Gear shift labels at segment midpoints
    if gear_col:
        gears = tel[gear_col].values
        changes = np.where(np.diff(gears, prepend=gears[0]) != 0)[0]
        segments = np.split(np.arange(len(tel)), changes)

        for seg in segments:
            if len(seg) == 0:
                continue
            mid = seg[len(seg) // 2]
            g = int(gears[mid])
            if g < 1:
                continue
            fig.add_annotation(
                x=tel["X"].iloc[mid],
                y=tel["Y"].iloc[mid],
                text=f"<b>{g}</b>",
                showarrow=False,
                font=dict(size=10, color="white"),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor="#00D2BE",
                borderwidth=1,
                borderpad=3,
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        margin=dict(l=10, r=50, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        dragmode=False,
    )

    return pio.to_json(fig)


def get_telemetry_traces(lap_telemetry: pd.DataFrame):
    """
    Multi-row telemetry traces (Speed, Throttle, Brake, Gear) against distance.
    Returns JSON string for Plotly.js.
    """
    tel = lap_telemetry.copy()
    distance = tel["Distance"].values / 1000  # km
    speed = tel["Speed"].values
    throttle = tel["Throttle"].values
    brake = tel["Brake"].values
    gear_col = "nGear" if "nGear" in tel.columns else ("Gear" if "Gear" in tel.columns else None)

    fig = go.Figure()

    # Speed
    fig.add_trace(go.Scatter(
        x=distance, y=speed,
        mode="lines",
        line=dict(color="white", width=2),
        name="Speed",
        hovertemplate="Speed: %{y:.0f} km/h<extra></extra>",
    ))

    # Throttle
    fig.add_trace(go.Scatter(
        x=distance, y=throttle,
        mode="lines",
        line=dict(color="#00D2BE", width=2),
        name="Throttle",
        hovertemplate="Throttle: %{y:.0f}%<extra></extra>",
    ))

    # Brake
    fig.add_trace(go.Scatter(
        x=distance, y=brake,
        mode="lines",
        line=dict(color="#E94560", width=2),
        name="Brake",
        hovertemplate="Brake: %{y:.0f}<extra></extra>",
    ))

    # Gear
    if gear_col:
        gears = tel[gear_col].values
        fig.add_trace(go.Scatter(
            x=distance, y=gears,
            mode="lines+markers",
            marker=dict(size=4, color="#FFB347"),
            line=dict(color="#FFB347", width=1.5, shape="hv"),
            name="Gear",
            hovertemplate="Gear: %{y:.0f}<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        hovermode="x unified",
        height=400,
        margin=dict(l=50, r=20, t=10, b=40),
        xaxis=dict(title="Distance (km)", gridcolor="#2A2A2A"),
        yaxis=dict(title="", gridcolor="#2A2A2A"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    return pio.to_json(fig)


def get_performance_envelope_3d(telemetry_data: dict):
    """
    3D Performance Envelope (Lat G, Long G, Speed) with driver/track toggle.
    telemetry_data: dict of {label: dataframe}
    """
    fig = go.Figure()

    for i, (name, df) in enumerate(telemetry_data.items()):
        df = _compute_g_forces(df)
        fig.add_trace(go.Scatter3d(
            x=df["LateralG"],
            y=df["LongitudinalG"],
            z=df["Speed"],
            mode="markers",
            marker=dict(
                size=2,
                color=df["Speed"],
                colorscale="Turbo",
                opacity=0.7,
                colorbar=dict(title="Speed<br>(km/h)", x=1.02),
            ),
            name=name,
            visible=(i == 0),
        ))

    buttons = []
    for i, name in enumerate(telemetry_data.keys()):
        visible = [False] * len(telemetry_data)
        visible[i] = True
        buttons.append(dict(label=name, method="update", args=[{"visible": visible}]))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1A1A1A",
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=0.1,
            y=1.1,
            bgcolor="#2A2A2A",
            bordercolor="#00D2BE",
            font=dict(color="white"),
        )],
        scene=dict(
            xaxis_title="Lateral G",
            yaxis_title="Longitudinal G",
            zaxis_title="Speed (km/h)",
            bgcolor="#1A1A1A",
            xaxis=dict(gridcolor="#2A2A2A"),
            yaxis=dict(gridcolor="#2A2A2A"),
            zaxis=dict(gridcolor="#2A2A2A"),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600,
    )

    return pio.to_json(fig)
