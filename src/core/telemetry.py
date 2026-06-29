"""
FastF1 telemetry wrapper for F1 aerodynamics analysis.

Session loading, lap filtering, telemetry extraction, and track map visualization.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import fastf1
from fastf1 import plotting
from fastf1.core import Session, Lap


CACHE_DIR = Path.home() / ".cache" / "fastf1"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


class TelemetryLoader:
    """Load and filter F1 telemetry sessions via FastF1."""

    def __init__(self, year: int = 2024, gp: str = "Monaco", session_type: str = "R"):
        self.year = year
        self.gp = gp
        self.session_type = session_type
        self.session: Optional[Session] = None
        self._loaded = False

    def load(self) -> Session:
        """Load the session telemetry."""
        self.session = fastf1.get_session(self.year, self.gp, self.session_type)
        self.session.load()
        self._loaded = True
        return self.session

    def fastest_lap(self, driver: str = "VER") -> Optional[Lap]:
        """Get the fastest lap for a given driver."""
        if not self._loaded:
            self.load()
        return self.session.laps.pick_drivers(driver).pick_fastest()

    def get_laps(self, driver: str = "VER") -> pd.DataFrame:
        """Get all laps for a driver."""
        if not self._loaded:
            self.load()
        return self.session.laps.pick_drivers(driver)

    def lap_telemetry(self, driver: str = "VER", lap_number: int = None) -> pd.DataFrame:
        """Get telemetry for a specific driver and lap."""
        laps = self.get_laps(driver)
        if lap_number is not None:
            lap = laps[laps["LapNumber"] == lap_number].iloc[0]
        else:
            lap = laps.pick_fastest()
        return lap.get_telemetry()

    def driver_top_speed(self, driver: str = "VER") -> float:
        """Return the top speed (km/h) for a driver's fastest lap."""
        telemetry = self.lap_telemetry(driver)
        return telemetry["Speed"].max()

    def speed_trace(self, driver: str = "VER") -> pd.DataFrame:
        """Get speed, throttle, brake, DRS, gear telemetry in a clean DataFrame."""
        telemetry = self.lap_telemetry(driver)
        cols = ["Time", "Speed", "RPM", "nGear", "Throttle", "Brake", "DRS", "X", "Y", "Z"]
        available = [c for c in cols if c in telemetry.columns]
        return telemetry[available].copy()

    def speed_on_track(self, driver: str = "VER", ax=None, **kwargs):
        """Plot speed colored track map (requires FastF1 plotting utils)."""
        plotting.setup_mpl()
        lap = self.fastest_lap(driver)
        return lap.plot_track(ax=ax, **kwargs)

    def session_info(self) -> dict:
        """Return session metadata."""
        if not self._loaded:
            self.load()
        return {
            "event": self.session.event["EventName"],
            "year": self.session.event.year,
            "circuit": self.session.event["Location"],
            "session": self.session.name,
            "drivers": self.session.drivers,
        }
