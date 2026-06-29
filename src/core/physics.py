"""
Utility functions for F1 aerodynamics calculations.

ISA atmosphere, Reynolds/Mach number, dynamic pressure, and unit conversions.
"""

import numpy as np


AIR_DENSITY = 1.225
AIR_VISCOSITY = 1.81e-5
SPEED_OF_SOUND = 340.3
G = 9.81


def dynamic_pressure(velocity_ms: float, rho: float = AIR_DENSITY) -> float:
    """q = 0.5 * rho * v^2"""
    return 0.5 * rho * velocity_ms**2


def force_from_coefficient(coeff: float, q: float, area: float) -> float:
    return coeff * q * area


def reynolds_number(velocity_ms: float, chord: float, nu: float = AIR_VISCOSITY / AIR_DENSITY) -> float:
    """Re = v * L / nu"""
    return velocity_ms * chord / nu


def mach_number(velocity_ms: float, a: float = SPEED_OF_SOUND) -> float:
    """M = v / a"""
    return velocity_ms / a


def velocity_from_reynolds(re: float, chord: float, nu: float = AIR_VISCOSITY / AIR_DENSITY) -> float:
    """v = Re * nu / L"""
    return re * nu / chord


def kmh_to_ms(kmh: float) -> float:
    return kmh / 3.6


def ms_to_kmh(ms: float) -> float:
    return ms * 3.6


def mph_to_ms(mph: float) -> float:
    return mph * 0.44704


def air_density_isa(altitude_m: float) -> float:
    """ISA density at altitude (simplified)."""
    T0 = 288.15
    L = 0.0065
    p0 = 101325
    R = 287.058
    T = T0 - L * altitude_m
    p = p0 * (1 - L * altitude_m / T0) ** (9.80665 / (R * L))
    return p / (R * T)
