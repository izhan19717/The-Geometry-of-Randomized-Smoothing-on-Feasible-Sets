"""Exact-rational interval primitives for small numerical proof artifacts.

The routines in this module never evaluate a transcendental function with
binary floating point.  They use :class:`fractions.Fraction` throughout and
enclose the required values with elementary convergent series:

* ``exp(-y)`` and ``integral_0^t exp(-x^2/2) dx`` use alternating-series
  brackets after the terms become decreasing;
* ``log(x)`` uses ``2 * atanh((x-1)/(x+1))`` with a geometric tail bound;
* ``pi`` uses Machin's identity and alternating arctangent series;
* square roots are rounded outwards on a decimal rational grid.

Soundness therefore rests on Python integer/Fraction arithmetic and the
documented series remainder arguments, rather than on a platform libm.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from math import isqrt


@dataclass(frozen=True)
class RationalInterval:
    """Closed interval with exact rational endpoints."""

    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError("interval lower endpoint exceeds upper endpoint")

    @classmethod
    def point(cls, value: Fraction | int) -> "RationalInterval":
        rational = Fraction(value)
        return cls(rational, rational)

    def __add__(self, other: "RationalInterval | Fraction | int") -> "RationalInterval":
        rhs = as_interval(other)
        return RationalInterval(self.lo + rhs.lo, self.hi + rhs.hi)

    __radd__ = __add__

    def __neg__(self) -> "RationalInterval":
        return RationalInterval(-self.hi, -self.lo)

    def __sub__(self, other: "RationalInterval | Fraction | int") -> "RationalInterval":
        return self + (-as_interval(other))

    def __rsub__(self, other: "RationalInterval | Fraction | int") -> "RationalInterval":
        return as_interval(other) - self

    def __mul__(self, other: "RationalInterval | Fraction | int") -> "RationalInterval":
        rhs = as_interval(other)
        products = (
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        )
        return RationalInterval(min(products), max(products))

    __rmul__ = __mul__

    def reciprocal(self) -> "RationalInterval":
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError("cannot invert an interval containing zero")
        endpoints = (1 / self.lo, 1 / self.hi)
        return RationalInterval(min(endpoints), max(endpoints))

    def __truediv__(self, other: "RationalInterval | Fraction | int") -> "RationalInterval":
        return self * as_interval(other).reciprocal()

    def __rtruediv__(self, other: "RationalInterval | Fraction | int") -> "RationalInterval":
        return as_interval(other) / self

    @property
    def width(self) -> Fraction:
        return self.hi - self.lo


def as_interval(value: RationalInterval | Fraction | int) -> RationalInterval:
    return value if isinstance(value, RationalInterval) else RationalInterval.point(value)


def _alternating_exp_neg(y: Fraction, tolerance: Fraction) -> RationalInterval:
    """Enclose ``exp(-y)`` for rational ``y >= 0``.

    For ``A_n = y^n/n!``, once ``2m+1 >= y`` the terms from ``A_(2m)``
    onward decrease.  Consequently ``S_(2m+1) <= exp(-y) <= S_(2m)``.
    The loop also requires the bracket width ``A_(2m+1)`` to be at most the
    requested tolerance.
    """

    if y < 0:
        raise ValueError("y must be non-negative")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    term = Fraction(1)
    partial = Fraction(1)
    n = 0
    while True:
        next_n = n + 1
        next_term = term * y / next_n
        next_partial = partial - next_term if next_n % 2 else partial + next_term
        if n % 2 == 0 and next_n >= y and next_term <= tolerance:
            return RationalInterval(next_partial, partial)
        n = next_n
        term = next_term
        partial = next_partial


@cache
def gaussian_kernel_value(x: Fraction, tolerance: Fraction) -> RationalInterval:
    """Enclose ``exp(-x^2/2)`` at an exact rational point."""

    return _alternating_exp_neg(x * x / 2, tolerance)


def _gaussian_integral_from_zero_positive(
    t: Fraction, tolerance: Fraction
) -> RationalInterval:
    """Enclose ``integral_0^t exp(-x^2/2) dx`` for rational ``t >= 0``.

    The integrated power series has terms
    ``A_n=t^(2n+1)/(2^n*n!*(2n+1))``.  Their ratio is bounded by
    ``(t^2/2)/(n+1)``, so the same even/odd alternating bracket applies once
    ``n+1 >= t^2/2``.
    """

    if t < 0:
        raise ValueError("t must be non-negative")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    y = t * t / 2
    term = t
    partial = t
    n = 0
    while True:
        next_n = n + 1
        next_term = term * y * (2 * n + 1) / (next_n * (2 * next_n + 1))
        next_partial = partial - next_term if next_n % 2 else partial + next_term
        if n % 2 == 0 and next_n >= y and next_term <= tolerance:
            return RationalInterval(next_partial, partial)
        n = next_n
        term = next_term
        partial = next_partial


@cache
def gaussian_integral_from_zero(t: Fraction, tolerance: Fraction) -> RationalInterval:
    """Enclose the odd Gaussian-kernel antiderivative from zero to ``t``."""

    if t >= 0:
        return _gaussian_integral_from_zero_positive(t, tolerance)
    return -_gaussian_integral_from_zero_positive(-t, tolerance)


@cache
def gaussian_kernel_mass(
    lower: Fraction, upper: Fraction, tolerance: Fraction
) -> RationalInterval:
    """Enclose ``integral_lower^upper exp(-x^2/2) dx``."""

    if lower > upper:
        raise ValueError("lower integration endpoint exceeds upper endpoint")
    return gaussian_integral_from_zero(upper, tolerance) - gaussian_integral_from_zero(
        lower, tolerance
    )


def _atan_positive(z: Fraction, tolerance: Fraction) -> RationalInterval:
    """Alternating-series enclosure of ``atan(z)`` for ``0 <= z <= 1``."""

    if not 0 <= z <= 1:
        raise ValueError("atan series expects z in [0, 1]")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    term = z
    partial = z
    n = 0
    while True:
        next_n = n + 1
        next_term = term * z * z * (2 * n + 1) / (2 * next_n + 1)
        next_partial = partial - next_term if next_n % 2 else partial + next_term
        if n % 2 == 0 and next_term <= tolerance:
            return RationalInterval(next_partial, partial)
        n = next_n
        term = next_term
        partial = next_partial


def pi_interval(tolerance: Fraction) -> RationalInterval:
    """Enclose pi using ``pi = 16 atan(1/5) - 4 atan(1/239)``."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    component_tolerance = tolerance / 32
    return 16 * _atan_positive(Fraction(1, 5), component_tolerance) - 4 * _atan_positive(
        Fraction(1, 239), component_tolerance
    )


def sqrt_interval(value: RationalInterval, digits: int) -> RationalInterval:
    """Round ``sqrt(value)`` outwards to ``digits`` decimal places."""

    if value.lo < 0:
        raise ValueError("sqrt interval must be non-negative")
    if digits <= 0:
        raise ValueError("digits must be positive")
    scale = 10**digits

    def floor_sqrt(rational: Fraction) -> int:
        scaled_floor = rational.numerator * scale * scale // rational.denominator
        return isqrt(scaled_floor)

    lower_integer = floor_sqrt(value.lo)
    upper_floor = floor_sqrt(value.hi)
    if upper_floor * upper_floor * value.hi.denominator == value.hi.numerator * scale * scale:
        upper_integer = upper_floor
    else:
        upper_integer = upper_floor + 1
    return RationalInterval(Fraction(lower_integer, scale), Fraction(upper_integer, scale))


def _log_positive_at_least_one(x: Fraction, tolerance: Fraction) -> RationalInterval:
    """Enclose log(x) for rational ``x >= 1`` by the atanh series."""

    if x < 1:
        raise ValueError("x must be at least one")
    z = (x - 1) / (x + 1)
    if z == 0:
        return RationalInterval.point(0)
    z2 = z * z
    term = z
    partial = z
    n = 0
    while True:
        next_power = term * z2
        next_n = n + 1
        next_term = next_power * (2 * n + 1) / (2 * next_n + 1)
        remainder = next_term / (1 - z2)
        if 2 * remainder <= tolerance:
            return RationalInterval(2 * partial, 2 * (partial + remainder))
        term = next_term
        partial += term
        n = next_n


def log_rational(x: Fraction, tolerance: Fraction) -> RationalInterval:
    """Enclose the natural logarithm of a positive rational."""

    if x <= 0:
        raise ValueError("log argument must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if x >= 1:
        return _log_positive_at_least_one(x, tolerance)
    return -_log_positive_at_least_one(1 / x, tolerance)


def log_interval(value: RationalInterval, tolerance: Fraction) -> RationalInterval:
    """Monotone outward enclosure of log over a positive interval."""

    if value.lo <= 0:
        raise ValueError("log interval must be strictly positive")
    lower = log_rational(value.lo, tolerance).lo
    upper = log_rational(value.hi, tolerance).hi
    return RationalInterval(lower, upper)


def outward_decimal_interval(value: RationalInterval, digits: int) -> tuple[Fraction, Fraction]:
    """Return a decimal-grid interval containing ``value`` using exact rounding."""

    if digits < 0:
        raise ValueError("digits must be non-negative")
    scale = 10**digits
    lower_integer = value.lo.numerator * scale // value.lo.denominator
    upper_integer = -((-value.hi.numerator * scale) // value.hi.denominator)
    return Fraction(lower_integer, scale), Fraction(upper_integer, scale)


def decimal_string(value: Fraction, digits: int) -> str:
    """Format a decimal-grid fraction without converting through float."""

    scale = 10**digits
    scaled = value * scale
    if scaled.denominator != 1:
        raise ValueError("value is not on the requested decimal grid")
    integer = scaled.numerator
    sign = "-" if integer < 0 else ""
    absolute = abs(integer)
    if digits == 0:
        return f"{sign}{absolute}"
    whole, fractional = divmod(absolute, scale)
    return f"{sign}{whole}.{fractional:0{digits}d}"
