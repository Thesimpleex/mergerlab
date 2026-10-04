"""Number formatting shared by the terminal output and the HTML memo."""

from __future__ import annotations

_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "CHF": "CHF "}
_SUFFIX = {"million": "m", "billion": "bn", "thousand": "k", "": ""}


def pct(x: float | None, digits: int = 1, signed: bool = True) -> str:
    """Fraction as a percentage string; ``-`` for missing values."""
    if x is None:
        return "-"
    return f"{100 * x:+.{digits}f}%" if signed else f"{100 * x:.{digits}f}%"


def money(x: float | None, currency: str = "USD", unit: str = "", digits: int = 1) -> str:
    """Amount with currency symbol and unit suffix, for example ``$14.1m``."""
    if x is None:
        return "-"
    sym = _SYMBOLS.get(currency, f"{currency} ")
    suffix = _SUFFIX.get(unit, f" {unit}")
    sign = "-" if x < 0 else ""
    return f"{sign}{sym}{abs(x):,.{digits}f}{suffix}"


def points(x: float) -> str:
    """HHI points with thousands separator."""
    return f"{x:,.0f}"
