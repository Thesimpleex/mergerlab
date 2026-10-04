"""Exceptions raised by mergerlab."""

from __future__ import annotations


class CalibrationError(ValueError):
    """Inputs cannot be rationalised by the demand system (message carries diagnostics)."""


class EquilibriumNotFound(RuntimeError):
    """No price vector satisfied the first-order conditions within the residual gate."""

    def __init__(self, message: str, diagnostics: dict[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
