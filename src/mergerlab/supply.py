"""Multi-product Bertrand-Nash supply side: first-order conditions and equilibrium.

With unit sales ``q(p)``, demand Jacobian ``J[j, k] = dq_j/dp_k``, marginal costs
``c`` and ownership matrix ``Omega``, firm profits ``sum_k (p_k - c_k) q_k`` give

    F(p) = q(p) + (Omega * J(p)') (p - c) = 0,

where ``*`` is the elementwise product. Pre-merger this identifies ``c`` from
observed prices; post-merger it is solved for ``p``.

Solutions are accepted only if the scaled residual ``max_j |F_j| / q_j`` is below a
gate (default 1e-10); otherwise :class:`EquilibriumNotFound` is raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize

from .errors import CalibrationError, EquilibriumNotFound
from .units import FloatArray, Ownership

if TYPE_CHECKING:
    from .demand.base import Demand

__all__ = [
    "DEFAULT_GATE",
    "Equilibrium",
    "EquilibriumNotFound",
    "foc",
    "foc_jacobian",
    "markup_matrix",
    "recover_costs",
    "scaled_residual",
    "solve_bertrand",
]

DEFAULT_GATE = 1e-10
_COMPLEX_STEP = 1e-30


@dataclass(frozen=True)
class Equilibrium:
    """A Bertrand-Nash price vector that passed the solver gate."""

    prices: FloatArray
    quantities: FloatArray
    residual: float
    method: str
    iterations: int
    starts_tried: int
    attempts: list[dict[str, object]] = field(default_factory=list)


def _omega(ownership: Ownership | FloatArray) -> FloatArray:
    return ownership.matrix if isinstance(ownership, Ownership) else np.asarray(ownership)


def foc(demand: Demand, p: FloatArray, costs: FloatArray, omega: FloatArray) -> FloatArray:
    """First-order conditions ``q + (Omega * J') (p - c)`` (complex-safe)."""
    q = demand.quantities(p)
    jac = demand.jacobian(p)
    return q + (omega * jac.T) @ (p - costs)


def scaled_residual(
    demand: Demand, p: FloatArray, costs: FloatArray, ownership: Ownership | FloatArray
) -> float:
    """``max_j |F_j(p)| / q_j``: first-order-condition residual in units of own sales."""
    omega = _omega(ownership)
    q = demand.quantities(p)
    f = foc(demand, p, costs, omega)
    return float(np.max(np.abs(f) / np.abs(q)))


def foc_jacobian(
    demand: Demand, p: FloatArray, costs: FloatArray, ownership: Ownership | FloatArray
) -> FloatArray:
    """``dF/dp`` to machine precision by complex-step differentiation."""
    omega = _omega(ownership)
    n = p.size
    jac = np.empty((n, n))
    pc = p.astype(complex)
    for k in range(n):
        step = np.zeros(n, dtype=complex)
        step[k] = 1j * _COMPLEX_STEP
        jac[:, k] = foc(demand, pc + step, costs, omega).imag / _COMPLEX_STEP
    return jac


def markup_matrix(demand: Demand, p: FloatArray, ownership: Ownership | FloatArray) -> FloatArray:
    """``Delta = Omega * J'`` so that the first-order condition reads ``q + Delta (p - c) = 0``."""
    return _omega(ownership) * demand.jacobian(p).T


def recover_costs(demand: Demand, p: FloatArray, ownership: Ownership | FloatArray) -> FloatArray:
    """Marginal costs that rationalise ``p`` as a Bertrand-Nash outcome of ``demand``.

    Raises :class:`CalibrationError` if the implied margins are not in (0, 1).
    """
    q = demand.quantities(p)
    delta = markup_matrix(demand, p, ownership)
    try:
        markup = -np.linalg.solve(delta, q)
    except np.linalg.LinAlgError as exc:
        raise CalibrationError(
            "the first-order conditions are singular at the observed prices"
        ) from exc
    costs = p - markup
    margins = markup / p
    if not np.all(np.isfinite(costs)):
        raise CalibrationError("implied marginal costs are not finite")
    if np.any(margins <= 0) or np.any(margins >= 1):
        bad = np.flatnonzero((margins <= 0) | (margins >= 1))
        raise CalibrationError(
            "the calibrated demand implies Lerner margins outside (0, 1) for products "
            f"{bad.tolist()} (margins {np.round(margins[bad], 4).tolist()}); "
            "the inputs are not consistent with profit-maximising Bertrand pricing"
        )
    return costs


def _zeta_iterate(
    demand: Demand, p: FloatArray, costs: FloatArray, omega: FloatArray, max_iter: int
) -> tuple[FloatArray, int]:
    """Morrow-Skerlos (2011) markup fixed point.

    With ``J = Gamma - diag(lam)`` the first-order condition rearranges to
    ``p = c + [q + (Omega * Gamma') (p - c)] / lam``, iterated here.
    """
    for it in range(1, max_iter + 1):
        lam, gamma = demand.zeta_terms(p)
        q = demand.quantities(p)
        p_new = costs + (q + (omega * gamma.T) @ (p - costs)) / lam
        if not np.all(np.isfinite(p_new)) or np.any(p_new <= 0):
            return p, it
        if np.max(np.abs(p_new - p) / p) < 1e-13:
            return p_new, it
        p = p_new
    return p, max_iter


def _newton(
    demand: Demand,
    p: FloatArray,
    costs: FloatArray,
    omega: FloatArray,
    tol: float,
    max_iter: int = 80,
) -> tuple[FloatArray, float, int]:
    """Damped Newton iteration on ``F(p) = 0`` with an exact complex-step Jacobian."""

    def merit(x: FloatArray) -> float:
        if not demand.is_valid(x):
            return np.inf
        q = demand.quantities(x)
        return float(np.max(np.abs(foc(demand, x, costs, omega)) / q))

    r = merit(p)
    iterations = 0
    while iterations < max_iter:
        if r < tol * 1e-3:
            break
        iterations += 1
        jac = foc_jacobian(demand, p, costs, omega)
        f = foc(demand, p, costs, omega)
        try:
            step = -np.linalg.solve(jac, f)
        except np.linalg.LinAlgError:
            break
        lam = 1.0
        improved = False
        for _ in range(40):
            cand = p + lam * step
            r_new = merit(cand)
            if r_new < r:
                p, r, improved = cand, r_new, True
                break
            lam *= 0.5
        if not improved:
            break
    return p, r, iterations


def solve_bertrand(
    demand: Demand,
    costs: FloatArray,
    ownership: Ownership | FloatArray,
    start: FloatArray,
    *,
    tol: float = DEFAULT_GATE,
    extra_starts: list[FloatArray] | None = None,
    zeta_iterations: int = 500,
) -> Equilibrium:
    """Solve the Bertrand-Nash first-order conditions and enforce the residual gate.

    Strategy: for each start (the supplied one first, then perturbations of it) run
    the Morrow-Skerlos markup fixed point when the demand system supports it, polish
    with damped Newton steps using an exact Jacobian, and fall back to a hybrid
    Powell solver. A candidate is accepted only if ``max_j |F_j| / q_j < tol`` and the
    prices lie where the demand system is valid.

    Raises
    ------
    EquilibriumNotFound
        If no start produced a solution below the gate. ``diagnostics`` lists the
        residual reached from each start and by which method.
    """
    omega = _omega(ownership)
    costs = np.asarray(costs, dtype=float)
    start = np.asarray(start, dtype=float)
    markup = np.maximum(start - costs, 1e-6 * start)
    starts = [
        start,
        costs + 1.5 * markup,
        1.1 * start,
        costs + 0.5 * markup,
        costs + 3.0 * markup,
    ]
    if extra_starts:
        starts.extend(np.asarray(s, dtype=float) for s in extra_starts)

    attempts: list[dict[str, object]] = []
    for idx, p0 in enumerate(starts):
        if not demand.is_valid(p0):
            attempts.append({"start": idx, "method": "skipped", "residual": None})
            continue
        p = p0
        total_iter = 0
        method = "newton"
        if demand.supports_zeta:
            p, n_zeta = _zeta_iterate(demand, p, costs, omega, zeta_iterations)
            total_iter += n_zeta
            method = "zeta+newton"
            if not demand.is_valid(p):
                p = p0
        p, r, n_newton = _newton(demand, p, costs, omega, tol)
        total_iter += n_newton
        attempts.append({"start": idx, "method": method, "residual": r})
        if r < tol:
            return Equilibrium(p, demand.quantities(p), r, method, total_iter, idx + 1, attempts)

        def fun(x: FloatArray) -> FloatArray:
            q = demand.quantities(x)
            return foc(demand, x, costs, omega) / q

        try:
            sol = optimize.root(fun, p0, method="hybr", tol=1e-14)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            continue
        cand, _, _ = _newton(demand, np.asarray(sol.x, dtype=float), costs, omega, tol)
        r2 = (
            float(np.max(np.abs(foc(demand, cand, costs, omega)) / demand.quantities(cand)))
            if demand.is_valid(cand)
            else np.inf
        )
        attempts.append({"start": idx, "method": "hybr+newton", "residual": r2})
        if r2 < tol:
            return Equilibrium(
                cand, demand.quantities(cand), r2, "hybr+newton", total_iter, idx + 1, attempts
            )

    best = min((a["residual"] for a in attempts if isinstance(a["residual"], float)), default=None)
    raise EquilibriumNotFound(
        f"no price vector satisfied the first-order conditions below {tol:g} "
        f"(best scaled residual {best if best is not None else 'n/a'}); "
        "the result would not be a Bertrand-Nash equilibrium",
        {"gate": tol, "best_residual": best, "attempts": attempts, "demand": demand.name},
    )
