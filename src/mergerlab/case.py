"""Case files (TOML) and the end-to-end analysis they drive."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import EquilibriumNotFound
from .market import Market
from .metrics import HMTest, hypothetical_monopolist_test
from .screens import RULESETS
from .simulate import MODEL_KINDS, ModelSpec, SimulationResult, simulate_merger
from .uncertainty import MonteCarloResult, Priors, run_monte_carlo
from .units import Margins, Ownership, Shares


class CaseError(ValueError):
    """The case file is inconsistent or incomplete."""


@dataclass(frozen=True)
class Case:
    """A merger case: market data, merger, demand systems to run and uncertainty ranges."""

    name: str
    market: Market
    parties: tuple[str, ...]
    post: Ownership
    models: tuple[ModelSpec, ...]
    description: str = ""
    currency: str = "USD"
    period: str = "per year"
    revenue_unit: str = ""
    source: str = ""
    disclaimer: str = ""
    efficiency: float = 0.0
    rulesets: tuple[str, ...] = RULESETS
    outside: str = "atomistic"
    priors: Priors = field(default_factory=Priors)
    draws: int = 500
    seed: int = 1
    hm_products: tuple[int, ...] | None = None
    ssnip: float = 0.05
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Analysis:
    """Everything computed for a case."""

    case: Case
    simulation: SimulationResult
    monte_carlo: MonteCarloResult | None
    hypothetical_monopolist: dict[str, HMTest]


def _require(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise CaseError(f"missing '{key}' in [{where}]")
    return d[key]


def _num(value: Any, name: str) -> float:
    """A TOML number as float; anything else is a :class:`CaseError` naming the key."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CaseError(f"'{name}' must be a number, got {value!r}")
    return float(value)


def _int(value: Any, name: str) -> int:
    """A TOML integer; floats such as 2.5 are refused rather than truncated."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaseError(f"'{name}' must be a whole number, got {value!r}")
    return value


def _range(d: dict[str, Any], key: str) -> tuple[float, float] | None:
    if key not in d:
        return None
    v = d[key]
    if not (isinstance(v, list) and len(v) == 2):
        raise CaseError(f"'{key}' must be a two-element list [low, high]")
    return (_num(v[0], f"{key}[0]"), _num(v[1], f"{key}[1]"))


def parse_case(data: dict[str, Any]) -> Case:
    """Build a :class:`Case` from parsed TOML, with explicit errors for inconsistent input."""
    meta = data.get("case", {})
    mk = _require(data, "market", "root")
    products = data.get("products")
    if not products:
        raise CaseError("at least one [[products]] entry is required")
    names = [str(_require(p, "name", "products")) for p in products]
    if len(set(names)) != len(names):
        raise CaseError("product names must be unique")
    owners = [str(p.get("owner", p["name"])) for p in products]
    prices = [_num(p.get("price", 1.0), f"price of {p['name']}") for p in products]
    if any(x <= 0 for x in prices):
        raise CaseError("prices must be positive")
    basis = str(mk.get("share_basis", "revenue"))
    if basis not in ("revenue", "quantity"):
        raise CaseError("share_basis must be 'revenue' or 'quantity'")
    shares_v = [_num(_require(p, "share", "products"), f"share of {p['name']}") for p in products]
    try:
        if "outside_share" in mk:
            shares = Shares.total(shares_v, basis, _num(mk["outside_share"], "outside_share"))
        elif abs(sum(shares_v) - 1.0) < 1e-8:
            shares = Shares.within(shares_v, basis)
        else:
            shares = Shares.total(shares_v, basis)
    except ValueError as exc:
        raise CaseError(f"shares: {exc}") from exc
    margins: list[float] = []
    for p in products:
        if "margin" in p and "margin_absolute" in p:
            raise CaseError(f"product {p['name']!r}: give either margin or margin_absolute")
        if "margin" in p:
            margins.append(_num(p["margin"], f"margin of {p['name']}"))
        elif "margin_absolute" in p:
            margins.append(
                _num(p["margin_absolute"], f"margin_absolute of {p['name']}")
                / _num(p.get("price", 1.0), f"price of {p['name']}")
            )
        else:
            margins.append(float("nan"))
    try:
        market = Market.build(
            prices,
            shares,
            Margins.lerner(margins),
            owners,
            names,
            revenue=_num(mk["revenue"], "revenue") if "revenue" in mk else None,
            outside_price=_num(mk["outside_price"], "outside_price")
            if "outside_price" in mk
            else None,
        )
    except ValueError as exc:
        raise CaseError(str(exc)) from exc

    merger = _require(data, "merger", "root")
    parties = tuple(str(x) for x in _require(merger, "parties", "merger"))
    try:
        post = market.ownership.merged(parties)
    except ValueError as exc:
        raise CaseError(f"merger: {exc}") from exc
    efficiency = _num(merger.get("efficiency", 0.0), "efficiency")
    if not 0 <= efficiency < 1:
        raise CaseError("efficiency must lie in [0, 1)")

    models_cfg = data.get("models", {})
    include = models_cfg.get("include", ["logit", "nested_logit", "ces", "linear", "pcaids"])
    if not isinstance(include, list) or not include:
        raise CaseError(f"[models] include must list at least one demand model of {MODEL_KINDS}")
    unknown = set(include) - set(MODEL_KINDS)
    if unknown:
        raise CaseError(f"unknown demand model(s) {sorted(unknown)}; choose from {MODEL_KINDS}")
    specs: list[ModelSpec] = []
    for kind in include:
        cfg = models_cfg.get(kind, {})
        if kind == "nested_logit":
            nests = [p.get("nest") for p in products]
            if any(n is None for n in nests):
                raise CaseError("nested_logit needs a 'nest' for every product")
            specs.append(
                ModelSpec(
                    kind,
                    nests=tuple(_int(n, "nest") for n in nests),
                    sigma=_num(cfg.get("sigma", 0.5), "sigma"),
                )
            )
        elif kind == "pcaids":
            product = cfg.get("product")
            if product is not None and product not in names:
                raise CaseError(f"[models.pcaids] product {product!r} is not in [[products]]")
            idx = names.index(product) if product is not None else 0
            specs.append(
                ModelSpec(
                    kind,
                    own_elasticity=_num(cfg["own_elasticity"], "own_elasticity")
                    if "own_elasticity" in cfg
                    else None,
                    market_elasticity=_num(cfg.get("market_elasticity", -1.0), "market_elasticity"),
                    known_index=idx,
                )
            )
        elif kind == "linear":
            fill = cfg.get("fill_margins")
            specs.append(ModelSpec(kind, fill_margins_from=str(fill) if fill is not None else None))
        else:
            specs.append(ModelSpec(kind))

    unc = data.get("uncertainty", {})
    try:
        priors = Priors(
            margin_halfwidth=_num(unc.get("margin_halfwidth", 0.10), "margin_halfwidth"),
            sigma_range=_range(unc, "sigma_range"),
            market_elasticity_range=_range(unc, "market_elasticity_range"),
            diversion_concentration=(
                _num(unc["diversion_concentration"], "diversion_concentration")
                if "diversion_concentration" in unc
                else None
            ),
        )
    except ValueError as exc:
        raise CaseError(f"uncertainty: {exc}") from exc

    screens_cfg = data.get("screens", {})
    rulesets = tuple(screens_cfg.get("rulesets", RULESETS))
    bad = set(rulesets) - set(RULESETS)
    if bad:
        raise CaseError(f"unknown rule set(s) {sorted(bad)}; choose from {RULESETS}")
    outside = str(screens_cfg.get("outside", "atomistic"))
    if outside not in ("atomistic", "exclude"):
        raise CaseError("screens.outside must be 'atomistic' or 'exclude'")

    hm = data.get("hypothetical_monopolist")
    hm_products = None
    ssnip = 0.05
    if hm is not None:
        sel = _require(hm, "products", "hypothetical_monopolist")
        missing = [s for s in sel if s not in names]
        if missing:
            raise CaseError(f"hypothetical_monopolist products not in [[products]]: {missing}")
        hm_products = tuple(names.index(s) for s in sel)
        ssnip = _num(hm.get("ssnip", 0.05), "ssnip")

    return Case(
        name=str(meta.get("name", "Merger case")),
        market=market,
        parties=parties,
        post=post,
        models=tuple(specs),
        description=str(meta.get("description", "")),
        currency=str(meta.get("currency", "USD")),
        period=str(meta.get("period", "per year")),
        revenue_unit=str(meta.get("revenue_unit", "")),
        source=str(meta.get("source", "")),
        disclaimer=str(meta.get("disclaimer", "")),
        efficiency=efficiency,
        rulesets=rulesets,
        outside=outside,
        priors=priors,
        draws=_int(unc.get("draws", 500), "draws"),
        seed=_int(unc.get("seed", 1), "seed"),
        hm_products=hm_products,
        ssnip=ssnip,
        notes=tuple(str(n) for n in data.get("assumptions", {}).get("notes", [])),
    )


def load_case(path: str | Path) -> Case:
    """Read a TOML case file."""
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise CaseError(f"{path}: invalid TOML ({exc})") from exc
    return parse_case(data)


def analyze(case: Case, draws: int | None = None, seed: int | None = None) -> Analysis:
    """Run screens, simulation, Monte Carlo bands and the hypothetical monopolist test."""
    sim = simulate_merger(
        case.market,
        case.post,
        case.models,
        efficiency=case.efficiency,
        rulesets=case.rulesets,
        outside=case.outside,
    )
    n_draws = case.draws if draws is None else draws
    mc = (
        run_monte_carlo(
            case.market,
            case.post,
            case.models,
            case.priors,
            draws=n_draws,
            seed=case.seed if seed is None else seed,
            efficiency=case.efficiency,
        )
        if n_draws > 0
        else None
    )
    hm: dict[str, HMTest] = {}
    if case.hm_products is not None:
        for o in sim.outcomes:
            if o.ok and o.calibration is not None:
                try:
                    hm[o.spec.kind] = hypothetical_monopolist_test(
                        o.calibration.demand,
                        case.market.prices,
                        o.calibration.costs,
                        np.asarray(case.hm_products),
                        ssnip=case.ssnip,
                    )
                except EquilibriumNotFound:
                    continue
    return Analysis(case, sim, mc, hm)
