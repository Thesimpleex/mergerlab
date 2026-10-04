"""Command line interface: ``mergerlab simulate case.toml`` and ``mergerlab screen``."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import __version__
from .case import Analysis, CaseError, analyze, load_case
from .formatting import money, pct, points
from .report import write_report
from .screens import RULESETS, concentration, screen
from .summary import Summary, summarize
from .supply import DEFAULT_GATE
from .units import Ownership, Shares

_MIN_PIPED_WIDTH = 120

_STATUS_STYLE = {
    "presumed_harmful": ("red", "presumption"),
    "possible_dominance": ("red", "may indicate dominance"),
    "warrants_scrutiny": ("yellow", "scrutiny"),
    "unlikely_concerns": ("green", "unlikely"),
    "no_presumption": ("default", "no presumption"),
}


def _console() -> Console:
    """A console that does not squeeze tables to 80 columns when output is piped."""
    console = Console(highlight=False)
    if not console.is_terminal and "COLUMNS" not in os.environ:
        return Console(highlight=False, width=max(console.width, _MIN_PIPED_WIDTH))
    return console


def _error(message: str) -> int:
    Console(stderr=True, highlight=False).print(f"mergerlab: {message}", style="red")
    return 2


def _band(b: tuple[float, float, float]) -> str:
    return f"{100 * b[1]:.1f}% [{100 * b[0]:.1f}, {100 * b[2]:.1f}]"


def _screens_table(console: Console, an_screens, conc) -> None:
    console.print(
        Text.assemble(
            ("Concentration screens", "bold"),
            (
                f"   HHI {points(conc.hhi_pre)} -> {points(conc.hhi_post)} "
                f"(+{points(conc.delta_hhi)}), merged share {conc.merged_share:.1%}",
                "dim",
            ),
        )
    )
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("Rule set", no_wrap=True)
    table.add_column("Reading", no_wrap=True)
    table.add_column("Reference", no_wrap=True)
    for r in an_screens:
        colour, word = _STATUS_STYLE[r.status]
        table.add_row(r.name, Text(word, style=colour), Text(r.reference, style="dim"))
    console.print(table)


def _market_table(console: Console, an: Analysis) -> None:
    case = an.case
    m = case.market
    parties = set(an.simulation.parties)
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("Product")
    table.add_column("Owner")
    table.add_column("Price", justify="right")
    table.add_column(f"{m.shares.basis.value.capitalize()} share", justify="right")
    table.add_column("Lerner margin", justify="right")
    lerner = m.lerner
    owners = m.ownership.owners or m.labels
    for i, name in enumerate(m.labels):
        mark = " *" if i in parties else ""
        table.add_row(
            name + mark,
            owners[i],
            f"{m.prices[i]:.2f}",
            f"{m.shares.values[i]:.1%}",
            "-" if lerner[i] != lerner[i] else f"{lerner[i]:.1%}",
        )
    if m.shares.has_outside:
        table.add_row("Outside good", "", "", f"{m.shares.outside:.1%}", "")
    console.print(Text.assemble(("Market", "bold"), ("   * merging parties", "dim")))
    console.print(table)


def _effects_table(console: Console, an: Analysis, s: Summary) -> None:
    case = an.case
    unit = f"{case.currency} {case.revenue_unit}".strip()
    console.print(
        Text.assemble(
            ("Price effects by demand form", "bold"),
            (
                f"   merging firms' products, same inputs; money in {unit} {case.period}"
                if case.market.revenue is not None
                else "   merging firms' products, same inputs; money per unit of inside sales",
                "dim",
            ),
        )
    )
    table = _effects_rows(an, s, with_parties=True)
    if console.measure(table, options=console.options.update_width(10_000)).maximum > console.width:
        table = _effects_rows(an, s, with_parties=False)
    console.print(table)
    for r in s.rows:
        if r.status != "ok":
            console.print(
                Text.assemble(
                    (f"{r.label}: ", "red"),
                    (f"{r.status.replace('_', ' ')}. {r.message or ''}", "dim"),
                )
            )
        for note in r.notes:
            console.print(Text(f"{r.label}: {note}", style="dim"))


def _effects_rows(an: Analysis, s: Summary, with_parties: bool) -> Table:
    case = an.case
    sim = an.simulation
    party_names = [case.market.labels[i] for i in sim.parties] if with_parties else []
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("Demand form", no_wrap=True)
    for n in party_names:
        table.add_column(n, justify="right")
    table.add_column("Average", justify="right")
    if s.mc_draws:
        table.add_column("MC median [5%, 95%]", justify="right", no_wrap=True)
    table.add_column("CMCR", justify="right")
    table.add_column("Synergies", justify="right")
    table.add_column("Consumer harm", justify="right", no_wrap=True)
    row: list[Text | str]
    for r in s.rows:
        if r.status != "ok":
            row = [r.label, *[""] * len(party_names), Text(r.status.replace("_", " "), style="red")]
            row += [""] * (len(table.columns) - len(row))
            table.add_row(*row)
            continue
        row = [r.label]
        if with_parties:
            row += [pct(x) for x in r.party_changes]
        row.append(Text(pct(r.average_change), style="bold"))
        if s.mc_draws:
            row.append("-" if r.band is None else _band(r.band))
        row += [
            pct(r.cmcr_average, signed=False),
            money(r.synergy_total, case.currency, case.revenue_unit),
            money(r.consumer_harm, case.currency, case.revenue_unit),
        ]
        table.add_row(*row)
    if s.ensemble_band is not None:
        row = ["Ensemble (equal weight)" if with_parties else "Ensemble"]
        row += [""] * (len(party_names) + 1)
        lo, mid, hi = s.ensemble_band
        row.append(_band((lo, mid, hi)))
        row += [
            pct(s.ensemble_cmcr_band[1], signed=False) if s.ensemble_cmcr_band else "-",
            "",
            "",
        ]
        table.add_row(*[Text(x, style="dim") if isinstance(x, str) else x for x in row])
    return table


def _footer(console: Console, an: Analysis, s: Summary) -> None:
    if s.mc_draws:
        share = s.failure_share or 0.0
        colour = "green" if share == 0 else "yellow"
        console.print(
            Text.assemble(
                ("Monte Carlo  ", "bold"),
                (f"{s.mc_draws:,} draws x {len(s.rows)} demand forms; ", "dim"),
                (f"{share:.1%} not calibrated or failed the solver gate", colour),
                (f"; margins +/-{100 * an.case.priors.margin_halfwidth:.0f} pp", "dim"),
            )
        )
    hm_rows = [r for r in s.rows if r.hm is not None]
    if hm_rows and an.case.hm_products is not None:
        names = ", ".join(an.case.market.labels[i] for i in an.case.hm_products)
        passes = sum(1 for r in hm_rows if r.hm and r.hm.passes)
        console.print(
            Text.assemble(
                ("Hypothetical monopolist  ", "bold"),
                (
                    f"{{{names}}}: {passes} of {len(hm_rows)} demand forms pass the "
                    f"{pct(an.case.ssnip, 0, signed=False)} SSNIP test",
                    "dim",
                ),
            )
        )
    if an.case.disclaimer:
        console.print(Text(an.case.disclaimer, style="dim"))


def render_analysis(console: Console, an: Analysis, s: Summary | None = None) -> None:
    """Print the terminal summary of an analysis."""
    s = s or summarize(an)
    case = an.case
    console.print(
        Text.assemble((f"mergerlab {__version__}", "bold"), ("  ", ""), (case.name, "bold"))
    )
    meta = f"{case.market.shares.basis.value} shares"
    if s.mc_draws:
        meta += f" · {s.mc_draws:,} Monte Carlo draws · seed {case.seed}"
    meta += f" · solver gate {DEFAULT_GATE:g}"
    console.print(Text(meta, style="dim"))
    console.print()
    _market_table(console, an)
    console.print()
    _screens_table(console, an.simulation.screens, an.simulation.concentration)
    console.print()
    _effects_table(console, an, s)
    console.print()
    _footer(console, an, s)


def _cmd_simulate(args: argparse.Namespace) -> int:
    console = _console()
    if args.draws is not None and args.draws < 0:
        return _error("--draws must be 0 (no Monte Carlo bands) or a positive number")
    try:
        case = load_case(args.case)
        an = analyze(case, draws=args.draws, seed=args.seed)
    except (CaseError, OSError) as exc:
        return _error(str(exc))
    render_analysis(console, an)
    if args.report:
        path = write_report(an, args.report)
        console.print(Text(f"Memo written to {path}", style="dim"))
    if not any(o.ok for o in an.simulation.outcomes):
        Console(stderr=True, highlight=False).print(
            "mergerlab: no demand form could be calibrated and solved for this case; "
            "see the reasons above",
            style="red",
        )
        return 1
    return 0


def _screen_inputs(shares_pct: Sequence[float], merge: Sequence[int]) -> list[float]:
    """Validate the percent shares and merging positions of ``mergerlab screen``."""
    if len(shares_pct) < 2:
        raise ValueError("--shares needs at least two firms")
    if any(x != x or x <= 0 for x in shares_pct):
        raise ValueError("--shares must be positive numbers")
    total = sum(shares_pct)
    if total < 10:
        hint = (
            f"; did you mean {' '.join(f'{100 * x:g}' for x in shares_pct)}?"
            if all(x <= 1 for x in shares_pct)
            else ""
        )
        raise ValueError(
            f"--shares are percentages but these sum to {total:g}, which looks like fractions "
            f"or a unit mix-up{hint}"
        )
    if total > 100 + 1e-9:
        raise ValueError(f"--shares are percentages and must sum to at most 100 (got {total:g})")
    bad = [i for i in merge if not 1 <= i <= len(shares_pct)]
    if bad:
        raise ValueError(
            f"--merge positions are 1-based and must lie between 1 and {len(shares_pct)} "
            f"(got {bad})"
        )
    if len(set(merge)) < 2:
        raise ValueError("--merge needs at least two different firms")
    return [x / 100.0 for x in shares_pct]


def _cmd_screen(args: argparse.Namespace) -> int:
    console = _console()
    try:
        shares = _screen_inputs(args.shares, args.merge)
        inferred = 1.0 - sum(shares)
        s = Shares.total(shares, "revenue") if inferred > 1e-9 else Shares.within(shares, "revenue")
        owners = [f"F{i + 1}" for i in range(len(shares))]
        pre = Ownership.from_owners(owners)
        post = pre.merged([owners[i - 1] for i in args.merge])
        conc = concentration(s, pre, post)
    except ValueError as exc:
        return _error(str(exc))
    if inferred > 1e-9:
        console.print(
            Text(
                f"Shares sum to {100 * (1 - inferred):g}%; the remaining {100 * inferred:.4g}% "
                "is treated as an atomistic outside good.",
                style="dim",
            )
        )
    _screens_table(console, screen(conc, tuple(args.rules)), conc)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mergerlab",
        description="Calibrated merger simulation and antitrust screening.",
    )
    parser.add_argument("--version", action="version", version=f"mergerlab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="simulate a merger described in a TOML case file")
    sim.add_argument("case", type=Path, help="path to the TOML case file")
    sim.add_argument("--report", type=Path, help="write a self-contained HTML merger memo here")
    sim.add_argument("--draws", type=int, help="Monte Carlo draws (0 disables bands)")
    sim.add_argument("--seed", type=int, help="random seed for the Monte Carlo draws")
    sim.set_defaults(func=_cmd_simulate)

    scr = sub.add_parser("screen", help="apply the guideline concentration screens to shares")
    scr.add_argument("--shares", type=float, nargs="+", required=True, help="shares in percent")
    scr.add_argument(
        "--merge", type=int, nargs="+", required=True, help="1-based positions of the merging firms"
    )
    scr.add_argument("--rules", nargs="+", default=list(RULESETS), choices=RULESETS)
    scr.set_defaults(func=_cmd_screen)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
