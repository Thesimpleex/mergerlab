"""Self-contained HTML merger memo (inline CSS and SVG, no external resources).

The layout follows the exhibit format of strategy consultancies: a row of key figures, findings
as titles, hairline rules instead of boxes, black and grey type in Helvetica Neue and a single
red accent for the one message of each chart.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from pathlib import Path

from . import __version__
from .case import Analysis
from .formatting import money, pct, points
from .summary import ModelRow, Summary, summarize

_CSS = """
:root { --ink:#111111; --ink2:#3d3d3d; --muted:#6b6b6b; --grid:#e8e8e8; --rule:#d9d9d9;
  --accent:#c8102e; --soft:#f3d0d6; }
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin:0; background:#fff; color:var(--ink); -webkit-font-smoothing:antialiased;
  font:15px/1.55 "Helvetica Neue", Helvetica, Arial, sans-serif; }
main { max-width:980px; margin:0 auto; padding:72px 56px 88px; }
b { font-weight:500; }
.mark { width:32px; height:3px; background:var(--accent); }
.eyebrow, .kicker { font-size:12px; font-weight:500; letter-spacing:.09em; text-transform:uppercase;
  color:var(--muted); }
.eyebrow { margin-top:18px; }
h1 { font-size:46px; line-height:1.06; font-weight:300; letter-spacing:-.035em; margin:18px 0;
  max-width:840px; }
.lede { font-size:19px; line-height:1.45; color:var(--ink2); margin:0 0 14px; max-width:760px;
  text-wrap:balance; }
.meta { font-size:13px; color:var(--muted); margin:0; }
section { margin-top:64px; }
h2 { font-size:26px; line-height:1.2; font-weight:400; letter-spacing:-.018em; margin:8px 0 10px;
  max-width:820px; text-wrap:balance; }
h3 { font-size:12px; font-weight:500; letter-spacing:.09em; text-transform:uppercase;
  color:var(--muted); margin:34px 0 10px; }
p { margin:0 0 12px; max-width:760px; }
.sub { font-size:14px; color:var(--ink2); margin:0 0 18px; }
.keyfigs { display:grid; margin:44px 0 0; border-top:1px solid var(--ink); }
.kf { padding:24px 24px 6px 0; }
.kf + .kf { padding-left:24px; border-left:1px solid var(--rule); }
.kf .v { font-size:38px; font-weight:300; letter-spacing:-.035em; line-height:1.05;
  font-variant-numeric:tabular-nums; white-space:nowrap; }
.kf .v.hot { color:var(--accent); }
.kf .l { font-size:13.5px; line-height:1.45; color:var(--ink2); margin-top:12px; max-width:230px; }
.findings { margin:20px 0 0; border-top:1px solid var(--ink); }
.findings .row { display:grid; grid-template-columns:190px 1fr; gap:24px; padding:15px 0;
  border-bottom:1px solid var(--grid); }
.findings dt { font-size:12px; font-weight:500; letter-spacing:.09em; text-transform:uppercase;
  color:var(--muted); padding-top:3px; }
.findings dd { margin:0; }
table { width:100%; border-collapse:collapse; font-size:14px; font-variant-numeric:tabular-nums; }
th { text-align:left; font-size:11.5px; font-weight:500; letter-spacing:.06em;
  text-transform:uppercase; color:var(--muted); padding:0 0 10px 18px; vertical-align:bottom;
  border-bottom:1px solid var(--ink); }
td { padding:12px 0 12px 18px; border-bottom:1px solid var(--grid); vertical-align:top; }
th:first-child, td:first-child { padding-left:0; }
th.n, td.n { text-align:right; white-space:nowrap; }
table.fit th:first-child, table.fit td:first-child { white-space:nowrap; }
tr.total td { font-weight:500; border-top:1px solid var(--ink); border-bottom:0; }
.tag { display:inline-flex; align-items:center; gap:8px; font-size:13px; font-weight:500;
  white-space:nowrap; }
.tag::before { content:""; width:8px; height:8px; background:var(--ink); }
.tag.red { color:var(--accent); } .tag.red::before { background:var(--accent); }
.tag.grey { color:var(--muted); font-weight:400; }
.tag.grey::before { background:none; border:1.5px solid var(--muted); }
.small { font-size:13px; color:var(--ink2); } .dim { color:var(--muted); }
.exh { margin-top:64px; }
.exh .chart { display:block; margin:6px 0 0; } .chart text { font-family:inherit; }
.chart .halo { paint-order:stroke; stroke:#fff; stroke-width:5px; stroke-linejoin:round; }
.src { border-top:1px solid var(--rule); margin-top:14px; padding-top:10px; font-size:12px;
  color:var(--muted); }
.note { font-size:12.5px; color:var(--muted); margin-top:10px; max-width:none; }
ul.plain { margin:0 0 12px; padding-left:20px; max-width:760px; } ul.plain li { margin:0 0 6px; }
footer { margin-top:72px; padding-top:16px; border-top:1px solid var(--rule); font-size:12.5px;
  color:var(--muted); }
footer p { max-width:none; margin:0 0 4px; }
@media print { main { padding:24px 0; } section, .exh { break-inside:avoid; }
  h2 { break-after:avoid; } }
@media (max-width: 760px) { main { padding:40px 22px 56px; } h1 { font-size:34px; }
  .keyfigs { grid-template-columns:1fr 1fr !important; }
  .kf:nth-child(odd) { padding-left:0; border-left:0; }
  .findings .row { grid-template-columns:1fr; gap:4px; } }
"""

_INK, _INK2, _MUTED, _GRID, _RULE = "#111111", "#3d3d3d", "#6b6b6b", "#e8e8e8", "#d9d9d9"
_ACCENT, _SOFT = "#c8102e", "#f3d0d6"

_TAG = {
    "presumed_harmful": ("red", "Presumption triggered"),
    "possible_dominance": ("red", "May indicate dominance"),
    "warrants_scrutiny": ("ink", "Warrants scrutiny"),
    "unlikely_concerns": ("grey", "Concerns unlikely"),
    "no_presumption": ("grey", "No presumption"),
}


def _range_text(r: tuple[float, float] | None, fmt) -> str:
    if r is None:
        return "-"
    lo, hi = r
    return fmt(lo) if abs(hi - lo) < 5e-4 * max(1.0, abs(hi)) else f"{fmt(lo)} to {fmt(hi)}"


def _short_range(lo: float, hi: float, fmt) -> str:
    """Key-figure range such as ``3.9-15.2%`` or ``$20.1-96.3m`` with an en dash."""
    first, last = fmt(lo), fmt(hi)
    if first == last:
        return first
    i = 0
    while i < min(len(first), len(last)) and not first[i].isdigit() and first[i] == last[i]:
        i += 1
    j = 0
    while (
        j < min(len(first), len(last)) - i
        and not first[-1 - j].isdigit()
        and (first[-1 - j] == last[-1 - j])
    ):
        j += 1
    return first[: len(first) - j] + "\u2013" + last[i:]


def _avg(r: ModelRow) -> float:
    assert r.average_change is not None
    return r.average_change


def _ticks(xmax: float) -> list[float]:
    """Axis ticks from zero up to the first tick at or above ``xmax``."""
    step = next(t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if xmax / t <= 8)
    n = int(xmax / step - 1e-9) + 1
    return [k * step for k in range(n + 1)]


_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def _count(n: int) -> str:
    """Small counts as words in running text."""
    return _WORDS.get(n, str(n))


def _join(names: Sequence[str]) -> str:
    return " and ".join(names) if len(names) <= 2 else ", ".join(names[:-1]) + " and " + names[-1]


def _forms_chart(
    rows: Sequence[tuple[str, float, tuple[float, float, float] | None]],
    ensemble: tuple[float, float, float] | None,
    callout_note: str,
    aria: str,
) -> str:
    """Dot-and-band exhibit: point estimate and 5th to 95th percentile band per demand form.

    The form with the largest point estimate is drawn in red and carries the callout; the
    ensemble of all forms sits below a hairline.
    """
    if not rows:
        return ""
    values = [v for _, v, _ in rows]
    highs = [b[2] for _, _, b in rows if b] + values + ([ensemble[2]] if ensemble else [])
    ticks = _ticks(max(max(highs) * 1.14, 0.02))
    left, right, top, rh, gap = 164, 28, 14, 50, 22
    width = 880
    n = len(rows)
    plot_h = rh * n + (gap + rh if ensemble else 0)
    bottom = top + plot_h
    height = bottom + 34

    def x(v: float) -> float:
        return left + (width - left - right) * v / ticks[-1]

    def y(i: int) -> float:
        return top + rh * i + rh / 2

    hot = max(range(n), key=lambda i: rows[i][1])
    out = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="{escape(aria)}">'
    ]
    for tk in ticks:
        out.append(
            f'<line x1="{x(tk):.1f}" x2="{x(tk):.1f}" y1="{top}" y2="{bottom}" '
            f'stroke="{_GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<line x1="{x(tk):.1f}" x2="{x(tk):.1f}" y1="{bottom}" y2="{bottom + 5}" '
            f'stroke="{_INK}" stroke-width="1"/>'
            f'<text x="{x(tk):.1f}" y="{bottom + 22}" font-size="12" fill="{_MUTED}" '
            f'text-anchor="middle">{100 * tk:.0f}%</text>'
        )
    out.append(
        f'<line x1="{left}" x2="{width - right}" y1="{bottom}" y2="{bottom}" '
        f'stroke="{_INK}" stroke-width="1"/>'
    )
    boxes: list[tuple[float, float, float, float]] = []
    for i, (label, value, band) in enumerate(rows):
        yy = y(i)
        red = i == hot
        out.append(
            f'<text x="{left - 16}" y="{yy + 4.5:.1f}" font-size="13.5" '
            f'fill="{_ACCENT if red else _INK2}" font-weight="{500 if red else 400}" '
            f'text-anchor="end">{escape(label)}</text>'
        )
        lo_x = hi_x = x(value)
        if band:
            lo_x, hi_x = x(band[0]), x(band[2])
            out.append(
                f'<line x1="{lo_x:.1f}" x2="{hi_x:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" '
                f'stroke="{_SOFT if red else _RULE}" stroke-width="10" stroke-linecap="round"/>'
            )
        boxes.append((min(lo_x, x(value)) - 14, yy - 22, max(hi_x, x(value)) + 14, yy + 14))
        if not red:
            out.append(
                f'<circle cx="{x(value):.1f}" cy="{yy:.1f}" r="6" fill="{_INK2}" '
                f'stroke="#fff" stroke-width="1.5"/>'
                f'<text x="{x(value):.1f}" y="{yy - 13:.1f}" font-size="12.5" fill="{_INK}" '
                f'text-anchor="middle">{100 * value:.1f}%</text>'
            )
    if ensemble:
        yy = y(n) + gap
        out.append(
            f'<line x1="{left - 150}" x2="{width - right}" y1="{top + rh * n + gap / 2:.1f}" '
            f'y2="{top + rh * n + gap / 2:.1f}" stroke="{_RULE}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{left - 16}" y="{yy + 4.5:.1f}" font-size="13.5" font-weight="500" '
            f'fill="{_INK}" text-anchor="end">Ensemble</text>'
            f'<line x1="{x(ensemble[0]):.1f}" x2="{x(ensemble[2]):.1f}" y1="{yy:.1f}" '
            f'y2="{yy:.1f}" stroke="{_INK2}" stroke-width="10" stroke-linecap="round"/>'
            f'<line x1="{x(ensemble[1]):.1f}" x2="{x(ensemble[1]):.1f}" y1="{yy - 6:.1f}" '
            f'y2="{yy + 6:.1f}" stroke="#fff" stroke-width="2"/>'
            f'<text x="{x(ensemble[2]) + 14:.1f}" y="{yy + 4.5:.1f}" font-size="12.5" '
            f'fill="{_INK}">median {100 * ensemble[1]:.1f}%</text>'
        )
    others = [b for i, b in enumerate(boxes) if i != hot]
    if ensemble:
        yy = y(n) + gap
        others.append((x(ensemble[0]) - 14, yy - 22, x(ensemble[2]) + 120, yy + 14))
    out.append(
        _callout(x(rows[hot][1]), y(hot), f"{100 * rows[hot][1]:.1f}%", callout_note, others)
    )
    out.append("</svg>")
    return "".join(out)


def _callout(
    cx: float,
    cy: float,
    value: str,
    note: str,
    boxes: list[tuple[float, float, float, float]],
) -> str:
    """Red key number tied to a data point by a hairline, placed where no other mark is."""
    lines = note.split("\n")
    w = max(len(s) for s in lines) * 6.3 + 6
    h = 30 + 17 * len(lines)
    slots = [(cx + 38, cy + 16), (cx - 38 - w, cy + 16)]
    slot = next(
        (
            s
            for s in slots
            if all(s[0] + w < b[0] or s[0] > b[2] or s[1] + h < b[1] or s[1] > b[3] for b in boxes)
            and s[0] >= 164
            and s[0] + w <= 860
        ),
        None,
    )
    if slot is None:
        return (
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="{_ACCENT}" stroke="#fff" '
            f'stroke-width="1.5"/><text x="{cx:.1f}" y="{cy - 13:.1f}" font-size="12.5" '
            f'font-weight="500" fill="{_ACCENT}" text-anchor="middle">{escape(value)}</text>'
        )
    sx, sy = slot
    anchor_x = sx - 6 if sx > cx else sx + w + 6
    parts = [
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{anchor_x:.1f}" y2="{sy + 4:.1f}" '
        f'stroke="{_ACCENT}" stroke-width="1"/>',
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="{_ACCENT}" stroke="#fff" '
        f'stroke-width="1.5"/>',
        f'<text class="halo" x="{sx:.1f}" y="{sy + 26:.1f}" font-size="30" font-weight="300" '
        f'fill="{_ACCENT}">{escape(value)}</text>',
    ]
    for k, line in enumerate(lines):
        parts.append(
            f'<text class="halo" x="{sx:.1f}" y="{sy + 48 + 17 * k:.1f}" font-size="12.5" '
            f'fill="{_INK2}">{escape(line)}</text>'
        )
    return "".join(parts)


def _exhibit(number: int, topic: str, title: str, subtitle: str, chart: str, source: str) -> str:
    return (
        f'<section class="exh" id="exhibit-{number}"><div class="kicker">Exhibit {number} '
        f"&middot; {escape(topic)}</div><h2>{escape(title)}</h2>"
        f'<p class="sub">{escape(subtitle)}</p>{chart}<div class="src">{escape(source)}</div>'
        "</section>"
    )


def _key_figures(an: Analysis, s: Summary) -> str:
    case = an.case
    sim = an.simulation
    c = sim.concentration
    hits = sum(1 for r in sim.screens if r.status == "presumed_harmful")
    cards = [
        (
            f"+{points(c.delta_hhi)}",
            f"change in HHI, to {points(c.hhi_post)}; {hits} of {len(sim.screens)} rule sets "
            "raise a presumption",
            False,
        )
    ]
    n_ok = len([r for r in s.rows if r.status == "ok"])
    if s.point_range:
        cards.append(
            (
                _short_range(*s.point_range, lambda v: pct(v, signed=False)),
                f"predicted price increase of the merging firms across {_count(n_ok)} demand "
                "forms (point estimates)",
                True,
            )
        )
    if s.cmcr_range:
        scale = ""
        if s.synergy_range and case.market.revenue is not None:
            money_range = _range_text(
                s.synergy_range, lambda v: money(v, case.currency, case.revenue_unit)
            )
            scale = f", or {money_range} {case.period}"
        cards.append(
            (
                _short_range(*s.cmcr_range, lambda v: pct(v, signed=False)),
                f"marginal-cost savings that would keep prices flat{scale}",
                False,
            )
        )
    if s.harm_range and case.market.revenue is not None:
        cards.append(
            (
                _short_range(*s.harm_range, lambda v: money(v, case.currency, case.revenue_unit)),
                f"consumer harm (compensating variation) {case.period}, before efficiencies",
                False,
            )
        )
    html = "".join(
        f'<div class="kf"><div class="v{" hot" if hot else ""}">{escape(v)}</div>'
        f'<div class="l">{escape(label)}</div></div>'
        for v, label, hot in cards
    )
    cols = f"grid-template-columns:repeat({len(cards)},1fr)"
    return f'<div class="keyfigs" style="{cols}">{html}</div>'


def _screens_section(an: Analysis) -> str:
    sim = an.simulation
    c = sim.concentration
    rows = []
    for r in sim.screens:
        cls, word = _TAG[r.status]
        rows.append(
            f"<tr><td>{escape(r.name)}</td><td><span class='tag {cls}'>{word}</span></td>"
            f"<td>{escape(r.summary)}<div class='dim small'>{escape(r.citation)}</div></td></tr>"
        )
    head = (
        f"HHI rises from {points(c.hhi_pre)} to {points(c.hhi_post)} and the merged firm holds "
        f"{c.merged_share:.1%}"
    )
    return (
        '<section id="screens"><div class="kicker">Structural screens</div>'
        f"<h2>{escape(head)}</h2>"
        "<table><thead><tr><th>Rule set</th><th>Reading</th><th>Basis</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<p class='note'>Screens are first indicators, not findings. The 2023 U.S. presumption "
        "is rebuttable and not meeting it is no safe harbour; EU paragraph 21 states that its "
        "levels give rise to no presumption in either direction.</p></section>"
    )


def _market_section(an: Analysis) -> str:
    case = an.case
    m = case.market
    parties = set(an.simulation.parties)
    owners = m.ownership.owners or m.labels
    lerner = m.lerner
    body = []
    for i, name in enumerate(m.labels):
        body.append(
            f"<tr><td>{escape(name)}{' <span class=dim>&middot; merging party</span>' if i in parties else ''}</td>"
            f"<td>{escape(owners[i])}</td><td class='n'>{m.prices[i]:.2f}</td>"
            f"<td class='n'>{m.shares.values[i]:.1%}</td>"
            f"<td class='n'>{'-' if lerner[i] != lerner[i] else f'{lerner[i]:.1%}'}</td></tr>"
        )
    if m.shares.has_outside:
        body.append(
            f"<tr><td class='dim'>Outside good</td><td></td><td class='n dim'>"
            f"{'' if m.outside_price is None else f'{m.outside_price:.2f}'}</td>"
            f"<td class='n dim'>{m.shares.outside:.1%}</td><td></td></tr>"
        )
    scale = (
        f"Total sales of the products shown: {money(m.revenue, case.currency, case.revenue_unit)} "
        f"{escape(case.period)}."
        if m.revenue is not None
        else "No absolute scale given: money amounts are per unit of inside sales."
    )
    src = f"<p class='note'>{escape(case.source)}</p>" if case.source else ""
    return (
        '<section id="market"><div class="kicker">Transaction and market</div>'
        "<h2>Products, prices, shares and margins as given</h2>"
        "<table><thead><tr><th>Product</th><th>Owner</th>"
        f"<th class='n'>Price</th><th class='n'>{m.shares.basis.value.capitalize()} share</th>"
        "<th class='n'>Lerner margin</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
        f"<p class='note'>{scale}</p>{src}</section>"
    )


def _mc_source(an: Analysis, s: Summary, what: str) -> str:
    if s.mc_draws:
        key = (
            f"Dot: point estimate; bar: 5th to 95th percentile of {s.mc_draws:,} Monte Carlo "
            "draws per form"
        )
        key += "; ensemble: all forms, equal weight. " if s.ensemble_band else ". "
    else:
        key = "Dot: point estimate. "
    return f"{key}Source: mergerlab {__version__}, {an.case.name}. {what}"


def _price_exhibit(an: Analysis, s: Summary) -> str:
    case = an.case
    sim = an.simulation
    ok = [r for r in s.rows if r.status == "ok" and r.average_change is not None]
    if not ok or not s.point_range:
        return ""
    ranked = sorted(ok, key=lambda r: -_avg(r))
    names = " and ".join(case.market.labels[i] for i in sim.parties)
    top = ranked[0]
    rest = ranked[1:]
    note = ""
    if rest:
        note = (
            f"{top.label}; the other {_count(len(rest))} predict\n"
            f"{_range_text((min(_avg(r) for r in rest), max(_avg(r) for r in rest)), lambda v: pct(v, signed=False))}"
        )
    chart = _forms_chart(
        [(r.label, _avg(r), r.band) for r in ranked],
        s.ensemble_band,
        note or top.label,
        "Predicted price increase by demand form with Monte Carlo bands",
    )
    title = (
        f"Predicted price increases range from {_range_text(s.point_range, lambda v: pct(v, signed=False))} "
        f"across {_count(len(ok))} demand forms"
        if len(ok) > 1
        else f"Predicted price increase of {pct(_avg(ok[0]), signed=False)}"
    )
    return _exhibit(
        1,
        "Price effects",
        title,
        f"Average price increase of {names} by demand form, %",
        chart,
        _mc_source(an, s, "The spread between forms is model risk."),
    )


def _effects_section(an: Analysis, s: Summary) -> str:
    case = an.case
    sim = an.simulation
    party_names = [case.market.labels[i] for i in sim.parties]
    if not any(r.status == "ok" for r in s.rows):
        failed = "".join(
            f"<tr><td>{escape(r.label)}</td><td class='dim'>{escape(r.status.replace('_', ' '))}: "
            f"{escape(r.message or '')}</td></tr>"
            for r in s.rows
        )
        return (
            '<section id="effects"><div class="kicker">Price effects</div>'
            "<h2>No demand form could be calibrated and solved</h2><p>There is no price effect, "
            "synergy or surplus result to report for these inputs. Reasons by demand form:</p>"
            "<table><thead><tr><th>Demand form</th><th>Reason</th></tr></thead>"
            f"<tbody>{failed}</tbody></table></section>"
        )
    head = "".join(f"<th class='n'>{escape(n)}</th>" for n in party_names)
    body = []
    for r in s.rows:
        if r.status != "ok":
            body.append(
                f"<tr><td>{escape(r.label)}</td><td colspan='{len(party_names) + 3}' "
                f"class='dim'>{escape(r.status.replace('_', ' '))}: {escape(r.message or '')}</td></tr>"
            )
            continue
        cells = "".join(f"<td class='n'>{pct(x)}</td>" for x in r.party_changes)
        band = (
            "-"
            if r.band is None
            else f"{100 * r.band[0]:.1f} to {100 * r.band[2]:.1f}% (median {100 * r.band[1]:.1f}%)"
        )
        body.append(
            f"<tr><td>{escape(r.label)}</td>{cells}<td class='n'><b>{pct(r.average_change)}</b></td>"
            f"<td class='n'>{pct(r.first_order_average)}</td><td class='n'>{band}</td></tr>"
        )
    if s.ensemble_band:
        b = s.ensemble_band
        body.append(
            f"<tr class='total'><td>Ensemble (equal weight per form)</td>"
            f"<td colspan='{len(party_names) + 2}'></td>"
            f"<td class='n'>{100 * b[0]:.1f} to {100 * b[2]:.1f}% (median {100 * b[1]:.1f}%)</td></tr>"
        )
    mc_note = ""
    if s.mc_draws:
        share = s.failure_share or 0.0
        mc_note = (
            f"<p class='note'>Bands: {s.mc_draws:,} Monte Carlo draws per demand form over margins "
            f"(+/-{100 * case.priors.margin_halfwidth:.0f} percentage points), the nesting "
            "parameter, the PCAIDS market elasticity and linear-demand diversion ratios as stated "
            f"under assumptions. {share:.1%} of draws could not be calibrated or failed the solver "
            "gate (first-order-condition residual below 1e-10); they are counted and excluded, so "
            "the bands are conditional on inputs that can be calibrated. Counts by form are listed "
            "under diagnostics.</p>"
        )
    return (
        '<section id="effects" style="margin-top:40px"><h3>Price effects by demand form</h3>'
        '<table class="fit"><thead><tr><th>Demand form</th>'
        + head
        + "<th class='n'>Average</th><th class='n'>First-order approx.</th>"
        "<th class='n'>Monte Carlo 5-95%</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>" + mc_note + "</section>"
    )


def _synergy_exhibit(an: Analysis, s: Summary) -> str:
    case = an.case
    sim = an.simulation
    ok = [r for r in s.rows if r.status == "ok" and r.cmcr_average is not None]
    if not ok or not s.cmcr_range:
        return ""
    ranked = sorted(ok, key=lambda r: -_avg(r))
    top = max(ranked, key=lambda r: r.cmcr_average or 0.0)
    low = min(ranked, key=lambda r: r.cmcr_average or 0.0)
    names = " and ".join(case.market.labels[i] for i in sim.parties)
    cmcr_money = (
        money(top.synergy_total, case.currency, case.revenue_unit)
        if case.market.revenue is not None
        else None
    )
    note = (
        "of marginal cost"
        + (f", or {cmcr_money} {case.period}" if cmcr_money else "")
        + f"\n{low.label} needs {pct(low.cmcr_average, signed=False)}"
    )
    chart = _forms_chart(
        [(r.label, r.cmcr_average or 0.0, r.cmcr_band) for r in ranked],
        s.ensemble_cmcr_band,
        note if len(ranked) > 1 else "of marginal cost",
        "Compensating marginal cost reduction by demand form with Monte Carlo bands",
    )
    title = (
        f"Keeping prices flat takes marginal-cost savings of "
        f"{_range_text(s.cmcr_range, lambda v: pct(v, signed=False))}, depending on the demand form"
        if len(ok) > 1 and s.cmcr_range[0] != s.cmcr_range[1]
        else f"Keeping prices flat takes marginal-cost savings of {pct(top.cmcr_average, signed=False)}"
    )
    return _exhibit(
        2,
        "Synergies",
        title,
        f"Compensating marginal cost reduction (CMCR) of {names}, % of marginal cost",
        chart,
        _mc_source(an, s, "CMCR after Werden (1996)."),
    )


def _synergy_section(an: Analysis, s: Summary) -> str:
    case = an.case
    if not any(r.status == "ok" for r in s.rows):
        return ""
    body = []
    for r in s.rows:
        if r.status != "ok":
            continue
        band = (
            "-"
            if r.cmcr_band is None
            else f"{100 * r.cmcr_band[0]:.1f} to {100 * r.cmcr_band[2]:.1f}%"
        )
        body.append(
            f"<tr><td>{escape(r.label)}</td><td class='n'>{pct(r.cmcr_average, signed=False)}</td>"
            f"<td class='n'>{band}</td>"
            f"<td class='n'>{money(r.synergy_total, case.currency, case.revenue_unit)}</td></tr>"
        )
    unit = (
        f"{case.currency} {case.revenue_unit} {case.period}".strip()
        if case.market.revenue is not None
        else "per unit of inside sales"
    )
    rng_c = _range_text(s.cmcr_range, lambda v: pct(v, signed=False))
    rng_s = _range_text(s.synergy_range, lambda v: money(v, case.currency, case.revenue_unit))
    return (
        '<section id="synergies" style="margin-top:40px"><h3>Synergies by demand form</h3>'
        "<p>The compensating marginal cost reduction (CMCR, Werden 1996) is the cost saving on the "
        "merging products that makes the pre-merger prices again a post-merger equilibrium. "
        f"Across demand forms it is <b>{rng_c}</b> of marginal cost, equivalent to "
        f"<b>{rng_s}</b> ({escape(unit)}).</p>"
        "<table class=\"fit\"><thead><tr><th>Demand form</th><th class='n'>CMCR (average)</th>"
        "<th class='n'>Monte Carlo 5-95%</th>"
        f"<th class='n'>Synergies, {escape(unit)}</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
        "<p class='note'>Synergies are the sum of the per-unit cost reductions times the "
        "pre-merger quantities of the merging products. They must be merger-specific, verifiable "
        "and pass through to marginal cost to count; this memo takes no view on that.</p></section>"
    )


def _welfare_section(an: Analysis, s: Summary) -> str:
    case = an.case
    if not any(r.status == "ok" for r in s.rows):
        return ""
    body = []
    for r in s.rows:
        if r.status != "ok":
            continue
        body.append(
            f"<tr><td>{escape(r.label)}</td>"
            f"<td class='n'>{money(r.consumer_harm, case.currency, case.revenue_unit)}</td>"
            f"<td class='n'>{pct(r.consumer_harm_share, signed=False)}</td>"
            f"<td class='n'>{money(r.party_surplus_change, case.currency, case.revenue_unit)}</td></tr>"
        )
    harm = ""
    if s.harm_range and case.market.revenue is not None:
        rng = _range_text(s.harm_range, lambda v: money(v, case.currency, case.revenue_unit))
        harm = f"Consumer harm is {rng} {escape(case.period)} before any efficiencies"
    return (
        '<section id="welfare"><div class="kicker">Consumer and producer surplus</div>'
        f"<h2>{harm or 'Consumer and producer surplus'}</h2>"
        "<table class=\"fit\"><thead><tr><th>Demand form</th><th class='n'>Consumer harm (CV)</th>"
        "<th class='n'>Share of pre-merger sales</th>"
        "<th class='n'>Change in variable profit of the parties</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
        "<p class='note'>Consumer harm is the compensating variation of the post-merger price "
        "change: closed form for logit, nested logit and CES; a Marshallian path integral for "
        "linear and PCAIDS, which is exact only without income effects. Variable profit is "
        "(p - c) q with the demand form's own implied costs.</p></section>"
    )


def _hm_section(an: Analysis, s: Summary) -> str:
    case = an.case
    rows = [r for r in s.rows if r.hm is not None]
    if not rows or case.hm_products is None:
        return ""
    body = []
    for r in rows:
        hm = r.hm
        assert hm is not None
        cls, word = ("ink", "Passes") if hm.passes else ("grey", "Fails")
        body.append(
            f"<tr><td>{escape(r.label)}</td><td class='n'>{pct(hm.max_increase)}</td>"
            f"<td><span class='tag {cls}'>{word}</span></td>"
            f"<td class='n'>{hm.actual_loss:.1%} vs {hm.critical_loss:.1%}</td></tr>"
        )
    passes = sum(1 for r in rows if r.hm and r.hm.passes)
    market = _join([case.market.labels[i] for i in case.hm_products])
    if passes == len(rows):
        head = f"Every demand form supports {market} as a relevant market"
    elif passes == 0:
        head = f"No demand form supports {market} as a relevant market"
    else:
        of = f"of {_count(len(rows))} demand forms"
        head = (
            f"{_count(passes).capitalize()} {of} support {market} as a relevant market"
            if passes > 1
            else f"Only one {of} supports {market} as a relevant market"
        )
    return (
        '<section id="hm"><div class="kicker">Hypothetical monopolist test</div>'
        f"<h2>{escape(head)}</h2>"
        f"<p>Would a hypothetical monopolist of {escape(market)} together profitably raise the "
        f"price of at least one product by {pct(case.ssnip, 0, signed=False)} (other prices held "
        "fixed)?</p>"
        '<table class="fit"><thead><tr><th>Demand form</th>'
        "<th class='n'>Profit-maximising increase</th>"
        "<th>Candidate market</th><th class='n'>Actual vs critical loss (uniform SSNIP)</th></tr>"
        f"</thead><tbody>{''.join(body)}</tbody></table>"
        "<p class='note'>Passing means the candidate set is a relevant market in the sense of "
        "section 4.1 of the 2010 U.S. Guidelines under that demand form.</p></section>"
    )


def _diagnostics_section(an: Analysis, s: Summary) -> str:
    mc = an.monte_carlo
    body = []
    for r in s.rows:
        notes = ""
        if r.status == "ok" and r.parameters is not None:
            params = ", ".join(f"{k.replace('_', ' ')} {v:.4g}" for k, v in r.parameters.items())
            notes = "".join(
                f"<div class='dim small'>{escape(n[:1].upper() + n[1:])}.</div>" for n in r.notes
            )
            resid = f"{r.margin_residual:.1e}" if r.margin_residual is not None else "-"
            res = f"{r.residual:.1e}" if r.residual is not None else "-"
        else:
            params, resid, res = "-", "-", "-"
        counts = "-"
        if mc is not None and r.kind in mc.status:
            c = mc.counts(r.kind)
            counts = f"{c['ok']:,} ok, {c['calibration_failed']:,} not calibrated, {c['solver_failed']:,} gate"
        body.append(
            f"<tr><td>{escape(r.label)}</td><td>{escape(params)}{notes}</td>"
            f"<td class='n'>{resid}</td><td class='n'>{res}</td><td class='n'>{counts}</td></tr>"
        )
    return (
        '<section id="diagnostics"><div class="kicker">Diagnostics</div>'
        "<h2>Calibration and solver checks by demand form</h2>"
        '<table class="fit"><thead><tr><th>Demand form</th><th>Calibrated parameters</th>'
        "<th class='n'>Max margin residual</th><th class='n'>FOC residual</th>"
        "<th class='n'>Monte Carlo draws</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></section>"
    )


def _assumptions_section(an: Analysis) -> str:
    case = an.case
    notes = "".join(f"<li>{escape(n)}</li>" for n in case.notes)
    notes_html = f"<h3>Inputs and assumptions</h3><ul class='plain'>{notes}</ul>" if notes else ""
    return (
        '<section id="assumptions"><div class="kicker">Assumptions and limitations</div>'
        "<h2>What the results depend on</h2>" + notes_html + "<h3>Method</h3><ul class='plain'>"
        "<li>Static Nash-Bertrand pricing by multi-product firms with constant marginal cost; "
        "marginal costs are recovered from pre-merger first-order conditions. Post-merger prices "
        "are accepted only if the first-order conditions hold to a residual below 1e-10.</li>"
        "<li>Each demand form is calibrated to the same prices, shares and margins. They agree on "
        "the pre-merger point and differ in curvature and substitution patterns, so the spread is "
        "functional-form risk, not sampling error.</li>"
        "<li>Predictions are model-based. They exclude entry, repositioning, coordinated effects, "
        "bargaining and dynamic or non-price effects, and they are sensitive to the definition of "
        "the outside good and to the margins.</li>"
        "<li>Accounting margins are not incremental margins; margins here are inputs to be "
        "challenged, which is what the bands are for.</li></ul></section>"
    )


def _summary_section(an: Analysis, s: Summary) -> str:
    case = an.case
    sim = an.simulation
    c = sim.concentration
    pr, cr, sr = s.point_range, s.cmcr_range, s.synergy_range
    words = {
        "presumed_harmful": "presumption triggered",
        "possible_dominance": "may indicate dominance",
        "warrants_scrutiny": "warrants scrutiny",
        "unlikely_concerns": "concerns unlikely",
        "no_presumption": "no presumption",
    }
    screen_txt = "; ".join(f"{r.name}: {words[r.status]}" for r in sim.screens)
    rows = [
        (
            "Structure",
            f"HHI rises from {points(c.hhi_pre)} to {points(c.hhi_post)} (+{points(c.delta_hhi)}); "
            f"the merged firm holds {c.merged_share:.1%}. {escape(screen_txt)}.",
        )
    ]
    if pr:
        band = (
            f" Allowing for unobserved inputs, the 5th to 95th percentile across forms is "
            f"{pct(s.ensemble_band[0])} to {pct(s.ensemble_band[2])} (median {pct(s.ensemble_band[1])})."
            if s.ensemble_band
            else ""
        )
        rows.append(
            (
                "Price effect",
                f"The merging firms' prices rise by <b>{_range_text(pr, pct)}</b> on average across "
                f"{_count(len([r for r in s.rows if r.status == 'ok']))} demand forms calibrated "
                f"to the same data.{band}",
            )
        )
    if cr:
        syn = (
            f", or {_range_text(sr, lambda v: money(v, case.currency, case.revenue_unit))} "
            f"{escape(case.period)}"
            if sr and case.market.revenue is not None
            else ""
        )
        rows.append(
            (
                "Synergies to offset",
                "Keeping prices flat requires marginal-cost savings of "
                f"<b>{_range_text(cr, lambda v: pct(v, signed=False))}</b> on the merging "
                f"products{syn}.",
            )
        )
    if s.harm_range and case.market.revenue is not None:
        rows.append(
            (
                "Consumer harm",
                "Compensating variation of "
                f"<b>{_range_text(s.harm_range, lambda v: money(v, case.currency, case.revenue_unit))}</b> "
                f"{escape(case.period)} before any efficiencies.",
            )
        )
    items = "".join(
        f'<div class="row"><dt>{label}</dt><dd>{text}</dd></div>' for label, text in rows
    )
    return (
        f'<section id="summary" style="margin-top:56px"><dl class="findings">{items}</dl></section>'
    )


def render_report(an: Analysis) -> str:
    """Render the analysis as one self-contained HTML document."""
    case = an.case
    s = summarize(an)
    eff = (
        f" &middot; {case.efficiency:.0%} cost saving on the merging products simulated"
        if case.efficiency > 0
        else ""
    )
    disclaimer = f"<p>{escape(case.disclaimer)}</p>" if case.disclaimer else ""
    sections = [
        _market_section(an),
        _screens_section(an),
        f'<div id="price-block">{_price_exhibit(an, s)}{_effects_section(an, s)}</div>',
        _synergy_exhibit(an, s),
        _synergy_section(an, s),
        _welfare_section(an, s),
        _hm_section(an, s),
        _diagnostics_section(an, s),
        _assumptions_section(an),
    ]
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(case.name)} - merger memo</title>
<style>{_CSS}</style></head>
<body><main>
<div id="front"><header><div class="mark"></div><div class="eyebrow">Merger screening memo</div>
<h1>{escape(case.name)}</h1>
<p class="lede">{escape(case.description)}</p>
<p class="meta">mergerlab {__version__} &middot; {case.market.shares.basis.value} shares
&middot; parties: {escape(", ".join(case.parties))}{eff}</p></header>
{_key_figures(an, s)}
{_summary_section(an, s)}</div>
{"".join(sections)}
<footer>{disclaimer}<p>Produced with mergerlab {__version__}. Calibrated model output, not an estimate.</p></footer>
</main></body></html>
"""
    return html


def write_report(an: Analysis, path: str | Path) -> Path:
    """Write the HTML memo to ``path`` and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(an), encoding="utf-8")
    return out
