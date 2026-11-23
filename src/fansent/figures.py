"""Static SVG figures for the README, drawn from a published schema-2 snapshot.

Each figure is one self-contained file (inline attributes, its own white background, no CSS,
script or font files), so GitHub renders it in a README in either colour theme. Like the
snapshot it is built from, a figure carries aggregate numbers and fixed labels only.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from .site import METRIC_LABELS, SCHEMA_V2, SnapshotError, fanbase_label, load_snapshot

INK = "#1b1f24"
MUTED = "#5f6b7a"
GRID = "#d8dee6"
ACCENT = "#b9553a"
FONT = "font-family=\"-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif\""
ROW_H = 26
LEFT = 230
RIGHT = 150
WIDTH = 860
FIELDING = ("ERRORS", "FLD_PCT")


def _f(v: float) -> str:
    return f"{v:+.3f}"


def _section(snap: dict, scope: str) -> list[dict]:
    for sec in snap["sections"]:
        if sec["scope"] == scope:
            return sec["rows"]
    raise SnapshotError(f"snapshot has no {scope!r} section")


def _text(x: float, y: float, s: str, *, anchor: str = "start", size: int = 14, fill: str = INK,
          weight: str = "400") -> str:  # fmt: skip
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" {FONT}>{escape(s)}</text>'
    )


def forest(title: str, rows: list[tuple[str, float, float, float, bool]], lo: float, hi: float,
           note: str) -> str:  # fmt: skip
    """rows: (label, effect, ci_low, ci_high, highlighted). One interval per row on a shared
    effect axis from lo to hi, with a line at zero."""
    top = 64
    height = top + len(rows) * ROW_H + 62
    x0, x1 = LEFT, WIDTH - RIGHT

    def sx(v: float) -> float:
        return x0 + (min(max(v, lo), hi) - lo) / (hi - lo) * (x1 - x0)

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" '
            f'width="{WIDTH}" height="{height}" role="img" aria-label="{escape(title)}">'
        ),
        f'<rect width="{WIDTH}" height="{height}" fill="#ffffff"/>',
        _text(20, 30, title, size=17, weight="600"),
    ]
    step = 0.05 if hi - lo > 0.2 else 0.02
    tick = round(lo / step) * step
    while tick <= hi + 1e-9:
        if tick >= lo - 1e-9:
            parts.append(
                f'<line x1="{sx(tick):.1f}" x2="{sx(tick):.1f}" y1="{top - 12}" '
                f'y2="{height - 50}" stroke="{INK if abs(tick) < 1e-9 else GRID}" '
                f'stroke-width="1"/>'
            )
            label = "0" if abs(tick) < 1e-9 else f"{tick:+.2f}"
            parts.append(_text(sx(tick), height - 30, label, anchor="middle", size=12, fill=MUTED))
        tick = round(tick + step, 6)
    for i, (label, mu, low, high, hot) in enumerate(rows):
        y = top + i * ROW_H + ROW_H / 2
        color = ACCENT if hot else INK
        parts.append(_text(x0 - 14, y + 5, label, anchor="end"))
        parts.append(
            f'<line x1="{sx(low):.1f}" x2="{sx(high):.1f}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>'
        )
        parts.append(f'<circle cx="{sx(mu):.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        parts.append(_text(x1 + 14, y + 5, f"{_f(mu)}  [{_f(low)}, {_f(high)}]", size=12))
    parts.append(_text(20, height - 8, note, size=12, fill=MUTED))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def confirmatory_figure(snap: dict) -> str:
    c = snap["confirmatory"]
    if c is None:
        raise SnapshotError("snapshot has no confirmatory result")
    rows = [
        (
            f"Two teams: {METRIC_LABELS[r['metric']]}, {fanbase_label(r['fanbase'])}",
            r["mu_mean"],
            r["ci_low"],
            r["ci_high"],
            False,
        )
        for r in sorted(_section(snap, "demo"), key=lambda r: r["mu_mean"])
        if r["metric"] in FIELDING and r["fanbase"] in ("CHICubs", "baseball")
    ]
    verdict = "confirmed" if c["confirmed"] else "not confirmed"
    rows.append(("Rest of league (pre-registered)", c["mu_mean"], c["ci_low"], c["ci_high"], True))
    return forest(
        f"Pre-registered fielding test: {verdict} (one-sided p = {c['perm_p_less']:.3f})",
        rows,
        -0.42,
        0.06,
        "Effect of fan sentiment on next-week fielding, sd per sd, with 95% credible intervals. "
        "Negative = worse fielding.",
    )


def league_figure(snap: dict) -> str:
    rows = [
        (METRIC_LABELS[r["metric"]], r["mu_mean"], r["ci_low"], r["ci_high"], False)
        for r in sorted(_section(snap, "league"), key=lambda r: list(METRIC_LABELS).index(r["metric"]))
        if not r["exploratory"]
    ]  # fmt: skip
    return forest(
        "Whole league, pooled: no metric clears correction",
        rows,
        -0.06,
        0.06,
        "Cubs and Red Sox players excluded. Positive = upbeat weeks came before better play. "
        "Exploratory.",
    )


def demo_figure(snap: dict) -> str:
    rows = [
        (
            f"{METRIC_LABELS[r['metric']]}, {fanbase_label(r['fanbase'])}",
            r["mu_mean"],
            r["ci_low"],
            r["ci_high"],
            False,
        )
        for r in _section(snap, "demo")
        if r["metric"] in FIELDING and r["fanbase"] != "news"
    ]
    return forest(
        "Cubs and Red Sox, fielding: borderline, and not replicated",
        rows,
        -0.42,
        0.26,
        "Corrected q is 0.07 to 0.09 for the r/CHICubs and r/baseball rows. Exploratory.",
    )


FIGURES = {
    "confirmatory.svg": confirmatory_figure,
    "league.svg": league_figure,
    "demo-fielding.svg": demo_figure,
}


def write_figures(snapshot_path: Path, out_dir: Path) -> list[Path]:
    snap = load_snapshot(snapshot_path)
    if snap["schema_version"] != SCHEMA_V2:
        raise SnapshotError("figures need a schema-2 snapshot (fansent snapshot --section ...)")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, build in FIGURES.items():
        path = out_dir / name
        path.write_text(build(snap), encoding="utf-8")
        paths.append(path)
    return paths
