"""Public results: aggregate-only snapshot of a `correlate` run, and a static site rendered from it.

The snapshot is the only thing that crosses from gitignored `data/` into the public repo, so it
is built from `analysis_fanbases.csv` alone (per-fanbase summary numbers) and every string in
it must match a closed vocabulary or a short identifier pattern. `validate_snapshot` enforces
that on write and on read, so a snapshot carrying free text cannot be published or rendered.

Schema 1 is one run (`rows`). Schema 2 (2026-11-22) is the final write-up: several runs as
`sections` keyed by a closed scope vocabulary, the permutation null used, and the summary
numbers of the pre-registered confirmatory test. All prose about a scope lives here in `SCOPES`,
never in the snapshot.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math
import re
from pathlib import Path

import pandas as pd

from .correlate import (
    EXPLORATORY_METRICS,
    FAMILY_COMPOSITE,
    FAMILY_EXPLORATORY,
    FAMILY_METRICS,
    FAMILY_OAA,
    FANBASES_OUT,
)

SCHEMA_VERSION = 1
DEFAULT_PUBLISHED_DIR = Path("results/published")
DEFAULT_SITE_DIR = Path("site")
REPO_URL = "https://github.com/SebastianFrazier26/mlb-fan-sentiment"

KNOWN_METRICS = ("COMPOSITE", "OPS", "AVG", "ERA", "WHIP", "FLD_PCT", "ERRORS", "OAA", "CS_PCT")
KNOWN_FAMILIES = (FAMILY_COMPOSITE, FAMILY_METRICS, FAMILY_OAA, FAMILY_EXPLORATORY)
FANBASE_RE = re.compile(r"^[A-Za-z0-9_]{2,32}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

TOP_KEYS = {
    "schema_version",
    "run_id",
    "created",
    "season",
    "ci_level",
    "perm_b",
    "alpha",
    "correction",
    "status",
    "rows",
}
STATUSES = ("preliminary", "final")
RUN_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
FLOAT_KEYS = ("mu_mean", "ci_low", "ci_high", "p_pos", "perm_p", "rhat_max")
NULLABLE_FLOAT_KEYS = ("q_bh", "p_holm")
INT_KEYS = ("n_units", "n_obs")
ROW_KEYS = {"family", "exploratory", "metric", "fanbase", *FLOAT_KEYS, *NULLABLE_FLOAT_KEYS}
ROW_KEYS |= set(INT_KEYS)

SCHEMA_V2 = 2
TOP_KEYS_V2 = (TOP_KEYS - {"rows"}) | {"null", "sections", "confirmatory"}
SECTION_KEYS = {"scope", "rows"}
NULLS = ("freedman-lane", "shuffle")
CONFIRM_METRIC = "FLD_Z"
CONFIRM_FLOAT_KEYS = ("mu_mean", "ci_low", "ci_high", "perm_p_less", "rhat_max")
CONFIRM_KEYS = {"metric", "confirmed", *CONFIRM_FLOAT_KEYS, *INT_KEYS}
# scope -> (heading, description). Fixed text: a snapshot can only pick a scope, not describe it.
SCOPES = {
    "league": (
        "Whole league",
        (
            "The other 28 teams' subreddits plus r/baseball, fitted as one pooled effect. Cubs "
            "and Red Sox players are left out because the first analysis was built on them."
        ),
    ),
    "demo": (
        "Cubs and Red Sox",
        (
            "The original two-team analysis: r/CHICubs, r/redsox, and r/baseball restricted to "
            "Cubs and Red Sox players (news headlines reach the minimum on too few players to "
            "read anything into). Its fielding rows are the signal the pre-registered league "
            "test was written to check."
        ),
    ),
    "league_fanbases": (
        "Every fanbase",
        "One effect per metric and fanbase across the league, news excluded.",
    ),
}
# Carried for readers of the snapshot (the results post). The static page has no chart for a
# table this long, and its palette stops at N_SLOTS fanbases.
DATA_ONLY_SCOPES = frozenset({"league_fanbases"})


class SnapshotError(ValueError):
    pass


def _num(v) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def snapshot_from_run(
    run_dir: Path,
    run_id: str,
    season: int,
    perm_b: int,
    alpha: float = 0.05,
    status: str = "preliminary",
) -> dict:
    df = pd.read_csv(run_dir / FANBASES_OUT)
    rows = []
    for r in df.to_dict("records"):
        rows.append(
            {
                "family": str(r["family"]),
                "exploratory": bool(r["exploratory"]),
                "metric": str(r["metric"]),
                "fanbase": str(r["fanbase"]),
                **{k: round(float(r[k]), 6) for k in FLOAT_KEYS},
                **{k: _round(_num(r[k])) for k in NULLABLE_FLOAT_KEYS},
                **{k: int(r[k]) for k in INT_KEYS},
            }
        )
    snap = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created": dt.datetime.now(dt.UTC).date().isoformat(),
        "season": season,
        "ci_level": 0.95,
        "perm_b": perm_b,
        "alpha": alpha,
        "correction": "bh",
        "status": status,
        "rows": rows,
    }
    validate_snapshot(snap)
    return snap


def snapshot_from_runs(
    run_dirs: dict[str, Path],
    run_id: str,
    season: int,
    perm_b: int,
    null: str,
    confirm_path: Path | None = None,
    alpha: float = 0.05,
    status: str = "preliminary",
) -> dict:
    """Schema-2 snapshot: one section per scope, each from that run's analysis_fanbases.csv."""
    sections = []
    for scope, run_dir in run_dirs.items():
        one = snapshot_from_run(run_dir, run_id, season, perm_b, alpha, status)
        sections.append({"scope": scope, "rows": one["rows"]})
    base = {k: v for k, v in one.items() if k != "rows"}
    snap = {
        **base,
        "schema_version": SCHEMA_V2,
        "null": null,
        "sections": sections,
        "confirmatory": _confirm_summary(confirm_path) if confirm_path else None,
    }
    validate_snapshot(snap)
    return snap


def _confirm_summary(path: Path) -> dict:
    """Picks the summary numbers out of `fansent confirm`'s result JSON; nothing else is copied."""
    c = json.loads(path.read_text(encoding="utf-8"))
    if c["metric"] != CONFIRM_METRIC or c["decision"] not in ("CONFIRMED", "NOT CONFIRMED"):
        raise SnapshotError("confirm result is not a decided FLD_Z test")
    return {
        "metric": CONFIRM_METRIC,
        "confirmed": c["decision"] == "CONFIRMED",
        **{k: round(float(c[k]), 6) for k in CONFIRM_FLOAT_KEYS},
        **{k: int(c[k]) for k in INT_KEYS},
    }


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 6)


def _is_num(v) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def validate_snapshot(snap: dict) -> None:
    if not isinstance(snap, dict):
        raise SnapshotError("snapshot must be an object")
    version = snap.get("schema_version")
    if version not in (SCHEMA_VERSION, SCHEMA_V2):
        raise SnapshotError(f"unsupported schema_version {version!r}")
    top = TOP_KEYS if version == SCHEMA_VERSION else TOP_KEYS_V2
    if set(snap) != top:
        raise SnapshotError(f"top-level keys must be exactly {sorted(top)}")
    if not isinstance(snap["run_id"], str) or not RUN_ID_RE.match(snap["run_id"]):
        raise SnapshotError("run_id must be a short identifier")
    try:
        dt.date.fromisoformat(snap["created"])
    except (TypeError, ValueError) as e:
        raise SnapshotError("created must be an ISO date") from e
    for k in ("season", "perm_b"):
        if not isinstance(snap[k], int) or isinstance(snap[k], bool) or snap[k] <= 0:
            raise SnapshotError(f"{k} must be a positive int")
    for k in ("ci_level", "alpha"):
        if not _is_num(snap[k]) or not 0 < snap[k] < 1:
            raise SnapshotError(f"{k} must be in (0, 1)")
    if snap["correction"] not in ("bh", "holm"):
        raise SnapshotError("correction must be 'bh' or 'holm'")
    if snap["status"] not in STATUSES:
        raise SnapshotError(f"status must be one of {STATUSES}")
    if version == SCHEMA_VERSION:
        _validate_rows(snap["rows"], snap["perm_b"])
        return
    if snap["null"] not in NULLS:
        raise SnapshotError(f"null must be one of {NULLS}")
    sections = snap["sections"]
    if not isinstance(sections, list) or not sections:
        raise SnapshotError("sections must be a non-empty list")
    seen = set()
    for sec in sections:
        if not isinstance(sec, dict) or set(sec) != SECTION_KEYS:
            raise SnapshotError(f"section keys must be exactly {sorted(SECTION_KEYS)}")
        if sec["scope"] not in SCOPES or sec["scope"] in seen:
            raise SnapshotError(f"section scope must be a distinct one of {sorted(SCOPES)}")
        seen.add(sec["scope"])
        _validate_rows(sec["rows"], snap["perm_b"])
    if snap["confirmatory"] is not None:
        _validate_confirm(snap["confirmatory"], snap["perm_b"])


def _validate_rows(rows, perm_b: int) -> None:
    if not isinstance(rows, list) or not rows:
        raise SnapshotError("rows must be a non-empty list")
    for i, r in enumerate(rows):
        _validate_row(i, r, perm_b)


def _on_perm_grid(p: float, perm_b: int) -> bool:
    # p = (1 + count) / (1 + B), so a mismatched perm_b shows up as a non-integer count.
    count = p * (perm_b + 1) - 1
    return abs(count - round(count)) <= 1e-3


def _validate_confirm(c, perm_b: int) -> None:
    if not isinstance(c, dict) or set(c) != CONFIRM_KEYS:
        raise SnapshotError(f"confirmatory keys must be exactly {sorted(CONFIRM_KEYS)}")
    if c["metric"] != CONFIRM_METRIC or not isinstance(c["confirmed"], bool):
        raise SnapshotError("confirmatory: metric must be FLD_Z and confirmed a bool")
    if not all(_is_num(c[k]) for k in CONFIRM_FLOAT_KEYS):
        raise SnapshotError("confirmatory: numbers must be finite")
    for k in INT_KEYS:
        if not isinstance(c[k], int) or isinstance(c[k], bool) or c[k] < 0:
            raise SnapshotError(f"confirmatory: {k} must be a non-negative int")
    if not c["ci_low"] <= c["mu_mean"] <= c["ci_high"]:
        raise SnapshotError("confirmatory: mu_mean outside its interval")
    if not 0 <= c["perm_p_less"] <= 1 or not _on_perm_grid(c["perm_p_less"], perm_b):
        raise SnapshotError("confirmatory: perm_p_less is not on the 1/(perm_b+1) grid")


def _validate_row(i: int, r: dict, perm_b: int) -> None:
    where = f"rows[{i}]"
    if not isinstance(r, dict) or set(r) != ROW_KEYS:
        raise SnapshotError(f"{where}: keys must be exactly {sorted(ROW_KEYS)}")
    if r["metric"] not in KNOWN_METRICS:
        raise SnapshotError(f"{where}: unknown metric {r['metric']!r}")
    if r["family"] not in KNOWN_FAMILIES:
        raise SnapshotError(f"{where}: unknown family {r['family']!r}")
    if not isinstance(r["fanbase"], str) or not FANBASE_RE.match(r["fanbase"]):
        raise SnapshotError(f"{where}: fanbase must be a short identifier")
    if not isinstance(r["exploratory"], bool):
        raise SnapshotError(f"{where}: exploratory must be a bool")
    if r["exploratory"] != (r["metric"] in EXPLORATORY_METRICS):
        raise SnapshotError(f"{where}: exploratory flag disagrees with metric")
    for k in FLOAT_KEYS:
        if not _is_num(r[k]):
            raise SnapshotError(f"{where}: {k} must be a finite number")
    for k in NULLABLE_FLOAT_KEYS:
        if r[k] is None:
            if not r["exploratory"]:
                raise SnapshotError(f"{where}: {k} is only null for exploratory rows")
        elif not _is_num(r[k]) or not 0 <= r[k] <= 1:
            raise SnapshotError(f"{where}: {k} must be a probability or null")
    for k in INT_KEYS:
        if not isinstance(r[k], int) or isinstance(r[k], bool) or r[k] < 0:
            raise SnapshotError(f"{where}: {k} must be a non-negative int")
    if not r["ci_low"] <= r["mu_mean"] <= r["ci_high"]:
        raise SnapshotError(f"{where}: mu_mean outside its interval")
    for k in ("p_pos", "perm_p"):
        if not 0 <= r[k] <= 1:
            raise SnapshotError(f"{where}: {k} must be a probability")
    if not _on_perm_grid(r["perm_p"], perm_b):
        raise SnapshotError(f"{where}: perm_p is not on the 1/(perm_b+1) grid")


def write_snapshot(snap: dict, out_dir: Path) -> Path:
    validate_snapshot(snap)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{snap['run_id']}.json"
    path.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return path


def load_snapshot(path: Path) -> dict:
    snap = json.loads(path.read_text(encoding="utf-8"))
    validate_snapshot(snap)
    return snap


def latest_snapshot(published_dir: Path) -> Path:
    """Run ids are date-prefixed, so the lexically last file is the newest run."""
    paths = sorted(published_dir.glob("*.json"))
    if not paths:
        raise SnapshotError(f"no snapshots in {published_dir}")
    return paths[-1]


FANBASE_ORDER = ("CHICubs", "redsox", "baseball", "news")
METRIC_LABELS = {
    "COMPOSITE": "Composite index",
    "OPS": "OPS",
    "AVG": "AVG",
    "ERA": "ERA",
    "WHIP": "WHIP",
    "FLD_PCT": "Fielding %",
    "ERRORS": "Errors",
    "OAA": "Outs above avg",
    "CS_PCT": "Caught stealing %",
}
N_SLOTS = 8


POOLED_FANBASE = "pooled"


def fanbase_label(fb: str) -> str:
    if fb == POOLED_FANBASE:
        return "All fanbases, pooled"
    return "News headlines" if fb == "news" else f"r/{fb}"


def _fanbases(rows: list[dict]) -> list[str]:
    present = {r["fanbase"] for r in rows}
    ordered = [f for f in FANBASE_ORDER if f in present]
    ordered += sorted(present - set(ordered))
    if len(ordered) > N_SLOTS:
        raise SnapshotError(f"more than {N_SLOTS} fanbases; the palette has no further slots")
    return ordered


def _ordered_rows(rows: list[dict], fanbases: list[str]) -> list[dict]:
    return sorted(
        rows, key=lambda r: (KNOWN_METRICS.index(r["metric"]), fanbases.index(r["fanbase"]))
    )


def _sig(r: dict, snap: dict) -> bool:
    q = r["q_bh"] if snap["correction"] == "bh" else r["p_holm"]
    return q is not None and q < snap["alpha"]


def _f(x: float, nd: int = 3, sign: bool = True) -> str:
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _p(x: float | None) -> str:
    return "n/a" if x is None else (f"{x:.3f}" if x >= 0.001 else "<0.001")


E = html.escape

W = 640
LEFT = 150
RIGHT = 24
ROW_H = 18
GROUP_GAP = 10
TOP = 30
BOTTOM = 40


def _layout(rows: list[dict]) -> tuple[list[float], list[tuple[str, float, float]], float]:
    """y centre per row, (metric, y0, y1) per metric group, total height."""
    ys, groups, y = [], [], TOP
    prev = None
    start = y
    for r in rows:
        if prev is not None and r["metric"] != prev:
            groups.append((prev, start, y))
            y += GROUP_GAP
            start = y
        ys.append(y + ROW_H / 2)
        y += ROW_H
        prev = r["metric"]
    groups.append((prev, start, y))
    return ys, groups, y + BOTTOM


def _nice_ticks(lo: float, hi: float) -> list[float]:
    step = 0.1 if hi - lo <= 1.2 else 0.25
    start = math.floor(lo / step) * step
    stop = math.ceil(hi / step - 1e-9) * step
    ticks, t = [], start
    while t <= stop + 1e-9:
        ticks.append(round(t, 6))
        t += step
    return ticks


def _metric_bands(groups, x0: float, x1: float) -> list[str]:
    out = []
    for i, (metric, y0, y1) in enumerate(groups):
        if i % 2 == 0:
            out.append(
                f'<rect class="band" x="{x0}" y="{y0 - 3:.1f}" width="{x1 - x0}" '
                f'height="{y1 - y0 + 6:.1f}" rx="4"/>'
            )
        out.append(
            f'<text class="lab" x="{x0 - 10}" y="{(y0 + y1) / 2 + 4:.1f}" '
            f'text-anchor="end">{E(METRIC_LABELS[metric])}</text>'
        )
    return out


def forest_svg(snap: dict, key: str = "forest") -> str:
    fanbases = _fanbases(snap["rows"])
    rows = _ordered_rows(snap["rows"], fanbases)
    ys, groups, height = _layout(rows)
    lo = min(min(r["ci_low"] for r in rows), -0.05)
    hi = max(max(r["ci_high"] for r in rows), 0.05)
    ticks = _nice_ticks(lo, hi)
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
    x0, x1 = LEFT, W - RIGHT

    def sx(v: float) -> float:
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    parts = _metric_bands(groups, x0, x1)
    for t in ticks:
        parts.append(
            f'<line class="grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{TOP - 6}" '
            f'y2="{height - BOTTOM + 4}"/>'
            f'<text class="tick" x="{sx(t):.1f}" y="{height - BOTTOM + 18}" '
            f'text-anchor="middle">{"0" if abs(t) < 1e-9 else f"{t:+.1f}"}</text>'
        )
    parts.append(
        f'<line class="zero" x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{TOP - 6}" '
        f'y2="{height - BOTTOM + 4}"/>'
    )
    parts.append(
        f'<text class="tick" x="{sx(0) - 6:.1f}" y="{TOP - 12}" text-anchor="end">'
        f"&#8592; worse next week</text>"
        f'<text class="tick" x="{sx(0) + 6:.1f}" y="{TOP - 12}">better next week &#8594;</text>'
    )
    for r, y in zip(rows, ys, strict=True):
        slot = fanbases.index(r["fanbase"]) + 1
        sig = _sig(r, snap)
        tip = (
            f"{fanbase_label(r['fanbase'])}, {METRIC_LABELS[r['metric']]}: "
            f"effect {_f(r['mu_mean'])} (95% CrI {_f(r['ci_low'])} to {_f(r['ci_high'])}), "
            f"permutation p {_p(r['perm_p'])}, BH q {_p(r['q_bh'])}"
        )
        parts.append(
            f'<g class="mark s{slot}"><title>{E(tip)}</title>'
            f'<rect class="hit" x="{x0}" y="{y - ROW_H / 2:.1f}" width="{x1 - x0}" '
            f'height="{ROW_H}"/>'
            f'<line class="ci" x1="{sx(r["ci_low"]):.1f}" x2="{sx(r["ci_high"]):.1f}" '
            f'y1="{y:.1f}" y2="{y:.1f}"/>'
            f'<circle class="{"pt sig" if sig else "pt"}" cx="{sx(r["mu_mean"]):.1f}" '
            f'cy="{y:.1f}" r="4.5"/></g>'
        )
        if sig:
            parts.append(
                f'<text class="note" x="{sx(r["ci_high"]) + 8:.1f}" y="{y + 4:.1f}">'
                f"q = {_p(r['q_bh'])}</text>"
            )
    parts.append(
        f'<text class="axis" x="{(x0 + x1) / 2:.1f}" y="{height - 6}" text-anchor="middle">'
        "fanbase effect &#956;: SD of next-week performance per SD of sentiment</text>"
    )
    desc = (
        "Forest plot of posterior mean fanbase effects with 95% credible intervals, one row per "
        "metric and fanbase. Filled dots are significant after correction. "
        "The data table below lists every value."
    )
    return _svg(key, "Fanbase effects with 95% credible intervals", desc, height, parts)


def perm_svg(snap: dict, key: str = "perm") -> str:
    fanbases = _fanbases(snap["rows"])
    rows = _ordered_rows(snap["rows"], fanbases)
    ys, groups, height = _layout(rows)
    floor = 1 / (snap["perm_b"] + 1)
    lo, hi = math.log10(min(floor, 0.001)), 0.0
    x0, x1 = LEFT, W - RIGHT

    def sx(p: float) -> float:
        return x0 + (math.log10(max(p, floor)) - lo) / (hi - lo) * (x1 - x0)

    parts = _metric_bands(groups, x0, x1)
    for t in (0.001, 0.01, 0.1, 1.0):
        if t >= 10**lo:
            parts.append(
                f'<line class="grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{TOP - 6}" '
                f'y2="{height - BOTTOM + 4}"/>'
                f'<text class="tick" x="{sx(t):.1f}" y="{height - BOTTOM + 18}" '
                f'text-anchor="middle">{t:g}</text>'
            )
    a = snap["alpha"]
    parts.append(
        f'<line class="zero" x1="{sx(a):.1f}" x2="{sx(a):.1f}" y1="{TOP - 6}" '
        f'y2="{height - BOTTOM + 4}"/>'
        f'<text class="tick" x="{sx(a) - 6:.1f}" y="{TOP - 12}" text-anchor="end">'
        f"&#945; = {a:g}</text>"
    )
    for r, y in zip(rows, ys, strict=True):
        slot = fanbases.index(r["fanbase"]) + 1
        q = r["q_bh"] if snap["correction"] == "bh" else r["p_holm"]
        tip = (
            f"{fanbase_label(r['fanbase'])}, {METRIC_LABELS[r['metric']]}: "
            f"permutation p {_p(r['perm_p'])}, corrected {_p(q)}"
        )
        seg = ""
        if q is not None:
            seg = (
                f'<line class="shift" x1="{sx(r["perm_p"]):.1f}" x2="{sx(q):.1f}" '
                f'y1="{y:.1f}" y2="{y:.1f}"/>'
                f'<rect class="q" x="{sx(q) - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" '
                f'transform="rotate(45 {sx(q):.1f} {y:.1f})"/>'
            )
        parts.append(
            f'<g class="mark s{slot}"><title>{E(tip)}</title>'
            f'<rect class="hit" x="{x0}" y="{y - ROW_H / 2:.1f}" width="{x1 - x0}" '
            f'height="{ROW_H}"/>{seg}'
            f'<circle class="pt sig" cx="{sx(r["perm_p"]):.1f}" cy="{y:.1f}" r="4.5"/></g>'
        )
    parts.append(
        f'<text class="axis" x="{(x0 + x1) / 2:.1f}" y="{height - 6}" text-anchor="middle">'
        f"two-sided p (log scale; floor 1/{snap['perm_b'] + 1})</text>"
    )
    desc = (
        "Dot plot of permutation-test p-values (dots) and multiplicity-corrected values "
        "(diamonds) per metric and fanbase on a log axis, with the significance threshold "
        "marked. The data table below lists every value."
    )
    return _svg(key, "Observed effects against the permutation null", desc, height, parts)


def _svg(key: str, title: str, desc: str, height: float, parts: list[str]) -> str:
    return (
        f'<svg class="chart" viewBox="0 0 {W} {height:.0f}" role="img" '
        f'aria-labelledby="{key}-t {key}-d" xmlns="http://www.w3.org/2000/svg">'
        f'<title id="{key}-t">{E(title)}</title><desc id="{key}-d">{E(desc)}</desc>'
        + "".join(parts)
        + "</svg>"
    )


def _legend(fanbases: list[str], extra: str = "") -> str:
    items = "".join(
        f'<li><span class="sw s{i}" aria-hidden="true"></span>{E(fanbase_label(f))}</li>'
        for i, f in enumerate(fanbases, 1)
    )
    return f'<ul class="legend">{items}{extra}</ul>'


def _tiles(snap: dict, fanbases: list[str]) -> str:
    comp = {r["fanbase"]: r for r in snap["rows"] if r["metric"] == "COMPOSITE"}
    out = []
    for i, fb in enumerate(fanbases, 1):
        r = comp.get(fb)
        if r is None:
            continue
        sig = _sig(r, snap)
        if sig and r["mu_mean"] < 0:
            verdict = "Upbeat weeks preceded <strong>worse</strong> play"
        elif sig:
            verdict = "Upbeat weeks preceded <strong>better</strong> play"
        else:
            verdict = "No detectable signal"
        out.append(
            f'<div class="tile"><div class="tile-h"><span class="sw s{i}" aria-hidden="true">'
            f"</span>{E(fanbase_label(fb))}</div>"
            f'<div class="big">{_f(r["mu_mean"], 2)}</div>'
            f'<div class="sub">95% CrI {_f(r["ci_low"], 2)} to {_f(r["ci_high"], 2)} &middot; '
            f"q = {_p(r['q_bh'])}</div>"
            f'<div class="verdict">{verdict}</div></div>'
        )
    return '<div class="tiles">' + "".join(out) + "</div>"


def _table(snap: dict, fanbases: list[str]) -> str:
    head = (
        "<tr><th scope=col>Metric</th><th scope=col>Fanbase</th><th scope=col>&#956;</th>"
        "<th scope=col>95% CrI</th><th scope=col>P(&#956;&gt;0)</th>"
        "<th scope=col>perm p</th><th scope=col>BH q</th><th scope=col>Holm p</th>"
        "<th scope=col>players</th><th scope=col>player-weeks</th>"
        "<th scope=col>max R-hat</th></tr>"
    )
    body = []
    for r in _ordered_rows(snap["rows"], fanbases):
        body.append(
            f"<tr><td>{E(METRIC_LABELS[r['metric']])}</td><td>{E(fanbase_label(r['fanbase']))}"
            f"</td><td>{_f(r['mu_mean'])}</td>"
            f"<td>{_f(r['ci_low'])} to {_f(r['ci_high'])}</td>"
            f"<td>{_f(r['p_pos'], 3, False)}</td><td>{_p(r['perm_p'])}</td>"
            f"<td>{_p(r['q_bh'])}</td><td>{_p(r['p_holm'])}</td><td>{r['n_units']}</td>"
            f"<td>{r['n_obs']}</td><td>{_f(r['rhat_max'], 3, False)}</td></tr>"
        )
    return f'<div class="scroll"><table>{head}{"".join(body)}</table></div>'


CSS = (
    """
:root{color-scheme:light;--bg:#fcfcfb;--panel:#f4f3ef;--ink:#0b0b0b;--ink2:#52514e;
--muted:#6f6d68;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--link:#1c5cab;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;
--s8:#e34948;--warn-bg:#fdf3dc;--warn-line:#d9a227;--warn-ink:#7a5200}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
--bg:#1a1a19;--panel:#222220;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9b9a94;--grid:#2c2c2a;
--axis:#55554f;--ring:rgba(255,255,255,.10);--link:#86b6ef;--s1:#3987e5;--s2:#d95926;
--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;
--warn-bg:#2e2716;--warn-line:#a87f1f;--warn-ink:#f0c45a}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--panel:#222220;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#9b9a94;--grid:#2c2c2a;--axis:#55554f;--ring:rgba(255,255,255,.10);
--link:#86b6ef;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;
--s6:#008300;--s7:#9085e9;--s8:#e66767;--warn-bg:#2e2716;--warn-line:#a87f1f;
--warn-ink:#f0c45a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:760px;margin:0 auto;padding:32px 16px 48px}
h1{font-size:1.9rem;line-height:1.2;margin:.2em 0 .3em;letter-spacing:-.01em}
h2{font-size:1.2rem;margin:2.2em 0 .5em}
p,li{color:var(--ink2)}
a{color:var(--link)}
.prelim{display:flex;gap:12px;align-items:flex-start;margin:0 0 22px;padding:12px 14px;
background:var(--warn-bg);border:1px solid var(--warn-line);border-left-width:4px;
border-radius:8px}
.prelim p{margin:0;color:var(--ink);font-size:.95rem}
.badge{flex:none;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
color:var(--warn-ink);border:1px solid var(--warn-line);border-radius:999px;padding:2px 9px;
margin-top:1px;white-space:nowrap}
.kicker{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0}
.lede{font-size:1.08rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--panel);border:1px solid var(--ring);border-radius:10px;padding:14px 16px}
.tile-h{font-weight:600;display:flex;align-items:center;gap:8px}
.big{font-size:2rem;font-weight:650;font-variant-numeric:tabular-nums;margin:4px 0 0}
.sub{font-size:.85rem;color:var(--muted);font-variant-numeric:tabular-nums}
.verdict{margin-top:6px;color:var(--ink2)}
figure{margin:12px 0 0;background:var(--panel);border:1px solid var(--ring);border-radius:10px;
padding:12px 8px 4px}
figcaption{font-size:.9rem;color:var(--ink2);padding:4px 8px 8px}
.fig-scroll{overflow-x:auto}
.chart{display:block;width:100%;min-width:560px;height:auto;font-family:inherit}
.chart .band{fill:var(--bg);opacity:.7}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .zero{stroke:var(--axis);stroke-width:1.5;stroke-dasharray:4 3}
.chart .lab{fill:var(--ink2);font-size:12px}
.chart .tick,.chart .axis{fill:var(--muted);font-size:11px}
.chart .note{fill:var(--ink2);font-size:11px;font-weight:600}
.chart .hit{fill:transparent}
.chart .ci,.chart .shift{stroke-width:2;stroke-linecap:round}
.chart .shift{stroke-dasharray:2 3}
.chart .pt{fill:var(--panel);stroke-width:2}
.chart .q{fill:var(--panel);stroke-width:2}
.chart .mark:hover .hit{fill:var(--grid);opacity:.5}
"""
    + "".join(
        f".s{i} .ci,.s{i} .shift{{stroke:var(--s{i})}}.s{i} .pt,.s{i} .q{{stroke:var(--s{i})}}"
        f".s{i} .pt.sig{{fill:var(--s{i})}}.sw.s{i}{{background:var(--s{i})}}"
        for i in range(1, N_SLOTS + 1)
    )
    + """
.legend{list-style:none;display:flex;flex-wrap:wrap;gap:6px 18px;padding:0 8px;margin:0 0 4px;
font-size:.9rem}
.legend li{display:flex;align-items:center;gap:6px;color:var(--ink2)}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%}
.key{display:inline-block;width:10px;height:10px;border:2px solid var(--muted);border-radius:50%}
.key.fill{background:var(--muted)}
.key.dia{border-radius:1px;transform:rotate(45deg);width:9px;height:9px}
details{margin-top:12px}
summary{cursor:pointer;color:var(--ink2)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font-size:.85rem;font-variant-numeric:tabular-nums;width:100%;
margin-top:8px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--grid);white-space:nowrap}
th{color:var(--muted);font-weight:600}
dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;margin:0}
dt{font-weight:600}
dd{margin:0;color:var(--ink2)}
footer{margin-top:3em;font-size:.85rem;color:var(--muted);border-top:1px solid var(--grid);
padding-top:12px}
@media (max-width:520px){dl{grid-template-columns:1fr}dd{margin-bottom:6px}
.prelim{padding:10px 12px;margin-bottom:16px}.prelim p{font-size:.88rem}.badge{display:none}}
"""
)


def _status_banner(snap: dict) -> str:
    if snap["status"] == "final":
        return ""
    m = RUN_DATE_RE.match(snap["run_id"])
    run = f"the {m.group(1)} run" if m else f"run {E(snap['run_id'])}"
    return (
        '<aside class="prelim" role="note" aria-label="Preliminary results">'
        '<span class="badge">Preliminary</span>'
        f"<p><strong>Preliminary results</strong> from {run}, scored with the baseline "
        "sentiment scorer before the target-aware classifier is trained. "
        "These numbers will change.</p></aside>"
    )


def _confirm_block(c: dict | None, alpha: float) -> str:
    if c is None:
        return ""
    verdict = "Confirmed" if c["confirmed"] else "Not confirmed"
    return f"""<section aria-labelledby="h-confirm">
<h2 id="h-confirm">Pre-registered test: {verdict.lower()}</h2>
<p>The two-team analysis suggested that upbeat weeks preceded <em>worse</em> fielding. Before
looking at any other team we <a href="{REPO_URL}/blob/main/docs/preregistration.md">registered
one test</a> of that on the rest of the league: a pooled fielding effect below zero, one-sided
p &lt; {alpha:g}.</p>
<div class="tiles"><div class="tile"><div class="tile-h">Fielding, rest of league</div>
<div class="big">{_f(c["mu_mean"], 3)}</div>
<div class="sub">95% CrI {_f(c["ci_low"], 3)} to {_f(c["ci_high"], 3)} &middot;
one-sided p = {_p(c["perm_p_less"])}</div>
<div class="verdict"><strong>{verdict}</strong> &middot; {c["n_units"]:,} fanbase-player
pairs, {c["n_obs"]:,} player-weeks</div></div></div>
</section>"""


def _section_block(snap: dict, sec: dict) -> str:
    scope = sec["scope"]
    title, blurb = SCOPES[scope]
    part = {**snap, "rows": sec["rows"]}
    fanbases = _fanbases(sec["rows"])
    n_tests = sum(not r["exploratory"] for r in sec["rows"])
    n_sig = sum(_sig(r, part) for r in sec["rows"])
    corr = "Benjamini-Hochberg" if snap["correction"] == "bh" else "Holm"
    sig_key = (
        '<li><span class="key fill" aria-hidden="true"></span>significant after correction</li>'
        '<li><span class="key" aria-hidden="true"></span>not significant</li>'
    )
    perm_key = (
        '<li><span class="key fill" aria-hidden="true"></span>permutation p</li>'
        f'<li><span class="key dia" aria-hidden="true"></span>{corr}-corrected</li>'
    )
    return f"""<section aria-labelledby="h-{scope}">
<h2 id="h-{scope}">{E(title)}</h2>
<p>{E(blurb)}</p>
{_tiles(part, fanbases)}
<p>{n_sig} of {n_tests} corrected tests clear q &lt; {snap["alpha"]:g}.</p>
<figure>
{_legend(fanbases, sig_key)}
<div class="fig-scroll">{forest_svg(part, f"forest-{scope}")}</div>
<figcaption>Posterior mean and {snap["ci_level"]:.0%} credible interval of each effect. Metrics
are oriented so that right is always better (fewer errors, lower ERA).</figcaption>
</figure>
<figure>
{_legend(fanbases, perm_key)}
<div class="fig-scroll">{perm_svg(part, f"perm-{scope}")}</div>
<figcaption>Permutation p for each effect; the diamond is the value after {corr} correction
within its family. Left of the dashed line is significant.</figcaption>
</figure>
<details><summary>All numbers (data table)</summary>{_table(part, fanbases)}</details>
</section>"""


NULL_TEXT = {
    "freedman-lane": (
        "Freedman-Lane permutation: within each player, the part of sentiment not explained by "
        "his recent performance is shuffled across his weeks and the model is refit. Fans "
        "reacting to last week's results is kept in the null, so the test asks whether sentiment "
        "adds anything beyond that. Two-sided p, centered on the null median."
    ),
    "shuffle": (
        "Within-player permutation of sentiment across weeks, model refit each time; two-sided p."
    ),
}


def _render_v2(snap: dict) -> str:
    shown = [sec for sec in snap["sections"] if sec["scope"] not in DATA_ONLY_SCOPES]
    rows = [r for sec in shown for r in sec["rows"]]
    c = snap["confirmatory"]
    prelim = snap["status"] == "preliminary"
    if c is not None and not c["confirmed"]:
        answer = (
            "<strong>We find no evidence that they do.</strong> The one test registered in "
            "advance was not confirmed, and no league-wide effect clears correction."
        )
    else:
        answer = "The sections below give the effect for each group of fanbases."
    banner = (
        '<aside class="prelim" role="note" aria-label="Preliminary results">'
        '<span class="badge">Preliminary</span>'
        "<p><strong>Preliminary results.</strong> These numbers may change.</p></aside>"
        if prelim
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Are the Fans Right?{" (preliminary)" if prelim else ""}</title>
<meta name="description" content="Does fan sentiment predict next-week MLB performance? \
Hierarchical Bayesian results for the {snap["season"]} season.">
<style>{CSS}</style>
</head>
<body>
<main>
{banner}
<p class="kicker">fansent &middot; {snap["season"]} season &middot; run {E(snap["run_id"])}</p>
<h1>Are the fans always right?</h1>
<p class="lede">When a fanbase is upbeat about a player one week, does that player actually play
better the next week, beyond what his current form already predicts? {answer}</p>
<p>Effects &#956; are in SD of next-week performance per SD of fan sentiment. Positive means
optimism preceded better play.</p>
{_confirm_block(c, snap["alpha"])}
{"".join(_section_block(snap, sec) for sec in shown)}
<section aria-labelledby="h-method">
<h2 id="h-method">Method</h2>
<dl>
<dt>Question</dt><dd>Does fan sentiment about a player in week t predict his performance in
week t+1, controlling for week-t performance and season-to-date form?</dd>
<dt>Data</dt><dd>Public Reddit comments from team subreddits (each counts only its own team's
players that week) and r/baseball, scored for sentiment toward the named player by a
classifier trained for this project; weekly player stats from the public MLB Stats API. Only
aggregate results are published; no text is shown or stored here.</dd>
<dt>Model</dt><dd>Hierarchical regression per metric, fit by a Gibbs sampler written from
scratch: each (fanbase, player) gets its own sentiment slope, drawn from a group-level mean
&#956; that is the reported effect.</dd>
<dt>Null</dt><dd>{NULL_TEXT[snap["null"]]} {snap["perm_b"]:,} refits per effect, so p cannot
go below 1/{snap["perm_b"] + 1}.</dd>
<dt>Corrections</dt><dd>Benjamini-Hochberg (and Holm) within separate families: per-metric
tests together, the composite index on its own.</dd>
</dl>
</section>
<section aria-labelledby="h-caveats">
<h2 id="h-caveats">Caveats</h2>
<ul>
<li>One season. The two-team tests rest on few players
({min(r["n_units"] for r in rows)} at the smallest), so their intervals are wide.</li>
<li>Only the pre-registered test is confirmatory. Everything else on this page is
exploratory.</li>
<li>The same players appear under more than one fanbase, so rows are not independent.</li>
<li>Sentiment is from an automatic classifier and name matching; both make errors.</li>
<li>Weekly thresholds and composite weights are documented assumptions, not tuned values.</li>
<li>Largest R-hat across these fits: {max(r["rhat_max"] for r in rows):.3f}.</li>
</ul>
</section>
<footer>
<p>Code, method and data contract: <a href="{REPO_URL}">github.com/SebastianFrazier26/\
mlb-fan-sentiment</a> &middot;
<a href="{REPO_URL}/blob/main/docs/results-2026-11.md">full results note</a>
&middot; <a href="data.json">snapshot JSON</a> &middot; Apache-2.0. Snapshot
{E(snap["created"])}.</p>
</footer>
</main>
</body>
</html>
"""


def render_html(snap: dict) -> str:
    validate_snapshot(snap)
    if snap["schema_version"] == SCHEMA_V2:
        return _render_v2(snap)
    fanbases = _fanbases(snap["rows"])
    n_tests = sum(not r["exploratory"] for r in snap["rows"])
    n_sig = sum(_sig(r, snap) for r in snap["rows"])
    fb_list = ", ".join(fanbase_label(f) for f in fanbases)
    corr = "Benjamini-Hochberg" if snap["correction"] == "bh" else "Holm"
    sig_key = (
        '<li><span class="key fill" aria-hidden="true"></span>significant after correction</li>'
        '<li><span class="key" aria-hidden="true"></span>not significant</li>'
    )
    perm_key = (
        '<li><span class="key fill" aria-hidden="true"></span>permutation p</li>'
        f'<li><span class="key dia" aria-hidden="true"></span>{corr}-corrected</li>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Are the Fans Right?{" (preliminary)" if snap["status"] == "preliminary" else ""}</title>
<meta name="description" content="Does fan sentiment predict next-week MLB performance? \
Hierarchical Bayesian results for {E(fb_list)}, {snap["season"]} season.">
<style>{CSS}</style>
</head>
<body>
<main>
{_status_banner(snap)}
<p class="kicker">fansent &middot; {snap["season"]} season &middot; run {E(snap["run_id"])}</p>
<h1>Are the fans always right?</h1>
<p class="lede">When a fanbase is upbeat about a player one week, does that player actually play
better the next week, beyond what his current form already predicts? Compared here:
{E(fb_list)}.</p>
<section aria-labelledby="h-headline">
<h2 id="h-headline">Headline: composite performance</h2>
<p>Effect &#956; in SD of next-week performance per SD of fan sentiment. Positive means
optimism preceded better play.</p>
{_tiles(snap, fanbases)}
<p>{n_sig} of {n_tests} corrected tests clear q &lt; {snap["alpha"]:g}.</p>
</section>
<section aria-labelledby="h-forest">
<h2 id="h-forest">Effects by metric</h2>
<figure>
{_legend(fanbases, sig_key)}
<div class="fig-scroll">{forest_svg(snap)}</div>
<figcaption>Posterior mean and {snap["ci_level"]:.0%} credible interval of each fanbase's
pooled effect. Metrics are oriented so that right is always better (fewer errors, lower ERA).
</figcaption>
</figure>
</section>
<section aria-labelledby="h-perm">
<h2 id="h-perm">Against the permutation null</h2>
<p>Each fanbase's effect is compared with {snap["perm_b"]:,} refits in which every player's
sentiment is shuffled across his own weeks. The p-value is the share of shuffled effects at
least as large in magnitude; the diamond shows it after {corr} correction within its family
(the composite is its own family).</p>
<figure>
{_legend(fanbases, perm_key)}
<div class="fig-scroll">{perm_svg(snap)}</div>
<figcaption>Left of the dashed line is significant. A run with {snap["perm_b"]:,} shuffles cannot
resolve p below 1/{snap["perm_b"] + 1}.</figcaption>
</figure>
<details><summary>All numbers (data table)</summary>{_table(snap, fanbases)}</details>
</section>
<section aria-labelledby="h-method">
<h2 id="h-method">Method</h2>
<dl>
<dt>Question</dt><dd>Does fan sentiment about a player in week t predict his performance in
week t+1, controlling for week-t performance and season-to-date form?</dd>
<dt>Data</dt><dd>Public Reddit comments from team subreddits (each counts only its own team's
players), scored for sentiment toward the named player; weekly player stats from the public
MLB Stats API. Only aggregate results are published; no text is shown or stored here.</dd>
<dt>Model</dt><dd>Hierarchical regression per metric, fit by a Gibbs sampler written from
scratch: each (fanbase, player) gets its own sentiment slope, drawn from a fanbase-level mean
&#956; that is the reported effect.</dd>
<dt>Null</dt><dd>Within-player permutation of sentiment across weeks, model refit each time;
two-sided p.</dd>
<dt>Corrections</dt><dd>Benjamini-Hochberg (and Holm) within separate families: per-metric
tests together, the composite index on its own.</dd>
</dl>
</section>
<section aria-labelledby="h-caveats">
<h2 id="h-caveats">Caveats</h2>
<ul>
<li>One season, {len(fanbases)} fanbase{"s" if len(fanbases) != 1 else ""}, \
{min(r["n_units"] for r in snap["rows"])} to {max(r["n_units"] for r in snap["rows"])} players per \
test: small samples, wide intervals.</li>
<li>The null breaks any sentiment-performance link, so it tests "sentiment carries no
information", not "sentiment merely reacts to last week".</li>
<li>A negative effect can be regression to the mean that fans fail to anticipate, not fans
causing anything.</li>
<li>Sentiment is from an automatic scorer and name matching; both make errors.</li>
<li>Weekly thresholds and composite weights are documented assumptions, not tuned values.</li>
<li>Some chains show R-hat up to {max(r["rhat_max"] for r in snap["rows"]):.2f}; values above
1.01 suggest longer sampling would tighten estimates.</li>
</ul>
</section>
<footer>
<p>Code, method and data contract: <a href="{REPO_URL}">github.com/SebastianFrazier26/\
mlb-fan-sentiment</a> &middot; <a href="{REPO_URL}#method-fansent-correlate">full method</a>
&middot; <a href="data.json">snapshot JSON</a> &middot; Apache-2.0. Snapshot
{E(snap["created"])}.</p>
</footer>
</main>
</body>
</html>
"""


def build_site(snapshot_path: Path, out_dir: Path) -> Path:
    snap = load_snapshot(snapshot_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "index.html"
    index.write_text(render_html(snap), encoding="utf-8")
    (out_dir / "data.json").write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
    return index
