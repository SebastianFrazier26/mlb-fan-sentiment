"""`fansent correlate` (hierarchical method): panel -> per-metric model + permutation null ->
multiple-comparison correction -> analysis_{fanbases,units,oaa}.csv."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .inference import InferenceConfig, MetricResult, analyze_metric
from .multitest import benjamini_hochberg, holm
from .panel import (
    PanelConfig,
    build_oaa_panel,
    build_weekly_panel,
    sentiment_periods,
)
from .util import DATA_DIR, OAA_CSV, SENTIMENT_CSV, STATS_CSV

FANBASES_OUT = DATA_DIR / "analysis_fanbases.csv"
UNITS_OUT = DATA_DIR / "analysis_units.csv"
OAA_OUT = DATA_DIR / "analysis_oaa.csv"
REDDIT_ATTRIBUTION_CSV = DATA_DIR / "reddit_attribution.csv"
NEWS_HEADLINES_CSV = DATA_DIR / "news_headlines.csv"

FANBASE_COLS = [
    "metric",
    "fanbase",
    "mu_mean",
    "ci_low",
    "ci_high",
    "p_pos",
    "perm_p",
    "q_bh",
    "p_holm",
    "n_units",
    "n_obs",
    "rhat_max",
]


@dataclass
class Outputs:
    fanbases: pd.DataFrame
    units: pd.DataFrame
    oaa_fanbases: pd.DataFrame
    oaa_units: pd.DataFrame
    skipped: list[str]
    seconds: float


def _read_optional(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, dtype=str) if path.exists() else None


def correct(df: pd.DataFrame) -> pd.DataFrame:
    """Add q_bh and p_holm over all (metric x fanbase) tests in df."""
    out = df.copy()
    out["q_bh"] = benjamini_hochberg(out["perm_p"].to_numpy())
    out["p_holm"] = holm(out["perm_p"].to_numpy())
    return out


def analyze(
    stats: pd.DataFrame,
    scores: pd.DataFrame,
    sidecars: list[pd.DataFrame],
    oaa: pd.DataFrame | None,
    pcfg: PanelConfig,
    icfg: InferenceConfig,
    metrics: list[str] | None = None,
    include_composite: bool = True,
) -> Outputs:
    t0 = time.perf_counter()
    weekly_sent, monthly_sent = sentiment_periods(scores, sidecars, pcfg)
    panel = build_weekly_panel(stats, weekly_sent, pcfg, metrics, include_composite)
    results: list[MetricResult] = []
    skipped: list[str] = []
    for metric in sorted(panel["metric"].unique()):
        try:
            results.append(analyze_metric(panel, metric, icfg))
        except ValueError as e:
            skipped.append(f"{metric}: {e}")
    if oaa is not None:
        oaa_panel = build_oaa_panel(oaa, monthly_sent, pcfg)
        if len(oaa_panel):
            try:
                results.append(analyze_metric(oaa_panel, "OAA", icfg))
            except ValueError as e:
                skipped.append(f"OAA: {e}")
    if not results:
        raise SystemExit(
            "No (metric, fanbase) had enough data; lower --min-weeks/--min-mentions or collect more."
        )
    fb = correct(pd.concat([r.fanbases for r in results], ignore_index=True))[FANBASE_COLS]
    units = pd.concat([r.units for r in results], ignore_index=True)
    is_oaa = fb["metric"] == "OAA"
    return Outputs(
        fanbases=fb[~is_oaa].reset_index(drop=True),
        units=units[units["metric"] != "OAA"].reset_index(drop=True),
        oaa_fanbases=fb[is_oaa].reset_index(drop=True),
        oaa_units=units[units["metric"] == "OAA"].reset_index(drop=True),
        skipped=skipped,
        seconds=time.perf_counter() - t0,
    )


def format_table(fb: pd.DataFrame, correction: str, alpha: float) -> str:
    col = "q_bh" if correction == "bh" else "p_holm"
    t = fb.copy()
    t["sig"] = (t[col] < alpha).map({True: "*", False: ""})
    for c in ("mu_mean", "ci_low", "ci_high"):
        t[c] = t[c].map("{:+.3f}".format)
    for c in ("p_pos", "perm_p", "q_bh", "p_holm", "rhat_max"):
        t[c] = t[c].map("{:.3f}".format)
    return t.to_string(index=False)


def run(
    pcfg: PanelConfig,
    icfg: InferenceConfig,
    metrics: list[str] | None,
    include_composite: bool,
    oaa_path: Path | None,
    correction: str,
    alpha: float,
) -> None:
    stats = pd.read_csv(STATS_CSV, dtype={"player_id": str})
    if not SENTIMENT_CSV.exists():
        raise SystemExit(f"{SENTIMENT_CSV} not found — run `fansent score` first.")
    scores = pd.read_csv(SENTIMENT_CSV, dtype={"text_id": str, "player_id": str})
    sidecars = [_read_optional(REDDIT_ATTRIBUTION_CSV), _read_optional(NEWS_HEADLINES_CSV)]
    path = oaa_path or (OAA_CSV if OAA_CSV.exists() else None)
    oaa = pd.read_csv(path, dtype={"player_id": str}) if path else None

    out = analyze(
        stats, scores, [s for s in sidecars if s is not None], oaa, pcfg, icfg, metrics,
        include_composite,
    )  # fmt: skip
    out.fanbases.to_csv(FANBASES_OUT, index=False)
    out.units.to_csv(UNITS_OUT, index=False)
    if len(out.oaa_fanbases):
        out.oaa_units.to_csv(OAA_OUT.with_name("analysis_oaa_units.csv"), index=False)
        out.oaa_fanbases.to_csv(OAA_OUT, index=False)

    print(
        f"\n=== Fanbase effect mu_k: sentiment_t -> better performance t+1 ({correction} < {alpha}) ==="
    )
    print(format_table(pd.concat([out.fanbases, out.oaa_fanbases]), correction, alpha))
    for s in out.skipped:
        print(f"skipped {s}")
    print(f"\n{FANBASES_OUT}, {UNITS_OUT}" + (f", {OAA_OUT}" if len(out.oaa_fanbases) else ""))
    print(f"mu_k is in sd(next-period performance) per sd(sentiment); {out.seconds:.0f}s.")
