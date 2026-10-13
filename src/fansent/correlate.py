"""`fansent correlate` (hierarchical method): panel -> per-metric model + permutation null ->
multiple-comparison correction -> analysis_{fanbases,units,oaa}.csv."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .inference import InferenceConfig, MetricResult, analyze_metric
from .multitest import benjamini_hochberg, holm
from .panel import (
    PanelConfig,
    build_oaa_panel,
    build_weekly_panel,
    sentiment_periods,
)
from .util import DEFAULT_DATA_DIR, OAA_FILE, SENTIMENT_FILE, STATS_FILE

FANBASES_OUT = "analysis_fanbases.csv"
UNITS_OUT = "analysis_units.csv"
OAA_OUT = "analysis_oaa.csv"
OAA_UNITS_OUT = "analysis_oaa_units.csv"
REDDIT_ATTRIBUTION_FILE = "reddit_attribution.csv"
NEWS_HEADLINES_FILE = "news_headlines.csv"

FAMILY_METRICS = "metrics"
FAMILY_COMPOSITE = "COMPOSITE"
FAMILY_OAA = "OAA"
FAMILY_EXPLORATORY = "exploratory"
FAMILY_ORDER = [FAMILY_COMPOSITE, FAMILY_METRICS, FAMILY_OAA, FAMILY_EXPLORATORY]
# CS_PCT only has catchers, so it is too thin to correct alongside the rest.
EXPLORATORY_METRICS = frozenset({"CS_PCT"})

FANBASE_COLS = [
    "family",
    "exploratory",
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


def family_of(metric: str) -> str:
    if metric in EXPLORATORY_METRICS:
        return FAMILY_EXPLORATORY
    if metric in (FAMILY_COMPOSITE, FAMILY_OAA):
        return metric
    return FAMILY_METRICS


def correct(df: pd.DataFrame) -> pd.DataFrame:
    """Assign `family`/`exploratory` and add q_bh and p_holm within each family only.

    Exploratory rows keep raw perm_p and get NA corrections."""
    out = df.copy()
    out["family"] = out["metric"].map(family_of)
    out["exploratory"] = out["family"] == FAMILY_EXPLORATORY
    out["q_bh"] = np.nan
    out["p_holm"] = np.nan
    for idx in out[~out["exploratory"]].groupby("family").groups.values():
        p = out.loc[idx, "perm_p"].to_numpy()
        out.loc[idx, "q_bh"] = benjamini_hochberg(p)
        out.loc[idx, "p_holm"] = holm(p)
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
        t[c] = t[c].map("{:.3f}".format).replace("nan", "NA")
    blocks = []
    for fam in FAMILY_ORDER:
        part = t[t["family"] == fam].drop(columns=["family", "exploratory"])
        if len(part):
            note = " (raw perm_p, not corrected)" if fam == FAMILY_EXPLORATORY else ""
            blocks.append(f"-- family: {fam}{note}\n{part.to_string(index=False)}")
    return "\n\n".join(blocks)


def run(
    pcfg: PanelConfig,
    icfg: InferenceConfig,
    metrics: list[str] | None,
    include_composite: bool,
    oaa_path: Path | None,
    correction: str,
    alpha: float,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> None:
    stats = pd.read_csv(data_dir / STATS_FILE, dtype={"player_id": str})
    sentiment_csv = data_dir / SENTIMENT_FILE
    if not sentiment_csv.exists():
        raise SystemExit(f"{sentiment_csv} not found — run `fansent score` first.")
    scores = pd.read_csv(sentiment_csv, dtype={"text_id": str, "player_id": str})
    sidecars = [
        _read_optional(data_dir / REDDIT_ATTRIBUTION_FILE),
        _read_optional(data_dir / NEWS_HEADLINES_FILE),
    ]
    default_oaa = data_dir / OAA_FILE
    path = oaa_path or (default_oaa if default_oaa.exists() else None)
    oaa = pd.read_csv(path, dtype={"player_id": str}) if path else None

    out = analyze(
        stats, scores, [s for s in sidecars if s is not None], oaa, pcfg, icfg, metrics,
        include_composite,
    )  # fmt: skip
    fanbases_out, units_out, oaa_out = (
        data_dir / FANBASES_OUT,
        data_dir / UNITS_OUT,
        data_dir / OAA_OUT,
    )
    out.fanbases.to_csv(fanbases_out, index=False)
    out.units.to_csv(units_out, index=False)
    if len(out.oaa_fanbases):
        out.oaa_units.to_csv(data_dir / OAA_UNITS_OUT, index=False)
        out.oaa_fanbases.to_csv(oaa_out, index=False)

    print(
        f"\n=== Fanbase effect mu_k: sentiment_t -> better performance t+1 ({correction} < {alpha}) ==="
    )
    print(format_table(pd.concat([out.fanbases, out.oaa_fanbases]), correction, alpha))
    for s in out.skipped:
        print(f"skipped {s}")
    print(f"\n{fanbases_out}, {units_out}" + (f", {oaa_out}" if len(out.oaa_fanbases) else ""))
    print(f"mu_k is in sd(next-period performance) per sd(sentiment); {out.seconds:.0f}s.")
