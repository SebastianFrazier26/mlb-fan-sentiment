"""Adapter: long-format panel DataFrame -> the plain arrays `hierarchical.fit` takes.
Column names are arguments so this stays independent of the stats.csv schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

POOLED = "pooled"
ESTIMATORS = ("pooled", "mundlak")
MUNDLAK_EXTRAS = ("between", "perf_mean", "std_mean")


@dataclass
class ModelInputs:
    y_next: np.ndarray
    perf: np.ndarray
    std: np.ndarray
    sent: np.ndarray
    unit_idx: np.ndarray
    fanbase_idx_of_unit: np.ndarray
    weights: np.ndarray | None
    fanbase_names: list[str]
    unit_names: list[tuple[str, str]]
    # Mundlak estimator: sent/perf/std above are then within-unit deviations and extras holds
    # the unit means (sent, perf, std), fed to the fit as fixed-coefficient covariates.
    extras: np.ndarray | None = None
    extra_names: tuple[str, ...] = ()

    def fit_kwargs(self) -> dict:
        return {
            "extra": self.extras,
            "extra_names": list(self.extra_names),
            "fanbase_names": self.fanbase_names,
            "unit_names": [f"{f}|{p}" for f, p in self.unit_names],
        }


def panel_to_inputs(
    panel: pd.DataFrame,
    *,
    y_next: str,
    perf: str,
    std: str,
    sent: str,
    fanbase: str = "fanbase",
    player: str = "player_id",
    weight: str | None = None,
    pool: bool = False,
    estimator: str = "pooled",
) -> ModelInputs:
    """pool=True puts every unit in one group named "pooled" (single mu). Units stay
    (fanbase, player), so a player in two fanbases is two units sharing the same outcomes.

    estimator="mundlak" splits sent, perf and std into unit mean + within-unit deviation (plain
    unweighted means, so permuting within a unit leaves the means fixed exactly): the deviations
    replace the raw columns and the three unit means become `extras`. gamma_u and mu then act on
    the within part; the coefficient on the sentiment unit mean is the between-unit association."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; expected one of {ESTIMATORS}")
    cols = [y_next, perf, std, sent, fanbase, player] + ([weight] if weight else [])
    df = panel.dropna(subset=cols)
    units = df.groupby([fanbase, player], sort=True).ngroup().to_numpy()
    unit_df = pd.DataFrame(
        {fanbase: df[fanbase], player: df[player], "_u": units}
    ).drop_duplicates()
    unit_df = unit_df.sort_values("_u")
    if pool:
        fb_names = [POOLED]
        fb_idx = np.zeros(len(unit_df), dtype=int)
    else:
        fb_names = sorted(unit_df[fanbase].astype(str).unique())
        fb_idx = unit_df[fanbase].astype(str).map({n: i for i, n in enumerate(fb_names)}).to_numpy()
    perf_a, std_a, sent_a = (df[c].to_numpy(float) for c in (perf, std, sent))
    extras, extra_names = None, ()
    if estimator == "mundlak":
        n = np.maximum(np.bincount(units), 1)
        means = [(np.bincount(units, weights=a) / n)[units] for a in (sent_a, perf_a, std_a)]
        sent_a, perf_a, std_a = (a - m for a, m in zip((sent_a, perf_a, std_a), means, strict=True))
        extras, extra_names = np.column_stack(means), MUNDLAK_EXTRAS
    return ModelInputs(
        y_next=df[y_next].to_numpy(float),
        perf=perf_a,
        std=std_a,
        sent=sent_a,
        unit_idx=units,
        fanbase_idx_of_unit=fb_idx,
        weights=df[weight].to_numpy(float) if weight else None,
        fanbase_names=fb_names,
        unit_names=list(zip(unit_df[fanbase].astype(str), unit_df[player].astype(str))),
        extras=extras,
        extra_names=extra_names,
    )
