"""Adapter: long-format panel DataFrame -> the plain arrays `hierarchical.fit` takes.
Column names are arguments so this stays independent of the stats.csv schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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

    def fit_kwargs(self) -> dict:
        return {
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
) -> ModelInputs:
    cols = [y_next, perf, std, sent, fanbase, player] + ([weight] if weight else [])
    df = panel.dropna(subset=cols)
    units = df.groupby([fanbase, player], sort=True).ngroup().to_numpy()
    unit_df = pd.DataFrame(
        {fanbase: df[fanbase], player: df[player], "_u": units}
    ).drop_duplicates()
    unit_df = unit_df.sort_values("_u")
    fb_names = sorted(unit_df[fanbase].astype(str).unique())
    fb_idx = unit_df[fanbase].astype(str).map({n: i for i, n in enumerate(fb_names)}).to_numpy()
    return ModelInputs(
        y_next=df[y_next].to_numpy(float),
        perf=df[perf].to_numpy(float),
        std=df[std].to_numpy(float),
        sent=df[sent].to_numpy(float),
        unit_idx=units,
        fanbase_idx_of_unit=fb_idx,
        weights=df[weight].to_numpy(float) if weight else None,
        fanbase_names=fb_names,
        unit_names=list(zip(unit_df[fanbase].astype(str), unit_df[player].astype(str))),
    )
