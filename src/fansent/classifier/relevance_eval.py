"""Relevance model: held-out split, grouped CV tuning, threshold selection, reports, drivers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .data import Example, load_labeled
from .evaluate import grouped_folds, load_players
from .relevance import (
    NA,
    REL,
    RelConfig,
    RelevanceClassifier,
    RelPrepped,
    WeeklyRoster,
    load_weekly_roster,
)

OPERATING_POINTS = ("f1", "precision")


@dataclass(frozen=True)
class RelGrid:
    Cs: tuple[float, ...] = (1.0, 3.0, 10.0)
    dense_weights: tuple[float, ...] = (1.0, 2.0)
    ctx_weights: tuple[float, ...] = (0.5, 1.0)


def y_na(examples: list[Example]) -> np.ndarray:
    return np.array([e.label == "na" for e in examples])


def _strata(e: Example) -> tuple[str, bool]:
    return e.subreddit, e.label == "na"


def split_holdout(
    examples: list[Example], frac: float, seed: int
) -> tuple[list[Example], list[Example]]:
    """Hold out ~``frac`` of the rows, whole texts at a time, stratified by subreddit x na."""
    if frac <= 0:
        return examples, []
    k = max(2, round(1 / frac))
    held = set(grouped_folds(examples, k, seed, _strata)[0].tolist())
    train = [e for i, e in enumerate(examples) if i not in held]
    test = [e for i, e in enumerate(examples) if i in held]
    return train, test


def average_precision(is_na: np.ndarray, p_na: np.ndarray) -> float:
    order = np.argsort(-p_na, kind="stable")
    hit = is_na[order]
    prec = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    return float((prec * hit).sum() / max(1, hit.sum()))


def prf(is_na: np.ndarray, pred_na: np.ndarray) -> tuple[float, float, float, int, int]:
    """(precision, recall, F1, #true na, #predicted na) for the na class."""
    tp = int((is_na & pred_na).sum())
    n_true, n_pred = int(is_na.sum()), int(pred_na.sum())
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_true if n_true else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, n_true, n_pred


def pr_points(is_na: np.ndarray, p_na: np.ndarray) -> list[tuple[float, float, float, float]]:
    """(threshold, precision, recall, F1) at every distinct score; rows with P(na) >= threshold
    are dropped."""
    out = []
    for t in sorted(set(p_na.tolist())):
        p, r, f, _, _ = prf(is_na, p_na >= t)
        out.append((t, p, r, f))
    return out


def pick_threshold(
    is_na: np.ndarray, p_na: np.ndarray, point: str = "f1", min_precision: float = 0.8
) -> float:
    pts = pr_points(is_na, p_na)
    if point == "f1":
        return max(pts, key=lambda x: (x[3], x[0]))[0]
    if point == "precision":
        ok = [x for x in pts if x[1] >= min_precision]
        # No threshold reaches the target precision: drop nothing rather than guess.
        return min(ok, key=lambda x: x[0])[0] if ok else float("inf")
    raise ValueError(f"operating point must be one of {OPERATING_POINTS}")


def _oof_p_na(
    prepped: list[RelPrepped],
    y: np.ndarray,
    splits: list[np.ndarray],
    cfg: RelConfig,
    players: list[dict],
    weekly: WeeklyRoster | None = None,
) -> np.ndarray:
    oof = np.zeros(len(y))
    for held in splits:
        mask = np.zeros(len(y), dtype=bool)
        mask[held] = True
        model = RelevanceClassifier(players, cfg, weekly)
        model.fit_prepped([d for d, m in zip(prepped, mask, strict=True) if not m], y[~mask])
        oof[mask] = model.p_na_prepped([d for d, m in zip(prepped, mask, strict=True) if m])
    return oof


def tune(
    examples: list[Example],
    players: list[dict],
    grid: RelGrid | None = None,
    folds: int = 5,
    seed: int = 0,
    log=print,
    weekly: WeeklyRoster | None = None,
) -> tuple[RelConfig, np.ndarray]:
    """Grouped CV over the grid scored by mean average precision of na. Returns the best config
    and its out-of-fold P(na) (the threshold is chosen on these, never on held-out data)."""
    grid = grid or RelGrid()
    prepped = RelevanceClassifier(players, weekly_teams=weekly).prep(examples)
    is_na = y_na(examples)
    y = np.where(is_na, NA, REL)
    splits = grouped_folds(examples, folds, seed, _strata)
    best: tuple[float, RelConfig, np.ndarray] | None = None
    for c, dw, cw in itertools.product(grid.Cs, grid.dense_weights, grid.ctx_weights):
        cfg = replace(RelConfig(), C=c, dense_weight=dw, ctx_weight=cw)
        oof = _oof_p_na(prepped, y, splits, cfg, players, weekly)
        ap = average_precision(is_na, oof)
        log(f"cv C={c:g} dense={dw:g} ctx={cw:g}: AP(na) {ap:.3f}")
        if best is None or ap > best[0]:
            best = (ap, cfg, oof)
    return best[1], best[2]


def format_pr(is_na: np.ndarray, p_na: np.ndarray, points: int = 12) -> str:
    pts = pr_points(is_na, p_na)
    idx = sorted({int(i) for i in np.linspace(0, len(pts) - 1, points)})
    lines = ["threshold  precision  recall     F1   (drop rows with P(na) >= threshold)"]
    lines += [
        f"{pts[i][0]:9.3f}  {pts[i][1]:9.3f}  {pts[i][2]:6.3f}  {pts[i][3]:5.3f}" for i in idx
    ]
    return "\n".join(lines)


def format_by_sub(examples: list[Example], pred_na: np.ndarray) -> str:
    is_na = y_na(examples)
    subs = np.array([e.subreddit for e in examples])
    lines = ["subreddit      n  true-na  pred-na  precision  recall     F1"]
    for sub in [*sorted(set(subs.tolist())), None]:
        m = np.ones(len(subs), dtype=bool) if sub is None else subs == sub
        p, r, f, nt, npred = prf(is_na[m], pred_na[m])
        lines.append(
            f"{sub or 'ALL':<10}{int(m.sum()):>6}  {nt:>7}  {npred:>7}  {p:9.3f}  {r:6.3f}  {f:5.3f}"
        )
    return "\n".join(lines)


def fit_with_threshold(
    examples: list[Example],
    players: list[dict],
    cv: int,
    grid: RelGrid,
    seed: int,
    point: str,
    min_precision: float,
    weekly: WeeklyRoster | None = None,
) -> RelevanceClassifier:
    """CV-tune, choose the threshold on out-of-fold scores, refit on all ``examples``."""
    cfg, oof = tune(examples, players, grid, cv, seed, weekly=weekly)
    is_na = y_na(examples)
    print(f"best: C={cfg.C:g} dense={cfg.dense_weight:g} ctx={cfg.ctx_weight:g}")
    print("\nout-of-fold precision-recall for na (training CV):")
    print(format_pr(is_na, oof))
    for name in OPERATING_POINTS:
        t = pick_threshold(is_na, oof, name, min_precision)
        p, r, f, _, _ = prf(is_na, oof >= t)
        print(f"  {name:<9} threshold {t:.3f}: precision {p:.3f} recall {r:.3f} F1 {f:.3f}")
    t = pick_threshold(is_na, oof, point, min_precision)
    print(f"operating point: {point} -> threshold {t:.3f}")
    return RelevanceClassifier(players, replace(cfg, threshold=t), weekly).fit(examples)


def run_eval(
    train: list[Path],
    test: Path,
    players_path: Path | None,
    cv: int,
    grid: RelGrid,
    seed: int,
    save_model: Path | None,
    holdout: float,
    point: str,
    min_precision: float,
) -> None:
    examples, n_na = load_labeled(train, keep_na=True)
    print(f"relevance: {len(examples)} rows ({n_na} na)")
    fit_set, held = split_holdout(examples, holdout, seed)
    print(f"held out {len(held)} rows (grouped by text, stratified by subreddit x na)\n")
    model = fit_with_threshold(
        fit_set,
        load_players(players_path),
        cv,
        grid,
        seed,
        point,
        min_precision,
        load_weekly_roster(players_path),
    )
    if held:
        print("\nheld-out Sonnet-labeled split:")
        print(format_by_sub(held, model.is_na(held)))
        print(format_pr(y_na(held), model.p_na(held)))
    human, _ = load_labeled([test], keep_na=True)
    print(f"\nhuman test ({test.name}, {len(human)} rows; not used for tuning):")
    print(format_by_sub(human, model.is_na(human)))
    print(format_pr(y_na(human), model.p_na(human)))
    if save_model:
        model.save(save_model)


def run_train(
    train: list[Path],
    out: Path,
    players_path: Path | None,
    cv: int,
    grid: RelGrid,
    seed: int,
    point: str,
    min_precision: float,
) -> None:
    examples, n_na = load_labeled(train, keep_na=True)
    print(f"relevance: {len(examples)} rows ({n_na} na)")
    model = fit_with_threshold(
        examples,
        load_players(players_path),
        cv,
        grid,
        seed,
        point,
        min_precision,
        load_weekly_roster(players_path),
    )
    model.save(out)
    print(f"saved {out} (drop threshold P(na) >= {model.config.threshold:.3f})")
