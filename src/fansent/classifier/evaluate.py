"""Metrics, grouped k-fold CV tuning, side-by-side report, and the clf-eval / clf-train drivers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .data import LABELS, Example, load_labeled
from .pipeline import CLASSES, ClfConfig, SentimentClassifier
from .scorers import ClfScorer, LexiconScorer, VaderScorer
from .text import Masker

LABEL_VALUE = np.array([-1.0, 0.0, 1.0])  # CLASSES order: neg, neu, pos


def y_of(examples: list[Example]) -> np.ndarray:
    return np.array([CLASSES.index(e.label) for e in examples])


def confusion(y: np.ndarray, pred: np.ndarray, k: int = 3) -> np.ndarray:
    m = np.zeros((k, k), dtype=int)
    np.add.at(m, (y, pred), 1)
    return m


def per_class(cm: np.ndarray) -> np.ndarray:
    """Rows = classes; columns = precision, recall, f1, support."""
    tp = np.diag(cm).astype(float)
    pred, true = cm.sum(axis=0), cm.sum(axis=1)
    p = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    r = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return np.column_stack([p, r, f, true])


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(per_class(confusion(y, pred))[:, 2].mean())


def ece(proba: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[tuple]]:
    """Expected calibration error on top-class confidence; also the reliability table."""
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    hit = (pred == y).astype(float)
    edges = np.linspace(1 / proba.shape[1], 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total, rows = 0.0, []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        c, a = conf[mask].mean(), hit[mask].mean()
        total += mask.mean() * abs(c - a)
        rows.append((edges[b], edges[b + 1], int(mask.sum()), float(c), float(a)))
    return float(total), rows


@dataclass
class Metrics:
    n: int
    accuracy: float
    macro_f1: float
    per_class: np.ndarray
    cm: np.ndarray
    spearman: float
    ece: float | None
    reliability: list[tuple]


def evaluate(scorer, examples: list[Example]) -> Metrics:
    y = y_of(examples)
    pred = scorer.predict(examples)
    cm = confusion(y, pred)
    rho = spearmanr(scorer.score(examples), LABEL_VALUE[y]).statistic
    proba = scorer.proba(examples)
    e, rel = ece(proba, y) if proba is not None else (None, [])
    return Metrics(
        len(y),
        float((pred == y).mean()),
        float(per_class(cm)[:, 2].mean()),
        per_class(cm),
        cm,
        float(rho),
        e,
        rel,
    )


def format_report(results: dict[str, Metrics]) -> str:
    lines = ["scorer     n  accuracy  macro-F1  spearman  ECE"]
    for name, m in results.items():
        ece_s = "n/a" if m.ece is None else f"{m.ece:.3f}"
        lines.append(
            f"{name:<8}{m.n:>5}  {m.accuracy:8.3f}  {m.macro_f1:8.3f}  {m.spearman:8.3f}  {ece_s}"
        )
    for name, m in results.items():
        lines += ["", f"== {name} ==", "class     P      R      F1   support"]
        for lab, row in zip(LABELS, m.per_class, strict=True):
            lines.append(f"{lab:<5}  {row[0]:.3f}  {row[1]:.3f}  {row[2]:.3f}  {int(row[3]):>6}")
        lines.append("confusion (rows=true, cols=pred; neg neu pos)")
        lines += ["  " + " ".join(f"{v:>5}" for v in r) for r in m.cm]
        if m.reliability:
            lines.append(f"reliability (ECE {m.ece:.3f}): conf-bin  n  mean-conf  accuracy")
            lines += [
                f"  {lo:.2f}-{hi:.2f} {n:>5}  {c:.3f}  {a:.3f}" for lo, hi, n, c, a in m.reliability
            ]
    return "\n".join(lines)


def grouped_folds(examples: list[Example], k: int, seed: int) -> list[np.ndarray]:
    """Folds that keep identical texts together (the same text can appear for several players,
    and a split across train/test would leak), stratified by each group's first label."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for i, e in enumerate(examples):
        groups.setdefault(e.text, []).append(i)
    by_label: dict[str, list[list[int]]] = {}
    for g in groups.values():
        by_label.setdefault(examples[g[0]].label, []).append(g)
    folds: list[list[int]] = [[] for _ in range(k)]
    j = 0
    for label in sorted(by_label):
        gs = by_label[label]
        for gi in rng.permutation(len(gs)):
            folds[j % k].extend(gs[gi])
            j += 1
    return [np.array(sorted(f)) for f in folds]


def _span(s: str) -> tuple[int, int]:
    lo, _, hi = s.partition("-")
    return int(lo), int(hi or lo)


@dataclass(frozen=True)
class Grid:
    Cs: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0)
    windows: tuple[int, ...] = (0, 3, 5)
    char_ngrams: tuple[str, ...] = ("0-0", "3-5")
    word_ngrams: tuple[str, ...] = ("1-2",)


def tune(
    examples: list[Example],
    masker: Masker,
    grid: Grid | None = None,
    base: ClfConfig | None = None,
    folds: int = 5,
    seed: int = 0,
    log=print,
) -> tuple[ClfConfig, list[tuple[ClfConfig, float]]]:
    """Grouped k-fold CV over the grid, scored by mean macro-F1. Features are fitted once per
    (feature config, fold) and reused across C."""
    grid, base = grid or Grid(), base or ClfConfig()
    splits = grouped_folds(examples, folds, seed)
    y = y_of(examples)
    results: list[tuple[ClfConfig, float]] = []
    for wn, cn, win in itertools.product(grid.word_ngrams, grid.char_ngrams, grid.windows):
        feats = replace(base.features, word_ngrams=_span(wn), char_ngrams=_span(cn), window=win)
        scores: dict[float, list[float]] = {c: [] for c in grid.Cs}
        for held in splits:
            mask = np.zeros(len(examples), dtype=bool)
            mask[held] = True
            train = [e for e, m in zip(examples, mask, strict=True) if not m]
            test = [e for e, m in zip(examples, mask, strict=True) if m]
            for c in grid.Cs:
                model = SentimentClassifier(replace(base, C=c, features=feats), masker)
                model.fit(train)
                pred = model.predict_proba(test).argmax(axis=1)
                scores[c].append(macro_f1(y[mask], pred))
        for c, s in scores.items():
            cfg = replace(base, C=c, features=feats)
            results.append((cfg, float(np.mean(s))))
            log(f"cv word={wn} char={cn} win={win} C={c:g}: macro-F1 {np.mean(s):.3f}")
    best = max(results, key=lambda r: r[1])
    return best[0], results


def load_players(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _prepare(train: list[Path], players: Path | None, cv: int, grid: Grid, seed: int):
    examples, n_na = load_labeled(train)
    if not examples:
        raise SystemExit("no trainable rows (all na or empty)")
    print(f"train: {len(examples)} rows ({n_na} 'na' rows excluded)")
    masker = Masker(load_players(players))
    cfg = ClfConfig()
    if cv > 1:
        cfg, _ = tune(examples, masker, grid, folds=cv, seed=seed)
        print(
            f"best: C={cfg.C:g} window={cfg.features.window} "
            f"char={cfg.features.char_ngrams} word={cfg.features.word_ngrams}"
        )
    return SentimentClassifier(cfg, masker).fit(examples)


def run_eval(train, test, players, cv, grid, seed, save_model=None) -> None:
    model = _prepare(train, players, cv, grid, seed)
    if save_model:
        model.save(save_model)
    held, n_na = load_labeled([test])
    print(f"test: {len(held)} rows ({n_na} 'na' rows excluded)\n")
    scorers = [VaderScorer(), LexiconScorer(), ClfScorer(model)]
    print(format_report({s.name: evaluate(s, held) for s in scorers}))


def run_train(train, out, players, cv, grid, seed) -> None:
    model = _prepare(train, players, cv, grid, seed)
    model.save(out)
    print(f"saved {out}")
