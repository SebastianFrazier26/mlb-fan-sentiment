"""TF-IDF feature blocks (own implementation): word n-grams, char n-grams, target window.

Vocabulary, not hashing: the corpus is small (thousands of texts), so a vocab with min_df costs
little, makes idf exact and collision-free, and lets us read the learned weights per n-gram.
Hashing would only pay off for a vocabulary that does not fit in memory.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .text import TGT


@dataclass(frozen=True)
class FeatureConfig:
    word_ngrams: tuple[int, int] = (1, 2)
    char_ngrams: tuple[int, int] = (3, 5)  # (0, 0) disables the block
    window: int = 4  # +-k tokens around __TGT__; 0 disables the block
    window_ngrams: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_features: int = 50000  # per block

    def blocks(self) -> list[str]:
        out = ["word"]
        if self.char_ngrams[1] > 0:
            out.append("char")
        if self.window > 0:
            out.append("win")
        return out


def _ngrams(seq: list[str], lo: int, hi: int) -> list[str]:
    return [" ".join(seq[i : i + n]) for n in range(lo, hi + 1) for i in range(len(seq) - n + 1)]


def word_terms(tokens: list[str], cfg: FeatureConfig) -> list[str]:
    return _ngrams(tokens, *cfg.word_ngrams)


def char_terms(tokens: list[str], cfg: FeatureConfig) -> list[str]:
    lo, hi = cfg.char_ngrams
    out = []
    for tok in tokens:
        if tok.startswith("__") or not tok[0].isalnum():
            continue
        padded = f" {tok} "
        for n in range(lo, hi + 1):
            out.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return out


def window_terms(tokens: list[str], cfg: FeatureConfig) -> list[str]:
    """n-grams inside +-k of every __TGT__, prefixed so they never collide with word features.

    Overlapping windows are merged first so a token is not counted twice, and n-grams never
    span the gap between two disjoint windows.
    """
    k = cfg.window
    spans: list[list[int]] = []
    for i, t in enumerate(tokens):
        if t == TGT:
            lo, hi = max(0, i - k), min(len(tokens), i + k + 1)
            if spans and lo <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], hi)
            else:
                spans.append([lo, hi])
    out: list[str] = []
    for lo, hi in spans:
        out.extend("win:" + g for g in _ngrams(tokens[lo:hi], *cfg.window_ngrams))
    return out


_EXTRACT = {"word": word_terms, "char": char_terms, "win": window_terms}


class TfidfBlock:
    """Sublinear tf (1 + ln tf) x smoothed idf (ln((1+n)/(1+df)) + 1), L2-normalised per row."""

    def __init__(self, min_df: int, max_features: int) -> None:
        self.min_df = min_df
        self.max_features = max_features
        self.terms: list[str] = []
        self.idf = np.zeros(0)
        self._index: dict[str, int] = {}

    def fit(self, docs: list[list[str]]) -> TfidfBlock:
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        kept = sorted(
            ((t, c) for t, c in df.items() if c >= self.min_df), key=lambda tc: (-tc[1], tc[0])
        )[: self.max_features]
        kept.sort()
        self.terms = [t for t, _ in kept]
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + c)) + 1.0 for _, c in kept])
        self._set_index()
        return self

    def _set_index(self) -> None:
        self._index = {t: i for i, t in enumerate(self.terms)}

    def transform(self, docs: list[list[str]]) -> sparse.csr_matrix:
        indptr, indices, data = [0], [], []
        for d in docs:
            counts = Counter(t for t in d if t in self._index)
            row_idx = np.fromiter(
                (self._index[t] for t in counts), dtype=np.int64, count=len(counts)
            )
            tf = np.array([1.0 + math.log(c) for c in counts.values()])
            vals = tf * self.idf[row_idx] if len(row_idx) else tf
            norm = np.linalg.norm(vals)
            if norm > 0:
                vals = vals / norm
            indices.append(row_idx)
            data.append(vals)
            indptr.append(indptr[-1] + len(row_idx))
        shape = (len(docs), len(self.terms))
        if not indices:
            return sparse.csr_matrix(shape)
        return sparse.csr_matrix(
            (np.concatenate(data), np.concatenate(indices), np.array(indptr)), shape=shape
        )


class Vectorizer:
    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        self.blocks: dict[str, TfidfBlock] = {
            b: TfidfBlock(cfg.min_df, cfg.max_features) for b in cfg.blocks()
        }

    def _terms(self, docs: list[list[str]]) -> dict[str, list[list[str]]]:
        return {b: [_EXTRACT[b](d, self.cfg) for d in docs] for b in self.blocks}

    def fit_transform(self, docs: list[list[str]]) -> sparse.csr_matrix:
        terms = self._terms(docs)
        for b, blk in self.blocks.items():
            blk.fit(terms[b])
        return self.transform(docs)

    def transform(self, docs: list[list[str]]) -> sparse.csr_matrix:
        terms = self._terms(docs)
        return sparse.hstack([blk.transform(terms[b]) for b, blk in self.blocks.items()]).tocsr()
