"""TF-IDF feature blocks (own implementation): word n-grams, char n-grams, target window.

Vocabulary, not hashing: the corpus is small (thousands of texts), so a vocab with min_df costs
little, makes idf exact and collision-free, and lets us read the learned weights per n-gram.
Hashing would only pay off for a vocabulary that does not fit in memory.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from functools import cache, lru_cache

import numpy as np
from scipy import sparse

from .embeddings import TokenVectors, mean_vector
from .overlay import LexiconVader
from .text import OTH, TGT


@dataclass(frozen=True)
class FeatureConfig:
    word_ngrams: tuple[int, int] = (1, 2)
    char_ngrams: tuple[int, int] = (3, 5)  # (0, 0) disables the block
    window: int = 4  # +-k tokens around __TGT__; 0 disables the block
    window_ngrams: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_features: int = 50000  # per block
    # Optional blocks. Weights scale the block after L2-normalisation, i.e. they act as a
    # per-block inverse regularisation strength (the penalty is on the weights, not the inputs).
    emb: bool = False
    dense_window: int = 4  # +-k for the lex/emb window features; independent of `window`
    emb_weight: float = 1.0
    lex: bool = False
    lex_weight: float = 1.0
    parent: bool = False
    parent_weight: float = 1.0

    def blocks(self) -> list[str]:
        out = ["word"]
        if self.char_ngrams[1] > 0:
            out.append("char")
        if self.window > 0:
            out.append("win")
        if self.parent:
            out.append("parent")
        return out


@dataclass(frozen=True)
class Prepped:
    """One example after masking and tokenising (parent text masked for the same target)."""

    tokens: list[str]
    masked: str
    parent_tokens: list[str]


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


def window_spans(tokens: list[str], k: int) -> list[list[int]]:
    """[lo, hi) spans of +-k tokens around every __TGT__, overlapping ones merged."""
    spans: list[list[int]] = []
    for i, t in enumerate(tokens):
        if t == TGT:
            lo, hi = max(0, i - k), min(len(tokens), i + k + 1)
            if spans and lo <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], hi)
            else:
                spans.append([lo, hi])
    return spans


def window_terms(tokens: list[str], cfg: FeatureConfig) -> list[str]:
    """n-grams inside +-k of every __TGT__, prefixed so they never collide with word features.

    Overlapping windows are merged first so a token is not counted twice, and n-grams never
    span the gap between two disjoint windows.
    """
    out: list[str] = []
    for lo, hi in window_spans(tokens, cfg.window):
        out.extend("win:" + g for g in _ngrams(tokens[lo:hi], *cfg.window_ngrams))
    return out


def parent_terms(parent_tokens: list[str], cfg: FeatureConfig) -> list[str]:
    return ["par:" + g for g in _ngrams(parent_tokens, *cfg.word_ngrams)]


_EXTRACT = {"word": word_terms, "char": char_terms, "win": window_terms}


@lru_cache(maxsize=1)
def _default_lexicon() -> LexiconVader:
    return LexiconVader()


@cache
def _lex_row(masked: str, window_text: str) -> tuple[float, float, float, float]:
    lex = _default_lexicon()
    pos, neg = lex.hits(masked)
    win = lex.compound(window_text) if window_text else 0.0
    return lex.compound(masked), win, math.log1p(pos), math.log1p(neg)


def lexicon_features(p: Prepped, k: int) -> np.ndarray:
    """[compound(comment), compound(target window), log1p(#pos hits), log1p(#neg hits)]."""
    toks = p.tokens
    win = " ".join(" ".join(toks[lo:hi]) for lo, hi in window_spans(toks, k))
    return np.array(_lex_row(p.masked, win))


def embedding_features(p: Prepped, k: int, tv: TokenVectors) -> np.ndarray:
    """[mean vector of the comment, mean vector of the +-k target window], placeholders dropped."""
    toks = p.tokens
    win = [t for lo, hi in window_spans(toks, k) for t in toks[lo:hi]]
    keep = lambda ts: [t for t in ts if t not in (TGT, OTH)]
    return np.concatenate([mean_vector(keep(toks), tv), mean_vector(keep(win), tv)])


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
    def __init__(self, cfg: FeatureConfig, vectors: TokenVectors | None = None) -> None:
        if cfg.emb and vectors is None:
            raise ValueError("emb block enabled but no embeddings were provided")
        self.cfg = cfg
        self.vectors = vectors
        self.blocks: dict[str, TfidfBlock] = {
            b: TfidfBlock(cfg.min_df, cfg.max_features) for b in cfg.blocks()
        }

    def _terms(self, docs: list[Prepped]) -> dict[str, list[list[str]]]:
        out = {}
        for b in self.blocks:
            if b == "parent":
                out[b] = [parent_terms(d.parent_tokens, self.cfg) for d in docs]
            else:
                out[b] = [_EXTRACT[b](d.tokens, self.cfg) for d in docs]
        return out

    def fit_transform(self, docs: list[Prepped]) -> sparse.csr_matrix:
        terms = self._terms(docs)
        for b, blk in self.blocks.items():
            blk.fit(terms[b])
        return self.transform(docs)

    def transform(self, docs: list[Prepped]) -> sparse.csr_matrix:
        terms = self._terms(docs)
        cfg = self.cfg
        mats = []
        for b, blk in self.blocks.items():
            m = blk.transform(terms[b])
            if b == "parent":
                is_reply = sparse.csr_matrix(
                    np.array([[1.0 if d.parent_tokens else 0.0] for d in docs])
                )
                m = sparse.hstack([m, is_reply]) * cfg.parent_weight
            mats.append(m)
        if cfg.lex:
            k = cfg.dense_window
            lex = np.array([lexicon_features(d, k) for d in docs]).reshape(len(docs), 4)
            mats.append(sparse.csr_matrix(lex * cfg.lex_weight))
        if cfg.emb:
            emb = np.array([embedding_features(d, cfg.dense_window, self.vectors) for d in docs])
            mats.append(sparse.csr_matrix(emb.reshape(len(docs), -1) * cfg.emb_weight))
        return sparse.hstack(mats).tocsr()
