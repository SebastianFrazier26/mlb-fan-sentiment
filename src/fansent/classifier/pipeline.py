"""Masker + tokenizer + TF-IDF blocks + softmax regression, with save/load."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import softmax
from .data import LABELS, Example
from .embeddings import FastTextVectors, TokenVectors
from .features import FeatureConfig, Prepped, Vectorizer
from .text import Masker, tokenize

CLASSES = LABELS  # column order of predict_proba: neg, neu, pos


@dataclass(frozen=True)
class ClfConfig:
    C: float = 10.0
    class_weight: str = "balanced"
    max_iter: int = 500
    features: FeatureConfig = field(default_factory=FeatureConfig)

    @classmethod
    def from_dict(cls, d: dict) -> ClfConfig:
        f = {k: tuple(v) if isinstance(v, list) else v for k, v in d.get("features", {}).items()}
        return cls(**{**d, "features": FeatureConfig(**f)})


class SentimentClassifier:
    def __init__(
        self,
        config: ClfConfig | None = None,
        masker: Masker | None = None,
        vectors: TokenVectors | None = None,
    ) -> None:
        self.config = config or ClfConfig()
        self.masker = masker or Masker()
        self.vectorizer = Vectorizer(self.config.features, vectors)
        self.w: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def prep(self, examples: list[Example]) -> list[Prepped]:
        def mask(text: str, e: Example) -> str:
            return self.masker.mask(text, name=e.player, player_id=e.player_id, aliases=e.aliases)

        out = []
        for e in examples:
            masked = mask(e.text, e)
            parent = tokenize(mask(e.parent_text, e)) if e.parent_text else []
            out.append(Prepped(tokenize(masked), masked, parent))
        return out

    def fit(self, examples: list[Example]) -> SentimentClassifier:
        y = np.array([CLASSES.index(e.label) for e in examples])
        x = self.vectorizer.fit_transform(self.prep(examples))
        self.w, self.b = softmax.fit_softmax(
            x, y, len(CLASSES), self.config.C, self.config.class_weight, self.config.max_iter
        )
        return self

    def predict_proba(self, examples: list[Example]) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("model is not fitted")
        return softmax.predict_proba(self.vectorizer.transform(self.prep(examples)), self.w, self.b)

    def score(self, examples: list[Example]) -> np.ndarray:
        """P(pos) - P(neg) in [-1, 1]."""
        p = self.predict_proba(examples)
        return p[:, CLASSES.index("pos")] - p[:, CLASSES.index("neg")]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "w": self.w,
            "b": self.b,
            "config": np.array(json.dumps(asdict(self.config))),
            "classes": np.array(CLASSES),
        }
        for name, blk in self.vectorizer.blocks.items():
            arrays[f"terms_{name}"] = np.array(blk.terms, dtype=str)
            arrays[f"idf_{name}"] = blk.idf
        if self.vectorizer.vectors is not None:
            arrays["emb_tokens"], arrays["emb_vecs"] = self.vectorizer.vectors.export()
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(
        cls, path: Path, masker: Masker | None = None, provider: FastTextVectors | None = None
    ) -> SentimentClassifier:
        """``provider`` (the fastText .bin) is only needed for tokens unseen in training; the
        model file carries the vectors of every token cached at save time."""
        with np.load(path, allow_pickle=False) as z:
            if tuple(z["classes"].tolist()) != CLASSES:
                raise ValueError(f"{path}: class order {z['classes'].tolist()} != {CLASSES}")
            config = ClfConfig.from_dict(json.loads(str(z["config"])))
            vectors = None
            if config.features.emb:
                vectors = TokenVectors.from_export(
                    z["emb_tokens"].tolist(), z["emb_vecs"], provider
                )
            model = cls(config, masker, vectors)
            model.w, model.b = z["w"], z["b"]
            for name, blk in model.vectorizer.blocks.items():
                blk.terms = z[f"terms_{name}"].tolist()
                blk.idf = z[f"idf_{name}"]
                blk._set_index()
        return model
