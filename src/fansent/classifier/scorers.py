"""Scorers share one interface: examples -> score in [-1, 1] (and probabilities, if any)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .data import Example
from .overlay import LexiconVader
from .pipeline import CLASSES, SentimentClassifier
from .text import Masker

# VADER's documented convention: |compound| < 0.05 is neutral.
NEUTRAL_BAND = 0.05

SCORERS = ("vader", "lexicon", "clf")


class Scorer(Protocol):
    name: str

    def score(self, examples: list[Example]) -> np.ndarray: ...

    def predict(self, examples: list[Example]) -> np.ndarray:
        """Class indices into CLASSES."""
        ...

    def proba(self, examples: list[Example]) -> np.ndarray | None: ...


def score_to_class(score: np.ndarray, band: float = NEUTRAL_BAND) -> np.ndarray:
    return np.where(score >= band, CLASSES.index("pos"), np.where(score <= -band, 0, 1))


@dataclass
class VaderScorer:
    name: str = "vader"

    def __post_init__(self) -> None:
        self._a = SentimentIntensityAnalyzer()

    def score(self, examples: list[Example]) -> np.ndarray:
        return np.array([self._a.polarity_scores(e.text)["compound"] for e in examples])

    def predict(self, examples):
        return score_to_class(self.score(examples))

    def proba(self, examples):
        return None


@dataclass
class LexiconScorer:
    name: str = "lexicon"
    entries: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self._v = LexiconVader(self.entries)

    def score(self, examples: list[Example]) -> np.ndarray:
        return np.array([self._v.compound(e.text) for e in examples])

    def predict(self, examples):
        return score_to_class(self.score(examples))

    def proba(self, examples):
        return None


@dataclass
class ClfScorer:
    model: SentimentClassifier
    name: str = "clf"

    def score(self, examples: list[Example]) -> np.ndarray:
        return self.model.score(examples)

    def predict(self, examples):
        return self.model.predict_proba(examples).argmax(axis=1)

    def proba(self, examples):
        return self.model.predict_proba(examples)


def build_scorer(name: str, model_path: Path | None = None, players: list[dict] | None = None):
    if name == "vader":
        return VaderScorer()
    if name == "lexicon":
        return LexiconScorer()
    if name == "clf":
        if model_path is None:
            raise ValueError("--model is required for the clf scorer")
        return ClfScorer(SentimentClassifier.load(model_path, Masker(players or [])))
    raise ValueError(f"unknown scorer {name!r}; choose from {SCORERS}")
