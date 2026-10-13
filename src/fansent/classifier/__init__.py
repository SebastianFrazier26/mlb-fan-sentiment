"""Target-aware sentiment classifier, built from scratch on numpy/scipy.sparse."""

from .data import Example
from .pipeline import CLASSES, ClfConfig, SentimentClassifier
from .text import OTH, TGT, Masker, tokenize

__all__ = [
    "CLASSES",
    "OTH",
    "TGT",
    "ClfConfig",
    "Example",
    "Masker",
    "SentimentClassifier",
    "tokenize",
]
