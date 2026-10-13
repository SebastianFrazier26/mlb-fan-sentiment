"""Labeled-example loading. Label files are CSVs: text, player, player_id, final_label |
draft_label, optional aliases (';'-separated). Other columns (parent_text, ...) are ignored."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

LABELS = ("neg", "neu", "pos")

_LABEL_ALIASES = {
    "pos": "pos",
    "positive": "pos",
    "+1": "pos",
    "1": "pos",
    "neu": "neu",
    "neutral": "neu",
    "0": "neu",
    "neg": "neg",
    "negative": "neg",
    "-1": "neg",
    "na": "na",
    "n/a": "na",
}


@dataclass(frozen=True)
class Example:
    text: str
    player: str
    player_id: str = ""
    aliases: tuple[str, ...] = ()
    label: str | None = None


def normalize_label(raw: str) -> str:
    key = raw.strip().lower()
    if key not in _LABEL_ALIASES:
        raise ValueError(f"unknown label {raw!r}")
    return _LABEL_ALIASES[key]


def load_labeled(paths: list[Path]) -> tuple[list[Example], int]:
    """Returns (examples with label in pos/neu/neg, number of 'na' rows dropped)."""
    out: list[Example] = []
    n_na = 0
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f), start=2):
                raw = (row.get("final_label") or "").strip() or (row.get("draft_label") or "")
                if not raw.strip():
                    raise ValueError(f"{path}:{i}: no final_label or draft_label")
                try:
                    label = normalize_label(raw)
                except ValueError as e:
                    raise ValueError(f"{path}:{i}: {e}") from None
                if label == "na":
                    n_na += 1
                    continue
                aliases = tuple(
                    a.strip() for a in (row.get("aliases") or "").split(";") if a.strip()
                )
                out.append(
                    Example(
                        text=row["text"],
                        player=row.get("player") or row.get("player_name") or "",
                        player_id=str(row.get("player_id") or ""),
                        aliases=aliases,
                        label=label,
                    )
                )
    return out, n_na
