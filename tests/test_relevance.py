"""Relevance model: features, training sanity, threshold selection, score-time filtering."""

import csv

import numpy as np
import pytest

from fansent import sentiment, util
from fansent.classifier import Example
from fansent.classifier.relevance import (
    DENSE_NAMES,
    RelConfig,
    RelevanceClassifier,
    relevance_prep,
)
from fansent.classifier.relevance_eval import (
    average_precision,
    pick_threshold,
    split_holdout,
)
from fansent.classifier.teams import teams_in
from fansent.classifier.text import Masker

PLAYERS = [
    {
        "player_id": "1",
        "full_name": "Edwin Diaz",
        "last_name": "Diaz",
        "team": "Chicago Cubs",
        "subreddit": "CHICubs",
    },
    {
        "player_id": "2",
        "full_name": "Rafael Devers",
        "last_name": "Devers",
        "team": "Boston Red Sox",
        "subreddit": "redsox",
    },
]


def feats(text, sub="baseball", player="Edwin Diaz", pid="1", parent=""):
    clf = RelevanceClassifier(PLAYERS)
    e = Example(text, player, pid, subreddit=sub, parent_text=parent)
    d = relevance_prep(e, clf.masker, clf.info).dense
    return dict(zip(DENSE_NAMES, d[: len(DENSE_NAMES)], strict=True))


def test_other_first_name_before_surname():
    assert feats("Jose Diaz hit one out")["prev_other_first"] == 1
    assert feats("Edwin Diaz hit one out")["prev_other_first"] == 0
    assert feats("E. Diaz hit one out")["prev_own_first"] == 1


def test_capitalized_stranger_before_but_not_sentence_start():
    assert feats("saw Mariano Diaz today")["prev_other_first"] == 0
    assert feats("saw Colonel Diaz today")["prev_cap_other"] == 1
    assert feats("Today Diaz was great")["prev_cap_other"] == 0


def test_full_name_match_type_and_casing():
    f = feats("Edwin Diaz is great")
    assert f["any_full"] == 1 and f["any_surname_only"] == 0
    f = feats("diaz is great")
    assert f["any_surname_only"] == 1 and f["all_lowercase"] == 1
    assert feats("nothing here")["has_mention"] == 0


def test_team_terms_own_vs_foreign():
    assert teams_in("the white sox lost") == {"Chicago White Sox"}
    assert teams_in("go sox") == {"Boston Red Sox"}
    f = feats("Diaz and the Yankees", parent="cubs game")
    assert f["foreign_team_comment"] == 1 and f["own_team_comment"] == 0
    assert f["own_team_parent"] == 1 and f["is_reply"] == 1
    f = feats("Diaz and the Cubs")
    assert f["own_team_comment"] == 1 and f["foreign_team_comment"] == 0


def test_baseball_sub_duplicates_dense_block():
    clf = RelevanceClassifier(PLAYERS)
    n = len(DENSE_NAMES)
    bb = relevance_prep(
        Example("Diaz", "Edwin Diaz", "1", subreddit="baseball"), clf.masker, clf.info
    )
    team = relevance_prep(
        Example("Diaz", "Edwin Diaz", "1", subreddit="CHICubs"), clf.masker, clf.info
    )
    assert np.allclose(bb.dense[n:], bb.dense[:n]) and not team.dense[n:].any()
    assert team.dense[DENSE_NAMES.index("sub_own")] == 1
    assert any(t.startswith("bb_") for t in bb.ctx_terms)


def _toy(n=40):
    ex = []
    for i in range(n):
        ex += [
            Example(
                f"Diaz struck out the side {i}",
                "Edwin Diaz",
                "1",
                label="pos",
                subreddit="baseball",
            ),
            Example(
                f"Jose Diaz of the Yankees homered {i}",
                "Edwin Diaz",
                "1",
                label="na",
                subreddit="baseball",
            ),
            Example(
                f"Devers crushed it {i}", "Rafael Devers", "2", label="neu", subreddit="redsox"
            ),
        ]
    return ex


def test_binary_training_separates_synthetic_classes():
    ex = _toy()
    model = RelevanceClassifier(PLAYERS, RelConfig(threshold=0.5)).fit(ex)
    pred = model.is_na(ex)
    assert (pred == np.array([e.label == "na" for e in ex])).mean() > 0.95


def test_save_load_roundtrip(tmp_path):
    ex = _toy(10)
    model = RelevanceClassifier(PLAYERS, RelConfig(threshold=0.7)).fit(ex)
    model.save(tmp_path / "r.npz")
    again = RelevanceClassifier.load(tmp_path / "r.npz", PLAYERS)
    assert again.config.threshold == 0.7
    assert np.allclose(model.p_na(ex), again.p_na(ex))


def test_threshold_selection():
    is_na = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
    p = np.array([0.95, 0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1])
    assert pick_threshold(is_na, p, "f1") == 0.4
    assert pick_threshold(is_na, p, "precision", 1.0) == 0.8
    assert pick_threshold(is_na, p, "precision", 0.75) == 0.4
    assert pick_threshold(is_na[::-1], p, "precision", 1.0) == float("inf")
    assert average_precision(is_na, p) == pytest.approx((1 + 1 + 1 + 4 / 5) / 4)


def test_holdout_is_grouped_and_stratified():
    ex = _toy(30) + [
        Example(
            "Diaz struck out the side 3", "Rafael Devers", "2", label="pos", subreddit="baseball"
        )
    ]
    train, held = split_holdout(ex, 0.15, 0)
    assert {e.text for e in train}.isdisjoint({e.text for e in held})
    assert len(train) + len(held) == len(ex)
    assert {e.subreddit for e in held} == {"baseball", "redsox"}


TEXT_FIELDS = [
    "text_id",
    "source",
    "subreddit",
    "date",
    "created_utc",
    "player_id",
    "player_name",
    "text",
]


@pytest.fixture
def data_dir(tmp_path):
    rows = [
        ("t1", "baseball", "Diaz struck out the side 3", "1", "Edwin Diaz"),
        ("t2", "baseball", "Jose Diaz of the Yankees homered 3", "1", "Edwin Diaz"),
        ("t3", "redsox", "Devers crushed it 3", "2", "Rafael Devers"),
    ]
    util.write_csv(
        tmp_path / util.TEXTS_FILE,
        [
            dict(
                zip(
                    TEXT_FIELDS,
                    ("id" + r[0], "reddit_arctic", r[1], "2026-05-01", "1", r[3], r[4], r[2]),
                    strict=True,
                )
            )
            | {"text_id": r[0]}
            for r in rows
        ],
        TEXT_FIELDS,
    )
    util.write_csv(tmp_path / util.PLAYERS_FILE, PLAYERS, list(PLAYERS[0]))
    RelevanceClassifier(PLAYERS, RelConfig(threshold=0.5)).fit(_toy()).save(
        tmp_path / util.MODELS_DIR / util.RELEVANCE_MODEL_FILE
    )
    return tmp_path


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_score_drops_na_writes_sidecar_and_keeps_schema(data_dir, capsys):
    sentiment.run("vader", data_dir=data_dir)
    out = _read(data_dir / util.SENTIMENT_FILE)
    assert [r["text_id"] for r in out] == ["t1", "t3"]
    assert list(out[0]) == sentiment.FIELDS
    side = _read(data_dir / util.RELEVANCE_FILE)
    assert list(side[0]) == sentiment.RELEVANCE_FIELDS
    assert {r["text_id"]: r["kept"] for r in side} == {"t1": "1", "t2": "0", "t3": "1"}
    assert all(0 <= float(r["p_relevant"]) <= 1 for r in side)
    assert "baseball" in capsys.readouterr().out


def test_no_relevance_flag_and_missing_model(data_dir):
    sentiment.run("vader", data_dir=data_dir, use_relevance=False)
    assert len(_read(data_dir / util.SENTIMENT_FILE)) == 3
    assert not (data_dir / util.RELEVANCE_FILE).exists()
    (data_dir / util.MODELS_DIR / util.RELEVANCE_MODEL_FILE).unlink()
    sentiment.run("vader", data_dir=data_dir)
    assert len(_read(data_dir / util.SENTIMENT_FILE)) == 3


def test_masker_target_pattern_public():
    assert Masker(PLAYERS).target_pattern("Edwin Diaz", "1").search("saw Diaz")
