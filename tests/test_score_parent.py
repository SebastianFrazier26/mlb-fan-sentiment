"""`fansent score` parent-context join (parent_texts.csv sidecar) for --parent models."""

import csv

import pytest

from fansent import sentiment, util
from fansent.classifier import ClfConfig, Example, Masker, SentimentClassifier
from fansent.classifier.features import FeatureConfig

PLAYERS = [{"player_id": "1", "full_name": "Seiya Suzuki", "last_name": "Suzuki"}]
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


def _train(parent: bool) -> SentimentClassifier:
    ex = []
    for i in range(10):
        ex += [
            Example(f"suzuki great {i}", "Seiya Suzuki", "1", label="pos", parent_text="love him"),
            Example(f"suzuki awful {i}", "Seiya Suzuki", "1", label="neg"),
            Example(f"suzuki bats second {i}", "Seiya Suzuki", "1", label="neu"),
        ]
    cfg = ClfConfig(features=FeatureConfig(min_df=1, parent=parent))
    return SentimentClassifier(cfg, Masker(PLAYERS)).fit(ex)


def _write(path, fields, rows):
    util.write_csv(path, [dict(zip(fields, r, strict=True)) for r in rows], fields)


@pytest.fixture
def data_dir(tmp_path):
    rows = [
        ("t1", "reddit_arctic", "CHICubs", "2026-05-01", "1", "1", "Seiya Suzuki", "he did it"),
        ("t2", "reddit_arctic", "CHICubs", "2026-05-01", "1", "1", "Seiya Suzuki", "same here"),
        ("t3", "reddit_arctic", "CHICubs", "2026-05-01", "1", "1", "Seiya Suzuki", "Suzuki rules"),
    ]
    _write(tmp_path / util.TEXTS_FILE, TEXT_FIELDS, rows)
    _write(tmp_path / util.PLAYERS_FILE, ["player_id", "full_name", "last_name"],
           [(p["player_id"], p["full_name"], p["last_name"]) for p in PLAYERS])  # fmt: skip
    _write(tmp_path / util.REDDIT_ATTRIBUTION_FILE,
           ["text_id", "player_id", "attribution", "parent_id"],
           [("t1", "1", "reply_inherited", "p1"), ("t2", "1", "reply_inherited", "p2"),
            ("t3", "1", "explicit", "")])  # fmt: skip
    _train(True).save(tmp_path / util.MODELS_DIR / util.MODEL_FILE)
    return tmp_path


@pytest.fixture
def seen(monkeypatch):
    captured: list[Example] = []
    real = sentiment.build_scorer

    def spy(*a, **k):
        s = real(*a, **k)
        inner = s.score

        def score(examples):
            captured.extend(examples)
            return inner(examples)

        s.score = score
        return s

    monkeypatch.setattr(sentiment, "build_scorer", spy)
    return captured


def _sentiment_rows(d):
    with open(d / util.SENTIMENT_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_sidecar_joins_reply_parent_and_truncates(data_dir, seen, capsys):
    _write(data_dir / util.PARENT_TEXTS_FILE, ["parent_id", "text"],
           [("p1", "x" * 500), ("p2", "  love him  ")])  # fmt: skip
    sentiment.run("clf", data_dir=data_dir)
    t1, t2, t3 = seen
    assert t1.parent_text == "x" * sentiment.PARENT_MAX_CHARS == "x" * 300
    assert t2.parent_text == "love him"
    assert t3.parent_text == ""  # explicit row: non-reply, as in training
    assert capsys.readouterr().err == ""


def test_reply_gets_parent_block_and_is_reply(data_dir):
    model = SentimentClassifier.load(data_dir / util.MODELS_DIR / util.MODEL_FILE, Masker(PLAYERS))
    plain = Example("same here", "Seiya Suzuki", "1")
    reply = Example("same here", "Seiya Suzuki", "1", parent_text="love him")
    xp, xr = (model.vectorizer.transform(model.prep([e])) for e in (plain, reply))
    assert xp[0, -1] == 0 and xr[0, -1] == 1  # is_reply is the last column
    assert (xr != xp).nnz > 1  # parent n-grams present too
    assert model.score([plain])[0] != model.score([reply])[0]


def test_missing_sidecar_warns_with_count(data_dir, seen, capsys):
    sentiment.run("clf", data_dir=data_dir)
    assert [e.parent_text for e in seen] == ["", "", ""]
    err = capsys.readouterr().err
    assert "warning: 2 reply_inherited rows have no parent text" in err
    assert util.PARENT_TEXTS_FILE in err


def test_partial_sidecar_counts_only_unresolved(data_dir, seen, capsys):
    _write(data_dir / util.PARENT_TEXTS_FILE, ["parent_id", "text"], [("p1", "love him")])
    sentiment.run("clf", data_dir=data_dir)
    assert [e.parent_text for e in seen] == ["love him", "", ""]
    assert "warning: 1 reply_inherited" in capsys.readouterr().err


def test_no_warning_for_model_without_parent_block(data_dir, seen, capsys):
    _train(False).save(data_dir / util.MODELS_DIR / util.MODEL_FILE)
    sentiment.run("clf", data_dir=data_dir)
    assert capsys.readouterr().err == ""
    assert {e.parent_text for e in seen} == {""}


def test_sentiment_schema_unchanged(data_dir):
    _write(data_dir / util.PARENT_TEXTS_FILE, ["parent_id", "text"], [("p1", "a"), ("p2", "b")])
    sentiment.run("clf", data_dir=data_dir)
    rows = _sentiment_rows(data_dir)
    assert len(rows) == 3 and list(rows[0]) == sentiment.FIELDS
