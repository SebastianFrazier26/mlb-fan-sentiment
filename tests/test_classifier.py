import csv

import numpy as np
import pytest
from scipy import sparse

from fansent import sentiment, util
from fansent.classifier import ClfConfig, Example, Masker, SentimentClassifier, tokenize
from fansent.classifier.data import load_labeled
from fansent.classifier.evaluate import Grid, evaluate, grouped_folds, tune
from fansent.classifier.features import FeatureConfig, window_terms
from fansent.classifier.overlay import LexiconVader
from fansent.classifier.scorers import ClfScorer, LexiconScorer, VaderScorer
from fansent.classifier.softmax import loss_grad
from fansent.classifier.text import OTH, TGT

PLAYERS = [
    {"player_id": "1", "full_name": "Seiya Suzuki", "last_name": "Suzuki"},
    {"player_id": "2", "full_name": "Willson Contreras", "last_name": "Contreras"},
    {"player_id": "3", "full_name": "Justin Turner", "last_name": "Turner"},
    {"player_id": "4", "full_name": "Trea Turner", "last_name": "Turner"},
    {"player_id": "5", "full_name": "Jhonny Álvarez", "last_name": "Álvarez"},
]


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    n, d, k = 12, 7, 3
    x = sparse.csr_matrix(rng.random((n, d)) * (rng.random((n, d)) > 0.5))
    y = rng.integers(0, k, n)
    s = rng.random(n) + 0.5
    theta = rng.normal(size=d * k + k)
    _, g = loss_grad(theta, x, y, s, 2.0, k)
    num = np.zeros_like(theta)
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = 1e-6
        num[i] = (
            loss_grad(theta + e, x, y, s, 2.0, k)[0] - loss_grad(theta - e, x, y, s, 2.0, k)[0]
        ) / 2e-6
    np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-6)


def _toy(n=40):
    pos = ["{p} is crushing the ball tonight", "great swing by {p} love it", "{p} was amazing"]
    neg = ["{p} is terrible and should sit", "awful at bat by {p} hate it", "{p} was dreadful"]
    neu = ["{p} bats second today", "lineup has {p} at short", "{p} is on the roster"]
    out = []
    for i in range(n):
        for lab, tpl in (("pos", pos), ("neg", neg), ("neu", neu)):
            out.append(Example(tpl[i % 3].format(p="Seiya Suzuki"), "Seiya Suzuki", "1", label=lab))
    return out


def test_separable_toy_problem_is_learned():
    ex = _toy()
    model = SentimentClassifier(ClfConfig(features=FeatureConfig(min_df=1)), Masker(PLAYERS)).fit(
        ex
    )
    pred = model.predict_proba(ex).argmax(axis=1)
    assert (pred == np.array([["neg", "neu", "pos"].index(e.label) for e in ex])).mean() > 0.99
    scores = model.score(ex)
    assert scores.min() >= -1 and scores.max() <= 1
    pos = [s for s, e in zip(scores, ex, strict=True) if e.label == "pos"]
    assert min(pos) > 0.5


def test_masking_target_other_aliases_and_accents():
    m = Masker(PLAYERS)
    out = m.mask(
        "Suzuki hit one, Contreras too. Alvarez and Trea Turner watched.",
        name="Seiya Suzuki",
        player_id="1",
    )
    assert out == f"{TGT} hit one, {OTH} too. {OTH} and {OTH} watched."
    out = m.mask("SEIYA SUZUKI and the Goat", name="Seiya Suzuki", player_id="1", aliases=["goat"])
    assert out == f"{TGT} and the {TGT}"
    out = m.mask("Álvarez's homer", name="Jhonny Álvarez", player_id="5")
    assert out.startswith(TGT + "'s")
    # ambiguous surname: not credited as "other", but masked as target when the row says so
    assert m.mask("Turner walked", name="Seiya Suzuki", player_id="1") == "Turner walked"
    assert m.mask("Turner walked", name="Justin Turner", player_id="3") == f"{TGT} walked"


def test_tokenizer_keeps_emoji_and_placeholders():
    toks = tokenize(f"{TGT} is 🔥🔥 don’t sit him!! {OTH} K's")
    assert toks == [TGT, "is", "🔥", "🔥", "don't", "sit", "him", "!", "!", OTH, "k's"]


def test_window_features_only_around_target():
    cfg = FeatureConfig(window=2, window_ngrams=(1, 1))
    toks = ["a", "b", "c", "d", TGT, "e", "f", "g", "h"]
    assert sorted(window_terms(toks, cfg)) == sorted(f"win:{t}" for t in ["c", "d", TGT, "e", "f"])
    assert window_terms(["no", "target", "here"], cfg) == []
    two = ["a", TGT, "b", "c", "d", "e", "f", TGT, "g"]
    bi = window_terms(two, FeatureConfig(window=1, window_ngrams=(2, 2)))
    assert "win:b c" not in bi and "win:a __TGT__" in bi and "win:__TGT__ b" in bi


def test_window_changes_prediction_context():
    # Same words, but sentiment attaches to the target only when it is inside the window.
    mk = lambda t, lab: Example(t, "Seiya Suzuki", "1", label=lab)
    ex = []
    for _ in range(10):
        ex += [
            mk("Suzuki is wonderful but Contreras is awful", "pos"),
            mk("Suzuki is awful but Contreras is wonderful", "neg"),
            mk("Contreras is wonderful and Suzuki is wonderful", "pos"),
            mk("Contreras is awful and Suzuki is awful", "neg"),
            mk("Suzuki is fine", "neu"),
        ]
    cfg = ClfConfig(features=FeatureConfig(min_df=1, window=2, char_ngrams=(0, 0)))
    model = SentimentClassifier(cfg, Masker(PLAYERS)).fit(ex)
    probe = [
        mk("Suzuki is wonderful but Contreras is awful", "pos"),
        mk("Suzuki is awful but Contreras is wonderful", "neg"),
    ]
    assert model.score(probe)[0] > 0 > model.score(probe)[1]


def test_overlay_flips_slang():
    plain, lex = VaderScorer(), LexiconScorer()
    ex = [Example("That slider is nasty", "x"), Example("He is absolutely filthy 🔥", "x")]
    assert (plain.score(ex)[0]) < 0
    assert lex.score(ex)[0] > 0.05 and lex.score(ex)[1] > 0.05
    assert lex.score([Example("He is washed", "x")])[0] < -0.05
    assert LexiconVader({"wicked": 0.0}).compound("wicked") == 0.0
    assert LexiconVader({"goat": 2.5}).compound("The GOAT!") > 0.05


def test_scorer_interface_and_sentiment_csv_schema(tmp_path, monkeypatch):
    ex = _toy(5)
    model = SentimentClassifier(ClfConfig(features=FeatureConfig(min_df=1)), Masker(PLAYERS)).fit(
        ex
    )
    model_path = tmp_path / "m.npz"
    model.save(model_path)

    texts = tmp_path / "texts.csv"
    fields = ["text_id", "source", "subreddit", "date", "player_id", "player_name", "text"]
    with open(texts, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(
            dict(
                zip(
                    fields,
                    [
                        "t1",
                        "reddit",
                        "CHICubs",
                        "2026-05-01",
                        "1",
                        "Seiya Suzuki",
                        "Suzuki was amazing",
                    ],
                    strict=True,
                )
            )
        )
    players = tmp_path / "players.csv"
    util.write_csv(players, PLAYERS, ["player_id", "full_name", "last_name"])
    monkeypatch.setattr(sentiment, "TEXTS_CSV", texts)
    monkeypatch.setattr(sentiment, "PLAYERS_CSV", players)
    for name in ("vader", "lexicon", "clf"):
        out = tmp_path / f"{name}.csv"
        monkeypatch.setattr(sentiment, "SENTIMENT_CSV", out)
        sentiment.run(name, model_path if name == "clf" else None)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0]) == sentiment.FIELDS
        assert -1 <= float(rows[0]["compound"]) <= 1


def test_model_save_load_roundtrip(tmp_path):
    ex = _toy(5)
    model = SentimentClassifier(
        ClfConfig(C=3.0, features=FeatureConfig(min_df=1, window=3)), Masker(PLAYERS)
    ).fit(ex)
    model.save(tmp_path / "m.npz")
    again = SentimentClassifier.load(tmp_path / "m.npz", Masker(PLAYERS))
    assert again.config == model.config
    np.testing.assert_allclose(again.predict_proba(ex), model.predict_proba(ex))


def test_load_labeled_fallback_and_na(tmp_path):
    p = tmp_path / "l.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "player", "player_id", "draft_label", "final_label", "aliases"])
        w.writerow(["a", "P One", "1", "pos", "", "po; one"])
        w.writerow(["b", "P One", "1", "pos", "neg", ""])
        w.writerow(["c", "P One", "1", "neu", "na", ""])
    ex, n_na = load_labeled([p])
    assert [e.label for e in ex] == ["pos", "neg"] and n_na == 1
    assert ex[0].aliases == ("po", "one")


def test_cv_folds_group_identical_texts_and_tune_runs():
    ex = _toy(8)
    folds = grouped_folds(ex, 4, seed=0)
    assert sorted(np.concatenate(folds).tolist()) == list(range(len(ex)))
    owner = {}
    for fi, idx in enumerate(folds):
        for i in idx:
            assert owner.setdefault(ex[i].text, fi) == fi
    grid = Grid(Cs=(1.0, 10.0), windows=(0, 3), char_ngrams=("0-0",), word_ngrams=("1-1",))
    best, results = tune(
        ex,
        Masker(PLAYERS),
        grid,
        base=ClfConfig(features=FeatureConfig(min_df=1)),
        folds=3,
        log=lambda *_: None,
    )
    assert len(results) == 4 and best.C in (1.0, 10.0)


def test_evaluate_metrics_and_calibration():
    ex = _toy(6)
    model = SentimentClassifier(ClfConfig(features=FeatureConfig(min_df=1)), Masker(PLAYERS)).fit(
        ex
    )
    m = evaluate(ClfScorer(model), ex)
    assert m.accuracy > 0.99 and m.macro_f1 > 0.99 and m.spearman > 0.9
    assert 0 <= m.ece <= 1 and m.cm.sum() == len(ex)
    v = evaluate(VaderScorer(), ex)
    assert v.ece is None


def test_class_weights_balanced():
    from fansent.classifier.softmax import class_weights

    y = np.array([0, 0, 0, 1])
    w = class_weights(y, 3, "balanced")
    assert w[:3].sum() == pytest.approx(w[3])
