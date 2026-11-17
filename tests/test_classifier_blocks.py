"""Optional feature blocks: fastText reader, embedding/lexicon/parent blocks, toggles, roundtrip."""

import struct

import numpy as np
import pytest

from fansent.classifier import ClfConfig, Example, Masker, SentimentClassifier
from fansent.classifier.data import load_labeled
from fansent.classifier.embeddings import (
    FastTextVectors,
    TokenVectors,
    fnv1a,
    mean_vector,
    subword_ids,
)
from fansent.classifier.evaluate import Grid, tune
from fansent.classifier.features import FeatureConfig, Prepped, window_spans
from fansent.classifier.overlay import LexiconVader

DIM, BUCKET, MINN, MAXN = 4, 40, 3, 4
WORDS = ["good", "bad", "great"]


def write_fake_bin(path, seed=0):
    rng = np.random.default_rng(seed)
    nwords = len(WORDS)
    mat = rng.normal(size=(nwords + BUCKET, DIM)).astype("<f4")
    out = struct.pack("<ii", 793712314, 12)
    out += struct.pack("<12i", DIM, 5, 5, 1, 5, 1, 1, 2, BUCKET, MINN, MAXN, 100) + b"\0" * 8
    out += struct.pack("<iiiqq", nwords, nwords, 0, 100, -1)
    for w in WORDS:
        out += w.encode() + b"\0" + struct.pack("<qB", 10, 0)
    out += b"\0" + struct.pack("<qq", *mat.shape) + mat.tobytes()
    out += b"\0" + struct.pack("<qq", 1, DIM) + b"\0" * (4 * DIM)  # output matrix, unread
    path.write_bytes(out)
    return mat


@pytest.fixture
def fake_bin(tmp_path):
    p = tmp_path / "fake.bin"
    mat = write_fake_bin(p)
    return p, mat


def test_fnv1a_reference_and_sign_extension():
    assert fnv1a(b"a") == 0xE40C292C  # published FNV-1a 32-bit test vector
    assert fnv1a(b"\xc3") != fnv1a(b"\x43")  # high bytes are sign-extended, not masked


def test_subword_ids_boundaries():
    ids = subword_ids("ab", nwords=3, bucket=BUCKET, minn=2, maxn=3)
    # "<ab>" -> 2-grams <a ab b>, 3-grams <ab ab>; single chars never emitted at minn=2
    assert len(ids) == 5 and all(3 <= i < 3 + BUCKET for i in ids)
    assert subword_ids("ab", 3, BUCKET, 1, 1) == [
        3 + fnv1a(b"a") % BUCKET,
        3 + fnv1a(b"b") % BUCKET,
    ]  # n==1 at the first/last position (the < > markers) is skipped


def test_vector_lookup_in_vocab_and_oov(fake_bin):
    path, mat = fake_bin
    ft = FastTextVectors(path)
    assert (ft.dim, ft.nwords, ft.bucket) == (DIM, 3, BUCKET)
    ng = subword_ids("good", 3, BUCKET, MINN, MAXN)
    np.testing.assert_allclose(ft.vector("good"), mat[[0, *ng]].mean(axis=0), rtol=1e-5)
    oov = subword_ids("goood", 3, BUCKET, MINN, MAXN)
    np.testing.assert_allclose(ft.vector("goood"), mat[oov].mean(axis=0), rtol=1e-5)
    assert ft.vector("") is None  # "<>" is shorter than minn, not in vocab


def test_rejects_bad_files(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"\0" * 200)
    with pytest.raises(ValueError, match="not a fastText"):
        FastTextVectors(p)


def test_token_vectors_unit_norm_cached_and_mean(fake_bin):
    tv = TokenVectors(FastTextVectors(fake_bin[0]))
    v = tv.get("good")
    assert tv.get("good") is v
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)
    assert tv.get("") is None
    m = mean_vector(["good", "", "bad"], tv)
    assert np.linalg.norm(m) == pytest.approx(1.0, abs=1e-5)
    assert not mean_vector([""], tv).any()


def test_window_spans_merge():
    toks = ["a", "__TGT__", "b", "c", "d", "e", "__TGT__"]
    assert window_spans(toks, 1) == [[0, 3], [5, 7]]
    assert window_spans(toks, 2) == [[0, 7]]


PLAYERS = [{"player_id": "1", "full_name": "Seiya Suzuki", "last_name": "Suzuki"}]


def _ex(text, label="pos", parent=""):
    return Example(text, "Seiya Suzuki", "1", label=label, parent_text=parent)


def _data():
    out = []
    for i in range(12):
        out += [
            _ex(f"suzuki is great good {i}", "pos"),
            _ex(f"suzuki is bad terrible {i}", "neg", parent="what about him" if i % 2 else ""),
            _ex(f"suzuki bats second {i}", "neu", parent="lineup is up" if i % 3 == 0 else ""),
        ]
    return out


def _model(fake_bin, **feat):
    cfg = ClfConfig(features=FeatureConfig(min_df=1, **feat))
    vec = TokenVectors(FastTextVectors(fake_bin[0])) if feat.get("emb") else None
    return SentimentClassifier(cfg, Masker(PLAYERS), vec)


def test_block_shapes_and_toggles(fake_bin):
    ex = _data()
    base = _model(fake_bin)
    x0 = base.vectorizer.fit_transform(base.prep(ex))
    n0 = x0.shape[1]

    lex = _model(fake_bin, lex=True)
    assert lex.vectorizer.fit_transform(lex.prep(ex)).shape[1] == n0 + 4

    emb = _model(fake_bin, emb=True)
    assert emb.vectorizer.fit_transform(emb.prep(ex)).shape[1] == n0 + 2 * DIM

    par = _model(fake_bin, parent=True)
    xp = par.vectorizer.fit_transform(par.prep(ex))
    assert "parent" in par.vectorizer.blocks
    assert xp.shape[1] == n0 + len(par.vectorizer.blocks["parent"].terms) + 1

    allb = _model(fake_bin, lex=True, emb=True, parent=True)
    assert allb.vectorizer.fit_transform(allb.prep(ex)).shape[1] == xp.shape[1] + 4 + 2 * DIM


def test_parent_block_only_on_replies(fake_bin):
    ex = _data()
    m = _model(fake_bin, parent=True)
    x = m.vectorizer.fit_transform(m.prep(ex)).tocsr()
    n_par = len(m.vectorizer.blocks["parent"].terms)
    start = x.shape[1] - n_par - 1
    parent_part = x[:, start:]
    for row, e in enumerate(ex):
        nnz = parent_part[row].nnz
        assert (nnz > 0) == bool(e.parent_text)
    flag = parent_part[:, -1].toarray().ravel()
    np.testing.assert_array_equal(flag, [1.0 if e.parent_text else 0.0 for e in ex])


def test_parent_text_is_masked_for_target():
    m = SentimentClassifier(ClfConfig(), Masker(PLAYERS))
    p = m.prep([_ex("x", parent="Suzuki is hitting")])[0]
    assert "__TGT__" in p.parent_tokens and "suzuki" not in p.parent_tokens


def test_weights_scale_blocks(fake_bin):
    ex = _data()
    a = _model(fake_bin, lex=True, lex_weight=1.0)
    b = _model(fake_bin, lex=True, lex_weight=3.0)
    xa = a.vectorizer.fit_transform(a.prep(ex))[:, -4:].toarray()
    xb = b.vectorizer.fit_transform(b.prep(ex))[:, -4:].toarray()
    np.testing.assert_allclose(xb, 3 * xa)


def test_emb_requires_vectors():
    with pytest.raises(ValueError, match="no embeddings"):
        SentimentClassifier(ClfConfig(features=FeatureConfig(emb=True)))


def test_lexicon_hits_and_features():
    lex = LexiconVader()
    assert lex.hits("he is filthy and washed") == (1, 1)
    assert lex.hits("nothing here") == (0, 0)
    m = SentimentClassifier(ClfConfig(), Masker(PLAYERS))
    from fansent.classifier.features import lexicon_features

    p = m.prep([_ex("suzuki is filthy. the weather is washed out today and then more words")])[0]
    comp, win, pos, neg = lexicon_features(p, 2)
    assert comp != 0 and win > 0 and pos > 0 and neg > 0
    noname = Prepped(["hello"], "hello", [])
    assert lexicon_features(noname, 2)[1] == 0.0


def test_roundtrip_with_all_blocks(fake_bin, tmp_path):
    ex = _data()
    model = _model(fake_bin, lex=True, emb=True, parent=True, window=3).fit(ex)
    model.save(tmp_path / "m.npz")
    again = SentimentClassifier.load(tmp_path / "m.npz", Masker(PLAYERS))  # no .bin needed
    assert again.config == model.config
    np.testing.assert_allclose(again.predict_proba(ex), model.predict_proba(ex), atol=1e-5)


def test_old_model_files_still_load(tmp_path):
    model = SentimentClassifier(ClfConfig(features=FeatureConfig(min_df=1)), Masker(PLAYERS))
    model.fit(_data())
    model.save(tmp_path / "m.npz")
    assert SentimentClassifier.load(tmp_path / "m.npz").vectorizer.vectors is None


def test_tune_enumerates_weights_only_for_enabled_blocks(fake_bin):
    ex = _data()
    grid = Grid(
        Cs=(1.0,),
        windows=(0,),
        char_ngrams=("0-0",),
        emb_weights=(1.0, 2.0),
        lex_weights=(1.0, 2.0, 3.0),
        parent_weights=(0.5,),
    )
    base = ClfConfig(features=FeatureConfig(min_df=1, lex=True))
    best, res = tune(ex, Masker(PLAYERS), grid, base, folds=3, log=lambda *_: None)
    assert len(res) == 3  # lex weights only
    assert best.features.lex and not best.features.emb


def test_load_labeled_reads_parent_text(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("text,player,player_id,draft_label,parent_text\nhi,P,1,pos, synthetic parent \n")
    ex, _ = load_labeled([p])
    assert ex[0].parent_text == "synthetic parent"
