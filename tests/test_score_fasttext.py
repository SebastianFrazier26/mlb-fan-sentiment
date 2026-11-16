from types import SimpleNamespace

from fansent import sentiment
from fansent.classifier import embeddings


def _model():
    return SimpleNamespace(vectorizer=SimpleNamespace(vectors=embeddings.TokenVectors(dim=3)))


def test_attaches_bin_for_unseen_tokens(tmp_path, monkeypatch):
    (tmp_path / embeddings.EMBEDDINGS_DIR).mkdir()
    (tmp_path / embeddings.EMBEDDINGS_DIR / embeddings.DEFAULT_BIN).write_bytes(b"")
    monkeypatch.setattr(embeddings, "FastTextVectors", lambda p: ("ft", p.name))
    m = _model()
    sentiment._attach_fasttext(m, tmp_path)
    assert m.vectorizer.vectors.provider == ("ft", embeddings.DEFAULT_BIN)


def test_missing_bin_warns_and_leaves_cache_only(tmp_path, capsys):
    m = _model()
    sentiment._attach_fasttext(m, tmp_path)
    assert m.vectorizer.vectors.provider is None
    assert "unseen tokens get no embedding" in capsys.readouterr().err
