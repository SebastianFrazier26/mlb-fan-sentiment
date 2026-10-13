import csv
import datetime as dt
import json
import threading
import urllib.error
import urllib.request

import pytest

from fansent import label
from fansent.label import LabelStore, load_rows, save_rows, serve

FIELDS = ["id", "player", "text", "parent_text", "draft_label", "sarcasm", "uncertain"]


def _write(path, rows, fields=FIELDS):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _rows():
    return [
        {"id": "a1", "player": "Seiya Suzuki", "text": "Seiya, rake", "draft_label": "pos"},
        {"id": "a2", "player": "Pete Crow-Armstrong", "text": 'PCA "dealt", line\nbreak'},
    ]


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "s.csv"
    _write(path, _rows())
    return path


def test_load_appends_output_columns(sample):
    fields, rows = load_rows(sample)
    assert fields[: len(FIELDS)] == FIELDS
    assert fields[len(FIELDS) :] == ["final_label", "reviewed", "reviewed_at"]
    assert rows[1]["text"] == 'PCA "dealt", line\nbreak'
    assert rows[0]["final_label"] == "" and rows[1]["parent_text"] == ""


def test_load_rejects_missing_required(tmp_path):
    path = tmp_path / "bad.csv"
    _write(path, [{"id": "x", "text": "t"}], ["id", "text"])
    with pytest.raises(ValueError, match="player"):
        load_rows(path)


def test_save_roundtrip_leaves_no_temp(sample):
    fields, rows = load_rows(sample)
    rows[0]["final_label"] = "neg"
    save_rows(sample, fields, rows)
    assert _read(sample)[0]["final_label"] == "neg"
    assert _read(sample)[1]["text"] == 'PCA "dealt", line\nbreak'
    assert [p.name for p in sample.parent.iterdir()] == ["s.csv"]


def test_save_failure_keeps_original(sample, monkeypatch):
    before = sample.read_bytes()
    fields, rows = load_rows(sample)
    rows[0]["final_label"] = "neg"

    def boom(src, dst):
        raise OSError("disk gone")

    monkeypatch.setattr(label.os, "replace", boom)
    with pytest.raises(OSError):
        save_rows(sample, fields, rows)
    assert sample.read_bytes() == before
    assert [p.name for p in sample.parent.iterdir()] == ["s.csv"]


def test_apply_persists_immediately(sample):
    store = LabelStore(sample)
    now = dt.datetime(2026, 10, 13, 12, 0, tzinfo=dt.UTC)
    row = store.apply(1, "neu", sarcasm=True, uncertain=False, now=now)
    assert row["final_label"] == "neu" and row["reviewed"] == "1"
    disk = _read(sample)[1]
    assert disk["final_label"] == "neu"
    assert (disk["sarcasm"], disk["uncertain"]) == ("1", "0")
    assert disk["reviewed_at"] == "2026-10-13T12:00:00+00:00"
    assert _read(sample)[0]["reviewed"] == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"index": 0, "label": "great"},
        {"index": 5, "label": "pos"},
        {"index": -1, "label": "pos"},
        {"index": "0", "label": "pos"},
        {"index": True, "label": "pos"},
        {"index": 0, "label": "pos", "sarcasm": "yes"},
    ],
)
def test_apply_rejects_bad_input(sample, kwargs):
    before = sample.read_bytes()
    with pytest.raises((ValueError, TypeError)):
        LabelStore(sample).apply(**kwargs)
    assert sample.read_bytes() == before


def test_apply_rolls_back_memory_on_save_failure(sample, monkeypatch):
    store = LabelStore(sample)
    monkeypatch.setattr(label, "save_rows", lambda *a: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        store.apply(0, "neg")
    assert store.rows[0]["final_label"] == "" and store.rows[0]["reviewed"] == ""


@pytest.fixture
def server(sample):
    srv = serve(sample, port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _req(url, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(base, obj, ctype="application/json"):
    return _req(base + "/api/label", json.dumps(obj).encode(), {"Content-Type": ctype}, "POST")


def test_binds_loopback_only(server):
    srv, _ = server
    assert srv.server_address[0] == "127.0.0.1"


def test_get_page_and_rows(server):
    _, base = server
    status, body = _req(base + "/")
    assert status == 200 and b"fansent label" in body and b"<script" in body
    assert b"https://" not in body and b"http://" not in body
    status, body = _req(base + "/api/rows")
    data = json.loads(body)
    assert status == 200 and data["file"] == "s.csv"
    assert data["labels"] == ["pos", "neu", "neg", "na"]
    assert [r["id"] for r in data["rows"]] == ["a1", "a2"]


def test_post_label_writes_csv(server, sample):
    _, base = server
    status, body = _post(base, {"index": 0, "label": "na", "uncertain": True})
    assert status == 200
    assert json.loads(body)["row"]["final_label"] == "na"
    disk = _read(sample)[0]
    assert (disk["final_label"], disk["uncertain"], disk["reviewed"]) == ("na", "1", "1")


@pytest.mark.parametrize(
    "obj",
    [{"index": 0, "label": "meh"}, {"index": 99, "label": "pos"}, {"label": "pos"}, [1, 2]],
)
def test_post_bad_payload_is_400(server, sample, obj):
    _, base = server
    before = sample.read_bytes()
    status, _ = _post(base, obj)
    assert status == 400
    assert sample.read_bytes() == before


def test_post_requires_json_content_type(server, sample):
    _, base = server
    before = sample.read_bytes()
    status, _ = _post(base, {"index": 0, "label": "pos"}, ctype="text/plain")
    assert status == 415
    assert sample.read_bytes() == before


def test_post_malformed_json_is_400(server):
    _, base = server
    status, _ = _req(base + "/api/label", b"{nope", {"Content-Type": "application/json"}, "POST")
    assert status == 400


def test_foreign_host_header_rejected(server):
    _, base = server
    status, _ = _req(base + "/api/rows", headers={"Host": "evil.example:80"})
    assert status == 403


def test_unknown_path_404(server):
    _, base = server
    assert _req(base + "/nope")[0] == 404
