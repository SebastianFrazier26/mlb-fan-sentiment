import datetime as dt

import numpy as np

from fansent.analyze import partial_corr
from fansent.names import build_matchers, find_mentions
from fansent.util import parse_ip, week_start


def test_week_start_is_monday():
    assert week_start(dt.date(2025, 7, 16)) == dt.date(2025, 7, 14)  # Wed -> Mon
    assert week_start(dt.date(2025, 7, 14)) == dt.date(2025, 7, 14)  # Mon -> itself
    assert week_start(dt.date(2025, 7, 20)) == dt.date(2025, 7, 14)  # Sun -> prior Mon


def test_parse_ip_thirds():
    assert parse_ip("5.2") == 5 + 2 / 3
    assert parse_ip("0.1") == 1 / 3
    assert parse_ip("7") == 7.0


def _players():
    return [
        {"player_id": "1", "full_name": "Seiya Suzuki", "last_name": "Suzuki"},
        {"player_id": "2", "full_name": "Justin Turner", "last_name": "Turner"},
        {"player_id": "3", "full_name": "Trea Turner", "last_name": "Turner"},
    ]


def test_unique_last_name_matches():
    matchers = build_matchers(_players())
    hits = find_mentions("suzuki is raking right now", matchers)
    assert [p["player_id"] for p in hits] == ["1"]


def test_shared_last_name_needs_full_name():
    matchers = build_matchers(_players())
    assert find_mentions("Turner struck out again", matchers) == []
    hits = find_mentions("Justin Turner struck out again", matchers)
    assert [p["player_id"] for p in hits] == ["2"]


def test_no_substring_false_positive():
    matchers = build_matchers(_players())
    assert find_mentions("the suzukimobile", matchers) == []


def test_partial_corr_detects_lagged_signal():
    rng = np.random.default_rng(42)
    n = 200
    sent = rng.normal(size=n)
    perf_now = rng.normal(size=n)
    perf_next = 0.8 * sent + 0.5 * perf_now + rng.normal(scale=0.3, size=n)
    r = partial_corr(sent, perf_next, perf_now)
    assert r.statistic > 0.8 and r.pvalue < 0.001


def test_partial_corr_no_signal():
    rng = np.random.default_rng(7)
    n = 200
    perf_now = rng.normal(size=n)
    sent = 0.9 * perf_now + rng.normal(scale=0.2, size=n)  # sentiment just tracks stats
    perf_next = 0.5 * perf_now + rng.normal(scale=0.5, size=n)  # next week only from stats
    r = partial_corr(sent, perf_next, perf_now)
    assert abs(r.statistic) < 0.2  # controlling perf_now removes the spurious link
