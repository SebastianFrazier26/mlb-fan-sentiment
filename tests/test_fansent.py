import datetime as dt

import numpy as np

from fansent.analyze import partial_corr
from fansent.names import build_matchers, find_mentions, fold, name_pattern
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


def _ids(text, players):
    return [p["player_id"] for p in find_mentions(text, build_matchers(players))]


def _p(pid, full, last=None):
    return {"player_id": pid, "full_name": full, "last_name": last or full.split()[-1]}


def test_accents_fold_both_directions():
    players = [_p("1", "Jose Alcántara"), _p("2", "Yacksel Rios"), _p("3", "Moisés Ballesteros")]
    assert _ids("Alcantara homered", players) == ["1"]
    assert _ids("Ríos came in to pitch", players) == ["2"]
    assert _ids("Moises Ballesteros is hitting", players) == ["3"]
    assert _ids("ALCÁNTARA!", players) == ["1"]


def test_fold_preserves_length_and_offsets():
    text = "Hoy Alcántara conectó"
    assert len(fold(text)) == len(text)
    m = name_pattern("Alcantara", full_name="Jose Alcantara").search(text)
    assert text[m.start() : m.end()] == "Alcántara"


_ANTHONY = [
    _p("1", "Roman Anthony"),
    _p("2", "Anthony Volpe"),
    _p("3", "Anthony Rizzo"),
    _p("4", "Anthony Seigler"),
]


def test_surname_that_is_first_name_not_credited_before_capitalized_word():
    assert _ids("Anthony Volpe went deep", _ANTHONY) == ["2"]
    assert _ids("Anthony Rizzo went deep", _ANTHONY) == ["3"]
    assert _ids("Anthony Seigler went deep", _ANTHONY) == ["4"]


def test_surname_that_is_first_name_still_matches_otherwise():
    assert _ids("Anthony is raking", _ANTHONY) == ["1"]
    assert _ids("Great game, Anthony.", _ANTHONY) == ["1"]
    assert _ids("Roman Anthony is raking", _ANTHONY) == ["1"]
    assert _ids("anthony volpe", _ANTHONY) == ["1", "2"]  # lowercase: no capital cue
    assert _ids("Anthony\n\nVolpe", _ANTHONY) == ["1", "2"]


def test_all_caps_following_word_is_emphasis_not_surname():
    nico = name_pattern("Nico", full_name="Nico Hoerner", guard=True)
    seiya = name_pattern("Seiya", full_name="Seiya Suzuki", guard=True)
    assert nico.search("NICO WTF") and nico.search("Nico WTF")
    assert seiya.search("THE SEIYA SPECIAL")
    assert not nico.search("Nico Hischier scored")


def test_all_caps_surname_still_blocks_only_when_tracked():
    assert _ids("ANTHONY VOLPE went deep", _ANTHONY) == ["2"]
    assert _ids("Anthony VOLPE went deep", _ANTHONY) == ["2"]
    assert _ids("ANTHONY IS RAKING", _ANTHONY) == ["1"]
    assert _ids("ANTHONY HISCHIER", _ANTHONY) == ["1"]  # untracked all-caps surname: accepted


def test_own_suffix_and_name_tokens_do_not_block():
    players = [_p("1", "Roman Anthony")]
    assert _ids("Anthony Jr hit one", players) == ["1"]


def test_sentence_start_capitalization_is_a_known_limitation():
    # 'Then' is indistinguishable from a surname without a name list.
    assert _ids("Anthony Then homered", _ANTHONY) == []


def test_alias_guard_hook():
    nico = name_pattern("Nico", full_name="Nico Hoerner", guard=True)
    assert nico.search("Nico Hoerner homered")
    assert nico.search("Nico is hot")
    assert not nico.search("Nico Hischier scored")
    assert nico.search("NICO WTF")
    assert not name_pattern(
        "Nico", full_name="Nico Hoerner", guard=True, surnames=frozenset({"hischier"})
    ).search("NICO HISCHIER")
    assert name_pattern("Moisés", full_name="Moisés Ballesteros", guard=True).search("MOISES!")
    cs = name_pattern("PCA", full_name="Pete Crow-Armstrong", case_sensitive=True)
    assert cs.search("PCA rules") and not cs.search("pca rules")


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
