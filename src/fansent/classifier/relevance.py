"""Relevance model: is a matched comment actually about the matched player?

The sentiment masker hides the name, but relevance lives in the name's surroundings: a bare
'Martinez' preceded by 'Jose' or followed by 'Yankees' is rarely our roster's player. So the
features keep the mention's context unmasked: match type, neighbouring words, whether the word
before the surname is the player's own first name or some other first name or capitalized word,
mention casing, own-team versus other-team terms in the comment and its parent, and the
subreddit. Surname collisions concentrate in r/baseball (the roster is Cubs + Red Sox only), so
every dense feature is duplicated times an is-r/baseball flag; a linear model then learns
separate rules for the league-wide sub and the team subs.

Model: the same from-scratch L2 logistic regression (``softmax.fit_softmax``, k=2) over
word TF-IDF + mention-context TF-IDF + dense flags.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from fansent.common_words import COMMON_WORDS
from fansent.first_names import COMMON_FIRST_NAMES
from fansent.names import NAME_SUFFIXES, fold
from fansent.panel import load_team_subreddit_rows
from fansent.util import PLAYER_TEAMS_FILE, TEAMS_FILE, week_start

from . import softmax
from .data import Example
from .features import TfidfBlock, _ngrams
from .teams import canonical_team, teams_in
from .text import Masker, tokenize

REL_CLASSES = ("na", "relevant")
NA, REL = 0, 1

_PREV = re.compile(r"([^\W\d_][\w'’.-]*)[^\S\n]+$")
_NEXT = re.compile(r"^[^\S\n]+([^\W\d_][\w'’-]*)")
_SENT_END = re.compile(r"(?:^|[.!?\n:\"“(])\s*$")
_MAX_MENTIONS = 3

DENSE_NAMES = (
    "one",
    "has_mention",
    "any_full",
    "any_alias",
    "any_surname_only",
    "log_n_mentions",
    "any_capitalized",
    "all_lowercase",
    "prev_own_first",
    "prev_other_first",
    "prev_cap_other",
    "next_cap",
    "is_reply",
    "target_in_parent",
    "own_team_comment",
    "foreign_team_comment",
    "own_team_parent",
    "foreign_team_parent",
    "log_n_foreign_comment",
    "sub_own",
    "log_len",
    "surname_is_first_name",
    "surname_is_common_word",
)
N_DENSE = len(DENSE_NAMES)


@dataclass(frozen=True)
class RelConfig:
    C: float = 3.0
    class_weight: str = "balanced"
    max_iter: int = 500
    min_df: int = 2
    word_weight: float = 1.0
    ctx_weight: float = 1.0
    dense_weight: float = 1.0
    threshold: float = 0.5  # drop a row when P(na) >= threshold

    @classmethod
    def from_dict(cls, d: dict) -> RelConfig:
        return cls(**d)


@dataclass(frozen=True)
class RelPrepped:
    tokens: list[str]
    ctx_terms: list[str]
    dense: np.ndarray


@dataclass(frozen=True)
class WeeklyRoster:
    """player_teams.csv in lookup form: (player_id, Monday ISO date) -> team_id, plus team_id ->
    full name (teams.csv)."""

    team_of: Mapping[tuple[str, str], int]
    names: Mapping[int, str]


def load_weekly_roster(players_path: Path | None) -> WeeklyRoster | None:
    """Reads player_teams.csv and teams.csv beside players.csv; None if either is missing."""
    if players_path is None:
        return None
    weekly_path = players_path.with_name(PLAYER_TEAMS_FILE)
    teams_path = players_path.with_name(TEAMS_FILE)
    if not (weekly_path.exists() and teams_path.exists()):
        return None
    with open(teams_path, newline="", encoding="utf-8") as f:
        names = {int(r["team_id"]): r["name"] for r in csv.DictReader(f)}
    with open(weekly_path, newline="", encoding="utf-8") as f:
        team_of = {
            (r["player_id"], r["week_start"]): int(r["team_id"])
            for r in csv.DictReader(f)
            if int(r["team_id"]) in names
        }
    return WeeklyRoster(team_of, names)


def _team_subreddits() -> dict[int, frozenset[str]]:
    out: dict[int, set[str]] = {}
    for r in load_team_subreddit_rows():
        out.setdefault(int(r["team_id"]), set()).add(r["subreddit"].lower())
    return {k: frozenset(v) for k, v in out.items()}


class PlayerInfo:
    """Roster lookups the features need: team, team subreddit, first name per player.

    With ``weekly``, own-team is the set of teams the player was on in the Mon-Sun week of the
    text's date plus the previous week (a traded player counts for both teams in the week he
    moved); without it, or when the player-week is absent, the players.csv team."""

    def __init__(self, players: list[dict], weekly: WeeklyRoster | None = None) -> None:
        self.by_id = {str(p["player_id"]): p for p in players}
        self.weekly = weekly
        self.team_subs = _team_subreddits() if weekly else {}
        self.by_name = {fold(p["full_name"]).lower(): p for p in players}
        self.first_names = set(COMMON_FIRST_NAMES)
        for p in players:
            self.first_names.add(self.first_of(p["full_name"]))

    @staticmethod
    def first_of(full_name: str) -> str:
        return fold(full_name).lower().split()[0] if full_name.split() else ""

    def own_teams(self, p: dict, day: str) -> tuple[set[str], set[int] | None]:
        """(full team names, team ids) the player counts for on ``day``; ids are None when the
        players.csv fallback applies (date unusable, no weekly data, or no row for that week)."""
        if day and self.weekly:
            try:
                week = week_start(dt.date.fromisoformat(day[:10]))
            except ValueError:
                week = None
            pid = str(p.get("player_id", ""))
            team_of = self.weekly.team_of
            now = team_of.get((pid, week.isoformat())) if week else None
            if now is not None:
                ids = {now}
                prev = team_of.get((pid, (week - dt.timedelta(days=7)).isoformat()))
                if prev is not None:
                    ids.add(prev)
                return {self.weekly.names[i] for i in ids}, ids
        return {p.get("team_name") or p.get("team", "")}, None

    def lookup(self, e: Example) -> dict:
        return self.by_id.get(str(e.player_id)) or self.by_name.get(fold(e.player).lower()) or {}


def _is_cap(word: str) -> bool:
    return word[0].isupper()


def _norm(word: str) -> str:
    return fold(word).lower().strip(".'’-")


def _mention_context(folded: str, m: re.Match) -> tuple[str, bool, str]:
    """(previous word, previous word starts a sentence, next word); '' if none."""
    before = folded[: m.start()]
    pm = _PREV.search(before)
    prev = pm.group(1) if pm else ""
    prev_initial = bool(pm) and bool(_SENT_END.search(before[: pm.start()]))
    nm = _NEXT.match(folded, m.end())
    nxt = nm.group(1) if nm else ""
    return prev, prev_initial, nxt


def relevance_prep(e: Example, masker: Masker, info: PlayerInfo) -> RelPrepped:
    p = info.lookup(e)
    first = info.first_of(p.get("full_name") or e.player)
    pat = masker.target_pattern(e.player, e.player_id, e.aliases)
    own_keys = {fold(n).lower() for n in (e.player, *e.aliases)}
    text = fold(e.text)
    matches = list(pat.finditer(text)) if pat else []
    parent_matches = bool(pat and e.parent_text and pat.search(fold(e.parent_text)))

    ctx: list[str] = []
    flags = dict.fromkeys(DENSE_NAMES, 0.0)
    flags["one"] = 1.0
    flags["has_mention"] = float(bool(matches))
    cap_any = False
    last = fold(p.get("last_name") or "").lower()
    for i, m in enumerate(matches):
        surface = m.group()
        key = " ".join(surface.lower().split())
        if key in own_keys and " " in key:
            mt = "full"
            flags["any_full"] = 1.0
        elif key in own_keys:
            mt = "alias"
            flags["any_alias"] = 1.0
        else:
            mt = "sur"
            flags["any_surname_only"] = 1.0
        cap_any |= _is_cap(surface)
        prev, prev_initial, nxt = _mention_context(text, m)
        pn, nn = _norm(prev), _norm(nxt)
        if prev:
            if pn == first or (len(pn) == 1 and pn == first[:1]):
                flags["prev_own_first"] = 1.0
            elif pn in info.first_names:
                flags["prev_other_first"] = 1.0
            elif _is_cap(prev) and not prev_initial:
                flags["prev_cap_other"] = 1.0
        if nxt and _is_cap(nxt) and nn not in NAME_SUFFIXES and nn not in key.split():
            flags["next_cap"] = 1.0
        if i < _MAX_MENTIONS:
            ctx += [f"mt:{mt}", f"L1:{pn or '^'}", f"R1:{nn or '$'}"]
            if mt == "sur":
                ctx.append(f"sur:{_norm(surface)}")
    flags["log_n_mentions"] = math.log1p(len(matches))
    flags["any_capitalized"] = float(cap_any)
    flags["all_lowercase"] = float(bool(matches) and not cap_any)

    own_names, own_ids = info.own_teams(p, e.date)
    own_teams = {canonical_team(n) for n in own_names}
    comment_teams = teams_in(e.text)
    parent_teams = teams_in(e.parent_text) if e.parent_text else set()
    foreign_c = comment_teams - own_teams
    flags["own_team_comment"] = float(bool(own_teams & comment_teams))
    flags["foreign_team_comment"] = float(bool(foreign_c))
    flags["own_team_parent"] = float(bool(own_teams & parent_teams))
    flags["foreign_team_parent"] = float(bool(parent_teams - own_teams))
    flags["log_n_foreign_comment"] = math.log1p(len(foreign_c))
    flags["is_reply"] = float(bool(e.parent_text))
    flags["target_in_parent"] = float(parent_matches)
    if own_ids is None:
        flags["sub_own"] = float(bool(e.subreddit) and e.subreddit == p.get("subreddit"))
    else:
        flags["sub_own"] = float(
            any(e.subreddit.lower() in info.team_subs.get(i, ()) for i in own_ids)
        )
    flags["log_len"] = math.log1p(len(text.split())) / 5.0
    flags["surname_is_first_name"] = float(last in info.first_names)
    flags["surname_is_common_word"] = float(last in COMMON_WORDS)

    bb = e.subreddit.lower() == "baseball"
    if bb:
        ctx += [f"bb_{c}" for c in ctx]
    if not matches:
        ctx.append("nomention")
    vec = np.array([flags[n] for n in DENSE_NAMES])
    dense = np.concatenate([vec, vec * float(bb)])

    masked = masker.mask(e.text, name=e.player, player_id=e.player_id, aliases=e.aliases)
    return RelPrepped(tokenize(masked), ctx, dense)


class RelevanceClassifier:
    def __init__(
        self,
        players: list[dict] | None = None,
        config: RelConfig | None = None,
        weekly_teams: WeeklyRoster | None = None,
    ) -> None:
        players = players or []
        self.config = config or RelConfig()
        self.masker = Masker(players)
        self.info = PlayerInfo(players, weekly_teams)
        self.word = TfidfBlock(self.config.min_df, 50000)
        self.ctx = TfidfBlock(self.config.min_df, 50000)
        self.w: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def prep(self, examples: list[Example]) -> list[RelPrepped]:
        return [relevance_prep(e, self.masker, self.info) for e in examples]

    def _matrix(self, docs: list[RelPrepped], fit: bool) -> sparse.csr_matrix:
        cfg = self.config
        words = [_ngrams(d.tokens, 1, 2) for d in docs]
        ctxs = [d.ctx_terms for d in docs]
        if fit:
            self.word.fit(words)
            self.ctx.fit(ctxs)
        dense = np.array([d.dense for d in docs]).reshape(len(docs), -1)
        return sparse.hstack(
            [
                self.word.transform(words) * cfg.word_weight,
                self.ctx.transform(ctxs) * cfg.ctx_weight,
                sparse.csr_matrix(dense * cfg.dense_weight),
            ]
        ).tocsr()

    def fit_prepped(self, docs: list[RelPrepped], y: np.ndarray) -> RelevanceClassifier:
        x = self._matrix(docs, fit=True)
        self.w, self.b = softmax.fit_softmax(
            x, y, 2, self.config.C, self.config.class_weight, self.config.max_iter
        )
        return self

    def fit(self, examples: list[Example]) -> RelevanceClassifier:
        y = np.array([NA if e.label == "na" else REL for e in examples])
        return self.fit_prepped(self.prep(examples), y)

    def p_na_prepped(self, docs: list[RelPrepped]) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("model is not fitted")
        return softmax.predict_proba(self._matrix(docs, fit=False), self.w, self.b)[:, NA]

    def p_na(self, examples: list[Example]) -> np.ndarray:
        return self.p_na_prepped(self.prep(examples))

    def is_na(self, examples: list[Example]) -> np.ndarray:
        return self.p_na(examples) >= self.config.threshold

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            w=self.w,
            b=self.b,
            config=np.array(json.dumps(asdict(self.config))),
            classes=np.array(REL_CLASSES),
            terms_word=np.array(self.word.terms, dtype=str),
            idf_word=self.word.idf,
            terms_ctx=np.array(self.ctx.terms, dtype=str),
            idf_ctx=self.ctx.idf,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        players: list[dict] | None = None,
        weekly_teams: WeeklyRoster | None = None,
    ) -> RelevanceClassifier:
        with np.load(path, allow_pickle=False) as z:
            if tuple(z["classes"].tolist()) != REL_CLASSES:
                raise ValueError(f"{path} is not a relevance model")
            model = cls(players, RelConfig.from_dict(json.loads(str(z["config"]))), weekly_teams)
            model.w, model.b = z["w"], z["b"]
            for name in ("word", "ctx"):
                blk = getattr(model, name)
                blk.terms = z[f"terms_{name}"].tolist()
                blk.idf = z[f"idf_{name}"]
                blk._set_index()
        return model
