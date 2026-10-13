"""Synthetic stats + scored texts with a planted sentiment -> next-week OPS effect."""

import numpy as np
import pandas as pd

FANBASE_SUB = {"CHICubs": "CHICubs", "redsox": "redsox", "baseball": "baseball"}


def make_stats(n_players=6, weeks=20, seed=0, start="2026-04-06"):
    rng = np.random.default_rng(seed)
    mondays = pd.date_range(start, periods=weeks, freq="7D")
    rows = []
    for i in range(n_players):
        pid = str(1000 + i)
        ability = rng.normal(0.75, 0.08)
        ops = np.clip(ability + rng.normal(0, 0.15, weeks), 0.2, 1.5)
        pa = rng.integers(14, 30, weeks)
        cum_pa = np.cumsum(pa)
        std = np.cumsum(ops * pa) / cum_pa
        for t in range(weeks):
            rows.append(
                {
                    "player_id": pid,
                    "full_name": f"Player {i}",
                    "team": "T",
                    "season": 2026,
                    "week_start": mondays[t].date().isoformat(),
                    "group": "hitting",
                    "metric": "OPS",
                    "value": ops[t],
                    "std_value": std[t],
                    "weight": float(pa[t]),
                    "std_weight": float(cum_pa[t]),
                    "higher_is_better": True,
                }
            )
    return pd.DataFrame(rows)


def make_scores(stats, effects, seed=1, texts_per_week=4, noise=0.5):
    """effects: fanbase -> coefficient of next-week OPS z-score in that fanbase's mean sentiment."""
    rng = np.random.default_rng(seed)
    rows = []
    n = 0
    for pid, g in stats.groupby("player_id"):
        g = g.sort_values("week_start")
        z = ((g["value"] - g["value"].mean()) / g["value"].std()).to_numpy()
        nxt = np.append(z[1:], 0.0)
        for fb, beta in effects.items():
            for t, week in enumerate(g["week_start"]):
                mean = 0.2 * rng.normal() + beta * nxt[t]
                for k in range(texts_per_week):
                    n += 1
                    day = pd.Timestamp(week) + pd.Timedelta(days=int(rng.integers(0, 7)))
                    rows.append(
                        {
                            "text_id": f"t{n}",
                            "source": "gdelt_headline" if fb == "news" else "reddit_arctic",
                            "subreddit": "" if fb == "news" else fb,
                            "date": day.date().isoformat(),
                            "player_id": pid,
                            "player_name": f"Player {pid}",
                            "compound": float(np.clip(mean + rng.normal(0, noise), -1, 1)),
                        }
                    )
    return pd.DataFrame(rows)
