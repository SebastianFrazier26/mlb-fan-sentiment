"""fansent CLI — pipeline: rosters -> stats -> [fansent-collect: texts/news] -> score -> correlate"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fansent",
        description="Correlate fan sentiment with MLB player performance.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("rosters", help="Fetch Cubs + Red Sox rosters into players.csv")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument(
        "--roster-type", default="fullSeason", help="MLB API rosterType (fullSeason, active, 40Man)"
    )

    p = sub.add_parser("stats", help="Pull game logs, aggregate to weekly stats.csv")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--limit", type=int, help="Only first N players (smoke testing)")

    p = sub.add_parser(
        "oaa", help="Load a Baseball Savant monthly OAA CSV export into oaa_monthly.csv"
    )
    p.add_argument("path", help="Path to the Savant Outs Above Average CSV export")

    sub.add_parser("score", help="VADER-score texts.csv (from fansent-collect) into sentiment.csv")

    p = sub.add_parser("correlate", help="Lagged partial correlations + fanbase comparison")
    p.add_argument(
        "--min-weeks", type=int, default=8, help="Min usable weeks per (fanbase, player)"
    )
    p.add_argument(
        "--min-mentions", type=int, default=3, help="Min Reddit mentions for a player-week to count"
    )

    args = parser.parse_args()

    if args.command == "rosters":
        from . import rosters

        rosters.run(args.season, args.roster_type)
    elif args.command == "stats":
        from . import stats

        stats.run(args.season, args.limit)
    elif args.command == "oaa":
        from . import oaa

        oaa.run(args.path)
    elif args.command == "score":
        from . import sentiment

        sentiment.run()
    elif args.command == "correlate":
        from . import analyze

        analyze.run(args.min_weeks, args.min_mentions)


if __name__ == "__main__":
    main()
