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

    p = sub.add_parser(
        "correlate", help="Fanbase effect of sentiment on next-week performance (hierarchical)"
    )
    p.add_argument(
        "--method",
        choices=["hierarchical", "partial"],
        default="hierarchical",
        help="'partial' = legacy per-pair partial correlations (OPS/ERA only), for comparison",
    )
    p.add_argument("--min-weeks", type=int, default=8, help="Min usable weeks per unit")
    p.add_argument("--min-mentions", type=int, default=3, help="Min mentions for a player-week")
    p.add_argument("--min-units", type=int, default=3, help="Min units per (metric, fanbase)")
    p.add_argument(
        "--reply-weight",
        type=float,
        default=1.0,
        help="Weight of reply_inherited mentions in the weekly mean (1 = same as explicit, 0 = drop)",
    )
    p.add_argument("--metrics", nargs="+", help="Subset of metrics (default: all with data)")
    p.add_argument("--no-composite", action="store_true", help="Skip the COMPOSITE index")
    p.add_argument("--oaa", help="OAA monthly CSV (default: data/oaa_monthly.csv if present)")
    p.add_argument("--perm-b", type=int, default=1000, help="Permutation replicates per metric")
    p.add_argument("--workers", type=int, default=1, help="Processes for the permutation null")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--correction", choices=["bh", "holm"], default="bh")
    p.add_argument("--alpha", type=float, default=0.05)

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
        if args.method == "partial":
            from . import analyze

            analyze.run(args.min_weeks, args.min_mentions)
        else:
            from pathlib import Path

            from . import correlate
            from .inference import InferenceConfig
            from .panel import PanelConfig

            correlate.run(
                PanelConfig(
                    min_mentions=args.min_mentions,
                    min_weeks=args.min_weeks,
                    min_units_per_fanbase=args.min_units,
                    reply_inherited_weight=args.reply_weight,
                ),
                InferenceConfig(perm_b=args.perm_b, workers=args.workers, seed=args.seed),
                args.metrics,
                not args.no_composite,
                Path(args.oaa) if args.oaa else None,
                args.correction,
                args.alpha,
            )


if __name__ == "__main__":
    main()
