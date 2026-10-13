"""fansent CLI — pipeline: rosters -> stats -> [fansent-collect: texts/news] -> score -> correlate"""

from __future__ import annotations

import argparse
from pathlib import Path


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

    p = sub.add_parser("score", help="Score texts.csv (from fansent-collect) into sentiment.csv")
    p.add_argument("--scorer", choices=["vader", "lexicon", "clf"], default="vader")
    p.add_argument("--model", help="Model .npz from clf-train (required for --scorer clf)")

    def add_grid(p):
        p.add_argument(
            "--players", default="data/players.csv", help="Roster CSV for __OTH__ masking"
        )
        p.add_argument("--cv", type=int, default=5, help="CV folds for tuning (<2 = no tuning)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--Cs", type=float, nargs="+", default=[1.0, 3.0, 10.0, 30.0])
        p.add_argument(
            "--windows", type=int, nargs="+", default=[0, 3, 5], help="0 = no window block"
        )
        p.add_argument("--char-ngrams", nargs="+", default=["0-0", "3-5"], help="lo-hi; 0-0 = off")
        p.add_argument("--word-ngrams", nargs="+", default=["1-2"])

    p = sub.add_parser(
        "clf-eval", help="Train on labeled CSVs, compare vader/lexicon/clf on a test CSV"
    )
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--save-model", help="Also save the trained model here")
    add_grid(p)

    p = sub.add_parser("clf-train", help="Train the classifier and save the model artifact")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--out", default="data/models/clf.npz")
    add_grid(p)

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

        sentiment.run(args.scorer, Path(args.model) if args.model else None)
    elif args.command in ("clf-eval", "clf-train"):
        from .classifier import evaluate

        grid = evaluate.Grid(
            tuple(args.Cs), tuple(args.windows), tuple(args.char_ngrams), tuple(args.word_ngrams)
        )
        train = [Path(t) for t in args.train]
        players = Path(args.players)
        if args.command == "clf-eval":
            save = Path(args.save_model) if args.save_model else None
            evaluate.run_eval(train, Path(args.test), players, args.cv, grid, args.seed, save)
        else:
            evaluate.run_train(train, Path(args.out), players, args.cv, grid, args.seed)
    elif args.command == "correlate":
        if args.method == "partial":
            from . import analyze

            analyze.run(args.min_weeks, args.min_mentions)
        else:
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
