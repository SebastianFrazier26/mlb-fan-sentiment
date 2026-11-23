"""fansent CLI — pipeline: rosters -> stats -> [fansent-collect: texts/news] -> score -> correlate"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from .util import (
    ENV_VAR,
    MODEL_FILE,
    MODELS_DIR,
    PLAYERS_FILE,
    RELEVANCE_MODEL_FILE,
    TEAMS_FILE,
    resolve_data_dir,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fansent",
        description="Correlate fan sentiment with MLB player performance.",
    )
    parser.add_argument(
        "--data-dir",
        metavar="DIR",
        help=f"Directory for all CSV inputs/outputs (default: ${ENV_VAR}, else ./data); "
        "give before the subcommand",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "rosters",
        help="Fetch weekly 40-man rosters for all 30 teams -> players.csv, player_teams.csv",
    )
    p.add_argument("--season", type=int, default=2026)
    p.add_argument(
        "--teams", nargs="+", metavar="TEAM", help="Restrict to these abbrevs/ids (e.g. CHC 111)"
    )
    p.add_argument("--first-week", type=dt.date.fromisoformat, help="Monday; needed off 2026")
    p.add_argument("--last-week", type=dt.date.fromisoformat, help="Monday; needed off 2026")

    p = sub.add_parser("stats", help="Pull game logs, aggregate to weekly stats.csv")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--limit", type=int, help="Only first N players (smoke testing)")
    p.add_argument(
        "--refresh", action="store_true", help="Ignore cached raw/mlb/gamelogs and refetch"
    )

    p = sub.add_parser(
        "oaa", help="Load a Baseball Savant monthly OAA CSV export into oaa_monthly.csv"
    )
    p.add_argument("path", help="Path to the Savant Outs Above Average CSV export")

    p = sub.add_parser("score", help="Score texts.csv (from fansent-collect) into sentiment.csv")
    p.add_argument("--scorer", choices=["vader", "lexicon", "clf"], default="vader")
    p.add_argument(
        "--model",
        help="Model .npz from clf-train (--scorer clf; default: <data-dir>/models/clf.npz)",
    )
    p.add_argument(
        "--relevance-model",
        help="Relevance model .npz (clf-train --task relevance); rows it predicts 'na' are "
        "dropped and audited in relevance.csv (default: <data-dir>/models/relevance.npz if present)",
    )
    p.add_argument(
        "--no-relevance", action="store_true", help="Score every matched row, no relevance filter"
    )

    def add_task(p):
        p.add_argument(
            "--task",
            choices=["sentiment", "relevance"],
            default="sentiment",
            help="relevance = predict whether a matched comment is about the matched player",
        )
        p.add_argument(
            "--operating-point",
            choices=["f1", "precision"],
            default="precision",
            help="relevance threshold: max na-F1, or the lowest threshold with CV precision "
            ">= --min-precision",
        )
        p.add_argument("--min-precision", type=float, default=0.8)

    def add_grid(p):
        p.add_argument(
            "--players",
            help="Roster CSV for __OTH__ masking (default: <data-dir>/players.csv)",
        )
        p.add_argument("--cv", type=int, default=5, help="CV folds for tuning (<2 = no tuning)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--Cs", type=float, nargs="+", default=[1.0, 3.0, 10.0, 30.0])
        p.add_argument(
            "--windows", type=int, nargs="+", default=[0, 3, 5], help="0 = no window block"
        )
        p.add_argument("--char-ngrams", nargs="+", default=["0-0", "3-5"], help="lo-hi; 0-0 = off")
        p.add_argument("--word-ngrams", nargs="+", default=["1-2"])
        p.add_argument("--emb", action="store_true", help="Add fastText mean-vector block")
        p.add_argument("--lex", action="store_true", help="Add lexicon-score block")
        p.add_argument("--parent", action="store_true", help="Add parent-text block + is_reply")
        p.add_argument(
            "--emb-path", help="fastText .bin (default: <data-dir>/embeddings/<crawl-300d-2M>.bin)"
        )
        p.add_argument("--emb-weights", type=float, nargs="+", default=[1.0, 2.0])
        p.add_argument("--lex-weights", type=float, nargs="+", default=[1.0, 2.0])
        p.add_argument("--parent-weights", type=float, nargs="+", default=[0.5, 1.0])

    p = sub.add_parser(
        "clf-eval", help="Train on labeled CSVs, compare vader/lexicon/clf on a test CSV"
    )
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--save-model", help="Also save the trained model here")
    p.add_argument(
        "--holdout",
        type=float,
        default=0.15,
        help="relevance only: fraction of --train held out before tuning for the Sonnet-split report",
    )
    add_task(p)
    add_grid(p)

    p = sub.add_parser("clf-train", help="Train the classifier and save the model artifact")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument(
        "--out",
        help="Model path (default: <data-dir>/models/clf.npz, or relevance.npz for --task relevance)",
    )
    add_task(p)
    add_grid(p)

    p = sub.add_parser(
        "clf-fetch-embeddings",
        help="Download the official fastText crawl-300d-2M-subword model (~5.8 GB zip, 7.2 GB bin)",
    )
    p.add_argument("--keep-zip", action="store_true")

    p = sub.add_parser("label", help="Label a sentiment-eval CSV in a local browser page")
    p.add_argument(
        "--file",
        required=True,
        help="CSV with at least id, player, text columns; relative paths resolve from the "
        "current directory, not --data-dir",
    )
    p.add_argument("--port", type=int, default=8765, help="Port on 127.0.0.1 (0 = any free)")
    p.add_argument("--open", action="store_true", help="Open the page in the default browser")

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
    p.add_argument(
        "--all-teams",
        action="store_true",
        help="Disable the own-team filter: team subreddits count every player (sensitivity check)",
    )
    p.add_argument("--metrics", nargs="+", help="Subset of metrics (default: all with data)")
    p.add_argument("--no-composite", action="store_true", help="Skip the COMPOSITE index")
    p.add_argument("--oaa", help="OAA monthly CSV (default: <data-dir>/oaa_monthly.csv if present)")
    p.add_argument("--perm-b", type=int, default=1000, help="Permutation replicates per metric")
    p.add_argument("--chains", type=int, default=4, help="Chains for the main fit")
    p.add_argument("--iter", type=int, default=6000, help="Iterations per chain, main fit")
    p.add_argument("--burn", type=int, default=3000, help="Burn-in per chain, main fit")
    p.add_argument(
        "--perm-iter", type=int, default=600, help="Iterations per chain, permutation fits"
    )
    p.add_argument("--perm-burn", type=int, default=300, help="Burn-in per chain, permutation fits")
    p.add_argument(
        "--teams",
        nargs="+",
        metavar="TEAM_ID|ABBR",
        help="Keep only sentiment rows whose player was on one of these teams in the Mon-Sun "
        "week of the text date (player_teams.csv; abbreviations via teams.csv)",
    )
    p.add_argument(
        "--exclude-teams",
        nargs="+",
        metavar="TEAM_ID|ABBR",
        help="Inverse of --teams: drop sentiment rows whose player was on one of these teams "
        "in the Mon-Sun week of the text date (needs player_teams.csv)",
    )
    p.add_argument(
        "--pool-fanbases",
        action="store_true",
        help="Fit all included fanbases' units as one group (single mu, reported as 'pooled'). "
        "Units stay (fanbase, player): a player in two fanbases is two units sharing outcomes",
    )
    p.add_argument(
        "--alternative",
        choices=["two-sided", "less", "greater"],
        default="two-sided",
        help="Which permutation p (perm_p / perm_p_less / perm_p_greater) feeds the q-values "
        "and the * marker; all three are always written",
    )
    p.add_argument(
        "--null",
        choices=["freedman-lane", "shuffle"],
        default="freedman-lane",
        help="Permutation null: freedman-lane permutes within-unit residuals of sentiment after "
        "regressing on the model's controls (valid when fans react to performance); shuffle "
        "permutes sentiment itself (the original null, biased in that case)",
    )
    p.add_argument(
        "--estimator",
        choices=["pooled", "mundlak"],
        default="pooled",
        help="pooled: gamma_u multiplies raw sentiment (mu mixes between- and within-player "
        "association; the registered estimator). mundlak: within-player effect, with unit "
        "means of sentiment/perf/std as extra covariates (adds beta_between); a different "
        "estimand, exploratory until pre-registered",
    )
    p.add_argument(
        "--pool-stats",
        choices=["league", "teams"],
        default="league",
        help="COMPOSITE z-score pool: every player (league) or only the players left after "
        "--teams/--fanbases/--exclude-fanbases (teams)",
    )
    p.add_argument(
        "--fanbases", nargs="+", metavar="NAME", help="Include only these fanbases (e.g. Yankees)"
    )
    p.add_argument(
        "--exclude-fanbases",
        nargs="+",
        default=[],
        metavar="NAME",
        help="Drop these fanbases (e.g. news)",
    )
    p.add_argument("--workers", type=int, default=1, help="Processes for the permutation null")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--correction", choices=["bh", "holm"], default="bh")
    p.add_argument("--alpha", type=float, default=0.05)

    p = sub.add_parser(
        "confirm",
        help="Run the pre-registered league-wide fielding confirmatory test "
        "(spec: docs/preregistration.md)",
        description="Run the confirmatory test exactly as pre-registered in "
        "docs/preregistration.md: FLD_Z only, CHC/BOS player-weeks and news excluded, fanbases "
        "pooled, 4 x 6000 iterations (burn 3000), B = 1000, one-sided permutation p, max R-hat "
        "< 1.05. The specification is hard-coded and has no flags to change it; --workers and "
        "--seed do not alter the estimand. Run once, after the data are final. Writes "
        "confirm_result.json to the data directory.",
    )
    p.add_argument("--workers", type=int, default=1, help="Processes for the permutation null")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser(
        "power",
        help="Simulated power / minimum detectable effect of the fanbase test",
    )
    p.add_argument("--reps", type=int, default=100, help="Simulated datasets per grid point")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=["team", "baseball", "league", "pooled"],
        help="team, baseball, league, pooled, pitching",
    )
    p.add_argument("--mus", nargs="+", type=float, help="True fanbase effects (sd per sd)")
    p.add_argument("--pooled-mus", nargs="+", type=float, help="Grid for the pooled scenario")
    p.add_argument("--league-units", type=int, default=25, help="Units per fanbase, league")
    p.add_argument("--league-fanbases", type=int, default=30)
    p.add_argument("--tau", type=float, default=0.1, help="Between-player slope sd")
    p.add_argument("--sigma", type=float, default=1.0, help="Residual noise sd (signal sd ~0.3)")
    p.add_argument("--perm-b", type=int, default=0, help="Confirm with permutation test at B")
    p.add_argument("--perm-reps", type=int, default=10, help="Replicates per point to confirm")
    p.add_argument("--target", type=float, default=0.8)
    p.add_argument("--out-dir", help="Where to write power_*.csv (default: data dir)")

    p = sub.add_parser(
        "snapshot",
        help="Write the aggregate-only public snapshot of a correlate run (results/published)",
    )
    p.add_argument("run_dir", nargs="?", help="Directory holding analysis_fanbases.csv")
    p.add_argument(
        "--section",
        action="append",
        default=[],
        metavar="SCOPE=RUN_DIR",
        help="Schema-2 snapshot with one section per scope (league, demo, league_fanbases); "
        "repeatable, replaces "
        "run_dir and needs --run-id",
    )
    p.add_argument(
        "--null",
        choices=["freedman-lane", "shuffle"],
        default="freedman-lane",
        help="Permutation null the --section runs used",
    )
    p.add_argument("--confirm", help="confirm_result.json to summarize (with --section)")
    p.add_argument("--run-id", help="Snapshot id (default: the run directory's name)")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--perm-b", type=int, default=1000, help="--perm-b the run used")
    p.add_argument(
        "--status",
        choices=["preliminary", "final"],
        default="preliminary",
        help="final drops the site's preliminary banner",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out-dir", default="results/published")

    p = sub.add_parser(
        "figures", help="Write the README's static SVG figures from a published snapshot"
    )
    p.add_argument("--snapshot", help="Snapshot JSON (default: newest in results/published)")
    p.add_argument("--out", default="docs/img", help="Output directory (default: docs/img)")

    p = sub.add_parser("site", help="Render the static results site from a published snapshot")
    p.add_argument("--snapshot", help="Snapshot JSON (default: newest in results/published)")
    p.add_argument("--out", default="site", help="Output directory (default: ./site)")

    args = parser.parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)

    if args.command == "rosters":
        from . import rosters

        rosters.run(args.season, data_dir, args.teams, args.first_week, args.last_week)
    elif args.command == "stats":
        from . import stats

        stats.run(args.season, args.limit, data_dir=data_dir, refresh=args.refresh)
    elif args.command == "oaa":
        from . import oaa

        oaa.run(args.path, data_dir)
    elif args.command == "score":
        from . import sentiment

        sentiment.run(
            args.scorer,
            Path(args.model) if args.model else None,
            data_dir,
            Path(args.relevance_model) if args.relevance_model else None,
            not args.no_relevance,
        )
    elif args.command == "clf-fetch-embeddings":
        from .classifier import embeddings

        embeddings.fetch(data_dir / embeddings.EMBEDDINGS_DIR, keep_zip=args.keep_zip)
    elif args.command in ("clf-eval", "clf-train") and args.task == "relevance":
        from .classifier import relevance_eval

        players = Path(args.players) if args.players else data_dir / PLAYERS_FILE
        grid = relevance_eval.RelGrid()
        point = (args.operating_point, args.min_precision)
        train = [Path(t) for t in args.train]
        if args.command == "clf-eval":
            save = Path(args.save_model) if args.save_model else None
            relevance_eval.run_eval(
                train,
                Path(args.test),
                players,
                args.cv,
                grid,
                args.seed,
                save,
                args.holdout,
                *point,
            )
        else:
            out = Path(args.out) if args.out else data_dir / MODELS_DIR / RELEVANCE_MODEL_FILE
            relevance_eval.run_train(train, out, players, args.cv, grid, args.seed, *point)
    elif args.command in ("clf-eval", "clf-train"):
        from .classifier import embeddings, evaluate
        from .classifier.features import FeatureConfig
        from .classifier.pipeline import ClfConfig

        grid = evaluate.Grid(
            tuple(args.Cs),
            tuple(args.windows),
            tuple(args.char_ngrams),
            tuple(args.word_ngrams),
            tuple(args.emb_weights),
            tuple(args.lex_weights),
            tuple(args.parent_weights),
        )
        base = ClfConfig(features=FeatureConfig(emb=args.emb, lex=args.lex, parent=args.parent))
        vectors = None
        if args.emb:
            bin_path = (
                Path(args.emb_path)
                if args.emb_path
                else data_dir / embeddings.EMBEDDINGS_DIR / embeddings.DEFAULT_BIN
            )
            if not bin_path.exists():
                parser.error(f"{bin_path} not found; run `fansent clf-fetch-embeddings` first")
            vectors = embeddings.TokenVectors(embeddings.FastTextVectors(bin_path))
        train = [Path(t) for t in args.train]
        players = Path(args.players) if args.players else data_dir / PLAYERS_FILE
        if args.command == "clf-eval":
            save = Path(args.save_model) if args.save_model else None
            evaluate.run_eval(
                train, Path(args.test), players, args.cv, grid, args.seed, save, base, vectors
            )
        else:
            out = Path(args.out) if args.out else data_dir / MODELS_DIR / MODEL_FILE
            evaluate.run_train(train, out, players, args.cv, grid, args.seed, base, vectors)
    elif args.command == "label":
        from . import label

        label.run(args.file, args.port, args.open)
    elif args.command == "snapshot":
        from . import site

        if args.section:
            if args.run_dir or not args.run_id:
                parser.error("--section replaces run_dir and needs --run-id")
            snap = site.snapshot_from_runs(
                {k: Path(v) for k, v in (s.split("=", 1) for s in args.section)},
                args.run_id,
                args.season,
                args.perm_b,
                args.null,
                Path(args.confirm) if args.confirm else None,
                args.alpha,
                args.status,
            )
        else:
            if not args.run_dir:
                parser.error("run_dir or --section is required")
            run_dir = Path(args.run_dir)
            snap = site.snapshot_from_run(
                run_dir,
                args.run_id or run_dir.resolve().name,
                args.season,
                args.perm_b,
                args.alpha,
                args.status,
            )
        print(f"wrote {site.write_snapshot(snap, Path(args.out_dir))}")
    elif args.command == "figures":
        from . import figures, site

        snap_path = (
            Path(args.snapshot)
            if args.snapshot
            else site.latest_snapshot(site.DEFAULT_PUBLISHED_DIR)
        )
        for path in figures.write_figures(snap_path, Path(args.out)):
            print(f"wrote {path}")
    elif args.command == "site":
        from . import site

        snap_path = (
            Path(args.snapshot)
            if args.snapshot
            else site.latest_snapshot(site.DEFAULT_PUBLISHED_DIR)
        )
        print(f"wrote {site.build_site(snap_path, Path(args.out))} from {snap_path}")
    elif args.command == "power":
        from . import power

        power.run(
            data_dir,
            power.PowerConfig(
                reps=args.reps,
                seed=args.seed,
                workers=args.workers,
                dgp=power.Dgp(tau=args.tau, sigma=args.sigma),
                perm_b=args.perm_b,
                perm_reps=args.perm_reps,
                target=args.target,
            ),
            args.scenarios,
            tuple(args.mus or power.DEFAULT_MUS),
            tuple(args.pooled_mus or power.DEFAULT_POOLED_MUS),
            args.league_units,
            args.league_fanbases,
            Path(args.out_dir) if args.out_dir else None,
        )
    elif args.command == "confirm":
        from . import confirm

        confirm.run(data_dir, workers=args.workers, seed=args.seed)
    elif args.command == "correlate":
        if args.method == "partial":
            from . import analyze

            analyze.run(args.min_weeks, args.min_mentions, data_dir)
        else:
            from . import correlate
            from .inference import InferenceConfig
            from .panel import PanelConfig

            if not 0 < args.burn < args.iter or not 0 < args.perm_burn < args.perm_iter:
                raise SystemExit("burn must be positive and smaller than iter")
            scope = correlate.Scope(
                teams=correlate.resolve_teams(args.teams, data_dir / TEAMS_FILE)
                if args.teams
                else None,
                exclude_teams=correlate.resolve_teams(args.exclude_teams, data_dir / TEAMS_FILE)
                if args.exclude_teams
                else frozenset(),
                fanbases=frozenset(args.fanbases) if args.fanbases else None,
                exclude_fanbases=frozenset(args.exclude_fanbases),
                pool_stats=args.pool_stats,
            )
            correlate.run(
                PanelConfig(
                    min_mentions=args.min_mentions,
                    min_weeks=args.min_weeks,
                    min_units_per_fanbase=args.min_units,
                    reply_inherited_weight=args.reply_weight,
                    **({"fanbase_teams": {}} if args.all_teams else {}),
                ),
                InferenceConfig(
                    n_chains=args.chains,
                    n_iter=args.iter,
                    burn=args.burn,
                    perm_iter=args.perm_iter,
                    perm_burn=args.perm_burn,
                    perm_b=args.perm_b,
                    workers=args.workers,
                    seed=args.seed,
                    pool_fanbases=args.pool_fanbases,
                    alternative=args.alternative,
                    null=args.null,
                    estimator=args.estimator,
                ),
                args.metrics,
                not args.no_composite,
                Path(args.oaa) if args.oaa else None,
                args.correction,
                args.alpha,
                data_dir,
                scope,
            )


if __name__ == "__main__":
    main()
