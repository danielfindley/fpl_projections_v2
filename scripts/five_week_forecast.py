#!/usr/bin/env python
"""Build an isolated five-gameweek FPL forecast and standalone HTML report.

Nothing in this runner calls ``FPLPipeline.predict`` or ``save_run``. Its only
write target is ``output/five_week``; production predictions and pages are
hashed before and after the run as a guardrail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_loader
from src.five_week import (
    FiveWeekForecaster,
    HorizonRoleModel,
    backtest_role_variants,
    build_horizon_training_frame,
    current_durability,
)
from src.five_week_viz import build_report_payload, render_five_week_html
from src.pipeline import FPLPipeline


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_snapshot() -> dict:
    paths = [
        ROOT / "distributions.html",
        ROOT / "data" / "experiments.db",
        ROOT / "data" / "fpl_actual_points.csv",
        *sorted((ROOT / "data").glob("gw*.csv")),
    ]
    return {str(path): _file_hash(path) for path in paths if path.exists()}


def _season_and_gameweeks(count: int = 5) -> tuple[str, list[int]]:
    bootstrap = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20
    ).json()
    bootstrap_events = bootstrap.get("events", [])
    if not bootstrap_events:
        raise RuntimeError("FPL bootstrap returned no gameweeks")
    deadline = bootstrap_events[0]["deadline_time"]
    start_year = int(deadline[:4])
    season = f"{start_year}/{start_year + 1}"

    fixtures = requests.get(
        "https://fantasy.premierleague.com/api/fixtures/", timeout=20
    ).json()
    fixture_gws = sorted({int(f["event"]) for f in fixtures if f.get("event") is not None})
    unfinished = [int(event["id"]) for event in bootstrap_events if not event.get("finished")]
    if not unfinished:
        raise RuntimeError("No unfinished FPL gameweeks remain")
    start = min(unfinished)
    gameweeks = [gw for gw in fixture_gws if gw >= start][:count]
    if len(gameweeks) < count:
        raise RuntimeError(f"Only {len(gameweeks)} future gameweeks found; expected {count}")
    return season, gameweeks


def _latest_tuned_params() -> tuple[Path, dict]:
    candidates = sorted(
        (ROOT / "data" / "runs").glob("gw*/tuned_params.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No tuned_params.json found under data/runs")
    path = candidates[0]
    return path, json.loads(path.read_text(encoding="utf-8"))


def _inclusive_availability(live: dict) -> dict:
    """Keep zero-chance players in the row set so later horizons can recover."""
    inclusive = {}
    for name, value in live.items():
        item = dict(value) if isinstance(value, dict) else {}
        if item.get("chance_of_playing") == 0:
            item["chance_of_playing"] = 0.001
            item["status"] = "d"
        inclusive[name] = item
    return inclusive


def _write_long_csv(payload: dict, output_path: Path) -> None:
    rows = []
    for player in payload["players"]:
        for gameweek, week in player["weeks"].items():
            dist = week["distribution"]
            rows.append({
                "player_id": player["id"],
                "player_name": player["name"],
                "team": player["team"],
                "position": player["position"],
                "gameweek": int(gameweek),
                "fixture": week["fixture"],
                "expected_points": week["xpts"],
                "p10_points": dist["p10"],
                "median_points": dist["median"],
                "p90_points": dist["p90"],
                "appearance_probability": week["appearance"],
                "expected_minutes": week["minutes"],
                "minutes_if_playing": week["minutes_if_playing"],
                "expected_goals": week["xg"],
                "expected_assists": week["xa"],
                "clean_sheet_probability": week["clean_sheet"],
                "five_week_total": player["five_week_total"],
                "average_points": player["average_xpts"],
            })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=5000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet
    output_dir = ROOT / "output" / "five_week"
    output_dir.mkdir(parents=True, exist_ok=True)
    protected_before = _protected_snapshot()

    season, gameweeks = _season_and_gameweeks(5)
    params_path, tuned_params = _latest_tuned_params()
    print(f"Forecast window: {season}, GW{gameweeks[0]}-GW{gameweeks[-1]}")
    print(f"Component params: {params_path.parent.name}")

    cached_fpl_path = ROOT / "data" / "fpl_actual_points.csv"
    cached_fpl = pd.read_csv(cached_fpl_path) if cached_fpl_path.exists() else pd.DataFrame()
    # Loading normally refreshes this shared cache. Return the existing snapshot
    # instead so the experiment is read-only outside its output directory.
    with patch("src.data_loader.fetch_fpl_actual_points", return_value=cached_fpl):
        pipeline = FPLPipeline(str(ROOT / "data"), n_sims=args.n_sims)
        pipeline.load_data(verbose=verbose)
    pipeline.compute_features(verbose=verbose)

    print("\nBuilding direct GW+1..GW+5 role targets...")
    training_frame, role_grid = build_horizon_training_frame(pipeline.df)
    seasons = sorted(training_frame["season"].dropna().unique())
    completed = [value for value in seasons if value < season]
    if not completed:
        raise RuntimeError("No completed season is available for the role-model holdout")
    test_season = completed[-1]
    use_durability, metrics = backtest_role_variants(training_frame, test_season)
    role_model = HorizonRoleModel(use_durability=use_durability).fit(training_frame)
    metrics["final_training_rows"] = int(len(training_frame))
    metrics["feature_importance"] = role_model.feature_importance().head(25).to_dict("records")
    print(
        f"Role holdout {test_season}: baseline={metrics['baseline']['multiclass_log_loss']:.4f}, "
        f"durability={metrics['durability']['multiclass_log_loss']:.4f}; "
        f"selected={metrics['selected']}"
    )

    print("\nTraining existing component models in memory...")
    pipeline.tuned_params = tuned_params
    pipeline.train(verbose=verbose)

    live_availability = data_loader.get_fpl_availability()
    inclusive_availability = _inclusive_availability(live_availability)
    # `_build_test_set` normally drops 0%-chance players. The experiment retains
    # them, then FiveWeekForecaster restores the true chance and decays that live
    # injury/suspension evidence across future horizons.
    with patch("src.pipeline.get_fpl_availability", return_value=inclusive_availability), \
         patch("src.models.bonus.get_fpl_availability", return_value=inclusive_availability):
        base_rows = pipeline._build_test_set(gameweeks[0], season, verbose=verbose)
        if base_rows.empty:
            raise RuntimeError(f"No prediction rows generated for GW{gameweeks[0]}")
        forecaster = FiveWeekForecaster(
            pipeline,
            role_model,
            current_durability(role_grid),
            n_sims=args.n_sims,
            live_availability=live_availability,
        )
        forecasts = {}
        for horizon, gameweek in enumerate(gameweeks, start=1):
            print(f"\nForecasting GW{gameweek} (horizon {horizon})...")
            forecasts[gameweek] = forecaster.predict_week(
                base_rows, gameweek, season, horizon, verbose=verbose
            )

    payload = build_report_payload(forecasts, season, metrics)
    html_path = render_five_week_html(payload, output_dir / "five_week_predictions.html")
    _write_long_csv(payload, output_dir / "five_week_predictions.csv")
    (output_dir / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8"
    )
    with (output_dir / "role_model.pkl").open("wb") as handle:
        pickle.dump(role_model, handle)

    protected_after = _protected_snapshot()
    if protected_before != protected_after:
        changed = sorted(set(protected_before) | set(protected_after))
        changed = [path for path in changed if protected_before.get(path) != protected_after.get(path)]
        raise RuntimeError(f"Isolation guard failed; protected files changed: {changed}")

    print(f"\nCreated: {html_path}")
    print(f"Players: {len(payload['players'])}; simulations per fixture: {args.n_sims:,}")
    print("Production predictions, experiments DB, cache, and distributions.html are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
