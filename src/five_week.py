"""Isolated five-gameweek forecasting utilities.

This module deliberately does not call ``FPLPipeline.predict`` or ``save_run``:
those methods persist production predictions.  The experimental runner uses the
pipeline's in-memory feature/model helpers and writes only beneath
``output/five_week``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, mean_absolute_error

from .data_loader import get_fpl_positions, map_fpl_position
from .features import (APPEARANCE_FEATURES, _compute_calendar_minutes_features,
                       chronological_frame, prior_stat, resolve_defcon_positions)
from .pipeline import FPL_POINTS


ROLE_NAMES = ("Absent", "Sub appearance", "60-79 minutes", "80+ minutes")
ROLE_COLORS = ("#64748b", "#f59e0b", "#38bdf8", "#34d399")
DURABILITY_FEATURES = (
    "appear_rate_roll20",
    "starter_rate_roll20",
    "full90_rate_roll20",
    "career_appear_rate",
    "career_starter_rate",
    "career_full90_rate",
    "career_gws_prior",
)


def _role_state(minutes: pd.Series) -> np.ndarray:
    mins = pd.to_numeric(minutes, errors="coerce").fillna(0).to_numpy()
    return np.select(
        [mins <= 0, mins < 60, mins < 80],
        [0, 1, 2],
        default=3,
    ).astype(int)


def build_role_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe weekly role features and labels.

    Every feature at GW ``g`` is based on outcomes through GW ``g-1``.  The
    returned ``role_state``/``target_minutes`` columns describe GW ``g`` and are
    labels only.
    """
    grid = _compute_calendar_minutes_features(df, include_appeared=True, per_fixture=True)

    keys = ["player_id", "match_id"] if "match_id" in grid else ["player_id", "season", "gameweek"]
    minutes = df.groupby(keys, as_index=False)["minutes"].max().rename(columns={"minutes": "target_minutes"})
    grid = grid.merge(minutes, on=keys, how="left")
    grid["target_minutes"] = grid["target_minutes"].fillna(0).clip(0, 90)
    grid["role_state"] = _role_state(grid["target_minutes"])
    grid["started"] = (grid["target_minutes"] >= 60).astype(int)
    grid["full90"] = (grid["target_minutes"] >= 80).astype(int)

    pos_cols = [c for c in ("is_gk", "is_def", "is_mid", "is_fwd") if c in df.columns]
    if pos_cols:
        positions = chronological_frame(df).sort_values("match_date").groupby("player_id", as_index=False)[pos_cols].first()
        grid = grid.merge(positions, on="player_id", how="left")
    for col in ("is_gk", "is_def", "is_mid", "is_fwd"):
        if col not in grid:
            grid[col] = 0
        grid[col] = grid[col].fillna(0).astype(int)

    grid = chronological_frame(grid).sort_values(["player_id", "match_date"]).reset_index(drop=True)
    career_group = grid.groupby("player_id", sort=False)

    for source, prefix in (
        ("appeared", "appear"),
        ("started", "starter"),
        ("full90", "full90"),
    ):
        grid[f"{prefix}_rate_roll20"] = prior_stat(grid, "player_id", source, 20)
        grid[f"career_{prefix}_rate"] = prior_stat(grid, "player_id", source)
    grid["career_gws_prior"] = prior_stat(grid.assign(_one=1), "player_id", "_one", aggregation="sum").fillna(0)
    defaults = {
        "appear_rate_roll20": 0.75,
        "starter_rate_roll20": 0.55,
        "full90_rate_roll20": 0.45,
        "career_appear_rate": 0.75,
        "career_starter_rate": 0.55,
        "career_full90_rate": 0.45,
        "career_gws_prior": 0.0,
    }
    for col, default in defaults.items():
        grid[col] = pd.to_numeric(grid[col], errors="coerce").fillna(default)
    return grid


def build_horizon_training_frame(
    df: pd.DataFrame,
    horizons: Iterable[int] = range(1, 6),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Expand weekly player snapshots into direct GW+1 ... GW+5 targets."""
    grid = build_role_grid(df)
    first_gw = grid.groupby(["player_id", "season"])["gameweek"].transform("min")
    anchors = grid[grid["gameweek"] > first_gw].drop_duplicates(["player_id", "season", "gameweek"]).copy()
    target = grid[
        ["player_id", "season", "gameweek", "role_state", "target_minutes", "result_time"]
    ].rename(columns={"gameweek": "target_gameweek", "result_time": "target_result_time"})

    frames = []
    for horizon in horizons:
        part = anchors.drop(columns=["role_state", "target_minutes", "started", "full90"]).copy()
        # Horizon 1 is the next fixture: its pre-match feature row and label share
        # a GW. Horizon 5 uses that same information snapshot four GWs earlier.
        part["target_gameweek"] = part["gameweek"] + int(horizon) - 1
        part["horizon"] = int(horizon)
        part = part.merge(
            target,
            on=["player_id", "season", "target_gameweek"],
            how="inner",
        )
        frames.append(part)
    return pd.concat(frames, ignore_index=True), grid


def current_durability(grid: pd.DataFrame) -> pd.DataFrame:
    """Durability priors through the latest observed week for each player."""
    ordered = grid.sort_values(["player_id", "season", "gameweek"]).copy()
    current = ordered.groupby("player_id", as_index=False).agg(
        career_appear_rate=("appeared", "mean"),
        career_starter_rate=("started", "mean"),
        career_full90_rate=("full90", "mean"),
        career_gws_prior=("appeared", "size"),
    )
    latest = ordered.groupby("player_id", as_index=False).agg(
        appear_rate_roll20=("appeared", lambda values: values.tail(20).mean()),
        starter_rate_roll20=("started", lambda values: values.tail(20).mean()),
        full90_rate_roll20=("full90", lambda values: values.tail(20).mean()),
    )
    current = current.merge(
        latest[["player_id", "appear_rate_roll20", "starter_rate_roll20", "full90_rate_roll20"]],
        on="player_id",
        how="left",
    )
    return current


class HorizonRoleModel:
    """Pooled direct multi-horizon classifier for future playing-time states."""

    def __init__(self, use_durability: bool = True, **params):
        self.use_durability = use_durability
        self.features = list(APPEARANCE_FEATURES)
        if use_durability:
            self.features += list(DURABILITY_FEATURES)
        self.features += ["horizon"]
        defaults = {
            "objective": "multi:softprob",
            "num_class": 4,
            "eval_metric": "mlogloss",
            "n_estimators": 220,
            "max_depth": 5,
            "learning_rate": 0.045,
            "min_child_weight": 12,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }
        defaults.update(params)
        self.model = xgb.XGBClassifier(**defaults)
        self.state_minutes = np.array([0.0, 25.0, 70.0, 88.0])
        self.is_fitted = False

    def _prepare_x(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for feature in self.features:
            if feature not in out:
                out[feature] = 0.0
        return out[self.features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    def fit(self, frame: pd.DataFrame) -> "HorizonRoleModel":
        self.model.fit(self._prepare_x(frame), frame["role_state"].astype(int))
        means = frame.groupby("role_state")["target_minutes"].mean()
        self.state_minutes = np.array(
            [0.0] + [float(means.get(state, self.state_minutes[state])) for state in (1, 2, 3)]
        )
        self.is_fitted = True
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("HorizonRoleModel is not fitted")
        probabilities = self.model.predict_proba(self._prepare_x(frame))
        probabilities = np.asarray(probabilities, dtype=float)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.features,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)


def evaluate_role_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    use_durability: bool,
) -> Tuple[HorizonRoleModel, Dict]:
    if "target_result_time" in train and "forecast_time" in test:
        train = train[pd.to_datetime(train["target_result_time"], utc=True) <
                      pd.to_datetime(test["forecast_time"], utc=True).min()].copy()
    model = HorizonRoleModel(use_durability=use_durability).fit(train)
    probability = model.predict_proba(test)
    actual_role = test["role_state"].astype(int).to_numpy()
    actual_appear = (actual_role > 0).astype(float)
    predicted_appear = 1.0 - probability[:, 0]
    predicted_minutes = probability @ model.state_minutes

    metrics = {
        "description": (
            "direct pooled GW+1..GW+5 role model with leakage-safe durability features"
            if use_durability else
            "direct pooled GW+1..GW+5 role model baseline"
        ),
        "multiclass_log_loss": float(log_loss(actual_role, probability, labels=[0, 1, 2, 3])),
        "appearance_brier": float(np.mean((predicted_appear - actual_appear) ** 2)),
        "minutes_mae": float(mean_absolute_error(test["target_minutes"], predicted_minutes)),
        "by_horizon": {},
    }
    for horizon in range(1, 6):
        mask = test["horizon"].to_numpy() == horizon
        if not mask.any():
            continue
        metrics["by_horizon"][str(horizon)] = {
            "n": int(mask.sum()),
            "appearance_brier": float(np.mean(
                (predicted_appear[mask] - actual_appear[mask]) ** 2
            )),
            "minutes_mae": float(mean_absolute_error(
                test.loc[mask, "target_minutes"], predicted_minutes[mask]
            )),
        }
    return model, metrics


def backtest_role_variants(
    training_frame: pd.DataFrame,
    test_season: str,
) -> Tuple[bool, Dict]:
    """Compare the one-variable-family change against a direct-horizon baseline."""
    train = training_frame[training_frame["season"] < test_season].copy()
    test = training_frame[training_frame["season"] == test_season].copy()
    if train.empty or test.empty:
        raise ValueError(f"Insufficient data for chronological holdout {test_season}")

    _, baseline = evaluate_role_model(train, test, use_durability=False)
    _, durability = evaluate_role_model(train, test, use_durability=True)
    selected = durability["multiclass_log_loss"] <= baseline["multiclass_log_loss"]
    return selected, {
        "test_season": test_season,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "baseline": baseline,
        "durability": durability,
        "selected": "durability" if selected else "baseline",
    }


def prepare_role_input(
    rows: pd.DataFrame,
    durability: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    out = rows.copy()
    out = out.merge(durability, on="player_id", how="left", suffixes=("", "_dur"))
    for feature in DURABILITY_FEATURES:
        alternate = f"{feature}_dur"
        if alternate in out:
            out[feature] = out[alternate]
    out["horizon"] = int(horizon)
    return out


def apply_live_availability(
    probability: np.ndarray,
    chance_of_playing: pd.Series,
    horizon: int,
) -> np.ndarray:
    """Blend next-round FPL status chance out over longer horizons.

    Historical FPL injury snapshots are unavailable, so recovery cannot yet be
    learned.  Current status is authoritative for GW+1 and its effect halves each
    additional horizon instead of incorrectly treating a short injury as permanent.
    """
    out = np.asarray(probability, dtype=float).copy()
    chance = pd.to_numeric(chance_of_playing, errors="coerce").fillna(100).to_numpy() / 100.0
    status_weight = 0.5 ** (int(horizon) - 1)
    availability = 1.0 - (1.0 - chance) * status_weight
    old_play = np.clip(1.0 - out[:, 0], 1e-9, 1.0)
    new_play = old_play * availability
    out[:, 1:] *= (new_play / old_play)[:, None]
    out[:, 0] = 1.0 - new_play
    return out / out.sum(axis=1, keepdims=True)


@dataclass
class WeekForecast:
    predictions: pd.DataFrame
    simulations: Dict[int, np.ndarray]


class FiveWeekForecaster:
    """Run existing component models with horizon-aware role probabilities in memory."""

    _FROZEN_ROLE_COLUMNS = tuple(
        dict.fromkeys(list(APPEARANCE_FEATURES) + [
            "starter_score", "current_season_minutes", "current_season_apps",
            "current_season_mins_per_app", "minutes_trend", "last_app_prev_season",
        ])
    )

    def __init__(self, pipeline, role_model: HorizonRoleModel, durability: pd.DataFrame,
                 n_sims: int = 5000, seed: int = 42,
                 live_availability: Optional[Dict] = None):
        self.pipeline = pipeline
        self.role_model = role_model
        self.durability = durability
        self.n_sims = int(n_sims)
        self.seed = int(seed)
        self.fpl_positions = get_fpl_positions()
        self.live_availability = live_availability or {}

    def _live_chance(self, player_name: str, fallback: float = 100.0) -> float:
        """Resolve the true current FPL chance after the build-step inclusion shim."""
        name = str(player_name).lower().strip()
        info = self.live_availability.get(name)
        if info is None and " " in name:
            info = self.live_availability.get(name.split()[-1])
        if not isinstance(info, dict):
            return float(fallback)
        chance = info.get("chance_of_playing")
        status = info.get("status", "a")
        if chance is None:
            return 100.0
        if status in ("i", "s", "u") and chance == 0:
            return 0.0
        return float(chance)

    def _freeze_current_player_state(self, future: pd.DataFrame,
                                     base: pd.DataFrame) -> pd.DataFrame:
        future = future.copy()
        current = base.drop_duplicates("player_id").set_index("player_id")
        for col in self._FROZEN_ROLE_COLUMNS:
            if col in current and col in future:
                mapped = future["player_id"].map(current[col])
                future[col] = mapped.where(mapped.notna(), future[col])
        return future

    def predict_week(self, base_rows: pd.DataFrame, gameweek: int, season: str,
                     horizon: int, verbose: bool = True) -> WeekForecast:
        p = self.pipeline
        future = p._build_test_set(gameweek, season, verbose)
        if future.empty:
            return WeekForecast(future, {})
        future = self._freeze_current_player_state(future, base_rows)
        future = p._recompute_interaction_features(future)

        if self.live_availability:
            future["fpl_chance_of_playing"] = [
                self._live_chance(name, fallback)
                for name, fallback in zip(
                    future["player_name"],
                    future.get("fpl_chance_of_playing", pd.Series(100, index=future.index)),
                )
            ]

        role_input = prepare_role_input(future, self.durability, horizon)
        probability = self.role_model.predict_proba(role_input)
        probability = apply_live_availability(
            probability,
            future.get("fpl_chance_of_playing", pd.Series(100, index=future.index)),
            horizon,
        )
        for idx, name in enumerate(ROLE_NAMES):
            future[f"role_prob_{idx}"] = probability[:, idx]
        future["pred_appear_prob"] = 1.0 - probability[:, 0]
        unconditional_minutes = probability @ self.role_model.state_minutes
        future["pred_minutes_uncond"] = unconditional_minutes
        future["pred_minutes"] = np.divide(
            unconditional_minutes,
            future["pred_appear_prob"].to_numpy(),
            out=np.zeros(len(future), dtype=float),
            where=future["pred_appear_prob"].to_numpy() > 1e-8,
        ).clip(1, 90)

        cs, two_plus, against = p._predict_clean_sheet(future, gameweek, season, verbose)
        future["pred_cs_prob"] = cs
        future["pred_2plus_conceded"] = two_plus
        future["pred_goals_against"] = against
        future["pred_team_goals"] = p._get_pred_team_goals(future, gameweek, season)
        live_position = future.apply(
            lambda row: map_fpl_position(
                row.get("position"), row.get("player_name"), self.fpl_positions,
                fallback_to_fotmob=False,
            ),
            axis=1,
        ).astype("string").str.upper()
        cached_position = future.get(
            "fpl_position", pd.Series(pd.NA, index=future.index, dtype="string")
        ).astype("string").str.upper()
        future["fpl_position"] = live_position.where(
            live_position.isin(["GK", "DEF", "MID", "FWD"]), cached_position
        )
        future = resolve_defcon_positions(future)
        future["pred_exp_goals"] = p.models["goals"].predict(future)
        future["pred_exp_assists"] = p.models["assists"].predict(future)
        future["pred_defcon_prob"] = p.models["defcon"].predict_threshold_prob(
            future, future["pred_minutes"]
        )
        future["pred_yellow_prob"] = p.models["cards"].predict_yellow_prob(
            future, future["pred_minutes"].to_numpy()
        )
        future["pred_red_prob"] = p.models["cards"].predict_red_prob(
            future, future["pred_minutes"].to_numpy()
        )
        future["pred_exp_saves"] = 0.0
        gk = future["is_gk"] == 1
        if gk.any():
            future.loc[gk, "pred_exp_saves"] = p.models["saves"].predict_expected_saves(
                future.loc[gk], future.loc[gk, "pred_minutes"].to_numpy()
            )
        future["pred_exp_defcon"] = p.models["defcon"].predict(future)
        future, draws = p._simulate_points(
            future, probability, self.role_model.state_minutes,
            n_simulations=self.n_sims, seed=self.seed + int(gameweek) * 1009)
        future = future.reset_index(drop=True)
        simulations = {i: draws["total_points"][:, i] for i in range(len(future))}
        return WeekForecast(future, simulations)

    def _simulate(self, frame: pd.DataFrame, probability: np.ndarray,
                  gameweek: int) -> Dict[int, np.ndarray]:
        """Compatibility helper delegated to the weekly match simulator."""
        _, draws = self.pipeline._simulate_points(
            frame, probability, self.role_model.state_minutes,
            n_simulations=self.n_sims, seed=self.seed + int(gameweek) * 1009)
        return {i: draws["total_points"][:, i] for i in range(len(frame))}
