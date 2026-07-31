# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FPL prediction pipeline: predicts Fantasy Premier League player points using FotMob match data, XGBoost models, and Optuna hyperparameter tuning.

## Setup & Running

```bash
pip install pandas numpy xgboost scikit-learn scipy requests optuna lightgbm
```

Primary workflow is through `run_models.ipynb` or Python directly:
```python
from src.pipeline import FPLPipeline
pipeline = FPLPipeline('data')
pipeline.load_data()
pipeline.compute_features()
pipeline.tune(n_iter=100, use_subprocess=True)
pipeline.train()
predictions = pipeline.predict(gameweek=28, season='2025/2026')
```

CLI scripts: `python scripts/tune.py`, `python scripts/train.py`, `python scripts/predict.py`
Data scraping: `python scrape_update_data.py --gameweek 28` or `--auto`

## Architecture

### Pipeline stages (src/pipeline.py)
`load_data()` → `compute_features()` → `tune()` → `train()` → `predict()`

**Dependency order matters in both tuning and training**: MinutesModel first (generates `pred_minutes`), then CleanSheetModel (generates `pred_team_goals` via TimeSeriesSplit OOF). Goals and Assists models consume both `pred_minutes` and `pred_team_goals` as features. During tuning, OOF predictions from minutes and clean_sheet are generated on the training set so downstream models tune against realistic (noisy) feature values rather than ground truth.

### Model hierarchy (src/models/)
- **BaseModel** (`base.py`): Abstract class with XGBoost + minute-weighted training (no scaler — trees don't need it). Subclasses define `FEATURES`, `TARGET`, and `_get_y_max()`.
- **BaseModel subclasses**: GoalsModel (Poisson objective on raw match counts), AssistsModel (Poisson objective on raw match counts), DefconModel (Poisson objective on raw defensive contribution counts, uses `pred_minutes` as feature), SavesModel — all follow the same fit/predict pattern.
- **CardsModel** (`cards.py`): XGBoost binary classifier (`binary:logistic`) on actual yellow card data from the FPL API. Requires `yellow_cards` column — `load_data()` merges this via `merge_fpl_card_data()` and will raise if the FPL API is unreachable. Red cards use fouls-based prediction (too rare for classification). Not part of the tuning loop.
- **AppearClassifier** (`minutes.py`): XGBoost binary classifier for P(minutes >= 1) — the only model trained on non-appearances. Its training set is the calendar grid from `features.build_appearance_grid()`, not the per-match frame, because a frame of appearances contains no counterexamples. Isotonic-calibrated against the most recent season (raw XGB put P(blank) at .090 for likely starters whose actual rate was .050, which would double every bench weight). Exposed via `MinutesModel.predict_appear_proba()`; `predict()` is untouched and still returns E[minutes | appears], so no downstream model needs retuning. Not part of the tuning loop.
- **Custom models** (don't inherit BaseModel): MinutesModel (custom capping logic), CleanSheetModel (Poisson regression for goals-against lambda, raw Poisson CS probs with prior_lambda anchor, home/away split season stats, team-level aggregates), BonusModel (Monte Carlo BPS simulation — simulates goals, assists, clean sheets, **and yellow cards** per match, then ranks BPS to award 3-2-1 bonus).

### Feature engineering (src/features.py)
120+ features computed with rolling windows (3, 5, 10 games). **All rolling features use `shift(1)`** to prevent data leakage. Key groups: per-90 rolling rates, player share of team output, form trends, xG overperformance, team/opponent rolling stats, interaction features, lifetime profiles, current-season context.

Per-90 stats are capped (e.g., xg_per90 at 2.0) to prevent inflation from low-minute appearances.

**Manager embeddings (`add_manager_embeddings`, end of `compute_rolling_features`)**: 8-dim PCA over rolling-20-prior manager stats. For each (manager, match) the pipeline builds a per-game vector covering minute distribution (mean/median/std/max, num_players_used, num_full_90, num_subs_made, mins_concentration_top11, mins_entropy), goals for/against, and parsed formation (def/mid/fwd counts). Each manager's rolling-20 mean is computed with `shift(1)` (strictly prior); managers with fewer than 3 prior games get a zero vector (interim protection). PCA is fit on all valid rows; the same basis is reused at predict time. For the next-GW synthetic rows the prior manager is assumed to continue, and their *current* rolled state (last 20 played games, no shift) is projected through the trained PCA. Manager identity for an upcoming match is treated as a known input — every numeric feature is still derived from games strictly before the row's own match. Outputs are 8 columns (`manager_emb_0..7`) added to every model's `FEATURES` list. Source data: `data/match_managers.csv`, derived once from the raw JSONs in `data/matches/raw/`.

### Tuning (dependency-ordered, Optuna with integrated feature selection)
Tuning follows the model dependency chain with OOF feature propagation:
1. **Minutes** tunes first → generates OOF `pred_minutes` on training set
2. **Clean sheet** tunes next → generates OOF `pred_team_goals` on training set
3. **Goals, Assists, Defcon, Saves** tune using OOF predictions as features (matches inference conditions)

Feature rankings are pre-computed once per model using 5 methods (XGBoost gain, XGBoost cover, LightGBM, permutation importance, mutual information). Then Optuna jointly optimizes XGBoost hyperparams + feature selection (ranking method + number of features) via TimeSeriesSplit 5-fold CV.

Protected features (`pred_minutes`, `pred_team_goals`) are always included and don't count toward the feature count. Min features: 15 for goals, 5 for others.

Loss functions per model: Poisson deviance (goals, assists, defcon, clean sheet), MAE (saves), Huber (minutes). Goals, Assists, and Defcon use `count:poisson` objective + Poisson deviance eval metric on raw match counts (0, 1, 2, ...) — true Poisson count data. `pred_minutes` is a feature so the model learns the minutes-count relationship internally.

Feature ranking module: `src/feature_selection.py` — `compute_feature_rankings()` pre-computes all rankings, `select_features()` picks top-N from a given method.

### Prediction flow
Goals, Assists, and Defcon models output expected match counts directly (no per-90 scaling needed). Goals/Assists are mapped to FPL points using position-specific multipliers. Defcon counts are converted to threshold probability via Negative Binomial CDF: P(defcon >= threshold) — NB is used instead of Poisson because defcon counts are heavily overdispersed (var/mean ≈ 2.5–3.3x); the dispersion parameter `r` is estimated from Pearson residuals during `fit()`. BonusModel uses Monte Carlo simulation of match outcomes.

## Key Conventions

- Position codes: 0=GK, 1=DEF, 2=MID, 3=FWD
- Team names are normalized to lowercase for matching
- `player_id` = `player_name_team` (composite key)
- Each model class defines its own `FEATURES` list and `TARGET` column
- `selected_features` from Optuna-selected ranking stored in tuned params and used at train time
- GK-specific models (Saves) filter to `is_gk == 1`
- Training samples weighted by minutes played relative to mean
- `use_subprocess=True` in tune() for memory isolation
- **Do not create new files** unless explicitly asked — use existing files, inline scripts (`python -c` or stdin heredoc), or the notebook

## Testing

Minimal: `test_fixtures.py` and `test_bonus_fix.py` for spot checks. No pytest framework. Validation is primarily through notebook analysis and holdout test set metrics printed during tuning.

## Experiment Logging

All tuning runs are auto-logged to `data/experiments.db` (SQLite). Each run captures: CV scores, test metrics, hyperparams, selected features, and FPL points MAE (actual vs predicted).

### Quick reference
```python
pipeline.tune(n_iter=100, use_subprocess=True, description='my experiment')
pipeline.experiment_history()          # all runs as DataFrame
pipeline.experiment_history('goals')   # filter to one model
pipeline.best_run('goals')            # best test score for a model
```

### CLI
```bash
python scripts/experiment.py -t 100 -d "baseline"           # run experiment
python scripts/experiment.py --history                       # show all runs
python scripts/experiment.py --best                          # best per model
python scripts/experiment.py --compare 3                     # compare last 3 runs
python scripts/experiment.py -t 200 -m goals assists -d "more trials"  # specific models
```

### FPL Points MAE
Two values are tracked:
- **ex-bonus**: excludes bonus from both sides (apples-to-apples for tuned models)
- **inc-bonus**: includes actual bonus in target (shows full gap including bonus)

For agent-driven experimentation workflow, see `AGENTS.md`.

### FPL API data integration
`load_data()` automatically fetches yellow/red card data from the FPL API (`fetch_fpl_actual_points()`) and merges it into the training DataFrame by matching player name + team + gameweek. This enriches the FotMob data (which lacks card columns) with actual FPL yellow_cards and red_cards for the current season. The merge is cached to `data/fpl_actual_points.csv`. If the API is unreachable, `load_data()` will raise — internet access is required on first run (cached thereafter).

## Squad optimization (src/optimizer.py)

`pipeline.optimize_squad(predictions, gameweek, season)` picks the best 15 under a £100.0m budget, solved exactly as a MILP via `scipy.optimize.milp` (no new dependency; `pulp` is not installed). Constraints: 2/5/5/3 by position, max 3 per club, valid XI shape, and starters must be in the squad.

Two things drive the objective:
- **Unconditional expected points.** `exp_total_pts` is built from `pred_minutes`, which is E[minutes | appears], so it is really E[points | appears] and overstates a fringe player by `1/p_appear`. The optimizer multiplies by `pred_appear_prob` first (column `exp_pts_uncond`).
- **Bench slots weighted by how often they actually play.** An outfield bench slot only scores via autosub, which needs a starter to blank, so slot *k* is weighted by P(at least *k* starters blank) — the exact Poisson-binomial tail over the XI's appearance probabilities. The backup GK is weighted separately by `1 - p_appear(GK1)`, which is why it converges to the £4.0m floor. Since the weights depend on the XI and the XI depends on the weights, the MILP is re-solved until the squad stops changing (typically 3 iterations).

Captain is reported as the highest-EP starter but is **not** doubled in the objective, which would bias the squad toward a single premium.

### Prices
Prices exist **only in the FPL API** (`bootstrap-static` → `now_cost`, in tenths); nothing in the repo stores them, and `fetch_fpl_prices()` requires internet. Joined to predictions by name via the same exact → web_name/second_name → token-subset cascade used for card data (~98.6% match; unmatched players are excluded and printed).

`snapshot_prices()` appends every fetch to `data/fpl_prices.csv` keyed by `(season, gameweek, fpl_id)`. **This is not recoverable retroactively** — the API serves per-gameweek price history only for the current season and drops it at rollover, keeping just `start_cost`/`end_cost` per player in `history_past`. Weekly snapshots are the only way to build price history.

## Visualization (src/viz.py)

`generate_distribution_html()` produces a standalone HTML file (`distributions.html`) with:
- Optimal-squad pitch (`squad=` arg), laid out in the shape of the formation it selected — a colored dot per player with name and price below, bench in autosub order. Desktop `body` is `display:flex`, so the side column sits to the right of the ridge plot with the squad above the metrics tables; the mobile template stacks it first on the page.
- D3.js ridge plot showing Monte Carlo points distributions for top outfield players
- Sub-model metrics table, overall FPL points metrics, and a calibration plot (predicted vs actual by bucket)
- Responsive layout — desktop ridge plot and mobile card layout are both embedded; the correct one is selected at load time based on viewport width

The viz is fed by `pipeline.get_viz_metrics()` (formats `last_test_metrics` into sections + calibration buckets) and `pipeline.last_simulations` (Monte Carlo simulation arrays from BonusModel).

## Weekly Deploy Workflow

Use the `/deploy-predictions` skill (Claude Code slash command) to run the full weekly pipeline. It automates:
1. **Detect next gameweek** — queries FPL API for latest finished GW, compares against scraped data
2. **Scrape** — `python scrape_update_data.py --auto` (Playwright + Cloudflare bypass on FotMob)
3. **Tune** (optional, ~30-60 min) — asks whether to retune or reuse cached params from latest run
4. **Train + predict + viz** — loads tuned params, trains on all data, predicts next GW, generates `distributions.html`
5. **Save run** — `pipeline.save_run()` persists predictions, simulations, tuned params, and metrics to `data/runs/gw{N}_{timestamp}/`
6. **Deploy** — copies `distributions.html` to `C:/Users/dpfin/repos/danielfindley.com/projects/`, commits, pushes (asks before push)

When tuning is split from training (steps 3a/3b in the skill), tuned params are serialized to `data/_latest_tuned_params.json` so a crash in the training step doesn't lose the tuning work.

### Run directory structure (`data/runs/gw{N}_{timestamp}/`)
- `predictions.csv` — full prediction table
- `squad.csv` / `squad_meta.json` — optimized 15 with an `is_starter` flag, plus formation, cost, and the bench weights used
- `simulations/` — Monte Carlo simulation arrays (`.npy`)
- `tuned_params.json` — Optuna-selected hyperparams + features per model
- `test_metrics.json` — holdout test set metrics (sub-model + FPL points)
- `viz_metrics.json` — formatted metrics for the HTML viz
- `meta.json` — run metadata (description, timestamp)

## Data

- `data/players/player_stats.csv`: Player-match level FotMob stats (gitignored)
- `data/fixtures.csv`: Match schedule with gameweeks
- `data/fpl_actual_points.csv`: Cached FPL API data with yellow/red cards per gameweek
- `data/fpl_prices.csv`: Weekly price snapshots keyed by (season, gameweek, fpl_id). Append-only; cannot be backfilled after a season rolls over.
- `data/tuning_results/`: Cached Optuna tuning results (JSON)
- `data/predictions/`: Output CSVs per gameweek (gitignored)
- `data/experiments.db`: Experiment log (SQLite, gitignored)
- `data/runs/`: Saved pipeline runs with predictions, simulations, params, metrics (gitignored)
- `data/matches/raw/{match_id}.json.gz`: Full FotMob match-details payloads, gzipped. Written by `scrape_update_data.py` on every fetch (both `--gameweek N` and `--auto` modes call `save_raw_match()`). Source of truth for fields not flattened into `player_stats.csv` — currently used by the manager-embedding feature to extract `content.lineup.{home,away}Team.coach` and `.formation`.
- `data/match_managers.csv`: Cache of (match_id, home/away team, manager id+name, formation) parsed from the raw JSONs. Rebuildable from raw at any time. Consumed by `add_manager_embeddings()` in `src/features.py`.
