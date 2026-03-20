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
- **Custom models** (don't inherit BaseModel): MinutesModel (custom capping logic), CleanSheetModel (Poisson regression for goals-against lambda, raw Poisson CS probs with prior_lambda anchor, home/away split season stats, team-level aggregates), BonusModel (Monte Carlo BPS simulation — simulates goals, assists, clean sheets, **and yellow cards** per match, then ranks BPS to award 3-2-1 bonus).

### Feature engineering (src/features.py)
120+ features computed with rolling windows (3, 5, 10 games). **All rolling features use `shift(1)`** to prevent data leakage. Key groups: per-90 rolling rates, player share of team output, form trends, xG overperformance, team/opponent rolling stats, interaction features, lifetime profiles, current-season context.

Per-90 stats are capped (e.g., xg_per90 at 2.0) to prevent inflation from low-minute appearances.

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

## Data

- `data/players/player_stats.csv`: Player-match level FotMob stats (gitignored)
- `data/fixtures.csv`: Match schedule with gameweeks
- `data/fpl_actual_points.csv`: Cached FPL API data with yellow/red cards per gameweek
- `data/tuning_results/`: Cached Optuna tuning results (JSON)
- `data/predictions/`: Output CSVs per gameweek (gitignored)
- `data/experiments.db`: Experiment log (SQLite, gitignored)
