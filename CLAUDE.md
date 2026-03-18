# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FPL prediction pipeline: predicts Fantasy Premier League player points using FotMob match data, XGBoost models, and Optuna hyperparameter tuning.

## Setup & Running

```bash
pip install pandas numpy xgboost scikit-learn scipy requests optuna
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

**Training order matters**: MinutesModel trains first (generates `pred_minutes`), then CleanSheetModel (generates `pred_team_goals` via TimeSeriesSplit OOF). Goals and Assists models consume both `pred_minutes` and `pred_team_goals` as features (leak-free).

### Model hierarchy (src/models/)
- **BaseModel** (`base.py`): Abstract class with XGBoost + minute-weighted training (no scaler — trees don't need it). Subclasses define `FEATURES`, `TARGET`, and `_get_y_max()`.
- **BaseModel subclasses**: GoalsModel (Poisson objective on raw match counts), AssistsModel (Poisson objective on raw match counts), DefconModel, SavesModel, CardsModel — all follow the same fit/predict pattern.
- **Custom models** (don't inherit BaseModel): MinutesModel (custom capping logic), CleanSheetModel (Poisson regression for goals-against lambda, raw Poisson CS probs with prior_lambda anchor, home/away split season stats, team-level aggregates), BonusModel (Monte Carlo BPS simulation).

### Feature engineering (src/features.py)
120+ features computed with rolling windows (3, 5, 10 games). **All rolling features use `shift(1)`** to prevent data leakage. Key groups: per-90 rolling rates, player share of team output, form trends, xG overperformance, team/opponent rolling stats, interaction features, lifetime profiles, current-season context.

Per-90 stats are capped (e.g., xg_per90 at 2.0) to prevent inflation from low-minute appearances.

### Tuning (two-phase, inside pipeline.tune())
1. **Optuna** tunes XGBoost hyperparams on all features (TimeSeriesSplit 5-fold CV, `n_iter` trials)
2. **RFECV** selects optimal feature subset using best hyperparams from Phase 1
3. Results saved to `data/tuning_results/{model}_tuned.json`

Loss functions per model: Poisson deviance (goals, assists, clean sheet), MAE (saves), RMSE (defcon), Huber (minutes). Goals and Assists use `count:poisson` objective + Poisson deviance eval metric on raw match counts (0, 1, 2, ...) — true Poisson count data. `pred_minutes` is a feature so the model learns the minutes-count relationship internally. RFECV min_features_to_select is 15 for goals (to prevent over-pruning), 5 for other models.

### Prediction flow
Goals and Assists models output expected match counts directly (no per-90 scaling needed). These are mapped to FPL points using position-specific multipliers (see README.md for full formula). BonusModel uses Monte Carlo simulation of match outcomes.

## Key Conventions

- Position codes: 0=GK, 1=DEF, 2=MID, 3=FWD
- Team names are normalized to lowercase for matching
- `player_id` = `player_name_team` (composite key)
- Each model class defines its own `FEATURES` list and `TARGET` column
- `selected_features` from RFECV stored in tuned params and used at train time
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

## Data

- `data/players/player_stats.csv`: Player-match level FotMob stats (gitignored)
- `data/fixtures.csv`: Match schedule with gameweeks
- `data/tuning_results/`: Cached Optuna+RFECV results (JSON)
- `data/predictions/`: Output CSVs per gameweek (gitignored)
- `data/experiments.db`: Experiment log (SQLite, gitignored)
