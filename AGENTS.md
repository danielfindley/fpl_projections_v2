# AGENTS.md — Experimenter Sub-Agent Guide

This file tells Claude Code sub-agents how to run, evaluate, and iterate on FPL model experiments. Read this before doing any experimentation work.

## Goal

Improve FPL point prediction accuracy by iteratively tuning models, analyzing results, and making targeted changes. The north star metric is **FPL Points MAE (ex-bonus)** — the mean absolute error between actual and predicted total FPL points on the holdout test set, excluding bonus (since BonusModel is not part of tuning).

Individual model metrics also matter (lower is better):
- Goals, Assists: Poisson Deviance
- Defcon: Poisson Deviance (training) + Negative Binomial CDF (threshold probs)
- Saves: MAE
- Minutes: Huber Loss
- Clean Sheet: Poisson Deviance

## How to Run an Experiment

```python
from src.pipeline import FPLPipeline

pipeline = FPLPipeline('data')
pipeline.load_data()
pipeline.compute_features()
pipeline.tune(
    n_iter=100,              # Optuna trials per model
    use_subprocess=True,     # Memory isolation (recommended)
    test_size=0.2,           # Holdout fraction
    description='describe what you changed and why',
)
```

Every `tune()` call auto-logs to `data/experiments.db`. Always provide a descriptive `description` so runs are identifiable later.

## How to Check Results

```python
# After tuning, check history
history = pipeline.experiment_history()
print(history[['run_id', 'model_name', 'cv_score', 'test_score', 'mae', 'fpl_points_mae']].to_string())

# Best runs per model
for model in ['goals', 'assists', 'minutes', 'defcon', 'clean_sheet', 'saves']:
    best = pipeline.best_run(model)
    if best is not None:
        print(f"{model}: test={best['test_score']:.4f}, mae={best['mae']:.4f}")
```

Or via CLI:
```bash
python scripts/experiment.py --history
python scripts/experiment.py --best
python scripts/experiment.py --compare 3
```

## Experimentation Loop

Follow this cycle:

### 1. Establish baseline
Run the first experiment with default settings (100 trials). Record the baseline metrics.

### 2. Identify the weakest model
Look at test metrics. The model with the worst relative performance (highest MAE relative to target variance) is the best candidate for improvement.

### 3. Form a hypothesis
Examples:
- "Goals Poisson deviance is high → try more trials (200) to explore hyperparameter space better"
- "Assists model selected too many features → try increasing n_iter so Optuna explores more feature subset sizes"
- "Minutes model has high Huber loss → the Huber delta might need adjustment"
- "Clean sheet Poisson deviance is high → might need more team-level features"

### 4. Make a targeted change
Change ONE thing at a time so you can attribute improvement. Options:

**Quick wins (no code changes):**
- Increase `n_iter` (more Optuna trials)
- Change `test_size` (0.15 or 0.25)
- Tune a single model: `pipeline.tune(models=['goals'], n_iter=200, ...)`

**Code changes (in src/models/ or src/features.py):**
- Add/remove features from a model's `FEATURES` list
- Adjust search space bounds in `_tune_in_process`
- Change rolling window sizes in `src/features.py`
- Add new interaction features
- Adjust per-90 caps

### 5. Run and compare
```python
pipeline.tune(n_iter=100, use_subprocess=True, description='<what you changed>')
# Then compare:
history = pipeline.experiment_history()
```

### 6. Keep or revert
If the change improved the target metric, keep it. If not, revert the code change and try a different hypothesis. Document findings in the description.

## Key Files

| File | What it does |
|------|-------------|
| `src/pipeline.py` | Main pipeline — `tune()` runs experiments, `_evaluate_on_test_set()` computes metrics |
| `src/experiment_log.py` | SQLite logging — `log_experiment()`, `get_history()`, `get_best_run()` |
| `src/features.py` | Feature engineering — rolling windows, per-90 rates, interactions |
| `src/models/` | Model definitions — each has `FEATURES`, `TARGET`, search spaces |
| `scripts/experiment.py` | CLI for running experiments and viewing history |
| `data/experiments.db` | SQLite database with all experiment results |

## Model-Specific Notes

### Goals & Assists (Poisson Deviance)
- Predict per-90 rates using `count:poisson` objective + Poisson deviance eval
- Key features: xg/xa rolling, team context, lifetime profiles
- Converting rate → count depends on minutes prediction quality

### Minutes (Huber Loss)
- Most impactful model — minutes prediction affects ALL other point calculations
- Uses Huber loss (robust to outliers from injuries/subs)
- Key features: rolling minutes, lifetime minutes, is_home

### Clean Sheet (Poisson Deviance)
- Team-level model predicting goals conceded (not player-level)
- Uses `count:poisson` objective in XGBoost
- Clean sheet probability = P(goals_conceded = 0) from Poisson distribution

### Saves (MAE, GK only)
- Filtered to `is_gk == 1`
- Per-90 rate prediction
- Small sample size — be cautious with overfitting

### Defcon (Poisson Deviance + NB CDF)
- Defensive contributions (clearances + blocks + interceptions + tackles + recoveries)
- Only DEF/MID get FPL points for this
- Trained with `count:poisson` objective; Poisson deviance eval metric (consistent for mean estimation)
- Threshold probabilities use Negative Binomial CDF to handle heavy overdispersion (var/mean ≈ 2.5–3.3x)
- Dispersion parameter `r` estimated from Pearson residuals during `fit()`

## Rules for Sub-Agents

1. **Always use `description`** — every tune() call must have a clear description
2. **Change one thing at a time** — don't change features AND hyperparams simultaneously
3. **Compare against baseline** — use `experiment_history()` to compare, not just the printed output
4. **Don't modify experiment_log.py** — the logging infrastructure is stable
5. **Use `use_subprocess=True`** — prevents memory leaks during long tuning sessions
6. **Check for data leakage** — any new features must use `shift(1)` on rolling calculations
7. **Don't tune bonus/cards** — BonusModel (Monte Carlo) and CardsModel aren't part of the tuning loop
8. **Report results** — after each experiment, summarize what changed, what improved, what didn't
