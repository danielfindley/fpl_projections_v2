# FPL Prediction Pipeline

Predicts Fantasy Premier League (FPL) player points using historical FotMob data, XGBoost models, and Optuna hyperparameter tuning.

## Quick Start

```bash
# Install dependencies
pip install pandas numpy xgboost scikit-learn scipy requests optuna lightgbm

# Or use the notebook
jupyter notebook run_models.ipynb
```

```python
from src.pipeline import FPLPipeline

pipeline = FPLPipeline('data')
pipeline.load_data()
pipeline.compute_features()
pipeline.tune(n_iter=100, use_subprocess=True)  # Optuna tuning with integrated feature selection
pipeline.train()
predictions = pipeline.predict(gameweek=28, season='2025/2026')
```

## Project Structure

```
projecting_fpl_v2/
├── data/
│   ├── players/player_stats.csv        # Player-match level stats from FotMob
│   ├── matches/                        # Match details and shotmaps
│   ├── fixtures.csv                    # Fixture list
│   ├── predictions/                    # Output predictions per gameweek
│   └── tuning_results/                 # Cached tuning results
├── src/
│   ├── data_loader.py                  # Load/merge FotMob data, FPL API integration
│   ├── features.py                     # Rolling feature engineering (120+ features)
│   ├── feature_selection.py            # Pre-computed feature rankings for Optuna
│   ├── pipeline.py                     # Main pipeline: tune, train, predict, points
│   └── models/
│       ├── base.py                     # Abstract base model (XGBoost, minute-weighted)
│       ├── minutes.py                  # Minutes prediction (1-90)
│       ├── goals.py                    # Goals per match (Poisson counts)
│       ├── assists.py                  # Assists per match (Poisson counts)
│       ├── clean_sheet.py              # Team goals against (Poisson regression)
│       ├── defcon.py                   # Defensive contributions per match (Poisson counts)
│       ├── saves.py                    # GK saves per 90 rate
│       ├── cards.py                    # Yellow/red card probability (direct classifier or fouls fallback)
│       └── bonus.py                    # Bonus points (Monte Carlo BPS simulation with yellow cards)
├── run_models.ipynb                    # Main notebook: scrape, tune, train, predict
├── scrape_update_data.py               # Incremental FotMob data scraper
├── scrape_historical.py                # Historical data scraper
└── exploratory_data_analysis.ipynb     # EDA notebook
```

## Models

| Model | Predicts | Method | Key Features |
|-------|----------|--------|--------------|
| **Minutes** | Playing time (1-90) | XGBoost regression | Rolling minutes, starter rate, current season minutes, goal involvement |
| **Goals** | Goals per match (raw counts) | XGBoost Poisson regression | xG rolling, shots, player share of team output, opponent weakness, xG overperformance, form trends |
| **Assists** | Assists per match (raw counts) | XGBoost Poisson regression | xA rolling, key passes, player centrality, opponent weakness, xA overperformance, form trends |
| **Clean Sheet** | Team goals against (lambda) | XGBoost Poisson regression | Team conceded/xGA rolling (5 windows), opponent xG, prior lambda anchor |
| **Defcon** | Defensive contributions per match (raw counts) | XGBoost Poisson regression | Raw/per-90 defcon rolling, tackles, interceptions, clearances, blocks, recoveries, opponent context, pred_minutes |
| **Saves** | GK saves per 90 | XGBoost regression (GK only) | Saves rolling, xGoT faced, team defensive context, opponent attacking strength |
| **Cards** | Yellow/red card probability | XGBoost binary classifier (`binary:logistic`) | Yellow card rolling history, fouls per 90 rolling, defensive activity, yellow-per-foul rate. Trained on actual FPL API yellow card data (required) |
| **Bonus** | Expected bonus points (0-3) | Monte Carlo BPS simulation | Simulates goals/assists/CS/yellow cards, ranks BPS per match (including -3 BPS per yellow), awards 3-2-1 bonus |

## Feature Engineering

All rolling features use `shift(1)` to prevent data leakage.

### Feature Groups

| Group | Features | Used By |
|-------|----------|---------|
| **Per-90 rolling rates** | `xg_per90_roll{3,5,10}`, `shots_per90_roll{3,5,10}`, etc. | Goals, Assists |
| **Player share / centrality** | `xg_share_roll5`, `shot_share_roll5`, `goal_share_roll5` | Goals, Assists |
| **Form trends** | `xg_trend`, `goals_trend`, `xa_trend`, `assists_trend`, `minutes_trend`, `defcon_trend` | All models |
| **xG overperformance** | `xg_overperformance_roll10`, `lifetime_xg_overperformance`, `xa_overperformance_roll10` | Goals, Assists |
| **Opponent CS rate** | `opp_cs_rate_roll5`, `opp_cs_rate_roll10` | Goals, Assists |
| **Interaction features** | `xg_x_opp_conceded`, `xa_x_opp_conceded`, `team_goals_x_opp_conceded`, `defcon_x_opp_xg` | Goals, Assists, Defcon |
| **Team defensive** | `team_conceded_roll{1,3,5,10,30}`, `team_xga_roll{1,3,5,10,30}`, `team_cs_rate_roll{1,3,5,10,30}` | Clean Sheet, Saves |
| **Opponent offensive** | `opp_goals_roll{5,10}`, `opp_xg_roll{5,10}` | All models |
| **GK-specific** | `saves_per90_roll{3,5,10}`, `xgot_faced_per90_roll{3,5,10}`, `lifetime_saves_per90` | Saves |
| **Lifetime profile** | `lifetime_goals_per90`, `lifetime_xg_per90`, `lifetime_minutes`, etc. | All models |
| **Current season** | `current_season_minutes`, `current_season_apps`, `current_season_mins_per_app` | Minutes |
| **Fouls** | `fouls_committed_per90_roll{3,5,10}`, `lifetime_fouls_committed_per90` | Cards |
| **Yellow cards** | `yellow_cards_roll{3,5,10}`, `yellow_per_foul_roll10`, `lifetime_yellow_cards_per90` (from FPL API merge) | Cards |

## Hyperparameter Tuning

Dependency-ordered tuning with OOF feature propagation and joint hyperparameter + feature selection optimization:

1. **Minutes model** tunes first → generates OOF `pred_minutes` on training set
2. **Clean sheet model** tunes next → generates OOF `pred_team_goals` on training set
3. **Remaining models** (goals, assists, defcon, saves) tune using OOF predictions as features, matching what they'll see at inference time

For each model:
1. **Pre-compute feature rankings** using 5 methods: XGBoost gain, XGBoost cover, LightGBM importance, permutation importance, mutual information
2. **Optuna jointly tunes** XGBoost hyperparams + feature selection (`feat_method`, `n_features`) via TimeSeriesSplit 5-fold CV
   - Search space: `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `colsample_bytree`, `subsample`, `reg_alpha`, `reg_lambda`, `feat_method`, `n_features`
   - Loss functions: Poisson deviance (goals, assists, defcon, clean sheet), MAE (saves), Huber (minutes)
   - Protected features (`pred_minutes`, `pred_team_goals`) are always included

Tuning runs in **subprocess isolation** (`use_subprocess=True`) to prevent OOM from parallel XGBoost + cross-validation.

## Expected Points Formula

```
exp_pts = appearance + goals + assists + clean_sheet + conceded_penalty
        + saves + defcon + bonus + yellow + red

where:
  appearance       = 2 if mins >= 60, 1 if mins >= 1, else 0
  goals            = pred_goals x {GK/DEF: 6, MID: 5, FWD: 4}
  assists          = pred_assists x 3
  clean_sheet      = pred_cs_prob x {GK/DEF: 4, MID: 1} (if mins >= 60)
  conceded_penalty = -E[floor(k/2)] via Poisson (GK/DEF, if mins >= 60)
  saves            = (pred_saves / 3) x 1 (GK only)
  defcon           = pred_defcon_prob x 2 (DEF/MID, if mins >= 60)
  bonus            = pred_bonus (0-3)
  yellow           = pred_yellow_prob x -1
  red              = pred_red_prob x -3
```

## Data Pipeline

### Scraping

```bash
# Scrape specific gameweek
python scrape_update_data.py --gameweek 28

# Auto-detect latest gameweek
python scrape_update_data.py --auto
```

### FPL API Integration

- **Positions**: Maps FotMob position codes to FPL positions (GK/DEF/MID/FWD)
- **Availability**: Filters out injured/suspended players (0% chance of playing)
- **Fixtures**: Resolves DGW (double gameweek) fixtures, aggregating points across matches
- **Yellow/Red cards**: `load_data()` fetches actual card data from the FPL live endpoint and merges it into the training DataFrame, enabling direct yellow card classification instead of fouls-based estimation

## Data Format

### Player Stats CSV (`data/players/player_stats.csv`)

| Column | Description |
|--------|-------------|
| `match_id` | Unique match ID |
| `name` | Player name |
| `team` | Team name |
| `position` | Position code: 0=GK, 1=DEF, 2=MID, 3=FWD |
| `minutes_played` | Minutes in match |
| `goals`, `assists` | Goals and assists |
| `expected_goals_(xg)`, `expected_assists_(xa)` | Expected goals/assists |
| `total_shots`, `shots_on_target` | Shot stats |
| `tackles`, `interceptions`, `clearances`, `blocks`, `recoveries` | Defensive stats |
| `saves`, `xgot_faced`, `goals_conceded` | GK stats |
| `fouls_committed` | Fouls (for card prediction) |
| `season` | Season string (e.g., "2025/2026") |

### Fixtures CSV (`data/fixtures.csv`)

| Column | Description |
|--------|-------------|
| `match_id` | Unique match ID |
| `season` | Season string |
| `round` | Gameweek number |
| `home_team`, `away_team` | Team names |

## Output

Predictions saved to `data/predictions/gw{N}_{season}.csv` with columns:

- Player info: `player_name`, `team`, `fpl_position`, `opponent`, `is_home`
- Predictions: `pred_minutes`, `pred_exp_goals`, `pred_exp_assists`, `pred_cs_prob`, `pred_defcon_prob`, `pred_exp_saves`, `pred_yellow_prob`, `pred_red_prob`, `pred_bonus`
- Points breakdown: `exp_goals_pts`, `exp_assists_pts`, `exp_cs_pts`, `exp_conceded_penalty`, `exp_saves_pts`, `exp_defcon_pts`, `exp_bonus_pts`, `exp_yellow_pts`, `exp_red_pts`, `exp_total_pts`
