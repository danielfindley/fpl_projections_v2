# FPL Prediction Pipeline

Predicts Fantasy Premier League (FPL) player points using historical FotMob data and XGBoost models.

## Quick Start

```bash
# Install dependencies
pip install pandas numpy xgboost scikit-learn scipy requests

# Train models and generate predictions for GW22
python scripts/predict.py --gameweek 22 --season "2025/2026"

# Or use Python directly
python -c "
from src.pipeline import FPLPipeline
pipeline = FPLPipeline('data')
pipeline.load_data()
pipeline.compute_features()
pipeline.train()
predictions = pipeline.predict(gameweek=22, season='2025/2026')
print(predictions.nlargest(20, 'exp_total_pts')[['player_name', 'team', 'fpl_position', 'opponent', 'exp_total_pts']])
"
```

## Project Structure

```
projecting_fpl_v2/
├── data/
│   ├── players/player_stats_*.csv    # Player-match level stats from FotMob
│   ├── matches/                      # Match details and shotmaps
│   ├── all_fixtures_8_seasons.csv    # Fixture list
│   └── predictions/                  # Output predictions
├── src/
│   ├── data_loader.py                # Load/merge FotMob data
│   ├── features.py                   # Rolling feature engineering
│   ├── pipeline.py                   # Main prediction pipeline
│   └── models/                       # XGBoost models
│       ├── minutes.py                # Minutes prediction
│       ├── goals.py                  # Goals per 90
│       ├── assists.py                # Assists per 90
│       ├── clean_sheet.py            # Team clean sheet probability
│       ├── defcon.py                 # Defensive contribution
│       └── bonus.py                  # Bonus points (Monte Carlo)
├── scripts/
│   ├── train.py                      # Train all models
│   ├── predict.py                    # Generate predictions (trains + predicts)
│   └── tune.py                       # Hyperparameter tuning
└── fotmob_scraping_test_nb.ipynb     # Playground notebook
```

## Usage

### Command Line

```bash
# Generate predictions (trains models first)
python scripts/predict.py --gameweek 22 --season "2025/2026" --top 30

# Train only (useful for inspecting model performance)
python scripts/train.py

# Tune hyperparameters (slow, optional)
python scripts/tune.py --trials 50 --models goals assists minutes defcon
```

### Python API

```python
from src.pipeline import FPLPipeline

# Initialize
pipeline = FPLPipeline(data_dir='data')

# Load and prepare data
pipeline.load_data()
pipeline.compute_features()

# Train models
pipeline.train()

# Predict for a gameweek
predictions = pipeline.predict(gameweek=22, season='2025/2026')

# Get top players
top = pipeline.get_top_players(predictions, n=30)
print(top)

# Access individual model predictions
print(predictions[['player_name', 'pred_minutes', 'pred_exp_goals', 'pred_exp_assists', 
                   'pred_cs_prob', 'pred_defcon_prob', 'pred_bonus', 'exp_total_pts']])
```

## Models

| Model | Predicts | Key Features |
|-------|----------|--------------|
| **Minutes** | Expected playing time (0-90) | Rolling minutes, starter rate, full 90 rate |
| **Goals** | Goals per 90 rate | xG rolling, shots, team/opponent strength |
| **Assists** | Assists per 90 rate | xA rolling, key passes, team context |
| **Clean Sheet** | Team CS probability | Goals conceded, xGA, opponent xG |
| **Defcon** | P(defensive contribution ≥ threshold) | Tackles, interceptions, clearances, blocks |
| **Bonus** | Expected bonus points | Monte Carlo simulation using BPS rules |

## Expected Points Formula

```
exp_pts = appearance_pts + goal_pts + assist_pts + cs_pts + defcon_pts + bonus_pts

where:
  appearance_pts = 2 if mins≥60, 1 if mins≥1, else 0
  goal_pts       = pred_goals × {GK/DEF: 6, MID: 5, FWD: 4}
  assist_pts     = pred_assists × 3
  cs_pts         = pred_cs_prob × {GK/DEF: 4, MID: 1, FWD: 0} (if mins≥60)
  defcon_pts     = pred_defcon_prob × 2 (DEF/MID only, if mins≥60)
  bonus_pts      = pred_bonus (0-3)
```

## Data Format

### Required: Player Stats CSV (`data/players/player_stats_*.csv`)

| Column | Description |
|--------|-------------|
| `match_id` | Unique match ID |
| `name` | Player name |
| `team` | Team name |
| `position` | Position code: 0=GK, 1=DEF, 2=MID, 3=FWD |
| `minutes_played` | Minutes in match |
| `goals`, `assists` | Goals and assists |
| `expected_goals_(xg)` | Expected goals |
| `expected_assists_(xa)` | Expected assists |
| `tackles`, `interceptions`, `clearances`, `blocks`, `recoveries` | Defensive stats |
| `season` | Season string (e.g., "2025/2026") |

### Required: Fixtures CSV (`data/all_fixtures_8_seasons.csv`)

| Column | Description |
|--------|-------------|
| `match_id` | Unique match ID |
| `season` | Season string |
| `round` | Gameweek number |
| `home_team`, `away_team` | Team names |

## Point-in-Time Correctness

All rolling features use `shift(1)` to prevent data leakage:

```python
# Correct - excludes current match
df['xg_roll5'] = df.groupby('player_id')['xg'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
```

## Output

Predictions are saved to `data/predictions/gw{N}_{season}.csv` with columns:

- `player_name`, `team`, `fpl_position`, `opponent`, `is_home`
- `pred_minutes`, `pred_exp_goals`, `pred_exp_assists`
- `pred_cs_prob`, `pred_defcon_prob`, `pred_bonus`
- `exp_total_pts` (final expected FPL points)
