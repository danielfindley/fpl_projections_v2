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

pipeline.load_lineups(gameweek=1, season='2026/2027')   # optional: RotoWire predicted XIs
predictions = pipeline.predict(gameweek=1, season='2026/2027')
squad = pipeline.optimize_squad(predictions, gameweek=1, season='2026/2027')
```

`load_lineups()` must precede `predict()`; both it and `optimize_squad()` require internet.

## Project Structure

```
projecting_fpl_v2/
├── data/
│   ├── players/player_stats.csv        # Player-match level stats from FotMob
│   ├── matches/                        # Match details and shotmaps
│   │   └── raw/{match_id}.json.gz      # Full FotMob match-details payloads (gzipped). Source of truth for managers/lineups; the scraper writes here on every fetch.
│   ├── match_managers.csv              # Per-match (home_manager, away_manager, formations) — derived from raw JSONs (cache used by manager-embedding feature)
│   ├── fixtures.csv                    # Fixture list
│   ├── fpl_prices.csv                  # Weekly price snapshots (append-only; unrecoverable after season rollover)
│   ├── lineups/gw{N}_{ts}.csv          # RotoWire predicted-XI snapshots (same constraint — current slate only)
│   ├── predictions/                    # Output predictions per gameweek
│   └── tuning_results/                 # Cached tuning results
├── src/
│   ├── data_loader.py                  # Load/merge FotMob data, FPL API integration
│   ├── features.py                     # Rolling feature engineering (120+ features)
│   ├── feature_selection.py            # Pre-computed feature rankings for Optuna
│   ├── lineups.py                      # RotoWire predicted lineups: scrape, match, override, self-scoring
│   ├── optimizer.py                    # Budget-constrained squad selection (MILP via scipy)
│   ├── viz.py                          # Standalone HTML viz (ridge plot, metrics, squad pitch)
│   ├── pipeline.py                     # Main pipeline: tune, train, predict, points
│   └── models/
│       ├── base.py                     # Abstract base model (XGBoost, minute-weighted)
│       ├── minutes.py                  # Minutes (1-90) + P(appears) + P(starts)
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
| **Minutes** | Playing time (1-90), conditional on appearing | XGBoost, two-stage: P(start) x starter regressor + (1-P) x sub regressor | Rolling minutes, starter rate, current season minutes, goal involvement |
| **Appears** | P(minutes >= 1) | XGBoost binary classifier, isotonic-calibrated | Calendar-grid minutes features. The **only** model trained on non-appearances — see below |
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
| **Calendar minutes** | `minutes_roll{N}`, `starter_rate_roll{N}`, `full90_rate_roll{N}`, `gw_gap_since_last_appearance`, `last_app_prev_season` — computed on a 0-filled calendar so a missed gameweek degrades form instead of being invisible | Minutes, Appears |
| **Fouls** | `fouls_committed_per90_roll{3,5,10}`, `lifetime_fouls_committed_per90` | Cards |
| **Yellow cards** | `yellow_cards_roll{3,5,10}`, `yellow_per_foul_roll10`, `lifetime_yellow_cards_per90` (from FPL API merge) | Cards |
| **Manager embeddings** | `manager_emb_0..7` — 8-dim PCA over rolling-20-prior manager stats (minutes distribution, GF/GA, formation) | All models |

### Appearance model (the only model that sees non-appearances)

`player_stats.csv` contains one row per match a player *featured in*, so no model in the pipeline ever saw a player being left out — meaning nothing could estimate P(plays at all). `MinutesModel.predict()` is trained on `minutes >= 1` and clipped to `[1, 90]`, so it answers **E[minutes | appears]**, not "will he play".

`features.build_appearance_grid()` restores the missing rows: a `(player_id, season, gameweek)` calendar spanning each player's first-to-last appearance in a season, with missed gameweeks filled at 0 minutes. That span keeps rotation calls and mid-season injury gaps while excluding weeks before a January signing arrived or after a departure. **90k player-gameweeks, 27.7k of them non-appearances.**

`AppearClassifier` (`src/models/minutes.py`) trains on that grid with target `minutes > 0`, using only features defined on a week the player didn't play. Holdout (2025/26): **AUC 0.844, Brier 0.134 vs 0.198 base rate**.

Two details that matter:

- **Isotonic calibration**, fitted on the most recent season. Raw XGB put P(blank) at .090 for likely starters whose actual rate was .050, which would have doubled every bench weight in the squad optimizer. Calibration cuts mean gap from .039 to .014.
- **The season-opening row of each player-season is dropped.** It is an appearance by construction (the span starts at their first game) while carrying the largest cross-season `gw_gap`, so leaving it in taught the exact inverse of the truth — that a player absent 40+ gameweeks is certain to feature. A backup keeper scored 1.000 before this fix.

`MinutesModel.predict()` is deliberately left conditional. Folding P(appears) into `pred_minutes` would change the meaning of a feature that goals/assists/defcon/saves/bonus were all tuned against, forcing a full retune. Consumers use `pred_appear_prob` explicitly instead.

### Manager embeddings (leak-free)

Each team's manager for a given match is looked up from `data/match_managers.csv` (extracted from raw FotMob JSONs at `data/matches/raw/`). For every (manager, match) pair, the pipeline builds a per-game feature vector capturing playstyle and rotation:

- **Goals**: `gf`, `ga`
- **Minutes distribution** (rotation/management signal): `mins_mean`, `mins_median`, `mins_std`, `mins_max`, `num_players_used`, `num_full_90`, `num_subs_made`, `mins_concentration_top11`, `mins_entropy`
- **Formation** (from `homeTeam.formation` / `awayTeam.formation`): `form_def`, `form_mid`, `form_fwd`

For each row, these are rolled over the manager's prior 20 games using `shift(1)` (strictly prior). Standardize → fit PCA(8) on rows with ≥3 prior games (interim-manager protection — fewer prior games gives a zero vector). For the upcoming gameweek's synthetic rows, the previous manager is assumed to continue and their *current* rolled state (their last 20 played games, no shift) is projected via the trained PCA basis.

Manager identity for the upcoming game is treated as a known input — no future-data leakage in the features themselves, since every numeric feature is derived from games played strictly before the row's own match.

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

## Predicted Lineups

`pipeline.load_lineups(gameweek, season)` scrapes [RotoWire](https://www.rotowire.com/soccer/lineups.php) predicted XIs and **must be called before `predict()`**. Inference-only — nothing is retrained, so experiment history stays comparable.

**The problem it solves.** At a season opener every feature is three months stale, so the minutes model cannot separate a nailed starter from a squad player and hedges on both. On GW1 2026/27 it projected Chelsea's backup striker at **65 minutes** and the first-choice striker at **62** — the latter's late-season injury had both idled him and depressed the rolling form his cold-start cap is computed from. With the feed: **79 vs 33**.

### How it injects

| Point | Effect |
|-------|--------|
| `p_start` override in `MinutesModel._blend` | Feed replaces the classifier's guess (0.95 in XI, 0.10 benched). The two minute regressors are untouched — only the weight between them changes |
| Cold-start cap bypass | `_apply_caps` pins GW1 players to `minutes_roll5 x 1.2` and binds on **168 of 419** rows. A published XI is better evidence than the heuristic it stands in for |
| `pred_appear_prob` scaling | `OUT`/`SUS` → 0, `QUES` → 0.75 (`DOUBT_MULTIPLIER`) |

**Hedging is applied to minutes, not to `p_start`** (`lineup_weight=0.7`). `_sharpen(temp=0.3)` is nearly a step function, so blending probabilities collapses straight back to the feed's answer:

| | range | effect on blended minutes |
|---|---|---|
| `p_in` | 0.75 → 0.99 | **1.6 min** |
| `lineup_weight` | 0.0 → 1.0 | **23.5 min** |

Tune the weight, not the probabilities.

### Safety: benching is inferred from absence

A name that fails to match falls into the same bucket as a genuinely benched player, and the two causes are very different:

- **Player is outside our data entirely** (promoted-club squad, new signing) → benching the rest of that club is fine
- **Player is in our data under a different spelling** → he is actually *starting*, and we would be about to project a nailed starter as a 20-minute sub

The surname test separates them. A club is safe to bench only if every unmatched starter has **no surname collision** in our squad for that club; a collision is ambiguous, so those clubs get positive starter overrides only.

As of GW1 2026/27 there were **0 collisions across all 31 unmatched starters** — every unmatched name was a promoted-club player or a signing we have no history for. The guard is precautionary and has not yet fired in practice.

Names come from the anchor `title` attribute (full name) plus a stable RotoWire id, never the abbreviated display text (`R. Calafiori`). `_fold()` extends `normalize_player_name` for letters NFD cannot decompose — ø, ß, æ, å, đ, ł — without which Odegaard never matches Ødegaard.

### Self-scoring

`evaluate_snapshots()` runs automatically inside `load_lineups()`, joining past snapshots to real FotMob minutes and reporting the three quantities the override constants are guessing at:

```
P(started  | predicted starter)   <- p_in (0.95)
P(started  | listed non-starter)  <- p_out (0.10)
P(appeared | QUES)                <- DOUBT_MULTIPLIER (0.75)
```

Quiet until a gameweek completes. Snapshots land in `data/lineups/gw{N}_{timestamp}.csv`; **RotoWire serves only the current slate, so this history cannot be recovered later.** Scrape as late as possible before the deadline — predicted XIs firm up nearer kickoff, and late-kickoff fixtures are guesswork at deadline time.

## Squad Optimization

`pipeline.optimize_squad(predictions, gameweek, season)` picks the best 15 under a £100.0m budget, solved as an exact MILP via `scipy.optimize.milp` (no new dependency). Constraints: 2/5/5/3 by position, max 3 per club, valid XI shape, starters must be in the squad.

**Unconditional expected points.** `exp_total_pts` is built from `pred_minutes`, which is E[minutes | appears] — so it is E[points | appears] and overstates a fringe player by `1/p_appear`. The optimizer multiplies by `pred_appear_prob` first.

**Bench slots weighted by how often they actually play.** An outfield bench slot only scores via autosub, which needs a starter to blank, so slot *k* is weighted by P(at least *k* starters blank) — the exact Poisson-binomial tail over the XI. Typical values: `[0.40, 0.085, 0.011]`. The backup GK is weighted separately by `1 - p_appear(GK1)` (~0.035), which is what drives it to the £4.0m floor.

Slot assignment is a **decision variable**, not a post-hoc sort: the weights differ by more than an order of magnitude, so averaging them values the third slot ~15x too high and overpays for a seat that never plays. Since weights depend on the XI and the XI depends on the weights, the MILP is re-solved to a fixed point (typically 2-3 iterations, capped at 25).

**Captain is doubled** (`captain_multiplier=2.0`; 1.0 disables, 3.0 for Triple Captain) via a binary constrained to exactly one starter. You always captain someone, so those points are real — omitting them systematically undervalues premiums.

### Prices

Prices exist **only in the FPL API** (`bootstrap-static` → `now_cost`); nothing in the repo stores them, so `fetch_fpl_prices()` needs internet. Joined by name via the same cascade used for card data (~98.6% match; unmatched players are excluded and printed).

`snapshot_prices()` appends every fetch to `data/fpl_prices.csv`. **Not recoverable retroactively** — the API serves per-gameweek price history only for the current season and keeps just `start_cost`/`end_cost` per player in `history_past` afterwards.

## Data Pipeline

### Scraping

```bash
# Scrape specific gameweek
python scrape_update_data.py --gameweek 28

# Auto-detect latest gameweek
python scrape_update_data.py --auto
```

Both modes write the **full FotMob match-details JSON** to `data/matches/raw/{match_id}.json.gz` (gzipped) on every fetch (`save_raw_match()` in `scrape_update_data.py`). These raw payloads are the source of truth for downstream features that need fields not surfaced into `player_stats.csv` — currently the **manager embeddings** (manager identity + formation are pulled from `content.lineup.{home,away}Team.coach` and `.formation`). The cache used by the embedding feature, `data/match_managers.csv`, can be rebuilt at any time from these raw files.

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
- Predictions: `pred_minutes` (E[minutes | appears]), `pred_appear_prob` (P(plays at all)), `pred_exp_goals`, `pred_exp_assists`, `pred_cs_prob`, `pred_defcon_prob`, `pred_exp_saves`, `pred_yellow_prob`, `pred_red_prob`, `pred_bonus`
- Points breakdown: `exp_goals_pts`, `exp_assists_pts`, `exp_cs_pts`, `exp_conceded_penalty`, `exp_saves_pts`, `exp_defcon_pts`, `exp_bonus_pts`, `exp_yellow_pts`, `exp_red_pts`, `exp_total_pts`

## Visualization

`src/viz.py` generates a standalone HTML file (`distributions.html`) with:

- **Optimal-squad pitch** — the selected 15 laid out in the shape of the formation the optimizer chose, a coloured dot per player with name and price below, bench in autosub order, captain marked. Desktop `body` is `display:flex`, so the side column sits right of the ridge plot with the squad above the metrics; the mobile template stacks it first
- **D3.js ridge plot** showing Monte Carlo points distributions for top outfield players
- **Metrics dashboard** — sub-model holdout metrics, overall FPL points MAE/Poisson deviance/Spearman, and a calibration plot (predicted vs actual by bucket)
- **Responsive layout** — desktop ridge plot and mobile card layout embedded in a single file, selected at load time based on viewport width

Generated via:
```python
from src.viz import generate_distribution_html

viz_metrics = pipeline.get_viz_metrics()
generate_distribution_html(
    predictions,
    pipeline.last_simulations,
    output_path='distributions.html',
    top_n=100,
    gameweek=32,
    metrics=viz_metrics,
    squad=squad,          # from pipeline.optimize_squad(); omit to skip the pitch
)
```

## Weekly Deploy Workflow

The full weekly workflow (scrape, train, predict, generate viz, deploy to [danielfindley.com](https://danielfindley.com)) is automated via a Claude Code slash command (`/deploy-predictions`). The workflow:

1. **Detect next gameweek** — queries the FPL API for the latest finished GW and compares against scraped data
2. **Scrape** — `python scrape_update_data.py --auto` (Playwright + Cloudflare bypass on FotMob)
3. **Tune** (optional, ~30-60 min) — asks whether to retune hyperparameters or reuse cached params from the latest saved run
4. **Train**
5. **Scrape predicted lineups** — `pipeline.load_lineups()`, before `predict()`. Wrapped in try/except so a scrape failure warns and falls through to the model rather than killing the deploy
6. **Predict + optimize squad + viz** — predicts next GW, picks the £100m squad (fetching and snapshotting live prices), generates `distributions.html`
7. **Save run** — `pipeline.save_run()` persists predictions, simulations, squad, tuned params, and metrics to `data/runs/gw{N}_{timestamp}/`
8. **Deploy** — copies `distributions.html` to the website repo, commits, and pushes

Both the price and lineup snapshots are append-only histories that **cannot be backfilled**, so running the deploy weekly is what builds the data needed to later calibrate `lineup_weight` and model price changes.

### Saved runs (`data/runs/`)

Each `save_run()` creates a timestamped directory containing:

```
data/runs/gw32_20260323_235026/
├── predictions.csv          # Full prediction table
├── simulations/             # Monte Carlo arrays (.npy)
├── squad.csv                # Optimized 15 with is_starter flag
├── squad_meta.json          # Formation, cost, captain, bench weights used
├── tuned_params.json        # Optuna-selected hyperparams + features
├── test_metrics.json        # Holdout test set metrics
├── viz_metrics.json         # Formatted metrics for the HTML viz
└── meta.json                # Run metadata (description, timestamp)
```

Previous runs can be used to regenerate the viz or load tuned params without retraining.

## Experiment Tracking

All tuning runs are auto-logged to `data/experiments.db` (SQLite). See `AGENTS.md` for the full experimentation workflow.

```bash
python scripts/experiment.py --history             # all runs
python scripts/experiment.py --best                # best per model
python scripts/experiment.py --compare 3           # compare last 3 runs
```
