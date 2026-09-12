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
│   ├── viz.py                          # Standalone HTML viz (ridge plot and metrics)
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
| **Minutes** | Playing time (1-90), conditional on appearing | XGBoost mixture: P(60+ given appearance) × 60+ regressor + remaining probability × short-appearance regressor | Rolling minutes, starter rate, current season minutes, goal involvement |
| **Appears** | P(minutes >= 1) | XGBoost binary classifier, isotonic-calibrated | Calendar-grid minutes features. The **only** model trained on non-appearances — see below |
| **Goals** | Goals per match (raw counts) | XGBoost Poisson regression | xG rolling, shots, player share of team output, opponent weakness, xG overperformance, form trends |
| **Assists** | Assists per match (raw counts) | XGBoost Poisson regression | xA rolling, key passes, player centrality, opponent weakness, xA overperformance, form trends |
| **Clean Sheet** | Team goals against (lambda) | XGBoost Poisson regression | Team conceded/xGA rolling (5 windows), opponent xG, prior lambda anchor |
| **Defcon** | Defensive contributions per match (raw counts) | XGBoost Poisson regression | Raw/per-90 defcon rolling, tackles, interceptions, clearances, blocks, recoveries, opponent context, pred_minutes |
| **Saves** | GK saves per 90 | XGBoost regression (GK only) | Saves rolling, xGoT faced, team defensive context, opponent attacking strength |
| **Cards** | Yellow/red card probability | XGBoost binary classifier (`binary:logistic`) | Yellow card rolling history, fouls per 90 rolling, defensive activity, yellow-per-foul rate. Trained on actual FPL API yellow card data (required) |
| **Bonus** | Expected bonus points (0-3) | Monte Carlo BPS simulation | Simulates goals/assists/CS/yellow cards, ranks BPS per match (including -3 BPS per yellow), awards 3-2-1 bonus |

## Feature Engineering

Rolling statistics use only completed results before the forecast deadline; this extends `shift(1)` to real kickoff chronology and double gameweeks.

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

`features.build_appearance_grid()` restores eligible player-fixture rows, including terminal non-appearances. Club blank gameweeks have no fixture and are not treated as a player being dropped. Explicit registration intervals can be supplied through `df.attrs['roster_spells']` (`player_id, team, season, start_date, end_date`). Otherwise membership is inferred from first observed appearances and club changes: unknown departures and pre-debut eligibility remain a limitation. The following appearance/calibration metrics describe the previous grid and require revalidation.

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

Raw prior-20-match summaries are read strictly before each forecast deadline.
The team's latest observed manager is assumed to continue; unknown historical
manager rows are not filled from future managers. Each model fits its own scaler
and PCA(8) on training rows with at least three prior manager matches. The saved
basis is reused at prediction time. There is no full-dataset or per-forecast PCA fit.

## Hyperparameter Tuning

Minutes and clean sheet tune first; goals, assists and defcon consume causal,
cross-fitted upstream predictions. Training, validation and holdout evaluation
use the actual model wrappers, including minute weights, target caps, and the
clean-sheet log-prior offset and geometric blend. Early OOF rows use prior
rolling minutes / the matchup prior, never their own observed outcome.

Five expanding folds contain whole forecast deadlines. Rows whose results were
not available at a validation origin are purged, and matches cannot straddle a
fold. A gameweek's default information boundary is 90 minutes before its first
kickoff; explicit `forecast_time` values override this. Features use real kickoff
chronology rather than gameweek-number ordering, including postponed matches.

Feature ranking is now wrapper-based gain importance, fitted separately inside
each training fold. Optuna chooses the feature count and tree parameters;
`pred_minutes` / `pred_team_goals` stay protected where applicable. The previous
five-ranking-method search is no longer the production tuning path. The final
feature list is ranked again on the full training window only. Manager PCA is
also fitted inside each wrapper's training window and reused at prediction time.

Minutes trials score the deployed probability-weighted mixture with Huber loss.
The other primary losses remain Poisson deviance (goals, assists, defcon, clean
sheet) and MAE (saves). Upstream parameters already selected in the training
partition are held fixed during downstream CV; these CV scores are not a fully
nested estimate of the entire selection process. The untouched chronological
holdout is the final check.

`use_subprocess=True` runs the **same implementation** in a fresh spawned process
per model. Existing parameter files remain accepted, but **retuning is strongly
recommended** after this change. Old and new CV/points metrics are not directly
comparable. The aggregate holdout report still uses reconstructed, played-only
points; it has not been replaced with official all-roster FPL scoring.

## Shared Match Simulation and Expected Points

The weekly and experimental five-week forecasts use the same event/scoring
engine. It samples a playing-time state, samples each team's score once, and
allocates scorer and assister credits from minutes-adjusted player intensities.
At most one different assister is credited per goal. Intensities above the team
total are reconciled proportionally; a residual bucket represents unmodeled
players and unassisted goals.

Clean sheets and conceded penalties read that score. Saves, cards and
negative-binomial defensive contributions scale with sampled minutes. Bonus
ranks the BPS from those same events. Tied first place gives 3, 3, 1; tied second
gives 3, 2, 2. Baseline-BPS reliability uses cumulative exposure, not a five-match
mean mistaken for a minutes total. The baseline is trained as a per-90 rate;
when actual BPS is available, shared simulated event contributions (including
cards and conceded goals) are removed from its target to avoid counting them twice.

The weekly minutes states are absent, 1–59, and 60–90. Thus a 50/50 mixture of
20 and 90 minutes earns 1.5 expected appearance points, even though its mean is
55 minutes. There is no eligibility cliff at **expected** minutes = 60.

- `pred_minutes`: E[minutes | appears], retained as an upstream feature.
- `pred_minutes_uncond`, `pred_appear_prob`, `pred_60_prob`: explicit mixture outputs.
- `exp_total_pts` and `exp_total_pts_uncond`: unconditional mean of scored draws.
- `exp_total_pts_cond`: conditional points view for consumers that need it.
- Point-component `exp_*` columns: unconditional means from the same draws.
- Raw `pred_exp_goals/assists/defcon`: conditional model outputs **before** team reconciliation.

Charts consume saved `total_points` draws without independently resampling
penalties or cards. Double-gameweek distributions sum fixture draws by player
ID. Saved-run metadata records the new semantics; legacy archives retain their
compatibility path. The optimizer uses unconditional points once, without an
extra appearance discount.

This remains an approximate simulator: player role draws are independent,
unmodeled players do not compete for bonus, and exact substitution/event timing
is not modeled. Team scores do not yet respond dynamically to lineup strength.
Configured scoring constants/eligibility rules are retained; this is not a
season-specific scoring-rule audit.

## Predicted Lineups

`pipeline.load_lineups(gameweek, season)` scrapes [RotoWire](https://www.rotowire.com/soccer/lineups.php) predicted XIs and **must be called before `predict()`**. Inference-only — nothing is retrained, so experiment history stays comparable.

**The problem it solves.** At a season opener every feature is three months stale, so the minutes model cannot separate a nailed starter from a squad player and hedges on both. On GW1 2026/27 it projected Chelsea's backup striker at **65 minutes** and the first-choice striker at **62** — the latter's late-season injury had both idled him and depressed the rolling form his cold-start cap is computed from. With the feed: **79 vs 33**.

### How it injects

Lineup P(start) and P(60+ | appears) are different quantities. A lineup prior is
converted using the training starter-completion rate when genuine start labels
are available, otherwise an explicit 0.90 completion prior. The resulting joint role prior is
blended with the model distribution (`lineup_weight=0.7`), raising appearance
probability for a likely starter. Live availability then discounts it once.
Starter/sub minutes come from their conditional regressors; the former
temperature sharpening and cold-start caps are removed.

OUT/SUS/QUES status still scales appearance probability. The resulting role
distribution drives every downstream simulated points component. Historical
lineup performance numbers above describe the previous implementation, not a
validation of this new blend.

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

**Unconditional expected points.** New forecasts already include absence risk. The optimizer reads `exp_total_pts_uncond` directly. Only legacy forecasts without that field receive the old appearance-probability discount.

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


## Regression Checks

Run `python -B -m pytest tests -q -p no:cacheprovider`. Tests use isolated
synthetic histories and temporary output directories; they do not retune, replace,
or deploy production runs. Coverage includes time boundaries, whole-match folds,
fold-local manager PCA, appearance-grid blanks, mixture thresholds, shared event
invariants, bonus ties, visualization totals and the weekly/five-week contract.

A real temporal retune/backtest is still required before claiming better MAE.
