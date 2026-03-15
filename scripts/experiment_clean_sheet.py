"""
CleanSheetModel Improvement Experiment Script

Tests multiple strategies to improve the goals-against / clean-sheet model.
Does NOT modify any project source files.

Run with: python scripts/experiment_clean_sheet.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data via the pipeline
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("CLEAN SHEET MODEL IMPROVEMENT EXPERIMENTS")
print("=" * 70)
print("\nLoading pipeline data …")

from src.pipeline import FPLPipeline
pipeline = FPLPipeline('data')
pipeline.load_data(verbose=False)
pipeline.compute_features(verbose=False)
print(f"  Loaded {len(pipeline.df):,} player-match rows")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Helper utilities
# ─────────────────────────────────────────────────────────────────────────────
def poisson_deviance(y_true, y_pred):
    """Mean Poisson deviance loss (lower = better)."""
    y_pred = np.clip(y_pred, 1e-6, None)
    return float(np.mean(2 * (y_true * np.log(np.where(y_true > 0, y_true / y_pred, 1)) - (y_true - y_pred))))


def brier(y_true_count, y_pred_lambda):
    """Brier score for clean-sheet probability (P(goals=0) = e^{-lambda})."""
    actual_cs = (y_true_count == 0).astype(float)
    pred_cs = poisson.pmf(0, y_pred_lambda)
    return float(brier_score_loss(actual_cs, pred_cs))


def ece(y_true_count, y_pred_lambda, n_bins=10):
    """Expected calibration error for clean-sheet probability."""
    actual_cs = (y_true_count == 0).astype(float)
    pred_cs = poisson.pmf(0, y_pred_lambda)
    # bin by predicted probability
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (pred_cs >= bins[i]) & (pred_cs < bins[i + 1])
        if mask.sum() > 0:
            ece_val += mask.sum() / len(pred_cs) * abs(pred_cs[mask].mean() - actual_cs[mask].mean())
    return float(ece_val)


def run_cv(team_df, features, target='goals_conceded', xgb_params=None, n_splits=5, verbose=False):
    """
    TimeSeriesSplit CV with Poisson deviance, MAE, CS Brier score.
    Returns dict of mean metrics.
    """
    default_xgb = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='count:poisson',
        verbosity=0,
    )
    if xgb_params:
        default_xgb.update(xgb_params)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    team_df = team_df.sort_values(['season', 'gameweek']).reset_index(drop=True)

    X_full = team_df[features].fillna(0).astype(float)
    y_full = team_df[target].fillna(0).values

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
        model = xgb.XGBRegressor(**default_xgb)
        model.fit(X_full.values[train_idx], y_full[train_idx])
        y_val = y_full[val_idx]
        y_pred = np.clip(model.predict(X_full.values[val_idx]), 1e-6, 10.0)

        fold_results.append({
            'fold': fold_idx,
            'n_val': len(val_idx),
            'poisson_dev': poisson_deviance(y_val, y_pred),
            'mae': mean_absolute_error(y_val, y_pred),
            'brier': brier(y_val, y_pred),
            'ece': ece(y_val, y_pred),
        })

    results_df = pd.DataFrame(fold_results)
    # Weighted by fold size
    total = results_df['n_val'].sum()
    weights = results_df['n_val'] / total

    return {
        'poisson_dev': float((results_df['poisson_dev'] * weights).sum()),
        'mae': float((results_df['mae'] * weights).sum()),
        'brier': float((results_df['brier'] * weights).sum()),
        'ece': float((results_df['ece'] * weights).sum()),
        'n_features': len(features),
        'n_samples': len(y_full),
    }


def print_result(name, res, baseline=None):
    b_arrow = ""
    m_arrow = ""
    br_arrow = ""
    if baseline:
        b_arrow = f"  (Δ {res['poisson_dev'] - baseline['poisson_dev']:+.4f})"
        m_arrow = f"  (Δ {res['mae'] - baseline['mae']:+.4f})"
        br_arrow = f"  (Δ {res['brier'] - baseline['brier']:+.4f})"
    print(f"\n  {name}")
    print(f"    Poisson deviance : {res['poisson_dev']:.4f}{b_arrow}")
    print(f"    MAE              : {res['mae']:.4f}{m_arrow}")
    print(f"    CS Brier score   : {res['brier']:.4f}{br_arrow}")
    print(f"    CS ECE           : {res['ece']:.4f}")
    print(f"    Features used    : {res['n_features']}  |  Samples: {res['n_samples']:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build the baseline team-level dataset (replicates CleanSheetModel.prepare_team_features)
# ─────────────────────────────────────────────────────────────────────────────
print("\nPreparing baseline team-match dataset …")

from src.models.clean_sheet import CleanSheetModel
cs_model = CleanSheetModel()
df_raw = pipeline.df.copy()
team_df_base = cs_model.prepare_team_features(df_raw)
team_df_base = team_df_base.dropna(subset=['team_conceded_roll5', 'goals_conceded'])
team_df_base = team_df_base.sort_values(['season', 'gameweek']).reset_index(drop=True)
print(f"  Team-match rows: {len(team_df_base):,}")
print(f"  Avg goals conceded: {team_df_base['goals_conceded'].mean():.3f}")
print(f"  Actual CS rate: {(team_df_base['goals_conceded'] == 0).mean():.1%}")

BASELINE_FEATURES = CleanSheetModel.FEATURES[:]  # 20 features

# ─────────────────────────────────────────────────────────────────────────────
# 4. Baseline
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 0: BASELINE (current 20 features, count:poisson)")
print("=" * 70)
baseline = run_cv(team_df_base, BASELINE_FEATURES)
print_result("Baseline", baseline)
results_summary = {'Baseline': baseline}

# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPERIMENT A: Richer player-level aggregates for team-match
#    Aggregate from player_stats.csv: tackles, interceptions, clearances,
#    blocks, recoveries, shots_faced (saves+goals_conceded), xgot_faced,
#    accurate_passes (possession proxy), opponent key_passes / shots / big_chances
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT A: More aggregated player features")
print("=" * 70)

def _normalize(name):
    if pd.isna(name):
        return ''
    return str(name).lower().replace(' ', '_').replace("'", '').strip()

def build_extended_team_df(df_raw):
    """Build team-match dataset with extra aggregated defensive / shot-quality features."""
    df = df_raw.copy()
    df['team_norm'] = df['team'].apply(_normalize)
    df['opponent_norm'] = df['opponent'].apply(_normalize)

    # Rename cols that come from pipeline (may differ slightly)
    col_renames = {
        'minutes_played': 'minutes',
        'expected_goals_(xg)': 'xg',
        'expected_goals_on_target_(xgot)': 'xgot',
        'expected_assists_(xa)': 'xa',
        'chances_created': 'key_passes',
    }
    for old, new in col_renames.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Ensure numeric
    for col in ['tackles', 'interceptions', 'clearances', 'blocks', 'recoveries',
                'saves', 'xgot_faced', 'accurate_passes', 'key_passes',
                'goals', 'xg', 'shots_on_target', 'big_chances_created',
                'goals_conceded', 'touches']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    # ── Team aggregates per match ──
    team_agg = df.groupby(['team_norm', 'opponent_norm', 'season', 'gameweek', 'is_home']).agg(
        goals=('goals', 'sum'),
        xg=('xg', 'sum'),
        team=('team', 'first'),
        opponent=('opponent', 'first'),
        # defensive quality proxies
        tackles=('tackles', 'sum'),
        interceptions=('interceptions', 'sum'),
        clearances=('clearances', 'sum'),
        blocks=('blocks', 'sum'),
        recoveries=('recoveries', 'sum'),
        # GK workload
        saves=('saves', 'sum'),
        xgot_faced=('xgot_faced', 'sum'),
        # possession
        accurate_passes=('accurate_passes', 'sum'),
        touches=('touches', 'sum'),
        # attacking
        key_passes=('key_passes', 'sum'),
        shots_on_target=('shots_on_target', 'sum'),
        big_chances_created=('big_chances_created', 'sum'),
    ).reset_index()

    # Goals conceded = opponent goals
    opp_goals = team_agg[['team_norm', 'season', 'gameweek', 'goals', 'xg']].copy()
    opp_goals = opp_goals.rename(columns={
        'team_norm': 'opponent_norm',
        'goals': 'goals_conceded',
        'xg': 'xga',
    })
    team_agg = team_agg.merge(opp_goals, on=['opponent_norm', 'season', 'gameweek'], how='left')

    team_agg['clean_sheet'] = (team_agg['goals_conceded'] == 0).astype(int)
    team_agg = team_agg.sort_values(['team_norm', 'season', 'gameweek']).reset_index(drop=True)

    # Composite defensive actions
    team_agg['def_actions'] = team_agg['tackles'] + team_agg['interceptions'] + team_agg['clearances'] + team_agg['blocks']
    team_agg['gk_workload'] = team_agg['saves'] + team_agg['goals_conceded']  # shots faced

    # ── Rolling features for each team ──
    def team_roll(df, col, windows, prefix=None):
        if prefix is None:
            prefix = col
        for w in windows:
            df[f'{prefix}_roll{w}'] = df.groupby('team_norm')[col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
        return df

    # Existing (replicate baseline)
    for col, pfx in [('goals_conceded', 'team_conceded'), ('xga', 'team_xga')]:
        team_agg = team_roll(team_agg, col, [1, 3, 5, 10, 30], pfx)
    team_agg = team_roll(team_agg, 'clean_sheet', [1, 3, 5, 10, 30], 'team_cs')

    # New defensive aggregates
    for col, pfx in [
        ('def_actions', 'team_def_actions'),
        ('tackles', 'team_tackles'),
        ('interceptions', 'team_intercepts'),
        ('clearances', 'team_clearances'),
        ('recoveries', 'team_recoveries'),
        ('gk_workload', 'team_shots_faced'),
        ('xgot_faced', 'team_xgot_faced'),
        ('accurate_passes', 'team_passes'),
        ('touches', 'team_touches'),
    ]:
        team_agg = team_roll(team_agg, col, [5, 10], pfx)

    # ── Opponent attacking rolling (what the opponent typically scores) ──
    # From team's own attacking stats: key_passes, shots_on_target, big_chances_created
    opp_offense_src = team_agg[['team_norm', 'season', 'gameweek',
                                 'key_passes', 'shots_on_target', 'big_chances_created',
                                 'goals', 'xg']].copy()
    for col, pfx in [
        ('goals', 'opp_scored'),
        ('xg', 'opp_xg'),
        ('key_passes', 'opp_key_passes'),
        ('shots_on_target', 'opp_shots_ot'),
        ('big_chances_created', 'opp_big_chances'),
    ]:
        for w in [5, 10]:
            opp_offense_src[f'{pfx}_roll{w}'] = opp_offense_src.groupby('team_norm')[col].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )

    opp_lookup_cols = ['team_norm', 'season', 'gameweek'] + \
        [c for c in opp_offense_src.columns if any(c.startswith(p) for p in
         ['opp_scored_roll', 'opp_xg_roll', 'opp_key_passes_roll', 'opp_shots_ot_roll', 'opp_big_chances_roll'])]

    opp_lookup = opp_offense_src[opp_lookup_cols].rename(columns={'team_norm': 'opponent_norm'})
    team_agg = team_agg.merge(opp_lookup, on=['opponent_norm', 'season', 'gameweek'], how='left')

    team_agg = team_agg.drop(columns=['team_norm', 'opponent_norm'], errors='ignore')
    return team_agg


team_df_ext = build_extended_team_df(df_raw)
team_df_ext = team_df_ext.dropna(subset=['team_conceded_roll5', 'goals_conceded'])
print(f"  Extended dataset: {len(team_df_ext):,} rows, {len(team_df_ext.columns)} columns")

EXP_A_FEATURES = BASELINE_FEATURES + [
    # team defensive aggregate rolling
    'team_def_actions_roll5', 'team_def_actions_roll10',
    'team_tackles_roll5', 'team_intercepts_roll5',
    'team_clearances_roll5', 'team_recoveries_roll5',
    # GK workload
    'team_shots_faced_roll5', 'team_shots_faced_roll10',
    'team_xgot_faced_roll5', 'team_xgot_faced_roll10',
    # possession proxy
    'team_passes_roll5', 'team_touches_roll5',
    # opponent attacking
    'opp_key_passes_roll5', 'opp_key_passes_roll10',
    'opp_shots_ot_roll5', 'opp_shots_ot_roll10',
    'opp_big_chances_roll5', 'opp_big_chances_roll10',
]
# Keep only those actually in the dataset
EXP_A_FEATURES = [f for f in EXP_A_FEATURES if f in team_df_ext.columns]

exp_a = run_cv(team_df_ext, EXP_A_FEATURES)
print_result("Exp A – More player-aggregate features", exp_a, baseline)
results_summary['Exp A: More features'] = exp_a

# ─────────────────────────────────────────────────────────────────────────────
# 6. EXPERIMENT B: Interaction features
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT B: Interaction features")
print("=" * 70)

team_df_int = team_df_ext.copy()

# Interaction terms (shift already baked into the roll columns)
team_df_int['xga_x_opp_xg_roll5'] = team_df_int['team_xga_roll5'].fillna(0) * team_df_int['opp_xg_roll5'].fillna(0)
team_df_int['xga_x_opp_xg_roll10'] = team_df_int['team_xga_roll10'].fillna(0) * team_df_int['opp_xg_roll10'].fillna(0)
team_df_int['cs_rate_x_home'] = team_df_int['team_cs_roll5'].fillna(0) * team_df_int['is_home'].fillna(0)
team_df_int['conceded_x_opp_scored_roll5'] = team_df_int['team_conceded_roll5'].fillna(0) * team_df_int['opp_scored_roll5'].fillna(0)
team_df_int['def_actions_x_opp_shots_roll5'] = team_df_int.get('team_def_actions_roll5', 0).fillna(0) * team_df_int.get('opp_shots_ot_roll5', 0).fillna(0) if 'team_def_actions_roll5' in team_df_int.columns else 0
# Ratio: team xGA vs opponent xG (defensive strength vs opponent threat)
team_df_int['xga_vs_opp_xg_ratio_roll5'] = team_df_int['team_xga_roll5'].fillna(1) / (team_df_int['opp_xg_roll5'].fillna(1) + 0.01)
team_df_int['cs_rate_vs_opp_scored_ratio'] = team_df_int['team_cs_roll10'].fillna(0.25) / (team_df_int['opp_scored_roll10'].fillna(1) + 0.1)

INT_FEATURES = EXP_A_FEATURES + [
    'xga_x_opp_xg_roll5', 'xga_x_opp_xg_roll10',
    'cs_rate_x_home', 'conceded_x_opp_scored_roll5',
    'xga_vs_opp_xg_ratio_roll5', 'cs_rate_vs_opp_scored_ratio',
]
if 'team_def_actions_roll5' in team_df_int.columns:
    INT_FEATURES.append('def_actions_x_opp_shots_roll5')
INT_FEATURES = [f for f in INT_FEATURES if f in team_df_int.columns]

exp_b = run_cv(team_df_int, INT_FEATURES)
print_result("Exp B – Interaction features added", exp_b, baseline)
results_summary['Exp B: + Interactions'] = exp_b

# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPERIMENT C: Ratio / difference engineering on baseline features
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT C: Ratio and difference features")
print("=" * 70)

team_df_ratio = team_df_base.copy()

# Trend: short window vs long window (momentum)
team_df_ratio['conceded_trend'] = team_df_ratio['team_conceded_roll3'].fillna(0) - team_df_ratio['team_conceded_roll10'].fillna(0)
team_df_ratio['xga_trend'] = team_df_ratio['team_xga_roll3'].fillna(0) - team_df_ratio['team_xga_roll10'].fillna(0)
team_df_ratio['cs_trend'] = team_df_ratio['team_cs_roll3'].fillna(0) - team_df_ratio['team_cs_roll10'].fillna(0)

# Ratios: team conceded / opponent scored
team_df_ratio['conceded_vs_opp_scored_ratio5'] = team_df_ratio['team_conceded_roll5'].fillna(1) / (team_df_ratio['opp_scored_roll5'].fillna(1) + 0.01)
team_df_ratio['xga_vs_opp_xg_ratio5'] = team_df_ratio['team_xga_roll5'].fillna(1) / (team_df_ratio['opp_xg_roll5'].fillna(1) + 0.01)

# Difference: xGA vs actual conceded (model over/under-performance)
team_df_ratio['conceded_minus_xga_roll5'] = team_df_ratio['team_conceded_roll5'].fillna(0) - team_df_ratio['team_xga_roll5'].fillna(0)

# Opponent vs baseline (opp scoring rate relative to league avg)
league_avg_scored = team_df_ratio['opp_scored_roll5'].mean()
team_df_ratio['opp_scored_vs_avg5'] = team_df_ratio['opp_scored_roll5'].fillna(league_avg_scored) - league_avg_scored

RATIO_FEATURES = BASELINE_FEATURES + [
    'conceded_trend', 'xga_trend', 'cs_trend',
    'conceded_vs_opp_scored_ratio5', 'xga_vs_opp_xg_ratio5',
    'conceded_minus_xga_roll5', 'opp_scored_vs_avg5',
]
RATIO_FEATURES = [f for f in RATIO_FEATURES if f in team_df_ratio.columns]

exp_c = run_cv(team_df_ratio, RATIO_FEATURES)
print_result("Exp C – Ratio and difference features", exp_c, baseline)
results_summary['Exp C: + Ratios/diffs'] = exp_c

# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPERIMENT D: Extended rolling windows (add roll20)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT D: Extra rolling windows (roll20)")
print("=" * 70)

team_df_windows = cs_model.prepare_team_features(df_raw)
team_df_windows = team_df_windows.dropna(subset=['team_conceded_roll5', 'goals_conceded'])
team_df_windows = team_df_windows.sort_values(['season', 'gameweek']).reset_index(drop=True)

# Add roll20 for key metrics
for col, source in [('team_conceded_roll20', 'goals_conceded'), ('team_xga_roll20', 'xga')]:
    team_df_windows[col] = team_df_windows.sort_values(['team', 'season', 'gameweek']).groupby('team')[source].transform(
        lambda x: x.shift(1).rolling(20, min_periods=1).mean()
    ) if 'team' in team_df_windows.columns else 0

# Re-sort after groupby transform
team_df_windows = team_df_windows.sort_values(['season', 'gameweek']).reset_index(drop=True)

WIN_FEATURES = BASELINE_FEATURES + ['team_conceded_roll20', 'team_xga_roll20']
WIN_FEATURES = [f for f in WIN_FEATURES if f in team_df_windows.columns]

exp_d = run_cv(team_df_windows, WIN_FEATURES)
print_result("Exp D – + roll20 windows", exp_d, baseline)
results_summary['Exp D: + roll20'] = exp_d

# ─────────────────────────────────────────────────────────────────────────────
# 9. EXPERIMENT E: Different XGBoost objectives
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT E: Alternative XGBoost objectives")
print("=" * 70)

objectives = {
    'reg:squarederror': {'objective': 'reg:squarederror'},
    'reg:tweedie (p=1.5)': {'objective': 'reg:tweedie', 'tweedie_variance_power': 1.5},
    'reg:tweedie (p=1.2)': {'objective': 'reg:tweedie', 'tweedie_variance_power': 1.2},
    'count:poisson (baseline obj)': {'objective': 'count:poisson'},
}

exp_e_results = {}
for obj_name, params in objectives.items():
    r = run_cv(team_df_base, BASELINE_FEATURES, xgb_params=params)
    exp_e_results[obj_name] = r
    print_result(f"  Obj: {obj_name}", r, baseline)

# Pick best
best_obj_name = min(exp_e_results, key=lambda k: exp_e_results[k]['poisson_dev'])
results_summary['Exp E: Best objective'] = exp_e_results[best_obj_name]
print(f"\n  Best objective: {best_obj_name}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. EXPERIMENT F: Tuned XGBoost hyperparams (manual sweep)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT F: Hyperparameter sweep on baseline features")
print("=" * 70)

hp_grid = [
    dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=3),
    dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=2),
    dict(n_estimators=400, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, min_child_weight=5),
    dict(n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, min_child_weight=1),
    dict(n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.75, colsample_bytree=0.75, min_child_weight=4),
]

best_hp_result = None
best_hp_params = None
for hp in hp_grid:
    r = run_cv(team_df_base, BASELINE_FEATURES, xgb_params=hp)
    label = f"d{hp['max_depth']}_lr{hp['learning_rate']}_n{hp['n_estimators']}"
    print_result(f"  HP {label}", r, baseline)
    if best_hp_result is None or r['poisson_dev'] < best_hp_result['poisson_dev']:
        best_hp_result = r
        best_hp_params = hp

results_summary['Exp F: Best HP'] = best_hp_result
print(f"\n  Best HP: {best_hp_params}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. EXPERIMENT G: Combined best ideas
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT G: Combined best (ext. features + interactions + ratios + best HP)")
print("=" * 70)

# Merge ratio features into the extended dataset
team_df_combined = team_df_int.copy()

# Add ratio features
team_df_combined['conceded_trend'] = team_df_combined['team_conceded_roll3'].fillna(0) - team_df_combined['team_conceded_roll10'].fillna(0)
team_df_combined['xga_trend'] = team_df_combined['team_xga_roll3'].fillna(0) - team_df_combined['team_xga_roll10'].fillna(0)
team_df_combined['cs_trend'] = team_df_combined['team_cs_roll3'].fillna(0) - team_df_combined['team_cs_roll10'].fillna(0)
team_df_combined['conceded_vs_opp_scored_ratio5'] = team_df_combined['team_conceded_roll5'].fillna(1) / (team_df_combined['opp_scored_roll5'].fillna(1) + 0.01)
team_df_combined['conceded_minus_xga_roll5'] = team_df_combined['team_conceded_roll5'].fillna(0) - team_df_combined['team_xga_roll5'].fillna(0)
league_avg_scored_comb = team_df_combined['opp_scored_roll5'].mean()
team_df_combined['opp_scored_vs_avg5'] = team_df_combined['opp_scored_roll5'].fillna(league_avg_scored_comb) - league_avg_scored_comb

COMBINED_FEATURES = list(set(INT_FEATURES + [
    'conceded_trend', 'xga_trend', 'cs_trend',
    'conceded_vs_opp_scored_ratio5', 'conceded_minus_xga_roll5',
    'opp_scored_vs_avg5',
]))
COMBINED_FEATURES = [f for f in COMBINED_FEATURES if f in team_df_combined.columns]

# Use best HP params
exp_g = run_cv(team_df_combined, COMBINED_FEATURES, xgb_params=best_hp_params)
print_result("Exp G – Combined best", exp_g, baseline)
results_summary['Exp G: Combined'] = exp_g

# ─────────────────────────────────────────────────────────────────────────────
# 12. EXPERIMENT H: Feature selection on combined set
#     Drop low-importance features via a quick importance-based filter
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT H: Feature-selected subset of combined (importance threshold)")
print("=" * 70)

def get_importance_filtered_features(team_df, features, xgb_params=None, top_k=30):
    """Train once on 80% of data, keep top_k features by importance."""
    default_xgb = dict(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective='count:poisson', verbosity=0,
    )
    if xgb_params:
        default_xgb.update(xgb_params)

    team_df_s = team_df.sort_values(['season', 'gameweek']).reset_index(drop=True)
    split = int(len(team_df_s) * 0.8)
    X_tr = team_df_s[features].fillna(0).astype(float).values[:split]
    y_tr = team_df_s['goals_conceded'].fillna(0).values[:split]
    m = xgb.XGBRegressor(**default_xgb)
    m.fit(X_tr, y_tr)
    imp = pd.Series(m.feature_importances_, index=features).sort_values(ascending=False)
    return list(imp.head(top_k).index)

top_features = get_importance_filtered_features(team_df_combined, COMBINED_FEATURES, best_hp_params, top_k=30)
print(f"  Top 30 features selected from {len(COMBINED_FEATURES)}: {top_features[:10]} …")

exp_h = run_cv(team_df_combined, top_features, xgb_params=best_hp_params)
print_result("Exp H – Top-30 feature subset", exp_h, baseline)
results_summary['Exp H: Top-30 subset'] = exp_h

# Also try top 20
top20_features = top_features[:20]
exp_h20 = run_cv(team_df_combined, top20_features, xgb_params=best_hp_params)
print_result("Exp H – Top-20 feature subset", exp_h20, baseline)
results_summary['Exp H: Top-20 subset'] = exp_h20

# ─────────────────────────────────────────────────────────────────────────────
# 13. EXPERIMENT I: Stacking / blend of Poisson + Tweedie predictions
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT I: Poisson + Tweedie ensemble (OOF blend)")
print("=" * 70)

def run_cv_oof(team_df, features, target='goals_conceded', xgb_params=None, n_splits=5):
    """Return OOF predictions."""
    default_xgb = dict(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective='count:poisson', verbosity=0,
    )
    if xgb_params:
        default_xgb.update(xgb_params)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    team_df = team_df.sort_values(['season', 'gameweek']).reset_index(drop=True)
    X_full = team_df[features].fillna(0).astype(float)
    y_full = team_df[target].fillna(0).values
    oof_preds = np.full(len(y_full), np.nan)
    for train_idx, val_idx in tscv.split(X_full):
        m = xgb.XGBRegressor(**default_xgb)
        m.fit(X_full.values[train_idx], y_full[train_idx])
        oof_preds[val_idx] = np.clip(m.predict(X_full.values[val_idx]), 1e-6, 10.0)
    return oof_preds, y_full

# Use combined feature set, best HP
params_poisson = dict(best_hp_params, **{'objective': 'count:poisson'})
params_tweedie = dict(best_hp_params, **{'objective': 'reg:tweedie', 'tweedie_variance_power': 1.5})

oof_poisson, y_full = run_cv_oof(team_df_combined, top_features, xgb_params=params_poisson)
oof_tweedie, _ = run_cv_oof(team_df_combined, top_features, xgb_params=params_tweedie)

valid = ~np.isnan(oof_poisson) & ~np.isnan(oof_tweedie)
oof_blend = 0.6 * oof_poisson[valid] + 0.4 * oof_tweedie[valid]
y_v = y_full[valid]

blend_result = {
    'poisson_dev': poisson_deviance(y_v, np.clip(oof_blend, 1e-6, None)),
    'mae': float(mean_absolute_error(y_v, oof_blend)),
    'brier': brier(y_v, oof_blend),
    'ece': ece(y_v, oof_blend),
    'n_features': len(top_features),
    'n_samples': int(valid.sum()),
}
print_result("Exp I – Poisson/Tweedie blend (0.6/0.4)", blend_result, baseline)
results_summary['Exp I: Ensemble blend'] = blend_result

# ─────────────────────────────────────────────────────────────────────────────
# 14. Final summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

summary_rows = []
for name, res in results_summary.items():
    summary_rows.append({
        'Experiment': name,
        'Poisson Dev': f"{res['poisson_dev']:.4f}",
        'MAE': f"{res['mae']:.4f}",
        'CS Brier': f"{res['brier']:.4f}",
        'CS ECE': f"{res['ece']:.4f}",
        'N Features': res['n_features'],
        'Δ PoisDev': f"{res['poisson_dev'] - baseline['poisson_dev']:+.4f}" if name != 'Baseline' else '—',
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# Rank by Poisson deviance
best_exp = min(results_summary.items(), key=lambda x: x[1]['poisson_dev'])
print(f"\n>> BEST EXPERIMENT: {best_exp[0]}")
print(f"   Poisson Dev: {best_exp[1]['poisson_dev']:.4f}  "
      f"MAE: {best_exp[1]['mae']:.4f}  "
      f"Brier: {best_exp[1]['brier']:.4f}")
print(f"   Improvement vs baseline: Δ PoisDev = {best_exp[1]['poisson_dev'] - baseline['poisson_dev']:+.4f}")

# Show top-30 features from best config if applicable
if 'top_features' in dir():
    print("\nTop 30 features from combined/feature-selected model:")
    for i, f in enumerate(top_features[:30], 1):
        print(f"  {i:2d}. {f}")

print("\nDone.")
