"""
Investigation: Why does CleanSheetModel predict Arsenal CS ~33% vs betting market ~50%?
This is RESEARCH ONLY - no source files are modified.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from scipy.stats import poisson

print("=" * 70)
print("ARSENAL CLEAN SHEET INVESTIGATION")
print("=" * 70)

# ─── Load pipeline ────────────────────────────────────────────────────────────
from src.pipeline import FPLPipeline
from src.models.clean_sheet import CleanSheetModel

print("\n[1] Loading pipeline data...")
pipeline = FPLPipeline('data')
pipeline.load_data(verbose=False)
pipeline.compute_features(verbose=False)
print(f"    Player-match rows: {len(pipeline.df):,}")
print(f"    Seasons: {sorted(pipeline.df['season'].unique())}")

# ─── Prepare team-level features ──────────────────────────────────────────────
print("\n[2] Preparing team-match features...")
csm = CleanSheetModel()
team_df = csm.prepare_team_features(pipeline.df)
print(f"    Team-match rows: {len(team_df):,}")
print(f"    Columns: {list(team_df.columns)}")

# ─── Train the model ──────────────────────────────────────────────────────────
print("\n[3] Training CleanSheetModel...")

# Load tuned params if available
import json
from pathlib import Path
tuned_path = Path('data/tuning_results/clean_sheet_tuned.json')
if tuned_path.exists():
    with open(tuned_path) as f:
        tuned = json.load(f)
    print(f"    Loaded tuned params from {tuned_path}")
    print(f"    Selected features ({len(tuned.get('selected_features', []))}): {tuned.get('selected_features', [])}")
    csm = CleanSheetModel(**tuned)
else:
    print("    No tuned params found, using defaults")

csm.fit(team_df, verbose=True)

# ─── Find Arsenal's most recent rows ──────────────────────────────────────────
print("\n[4] Arsenal recent form (last 10 team-match rows)...")

def normalize(name):
    if pd.isna(name): return ''
    return str(name).lower().replace(' ', '_').replace("'", "").strip()

# Re-prepare with team_norm kept temporarily for filtering
team_df2 = csm.prepare_team_features(pipeline.df)
# Add back team_norm for filtering
team_df2_raw = pipeline.df.copy()
team_df2_raw['team_norm'] = team_df2_raw['team'].apply(normalize)
arsenal_teams = [t for t in team_df2_raw['team_norm'].unique() if 'arsenal' in t]
print(f"    Arsenal team_norm values: {arsenal_teams}")

# Get team_norm back into team_df2 temporarily
team_df2_with_norm = csm.prepare_team_features(pipeline.df)
# Reconstruct team_norm from 'team' column
team_df2_with_norm['team_norm'] = team_df2_with_norm['team'].apply(normalize)

arsenal_rows = team_df2_with_norm[team_df2_with_norm['team_norm'].str.contains('arsenal', na=False)].copy()
arsenal_rows = arsenal_rows.sort_values(['season', 'gameweek'])
print(f"    Total Arsenal rows: {len(arsenal_rows)}")

# Predict for Arsenal
arsenal_lambda = csm.predict_goals_against(arsenal_rows)
arsenal_raw_cs = poisson.pmf(0, arsenal_lambda)
arsenal_cal_cs = csm.predict_cs_prob(arsenal_rows)

arsenal_rows['pred_lambda'] = arsenal_lambda
arsenal_rows['raw_cs_prob'] = arsenal_raw_cs
arsenal_rows['cal_cs_prob'] = arsenal_cal_cs

print("\n    Last 10 Arsenal matches:")
display_cols = ['season', 'gameweek', 'opponent', 'is_home', 'goals_conceded',
                'team_xga_roll3', 'team_xga_roll10', 'team_xga_roll30',
                'team_cs_roll30', 'opp_xg_roll10',
                'pred_lambda', 'raw_cs_prob', 'cal_cs_prob']
display_cols = [c for c in display_cols if c in arsenal_rows.columns]
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.3f}'.format)
print(arsenal_rows[display_cols].tail(10).to_string(index=False))

# ─── GW30 specific row ────────────────────────────────────────────────────────
print("\n[5] Arsenal GW30 prediction (vs Everton at home)...")
current_season = pipeline.df['season'].max()
print(f"    Current season: {current_season}")

# Get Arsenal's most recent data row (proxy for GW30 features)
arsenal_latest = arsenal_rows[arsenal_rows['season'] == current_season].sort_values('gameweek').tail(1)
if len(arsenal_latest) == 0:
    print("    No current season Arsenal data found, using overall latest")
    arsenal_latest = arsenal_rows.tail(1)

print(f"\n    Arsenal latest row (GW{int(arsenal_latest['gameweek'].iloc[0])}):")
all_feature_cols = csm.features_to_use
feature_vals = arsenal_latest[all_feature_cols].iloc[0] if all(c in arsenal_latest.columns for c in all_feature_cols) else {}
for f in all_feature_cols:
    val = arsenal_latest[f].iloc[0] if f in arsenal_latest.columns else 'MISSING'
    print(f"      {f:40s}: {val}")

print(f"\n    Predicted lambda:     {arsenal_latest['pred_lambda'].iloc[0]:.4f}")
print(f"    Raw CS probability:   {arsenal_latest['raw_cs_prob'].iloc[0]:.3f}  ({arsenal_latest['raw_cs_prob'].iloc[0]*100:.1f}%)")
print(f"    Calibrated CS prob:   {arsenal_latest['cal_cs_prob'].iloc[0]:.3f}  ({arsenal_latest['cal_cs_prob'].iloc[0]*100:.1f}%)")

# ─── Compare to other top defensive teams ─────────────────────────────────────
print("\n[6] Comparison: Arsenal vs other teams (current season, last available GW)...")
top_teams = ['arsenal', 'liverpool', 'manchester_city', 'chelsea', 'nottingham_forest',
             'everton', 'newcastle_united', 'aston_villa']

current_df = team_df2_with_norm[team_df2_with_norm['season'] == current_season].copy()
current_lambda = csm.predict_goals_against(current_df)
current_cs = csm.predict_cs_prob(current_df)
current_df['pred_lambda'] = current_lambda
current_df['cal_cs_prob'] = current_cs

# Get latest GW for each team
latest_by_team = current_df.sort_values('gameweek').groupby('team_norm').last().reset_index()
latest_by_team = latest_by_team.sort_values('cal_cs_prob', ascending=False)

compare_cols = ['team_norm', 'gameweek', 'team_xga_roll3', 'team_xga_roll10', 'team_xga_roll30',
                'team_cs_roll30', 'opp_xg_roll10', 'is_home', 'pred_lambda', 'cal_cs_prob']
compare_cols = [c for c in compare_cols if c in latest_by_team.columns]
print(latest_by_team[compare_cols].head(15).to_string(index=False))

# ─── Feature importance ────────────────────────────────────────────────────────
print("\n[7] CleanSheetModel feature importance:")
fi = csm.feature_importance()
print(fi.to_string(index=False))

# ─── Arsenal vs league average for each feature ───────────────────────────────
print("\n[8] Arsenal feature values vs league average (current season)...")
arsenal_cur = current_df[current_df['team_norm'].str.contains('arsenal', na=False)].sort_values('gameweek').tail(1)
league_mean = current_df.groupby('team_norm').last().reset_index()[csm.features_to_use].mean()
league_std  = current_df.groupby('team_norm').last().reset_index()[csm.features_to_use].std()

print(f"\n    {'Feature':40s} {'Arsenal':>10} {'LeagMean':>10} {'Z-score':>10}")
print("    " + "-" * 72)
for f in csm.features_to_use:
    if f in arsenal_cur.columns:
        a_val = arsenal_cur[f].iloc[0]
        m_val = league_mean[f]
        s_val = league_std[f] if league_std[f] > 0 else 1.0
        z = (a_val - m_val) / s_val
        print(f"    {f:40s} {a_val:10.3f} {m_val:10.3f} {z:10.2f}")
    else:
        print(f"    {f:40s} {'MISSING':>10}")

# ─── Calibration analysis: predicted vs actual CS rate ────────────────────────
print("\n[9] Calibration analysis: predicted vs actual CS rates...")
# Use all historical data
all_lambda = csm.predict_goals_against(team_df2_with_norm.dropna(subset=['goals_conceded']))
all_cs_prob = csm.predict_cs_prob(team_df2_with_norm.dropna(subset=['goals_conceded']))
all_actual = (team_df2_with_norm.dropna(subset=['goals_conceded'])['goals_conceded'] == 0).values

# Bin into deciles by predicted CS prob
df_cal = pd.DataFrame({'pred_cs': all_cs_prob, 'actual_cs': all_actual.astype(float)})
df_cal['bin'] = pd.qcut(df_cal['pred_cs'], q=10, duplicates='drop')
cal_summary = df_cal.groupby('bin').agg(
    count=('actual_cs', 'count'),
    mean_pred=('pred_cs', 'mean'),
    mean_actual=('actual_cs', 'mean'),
).reset_index()
cal_summary['error'] = cal_summary['mean_pred'] - cal_summary['mean_actual']
print(cal_summary.to_string(index=False))

# Focus on 30-60% range
print("\n    High-prob region (pred CS > 30%):")
high_mask = df_cal['pred_cs'] > 0.30
hi_pred = df_cal[high_mask]['pred_cs'].mean()
hi_actual = df_cal[high_mask]['actual_cs'].mean()
print(f"    N={high_mask.sum()}, mean_pred={hi_pred:.3f}, mean_actual={hi_actual:.3f}, bias={hi_pred-hi_actual:+.3f}")

# ─── Rolling window adequacy ──────────────────────────────────────────────────
print("\n[10] Rolling window adequacy for Arsenal...")
# Show multiple windows for Arsenal
roll_cols = [c for c in arsenal_rows.columns if c.startswith('team_xga_roll') or c.startswith('team_cs_roll') or c.startswith('team_conceded_roll')]
roll_cols.sort()
print("\n    Arsenal defensive rolling stats (last 10 rows):")
show_cols = ['season', 'gameweek', 'goals_conceded'] + roll_cols
show_cols = [c for c in show_cols if c in arsenal_rows.columns]
print(arsenal_rows[show_cols].tail(10).to_string(index=False))

# Actual season stats
current_arsenal = arsenal_rows[arsenal_rows['season'] == current_season]
actual_cs_rate = current_arsenal['clean_sheet'].mean() if 'clean_sheet' in current_arsenal.columns else None
actual_xga = current_arsenal['xga'].mean() if 'xga' in current_arsenal.columns else None
actual_conceded = current_arsenal['goals_conceded'].mean() if 'goals_conceded' in current_arsenal.columns else None
n_games = len(current_arsenal)
print(f"\n    Arsenal {current_season} actual stats ({n_games} games):")
print(f"      Actual CS rate:       {actual_cs_rate:.3f}" if actual_cs_rate is not None else "      CS rate: N/A")
print(f"      Actual xGA/game:      {actual_xga:.3f}" if actual_xga is not None else "      xGA: N/A")
print(f"      Actual goals conceded/game: {actual_conceded:.3f}" if actual_conceded is not None else "      Goals conceded: N/A")

# ─── Opponent (Everton) attacking quality ─────────────────────────────────────
print("\n[11] Everton attacking quality (current season)...")
everton_teams = [t for t in team_df2_with_norm['team_norm'].unique() if 'everton' in t]
print(f"    Everton team_norm values: {everton_teams}")
everton_cur = current_df[current_df['team_norm'].str.contains('everton', na=False)].sort_values('gameweek').tail(1)
if len(everton_cur) > 0:
    opp_cols = [c for c in everton_cur.columns if 'opp_xg' in c or 'opp_shots' in c or 'opp_key' in c or c == 'xg']
    opp_cols = [c for c in opp_cols if c in everton_cur.columns]
    print("\n    Everton as opponent — their rolling attack features (these become 'opp_*' for Arsenal):")
    for c in sorted(opp_cols):
        print(f"      {c:40s}: {everton_cur[c].iloc[0]:.3f}")

    # What Everton's opp_xg_roll10 is (i.e., Everton's own xg rolled)
    everton_xg_roll10 = everton_cur['opp_xg_roll10'].values[0] if 'opp_xg_roll10' in everton_cur.columns else None
    print(f"\n    Everton opp_xg_roll10 (Everton's xg/game rolling 10): N/A (this col is OPPONENT's xg)")
    # The actual xg Everton generate per game
    everton_xg_src = current_df[current_df['team_norm'].str.contains('everton', na=False)].sort_values('gameweek')
    if 'xg' in everton_xg_src.columns:
        print(f"    Everton own xg recent (raw, last 10 games):")
        print(everton_xg_src[['gameweek','xg']].tail(10).to_string(index=False))

# ─── Home advantage analysis ──────────────────────────────────────────────────
print("\n[12] Home vs Away CS rate in training data...")
# Overall
home_cs = team_df2_with_norm[team_df2_with_norm['is_home'] == 1]['clean_sheet'].mean()
away_cs = team_df2_with_norm[team_df2_with_norm['is_home'] == 0]['clean_sheet'].mean()
print(f"    Overall home CS rate: {home_cs:.3f} ({home_cs*100:.1f}%)")
print(f"    Overall away CS rate: {away_cs:.3f} ({away_cs*100:.1f}%)")

# In model predictions
all_home_cs_pred = current_df[current_df['is_home'] == 1]['cal_cs_prob'].mean()
all_away_cs_pred = current_df[current_df['is_home'] == 0]['cal_cs_prob'].mean()
print(f"    Model home CS pred:   {all_home_cs_pred:.3f} ({all_home_cs_pred*100:.1f}%)")
print(f"    Model away CS pred:   {all_away_cs_pred:.3f} ({all_away_cs_pred*100:.1f}%)")

# is_home feature importance rank
fi_df = csm.feature_importance()
home_imp = fi_df[fi_df['feature'] == 'is_home']
total_imp = fi_df['importance'].sum()
if len(home_imp) > 0:
    home_rank = fi_df.index[fi_df['feature'] == 'is_home'].tolist()
    pct = home_imp['importance'].iloc[0] / total_imp * 100
    print(f"    is_home importance rank: {fi_df[fi_df['feature']=='is_home'].index[0]+1}/{len(fi_df)} ({pct:.1f}% of total)")

# ─── Team identity gap analysis ───────────────────────────────────────────────
print("\n[13] Team identity gap — season-long CS rates vs model average prediction...")
season_cs = (team_df2_with_norm
             .groupby(['team_norm', 'season'])
             .agg(actual_cs=('clean_sheet', 'mean'),
                  n_games=('clean_sheet', 'count'))
             .reset_index())
# Merge in model preds by re-predicting per team-season
all_preds_lambda = csm.predict_goals_against(team_df2_with_norm.dropna(subset=['goals_conceded']))
all_preds_cs = csm.predict_cs_prob(team_df2_with_norm.dropna(subset=['goals_conceded']))
temp = team_df2_with_norm.dropna(subset=['goals_conceded']).copy()
temp['pred_cs'] = all_preds_cs
team_season_pred = (temp.groupby(['team_norm', 'season'])
                    .agg(pred_cs_mean=('pred_cs', 'mean'))
                    .reset_index())
merged = season_cs.merge(team_season_pred, on=['team_norm', 'season'])
merged['bias'] = merged['actual_cs'] - merged['pred_cs_mean']
merged = merged[merged['season'] == current_season].sort_values('actual_cs', ascending=False)
print(f"\n    {current_season} -- Top 10 teams by actual CS rate:")
print(merged.head(10)[['team_norm', 'n_games', 'actual_cs', 'pred_cs_mean', 'bias']].to_string(index=False))

# ─── What lambda would give 50% CS? ───────────────────────────────────────────
print("\n[14] Target analysis: what lambda gives ~50% CS probability?")
target_cs = 0.50
# P(CS) = calibrated(e^{-lambda}) = 0.50
# For Poisson: P(0) = e^{-lambda} = 0.50 → lambda = -ln(0.50) ≈ 0.693
raw_lambda_for_50 = -np.log(0.50)
print(f"    Raw Poisson: e^(-lambda) = 0.50 -> lambda = {raw_lambda_for_50:.3f}")
print(f"    Arsenal's predicted lambda: {arsenal_latest['pred_lambda'].iloc[0]:.3f}")
print(f"    Arsenal's raw CS prob:      {arsenal_latest['raw_cs_prob'].iloc[0]:.3f}")
print(f"    Arsenal's calibrated CS:    {arsenal_latest['cal_cs_prob'].iloc[0]:.3f}")

# Check calibrator mapping
if csm.cs_calibrator is not None:
    test_probs = np.linspace(0.2, 0.8, 13)
    calibrated = csm.cs_calibrator.predict(test_probs)
    print(f"\n    Calibrator mapping (raw_prob -> calibrated):")
    for rp, cp in zip(test_probs, calibrated):
        print(f"      {rp:.2f} -> {cp:.3f}")

print("\n" + "=" * 70)
print("INVESTIGATION COMPLETE")
print("=" * 70)
