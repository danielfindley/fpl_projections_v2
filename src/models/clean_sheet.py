"""Goals Against model (team-level) - predicts expected goals conceded per match.
Uses Poisson distribution to derive clean sheet probability and 2+ conceded probability.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import poisson
from sklearn.metrics import mean_absolute_error


class CleanSheetModel:
    """Predicts expected goals against per match, then derives:
    - P(clean sheet) = P(goals_against = 0) via Poisson
    - P(2+ conceded) = 1 - P(goals_against <= 1) via Poisson

    Uses raw Poisson probabilities (no calibration needed since
    the count:poisson objective directly predicts lambda).
    """

    FEATURES = [
        # Prior lambda (strong anchor: team xGA × opp goal rate / league avg)
        'prior_lambda',
        # Naive CS probability = exp(-prior_lambda) — direct anchor for the model
        'naive_cs_prob',

        # Interaction features
        'xga_x_opp_xg_roll1', 'xga_x_opp_xg_roll2', 'xga_x_opp_xg_roll3',
        'xga_x_opp_xg_roll5', 'xga_x_opp_xg_roll7', 'xga_x_opp_xg_roll10',
        'def_actions_x_opp_shots_roll1', 'def_actions_x_opp_shots_roll2', 'def_actions_x_opp_shots_roll3',
        'def_actions_x_opp_shots_roll5', 'def_actions_x_opp_shots_roll7', 'def_actions_x_opp_shots_roll10',

        # Ratio features (team defensive strength relative to opponent)
        'xga_div_opp_xg_roll1', 'xga_div_opp_xg_roll2', 'xga_div_opp_xg_roll3',
        'xga_div_opp_xg_roll5', 'xga_div_opp_xg_roll7', 'xga_div_opp_xg_roll10',

        # Team defensive history
        'team_xga_roll1', 'team_xga_roll2', 'team_xga_roll3', 'team_xga_roll5', 'team_xga_roll7', 'team_xga_roll10', 'team_xga_roll30',
        'team_conceded_roll1', 'team_conceded_roll2', 'team_conceded_roll3', 'team_conceded_roll5', 'team_conceded_roll7', 'team_conceded_roll10', 'team_conceded_roll30',
        'team_cs_roll1', 'team_cs_roll2', 'team_cs_roll3', 'team_cs_roll5', 'team_cs_roll7', 'team_cs_roll10', 'team_cs_roll30',
        'team_xga_ewm',

        # Team identity proxy (season-to-date defensive level)
        'season_cs_rate', 'season_xga_per_game',

        # Home/away split season stats (stronger team identity)
        'ha_season_xga', 'ha_season_cs_rate',

        # Team defensive quality aggregates
        'team_def_actions_roll1', 'team_def_actions_roll2', 'team_def_actions_roll3', 'team_def_actions_roll5', 'team_def_actions_roll7', 'team_def_actions_roll10',
        'team_clearances_roll1', 'team_clearances_roll2', 'team_clearances_roll3', 'team_clearances_roll5', 'team_clearances_roll7', 'team_clearances_roll10',
        'team_shots_faced_roll1', 'team_shots_faced_roll2', 'team_shots_faced_roll3', 'team_shots_faced_roll5', 'team_shots_faced_roll7', 'team_shots_faced_roll10',

        # Team possession proxy
        'team_passes_roll1', 'team_passes_roll2', 'team_passes_roll3', 'team_passes_roll5', 'team_passes_roll7', 'team_passes_roll10',
        'team_touches_roll1', 'team_touches_roll2', 'team_touches_roll3', 'team_touches_roll5', 'team_touches_roll7', 'team_touches_roll10',

        # Opponent attacking quality
        'opp_xg_roll1', 'opp_xg_roll2', 'opp_xg_roll3', 'opp_xg_roll5', 'opp_xg_roll7', 'opp_xg_roll10',
        'opp_key_passes_roll1', 'opp_key_passes_roll2', 'opp_key_passes_roll3', 'opp_key_passes_roll5', 'opp_key_passes_roll7', 'opp_key_passes_roll10',
        'opp_shots_ot_roll1', 'opp_shots_ot_roll2', 'opp_shots_ot_roll3', 'opp_shots_ot_roll5', 'opp_shots_ot_roll7', 'opp_shots_ot_roll10',

        # Opponent identity proxy (season-to-date attacking level)
        'opp_season_goals_per_game', 'opp_season_xg_per_game',

        # Opponent home/away split (away teams score differently)
        'opp_ha_season_goals', 'opp_ha_season_xg',

        # Home advantage interaction
        'home_x_team_xga_roll10',
        'home_x_season_cs_rate',

        # Match context
        'is_home',

        # Manager embeddings (8-dim PCA over rolling-20-prior manager stats)
        'manager_emb_0', 'manager_emb_1', 'manager_emb_2', 'manager_emb_3',
        'manager_emb_4', 'manager_emb_5', 'manager_emb_6', 'manager_emb_7',
    ]

    TARGET = 'goals_conceded'

    def __init__(self, **xgb_params):
        # Extract selected_features if provided (from tuning)
        self.selected_features = xgb_params.pop('selected_features', None)

        default_params = {
            'n_estimators': 300,
            'max_depth': 3,
            'learning_rate': 0.02,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 2.0,
            'reg_lambda': 5.0,
            'random_state': 42,
            'objective': 'count:poisson',
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBRegressor(**default_params)
        self.is_fitted = False

    @property
    def features_to_use(self):
        """Return selected features if available, otherwise all FEATURES."""
        return self.selected_features if self.selected_features else self.FEATURES

    @staticmethod
    def _normalize_name(name):
        """Normalize team name for consistent matching."""
        if pd.isna(name):
            return ''
        return str(name).lower().replace(' ', '_').replace("'", "").strip()

    def prepare_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate player data to team-match level and compute team features."""
        df = df.copy()
        df['team_norm'] = df['team'].apply(self._normalize_name)
        df['opponent_norm'] = df['opponent'].apply(self._normalize_name)

        # Ensure numeric for aggregate columns
        for col in ['tackles', 'interceptions', 'clearances', 'blocks',
                     'saves', 'xgot_faced', 'accurate_passes', 'touches',
                     'key_passes', 'shots_on_target']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0

        # Ensure own_goal column exists
        if 'own_goal' not in df.columns:
            df['own_goal'] = 0

        # Manager embedding columns: constant per team-match, take first
        _mgr_emb_cols = [f'manager_emb_{i}' for i in range(8)]
        for c in _mgr_emb_cols:
            if c not in df.columns:
                df[c] = 0.0

        # Aggregate to team-match level
        agg_kwargs = dict(
            goals=('goals', 'sum'),
            own_goals=('own_goal', 'sum'),
            xg=('xg', 'sum'),
            team=('team', 'first'),
            opponent=('opponent', 'first'),
            tackles=('tackles', 'sum'),
            interceptions=('interceptions', 'sum'),
            clearances=('clearances', 'sum'),
            blocks=('blocks', 'sum'),
            saves=('saves', 'sum'),
            xgot_faced=('xgot_faced', 'sum'),
            accurate_passes=('accurate_passes', 'sum'),
            touches=('touches', 'sum'),
            key_passes=('key_passes', 'sum'),
            shots_on_target=('shots_on_target', 'sum'),
        )
        for c in _mgr_emb_cols:
            agg_kwargs[c] = (c, 'first')
        team_match = df.groupby(['team_norm', 'opponent_norm', 'season', 'gameweek', 'is_home']).agg(
            **agg_kwargs
        ).reset_index()

        # Goals conceded = opponent's player goals + this team's own goals
        opp_goals = team_match[['team_norm', 'season', 'gameweek', 'goals', 'xg']].copy()
        opp_goals = opp_goals.rename(columns={
            'team_norm': 'opponent_norm',
            'goals': 'opp_goals',
            'xg': 'xga'
        })
        team_match = team_match.merge(opp_goals, on=['opponent_norm', 'season', 'gameweek'], how='left')
        team_match['goals_conceded'] = team_match['opp_goals'] + team_match['own_goals']

        team_match['clean_sheet'] = (team_match['goals_conceded'] == 0).astype(int)
        team_match['def_actions'] = (team_match['tackles'] + team_match['interceptions']
                                     + team_match['clearances'] + team_match['blocks'])
        team_match['gk_workload'] = team_match['saves'] + team_match['goals_conceded'].fillna(0)
        team_match = team_match.sort_values(['team_norm', 'season', 'gameweek'])

        # --- Rolling features ---
        _TEAM_WINDOWS = [1, 2, 3, 5, 7, 10, 30]
        _PLAYER_WINDOWS = [1, 2, 3, 5, 7, 10]

        def _roll(source_col, prefix, windows):
            for w in windows:
                team_match[f'{prefix}_roll{w}'] = team_match.groupby('team_norm')[source_col].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                )

        # Team defensive
        _roll('goals_conceded', 'team_conceded', _TEAM_WINDOWS)
        _roll('xga', 'team_xga', _TEAM_WINDOWS)
        _roll('clean_sheet', 'team_cs', _TEAM_WINDOWS)

        # Team offensive (needed for opponent lookup at prediction time)
        _roll('goals', 'team_scored', _PLAYER_WINDOWS)
        _roll('xg', 'team_xg_scored', _PLAYER_WINDOWS)

        # EWMA for xGA (reacts faster to form changes than rolling mean)
        team_match['team_xga_ewm'] = team_match.groupby('team_norm')['xga'].transform(
            lambda x: x.shift(1).ewm(span=5, min_periods=1).mean()
        )

        # Defensive quality aggregates
        _roll('def_actions', 'team_def_actions', _PLAYER_WINDOWS)
        _roll('clearances', 'team_clearances', _PLAYER_WINDOWS)
        _roll('gk_workload', 'team_shots_faced', _PLAYER_WINDOWS)

        # Possession proxy
        _roll('accurate_passes', 'team_passes', _PLAYER_WINDOWS)
        _roll('touches', 'team_touches', _PLAYER_WINDOWS)

        # --- Season-to-date team identity features (shifted to avoid leakage) ---
        team_match['season_cs_rate'] = team_match.groupby(['team_norm', 'season'])['clean_sheet'].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
        team_match['season_xga_per_game'] = team_match.groupby(['team_norm', 'season'])['xga'].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

        # --- Opponent attacking stats (looked up via team_norm -> opponent_norm) ---
        opp_offense = team_match[['team_norm', 'season', 'gameweek',
                                   'goals', 'xg', 'key_passes', 'shots_on_target']].copy()
        for col, pfx in [('goals', 'opp_scored'), ('xg', 'opp_xg'),
                          ('key_passes', 'opp_key_passes'), ('shots_on_target', 'opp_shots_ot')]:
            for w in _PLAYER_WINDOWS:
                opp_offense[f'{pfx}_roll{w}'] = opp_offense.groupby('team_norm')[col].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                )

        # Opponent season-level identity (how good/bad they are this season)
        opp_offense['opp_season_goals_per_game'] = opp_offense.groupby(['team_norm', 'season'])['goals'].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
        opp_offense['opp_season_xg_per_game'] = opp_offense.groupby(['team_norm', 'season'])['xg'].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

        opp_roll_cols = [c for c in opp_offense.columns
                         if any(c.startswith(p) for p in ['opp_scored_roll', 'opp_xg_roll',
                                                          'opp_key_passes_roll', 'opp_shots_ot_roll',
                                                          'opp_season_'])]
        opp_lookup = opp_offense[['team_norm', 'season', 'gameweek'] + opp_roll_cols].copy()
        opp_lookup = opp_lookup.rename(columns={'team_norm': 'opponent_norm'})
        team_match = team_match.merge(opp_lookup, on=['opponent_norm', 'season', 'gameweek'], how='left')

        # --- Interaction and ratio features (computed for all standard windows) ---
        for w in _PLAYER_WINDOWS:
            team_match[f'xga_x_opp_xg_roll{w}'] = (team_match[f'team_xga_roll{w}'].fillna(0)
                                                     * team_match[f'opp_xg_roll{w}'].fillna(0))
            team_match[f'def_actions_x_opp_shots_roll{w}'] = (team_match[f'team_def_actions_roll{w}'].fillna(0)
                                                               * team_match[f'opp_shots_ot_roll{w}'].fillna(0))
            team_match[f'xga_div_opp_xg_roll{w}'] = (team_match[f'team_xga_roll{w}'].fillna(1.0)
                                                       / (team_match[f'opp_xg_roll{w}'].fillna(1.0) + 0.01))

        # --- Home/away split season stats ---
        # Arsenal's home xGA is much lower than overall; this captures venue-specific identity
        for venue, venue_val in [('home', 1), ('away', 0)]:
            venue_mask = team_match['is_home'] == venue_val
            # Season xGA for this venue only
            team_match[f'{venue}_season_xga'] = np.nan
            team_match[f'{venue}_season_cs_rate'] = np.nan
            for (tn, s), grp in team_match[venue_mask].groupby(['team_norm', 'season']):
                idx = grp.index
                team_match.loc[idx, f'{venue}_season_xga'] = grp['xga'].shift(1).expanding(min_periods=1).mean()
                team_match.loc[idx, f'{venue}_season_cs_rate'] = grp['clean_sheet'].shift(1).expanding(min_periods=1).mean()

        # ha_season_xga/cs_rate = the relevant venue's season stat (home stat for home games, away for away)
        team_match['ha_season_xga'] = np.where(
            team_match['is_home'] == 1,
            team_match['home_season_xga'],
            team_match['away_season_xga']
        )
        team_match['ha_season_cs_rate'] = np.where(
            team_match['is_home'] == 1,
            team_match['home_season_cs_rate'],
            team_match['away_season_cs_rate']
        )
        # Fill NaN (first game of season at venue) with overall season stat
        team_match['ha_season_xga'] = team_match['ha_season_xga'].fillna(team_match['season_xga_per_game'])
        team_match['ha_season_cs_rate'] = team_match['ha_season_cs_rate'].fillna(team_match['season_cs_rate'])
        # Forward-fill venue-specific stats so latest row has both home and away values
        # (otherwise only the venue of the last game has a value)
        for col in ['home_season_xga', 'away_season_xga', 'home_season_cs_rate', 'away_season_cs_rate']:
            team_match[col] = team_match.groupby('team_norm')[col].ffill()

        # --- Opponent home/away split season stats ---
        # How does the opponent perform when they are at the OTHER venue (away goals for our home games)
        for venue, venue_val in [('home', 1), ('away', 0)]:
            venue_mask = team_match['is_home'] == venue_val
            team_match[f'{venue}_season_goals'] = np.nan
            team_match[f'{venue}_season_xg'] = np.nan
            for (tn, s), grp in team_match[venue_mask].groupby(['team_norm', 'season']):
                idx = grp.index
                team_match.loc[idx, f'{venue}_season_goals'] = grp['goals'].shift(1).expanding(min_periods=1).mean()
                team_match.loc[idx, f'{venue}_season_xg'] = grp['xg'].shift(1).expanding(min_periods=1).mean()

        # opp_ha = opponent's away stats when we're home, opponent's home stats when we're away
        opp_ha_lookup = team_match[['team_norm', 'season', 'gameweek', 'is_home',
                                     'home_season_goals', 'away_season_goals',
                                     'home_season_xg', 'away_season_xg']].copy()
        # Flip: opponent's away stats go to our home games
        opp_ha_lookup['opp_ha_season_goals'] = np.where(
            opp_ha_lookup['is_home'] == 1,
            opp_ha_lookup['home_season_goals'],  # opponent is home here, but we want opponent's away
            opp_ha_lookup['away_season_goals']    # opponent is away here, but we want opponent's home
        )
        # Actually, we need to look this up from the opponent's perspective
        # When we're home, opponent is away -> we want opponent's AWAY scoring rate
        # The data is already per-team, so we need to flip
        opp_ha_lookup2 = team_match[['team_norm', 'season', 'gameweek',
                                      'away_season_goals', 'away_season_xg',
                                      'home_season_goals', 'home_season_xg']].copy()
        opp_ha_lookup2 = opp_ha_lookup2.rename(columns={'team_norm': 'opponent_norm'})
        # When our is_home=1, opponent is away, so use opponent's away_season_goals
        # When our is_home=0, opponent is home, so use opponent's home_season_goals
        team_match = team_match.merge(
            opp_ha_lookup2[['opponent_norm', 'season', 'gameweek',
                            'away_season_goals', 'away_season_xg',
                            'home_season_goals', 'home_season_xg']],
            on=['opponent_norm', 'season', 'gameweek'], how='left',
            suffixes=('', '_opp')
        )
        team_match['opp_ha_season_goals'] = np.where(
            team_match['is_home'] == 1,
            team_match['away_season_goals_opp'],   # opponent's away scoring
            team_match['home_season_goals_opp']    # opponent's home scoring
        )
        team_match['opp_ha_season_xg'] = np.where(
            team_match['is_home'] == 1,
            team_match['away_season_xg_opp'],
            team_match['home_season_xg_opp']
        )
        team_match['opp_ha_season_goals'] = team_match['opp_ha_season_goals'].fillna(
            team_match['opp_season_goals_per_game'])
        team_match['opp_ha_season_xg'] = team_match['opp_ha_season_xg'].fillna(
            team_match['opp_season_xg_per_game'])

        # Drop intermediate columns
        team_match = team_match.drop(columns=[
            'home_season_goals', 'away_season_goals', 'home_season_xg', 'away_season_xg',
            'away_season_goals_opp', 'home_season_goals_opp',
            'away_season_xg_opp', 'home_season_xg_opp',
        ], errors='ignore')

        # --- Prior lambda (strong direct anchor) ---
        # Estimate: team_season_xga × (opp_season_goals / league_avg_goals) × home_factor
        league_avg_goals = team_match['goals_conceded'].mean()  # ~1.38
        if league_avg_goals < 0.5:
            league_avg_goals = 1.3  # safety fallback

        team_xga = team_match['ha_season_xga'].fillna(team_match['season_xga_per_game']).fillna(league_avg_goals)
        opp_goals = team_match['opp_ha_season_goals'].fillna(
            team_match['opp_season_goals_per_game']).fillna(league_avg_goals)

        team_match['prior_lambda'] = team_xga * (opp_goals / league_avg_goals)

        # Naive CS probability: direct Poisson anchor from prior_lambda
        team_match['naive_cs_prob'] = np.exp(-team_match['prior_lambda'])

        # --- Home advantage interactions ---
        team_match['home_x_team_xga_roll10'] = (team_match['is_home'].fillna(0)
                                                  * team_match['team_xga_roll10'].fillna(1.0))
        team_match['home_x_season_cs_rate'] = (team_match['is_home'].fillna(0)
                                                 * team_match['season_cs_rate'].fillna(0.25))

        # Drop temp columns
        team_match = team_match.drop(columns=['team_norm', 'opponent_norm'], errors='ignore')

        return team_match

    def _prepare_X(self, df: pd.DataFrame):
        df = df.copy()
        features = self.features_to_use
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0
        return df[features].fillna(0).astype(float)

    @staticmethod
    def _get_base_margin(df: pd.DataFrame) -> np.ndarray:
        """Compute base_margin = log(prior_lambda) for offset-based Poisson regression.

        This tells XGBoost: "start from the naive matchup estimate and learn corrections."
        Without this, the model regresses everything toward the global mean.
        """
        prior = df['prior_lambda'].values if 'prior_lambda' in df.columns else None
        if prior is None:
            return None
        prior = np.clip(prior, 0.3, 4.0)  # safety bounds
        # Replace NaN with log of league average (~1.38)
        prior = np.where(np.isnan(prior), 1.38, prior)
        return np.log(prior)

    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train on team-match data to predict goals conceded."""
        team_df = self.prepare_team_features(df)
        team_df = team_df.dropna(subset=['team_conceded_roll5', 'goals_conceded'])

        team_df = team_df.sort_values(['season', 'gameweek']).reset_index(drop=True)

        X = self._prepare_X(team_df)
        y = team_df['goals_conceded'].fillna(0).values
        base_margin = self._get_base_margin(team_df)

        if verbose:
            n_features = len(self.features_to_use)
            feature_info = f"({n_features} features"
            if self.selected_features:
                feature_info += ", tuned selection)"
            else:
                feature_info += ")"
            print(f"Training CleanSheetModel (Goals Against) on {len(X):,} team-matches {feature_info}...")
            print(f"  Avg goals conceded: {y.mean():.3f}")
            print(f"  Actual CS rate: {(y == 0).mean():.1%}")
            if base_margin is not None:
                naive_lambda = np.exp(base_margin)
                print(f"  Prior lambda range: {naive_lambda.min():.2f} - {naive_lambda.max():.2f}")

        X_vals = X.values if hasattr(X, 'values') else X
        if base_margin is not None:
            self.model.fit(X_vals, y, base_margin=base_margin)
        else:
            self.model.fit(X_vals, y)
        self.is_fitted = True

        if verbose:
            y_pred = self.predict_goals_against(team_df)
            cs_probs = self.predict_cs_prob(team_df)
            two_plus_probs = self.predict_2plus_conceded_prob(team_df)
            print(f"  MAE (goals against): {mean_absolute_error(y, y_pred):.3f}")
            print(f"  Predicted CS prob (mean): {cs_probs.mean():.1%}")
            print(f"  Predicted 2+ conceded prob (mean): {two_plus_probs.mean():.1%}")
            print(f"  Predicted CS prob range: {cs_probs.min():.1%} - {cs_probs.max():.1%}")

        return self

    # Weight for blending prior_lambda with model prediction (0=pure model, 1=pure prior)
    PRIOR_WEIGHT = 0.5

    def predict_goals_against(self, df: pd.DataFrame) -> np.ndarray:
        """Predict expected goals against (lambda for Poisson).

        Uses base_margin = log(prior_lambda) so model predictions are
        adjustments on top of the naive matchup estimate, then blends
        the result with the prior to prevent over-correction.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X = self._prepare_X(df)
        base_margin = self._get_base_margin(df)

        X_vals = X.values if hasattr(X, 'values') else X
        if base_margin is not None:
            dmat = xgb.DMatrix(X_vals, feature_names=list(X.columns) if hasattr(X, 'columns') else None)
            dmat.set_base_margin(base_margin)
            model_pred = self.model.get_booster().predict(dmat)

            # Blend in log-space: geometric mean of prior and model
            prior_lambda = np.exp(base_margin)
            w = self.PRIOR_WEIGHT
            log_blended = w * np.log(prior_lambda) + (1 - w) * np.log(np.clip(model_pred, 1e-6, 10.0))
            raw_pred = np.exp(log_blended)
        else:
            raw_pred = self.model.predict(X_vals)

        return np.clip(raw_pred, 1e-6, 10.0)

    def predict_cs_prob(self, df: pd.DataFrame) -> np.ndarray:
        """Predict clean sheet probability: P(goals_against = 0) = e^(-lambda)."""
        lambda_pred = self.predict_goals_against(df)
        return poisson.pmf(0, lambda_pred)

    def predict_2plus_conceded_prob(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of conceding 2+ goals: 1 - P(0) - P(1)."""
        lambda_pred = self.predict_goals_against(df)
        return 1.0 - poisson.cdf(1, lambda_pred)

    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return pd.DataFrame({
            'feature': self.features_to_use,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
