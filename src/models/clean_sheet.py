"""Goals Against model (team-level) - predicts expected goals conceded per match.
Uses Poisson distribution to derive clean sheet probability and 2+ conceded probability.
"""
from ..features import ManagerFeatureMixin, chronological_frame, prior_stat
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import poisson
from sklearn.metrics import mean_absolute_error

from ..features import (TEAM_NAME_MAP, TEAM_ROLL_MIN_PERIODS, SEASON_STAT_WINDOW,
                        SEASON_STAT_MIN_GAMES, promoted_team_seasons)


class CleanSheetModel(ManagerFeatureMixin):
    """Predicts expected goals against per match, then derives:
    - P(clean sheet) = P(goals_against = 0) via Poisson
    - P(2+ conceded) = 1 - P(goals_against <= 1) via Poisson

    Uses raw Poisson probabilities (no calibration needed since
    the count:poisson objective directly predicts lambda).
    """

    FEATURES = [
        # Prior lambda (strong anchor: team xGA × blended opp goals/xG / league avg)
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

    # Blend promoted-club priors into the first five completed matches, then
    # rely entirely on the club's own Premier League evidence. This is separate
    # from TEAM_ROLL_MIN_PERIODS: rolling features can be numerically available
    # after three matches without yet being a stable team identity.
    PROMOTED_PRIOR_GAMES = 5
    GOALS_ATTACK_WEIGHT = 0.5
    PRIOR_LAMBDA_MIN = 0.3
    PRIOR_LAMBDA_MAX = 4.0

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
        # 'team' comes from player_stats, 'opponent' from fixtures, and the two
        # spell Bournemouth and Brighton differently.
        name = TEAM_NAME_MAP.get(str(name).strip(), name)
        return str(name).lower().replace(' ', '_').replace("'", "").strip()

    def prepare_team_features(self, df: pd.DataFrame,
                              prediction_fixtures: pd.DataFrame = None) -> pd.DataFrame:
        """Aggregate player data to team-match level and compute team features.

        ``prediction_fixtures`` optionally adds empty future team-match rows before
        rolling features are calculated. Because every rolling calculation shifts
        by one match, those rows provide the true pre-fixture feature state and
        include the most recently completed match without leaking future results.
        """
        df = chronological_frame(df)
        if 'match_id' not in df:
            pair = df.apply(lambda r: '|'.join(sorted([str(r['team']), str(r['opponent'])])), axis=1)
            df['match_id'] = df['season'].astype(str) + ':' + df['gameweek'].astype(str) + ':' + pair
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
        _mgr_emb_cols = [f'manager_emb_{i}' for i in range(8)] + [c for c in df if c.startswith('manager_raw_') or c == 'manager_prior_games']
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
        for c in ['match_date', 'forecast_time', 'result_time']:
            agg_kwargs[c] = (c, 'min')
        for c in _mgr_emb_cols:
            agg_kwargs[c] = (c, 'first')
        team_match = df.groupby(['team_norm', 'opponent_norm', 'season', 'gameweek', 'match_id', 'is_home']).agg(
            **agg_kwargs
        ).reset_index()
        team_match['_is_prediction'] = False

        if prediction_fixtures is not None and len(prediction_fixtures) > 0:
            required = ['team', 'opponent', 'season', 'gameweek', 'is_home']
            missing = [c for c in required if c not in prediction_fixtures.columns]
            if missing:
                raise ValueError(f"prediction_fixtures missing columns: {missing}")

            future = chronological_frame(prediction_fixtures)
            if 'match_id' not in future:
                pair = future.apply(lambda r: '|'.join(sorted([str(r['team']), str(r['opponent'])])), axis=1)
                future['match_id'] = future['season'].astype(str) + ':' + future['gameweek'].astype(str) + ':' + pair
            future['team_norm'] = future['team'].apply(self._normalize_name)
            future['opponent_norm'] = future['opponent'].apply(self._normalize_name)
            fixture_key = ['team_norm', 'opponent_norm', 'season', 'gameweek', 'match_id', 'is_home']
            future = future.drop_duplicates(subset=fixture_key, keep='first')

            # Do not duplicate a fixture if this method is deliberately used on a
            # gameweek that has already been added to the results data.
            existing_keys = pd.MultiIndex.from_frame(team_match[fixture_key])
            future_keys = pd.MultiIndex.from_frame(future[fixture_key])
            future = future.loc[~future_keys.isin(existing_keys)].copy()

            if len(future) > 0:
                future_team_match = future[fixture_key + ['team', 'opponent', 'match_date', 'forecast_time', 'result_time']].copy()
                for c in ['goals', 'xg', 'tackles', 'interceptions', 'clearances',
                          'blocks', 'saves', 'xgot_faced', 'accurate_passes',
                          'touches', 'key_passes', 'shots_on_target']:
                    future_team_match[c] = np.nan
                future_team_match['own_goals'] = 0.0
                for c in _mgr_emb_cols:
                    future_team_match[c] = pd.to_numeric(
                        future[c], errors='coerce'
                    ) if c in future.columns else 0.0
                future_team_match['_is_prediction'] = True
                team_match = pd.concat(
                    [team_match, future_team_match], ignore_index=True, sort=False
                )

        # Goals conceded = opponent's player goals + this team's own goals
        # Match on both sides of the fixture so DGWs cannot create a cartesian join.
        opp_goals = team_match[[
            'team_norm', 'opponent_norm', 'season', 'gameweek', 'match_id', 'goals', 'xg'
        ]].copy()
        opp_goals = opp_goals.rename(columns={
            'team_norm': 'opponent_norm',
            'opponent_norm': 'team_norm',
            'goals': 'opp_goals',
            'xg': 'xga'
        })
        team_match = team_match.merge(
            opp_goals,
            on=['team_norm', 'opponent_norm', 'season', 'gameweek', 'match_id'],
            how='left'
        )
        team_match['goals_conceded'] = team_match['opp_goals'] + team_match['own_goals']

        team_match['clean_sheet'] = (team_match['goals_conceded'] == 0).astype(int)
        team_match.loc[team_match['_is_prediction'], 'clean_sheet'] = np.nan
        team_match['def_actions'] = (team_match['tackles'] + team_match['interceptions']
                                     + team_match['clearances'] + team_match['blocks'])
        team_match['gk_workload'] = team_match['saves'] + team_match['goals_conceded'].fillna(0)
        team_match = team_match.sort_values(['team_norm', 'match_date'], kind='stable').reset_index(drop=True)

        # --- Rolling features ---
        _TEAM_WINDOWS = [1, 2, 3, 5, 7, 10, 30]
        _PLAYER_WINDOWS = [1, 2, 3, 5, 7, 10]

        def _roll(source_col, prefix, windows):
            for w in windows:
                team_match[f'{prefix}_roll{w}'] = prior_stat(team_match, 'team_norm', source_col, w, min_periods=min(TEAM_ROLL_MIN_PERIODS, w))

        # Team defensive
        _roll('goals_conceded', 'team_conceded', _TEAM_WINDOWS)
        _roll('xga', 'team_xga', _TEAM_WINDOWS)
        _roll('clean_sheet', 'team_cs', _TEAM_WINDOWS)

        # Team offensive (needed for opponent lookup at prediction time)
        _roll('goals', 'team_scored', _PLAYER_WINDOWS)
        _roll('xg', 'team_xg_scored', _PLAYER_WINDOWS)

        # EWMA for xGA (reacts faster to form changes than rolling mean)
        team_match['team_xga_ewm'] = prior_stat(team_match, 'team_norm', 'xga', span=5)

        # Defensive quality aggregates
        _roll('def_actions', 'team_def_actions', _PLAYER_WINDOWS)
        _roll('clearances', 'team_clearances', _PLAYER_WINDOWS)
        _roll('gk_workload', 'team_shots_faced', _PLAYER_WINDOWS)

        # Possession proxy
        _roll('accurate_passes', 'team_passes', _PLAYER_WINDOWS)
        _roll('touches', 'team_touches', _PLAYER_WINDOWS)

        # --- Team identity features: rolling ACROSS seasons, not season-to-date ---
        # Carrying last season's form is the point: in August a season-to-date mean
        # is a single match. Promoted clubs are masked below and never reach here
        # with real values.
        _n_season = prior_stat(team_match.assign(_one=1), ['team_norm', 'season'], '_one', aggregation='sum').fillna(0)
        _use_std = (_n_season >= SEASON_STAT_MIN_GAMES).values

        def _identity(frame, col, n_season):
            """Season-to-date once the club has enough games this season, else a
            cross-season rolling window so last season's form carries in August."""
            std = prior_stat(frame, ['team_norm', 'season'], col)
            roll = prior_stat(frame, 'team_norm', col, SEASON_STAT_WINDOW, min_periods=TEAM_ROLL_MIN_PERIODS)
            return np.where((n_season >= SEASON_STAT_MIN_GAMES).values, std, roll)

        team_match['season_cs_rate'] = _identity(team_match, 'clean_sheet', _n_season)
        team_match['season_xga_per_game'] = _identity(team_match, 'xga', _n_season)

        # --- Opponent attacking stats (looked up via team_norm -> opponent_norm) ---
        opp_offense = team_match[['team_norm', 'opponent_norm', 'season', 'gameweek', 'match_id',
                                   'goals', 'xg', 'key_passes', 'shots_on_target', 'match_date', 'forecast_time', 'result_time']].copy()
        for col, pfx in [('goals', 'opp_scored'), ('xg', 'opp_xg'),
                          ('key_passes', 'opp_key_passes'), ('shots_on_target', 'opp_shots_ot')]:
            for w in _PLAYER_WINDOWS:
                opp_offense[f'{pfx}_roll{w}'] = prior_stat(opp_offense, 'team_norm', col, w, min_periods=min(TEAM_ROLL_MIN_PERIODS, w))

        # Opponent scoring identity, rolling across seasons. This is the term that
        # zeroes prior_lambda when a club blanks, so it must not be a one-match mean.
        _opp_n_season = prior_stat(opp_offense.assign(_one=1), ['team_norm', 'season'], '_one', aggregation='sum').fillna(0)
        opp_offense['opp_season_goals_per_game'] = _identity(opp_offense, 'goals', _opp_n_season)
        opp_offense['opp_season_xg_per_game'] = _identity(opp_offense, 'xg', _opp_n_season)
        # Promoted-team blending needs genuine current-season evidence from the
        # first match; the general identity feature deliberately waits longer.
        for _source in ('goals', 'xg'):
            opp_offense[f'_opp_current_season_{_source}'] = prior_stat(opp_offense, ['team_norm', 'season'], _source)

        opp_roll_cols = [c for c in opp_offense.columns
                         if any(c.startswith(p) for p in ['opp_scored_roll', 'opp_xg_roll',
                                                          'opp_key_passes_roll', 'opp_shots_ot_roll',
                                                          'opp_season_'])]
        opp_current_cols = [
            '_opp_current_season_goals', '_opp_current_season_xg'
        ]
        opp_lookup = opp_offense[[
            'team_norm', 'opponent_norm', 'season', 'gameweek', 'match_id'
        ] + opp_roll_cols + opp_current_cols].copy()
        opp_lookup = opp_lookup.rename(columns={
            'team_norm': 'opponent_norm', 'opponent_norm': 'team_norm'
        })
        team_match = team_match.merge(
            opp_lookup,
            on=['team_norm', 'opponent_norm', 'season', 'gameweek', 'match_id'],
            how='left'
        )

        # --- League-average fallbacks for unknown teams ---
        # Aggregates are NaN until a club has TEAM_ROLL_MIN_PERIODS games (and for
        # newly promoted clubs with no PL history at all). _prepare_X ends in a
        # blanket fillna(0), which would read as "concedes 0.0 per game" -- better
        # than any real defence. Anchor the unknowns to league average instead.
        _observed = ~team_match['_is_prediction'].fillna(False).astype(bool)
        # Fixed cold-start defaults contain no future league outcomes.
        _lg_conceded, _lg_cs, _lg_xg = 1.3, 0.25, 1.3

        # Newly promoted clubs need a real prior while their current-season sample
        # is still small. Blend continuously over five completed matches instead
        # of switching abruptly as soon as rolling features become available.
        # Priors use early-season promoted clubs from PRIOR seasons, so every
        # training row remains leakage-safe.
        _promoted = promoted_team_seasons(team_match)
        team_match['_prior_games'] = prior_stat(team_match.assign(_one=1), ['team_norm', 'season'], '_one', aggregation='sum').fillna(0)
        team_match['_promoted_prior_weight'] = pd.Series([
            max(0.0, 1.0 - float(g) / self.PROMOTED_PRIOR_GAMES)
            if (t, sn) in _promoted else 0.0
            for t, sn, g in zip(team_match['team_norm'], team_match['season'],
                                team_match['_prior_games'])
        ], index=team_match.index)
        _opp_prior_lookup = team_match[[
            'team_norm', 'opponent_norm', 'season', 'gameweek', 'match_id',
            '_promoted_prior_weight'
        ]].rename(columns={
            'team_norm': 'opponent_norm', 'opponent_norm': 'team_norm',
            '_promoted_prior_weight': '_opp_promoted_prior_weight'
        })
        team_match = team_match.merge(
            _opp_prior_lookup,
            on=['team_norm', 'opponent_norm', 'season', 'gameweek', 'match_id'],
            how='left'
        )
        _prior_weight = team_match['_promoted_prior_weight'].fillna(0.0).clip(0.0, 1.0)
        _opp_prior_weight = team_match['_opp_promoted_prior_weight'].fillna(0.0).clip(0.0, 1.0)
        _cold = _prior_weight.gt(0.0)
        _opp_cold = _opp_prior_weight.gt(0.0)

        _source_defaults = {
            'goals_conceded': 1.3, 'xga': 1.3, 'clean_sheet': 0.25,
            'goals': 1.3, 'xg': 1.3, 'def_actions': 50.0,
            'clearances': 20.0, 'gk_workload': 5.0,
            'accurate_passes': 350.0, 'touches': 600.0,
            'key_passes': 9.0, 'shots_on_target': 4.0,
        }
        _gw_num = pd.to_numeric(team_match['gameweek'], errors='coerce')
        _is_promoted = pd.Series(
            list(zip(team_match['team_norm'], team_match['season'])),
            index=team_match.index
        ).isin(_promoted)
        _cohort_base = _observed & _is_promoted & (_gw_num <= 10)
        _prior_cache = {}

        def _promoted_prior(source_col):
            """Return row-aligned priors based only on earlier seasons."""
            cache_key = source_col
            if cache_key in _prior_cache:
                return _prior_cache[cache_key]
            result = pd.Series(index=team_match.index, dtype=float)
            for _season in sorted(team_match['season'].dropna().unique()):
                hist = _cohort_base & (team_match['season'] < _season)
                values = pd.to_numeric(
                    team_match.loc[hist, source_col], errors='coerce'
                ).dropna()
                if values.empty:
                    hist = _observed & (team_match['season'] < _season)
                    values = pd.to_numeric(
                        team_match.loc[hist, source_col], errors='coerce'
                    ).dropna()
                value = (values.mean() if not values.empty
                         else _source_defaults[source_col])
                result.loc[team_match['season'].eq(_season)] = value
            result = result.fillna(_source_defaults[source_col])
            _prior_cache[cache_key] = result
            return result

        def _blend_promoted(column, source_col, weight, current=None):
            """Blend observed values with a leakage-safe promoted-team prior."""
            prior = _promoted_prior(source_col)
            observed = pd.to_numeric(
                team_match[column] if current is None else current,
                errors='coerce'
            )
            # If a wider rolling feature is not available yet, the promoted prior
            # remains the best estimate regardless of the nominal blend weight.
            observed = observed.fillna(prior)
            blended = weight * prior + (1.0 - weight) * observed
            mask = weight.gt(0.0)
            team_match.loc[mask, column] = blended.loc[mask]

        _team_sources = {
            'team_conceded_roll': 'goals_conceded',
            'team_xga_roll': 'xga',
            'team_cs_roll': 'clean_sheet',
            'team_scored_roll': 'goals',
            'team_xg_scored_roll': 'xg',
            'team_def_actions_roll': 'def_actions',
            'team_clearances_roll': 'clearances',
            'team_shots_faced_roll': 'gk_workload',
            'team_passes_roll': 'accurate_passes',
            'team_touches_roll': 'touches',
        }
        _opp_sources = {
            'opp_scored_roll': 'goals',
            'opp_xg_roll': 'xg',
            'opp_key_passes_roll': 'key_passes',
            'opp_shots_ot_roll': 'shots_on_target',
        }

        for _prefix, _source in _team_sources.items():
            for _c in [c for c in team_match.columns if c.startswith(_prefix)]:
                _blend_promoted(_c, _source, _prior_weight)
                team_match[_c] = team_match[_c].fillna(_source_defaults[_source])
        for _prefix, _source in _opp_sources.items():
            for _c in [c for c in team_match.columns if c.startswith(_prefix)]:
                _blend_promoted(_c, _source, _opp_prior_weight)
                team_match[_c] = team_match[_c].fillna(_source_defaults[_source])

        _blend_promoted('team_xga_ewm', 'xga', _prior_weight)
        team_match['team_xga_ewm'] = team_match['team_xga_ewm'].fillna(_lg_xg)
        _current_team_xga = prior_stat(team_match, ['team_norm', 'season'], 'xga')
        _current_team_cs = prior_stat(team_match, ['team_norm', 'season'], 'clean_sheet')
        _blend_promoted(
            'season_xga_per_game', 'xga', _prior_weight, _current_team_xga
        )
        _blend_promoted(
            'season_cs_rate', 'clean_sheet', _prior_weight, _current_team_cs
        )
        _blend_promoted(
            'opp_season_goals_per_game', 'goals', _opp_prior_weight,
            team_match['_opp_current_season_goals']
        )
        _blend_promoted(
            'opp_season_xg_per_game', 'xg', _opp_prior_weight,
            team_match['_opp_current_season_xg']
        )

        # Keep the flags and weights: venue splits below use the same transition.
        team_match['_is_cold'] = _cold.values
        team_match['_is_opp_cold'] = _opp_cold.values
        team_match = team_match.drop(
            columns=['_prior_games'], errors='ignore')
        # Season-to-date is empty for the first TEAM_ROLL_MIN_PERIODS games of every
        # season, so cascade to the club's CROSS-SEASON rolling value before the
        # league mean -- otherwise Arsenal enters every August indistinguishable
        # from a promoted side. For a genuinely new club the rolling value is
        # itself the prior, so it still lands at league average.
        team_match['season_xga_per_game'] = (team_match['season_xga_per_game']
                                             .fillna(team_match['team_xga_roll5'])
                                             .fillna(_lg_conceded))
        team_match['season_cs_rate'] = (team_match['season_cs_rate']
                                        .fillna(team_match['team_cs_roll5'])
                                        .fillna(_lg_cs))
        team_match['opp_season_goals_per_game'] = team_match['opp_season_goals_per_game'].fillna(_lg_conceded)
        team_match['opp_season_xg_per_game'] = team_match['opp_season_xg_per_game'].fillna(_lg_xg)

        team_match = team_match.copy()  # consolidate before adding interaction columns
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
            for tn, grp in team_match[venue_mask].groupby('team_norm'):
                idx = grp.index
                _vn = prior_stat(grp.assign(_one=1), 'season', '_one', aggregation='sum').fillna(0)
                for _sfx, _src in (('xga', 'xga'), ('cs_rate', 'clean_sheet')):
                    _r = prior_stat(grp, 'team_norm', _src, SEASON_STAT_WINDOW, min_periods=TEAM_ROLL_MIN_PERIODS)
                    _sd = prior_stat(grp, 'season', _src)
                    team_match.loc[idx, f'{venue}_season_{_sfx}'] = np.where(
                        (_vn >= SEASON_STAT_MIN_GAMES).values, _sd, _r)

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
        # During the five-match ramp, use the already blended overall identity.
        # Venue splits are too sparse this early (often only one or two matches).
        _cold_rows = team_match['_is_cold'].fillna(False).astype(bool)
        team_match.loc[_cold_rows, 'ha_season_xga'] = team_match.loc[
            _cold_rows, 'season_xga_per_game']
        team_match.loc[_cold_rows, 'ha_season_cs_rate'] = team_match.loc[
            _cold_rows, 'season_cs_rate']

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
            for tn, grp in team_match[venue_mask].groupby('team_norm'):
                idx = grp.index
                _vn = prior_stat(grp.assign(_one=1), 'season', '_one', aggregation='sum').fillna(0)
                for _sfx, _src in (('goals', 'goals'), ('xg', 'xg')):
                    _r = prior_stat(grp, 'team_norm', _src, SEASON_STAT_WINDOW, min_periods=TEAM_ROLL_MIN_PERIODS)
                    _sd = prior_stat(grp, 'season', _src)
                    team_match.loc[idx, f'{venue}_season_{_sfx}'] = np.where(
                        (_vn >= SEASON_STAT_MIN_GAMES).values, _sd, _r)

        # When we're home, use the opponent's away scoring rate and vice versa.
        opp_ha_lookup2 = team_match[['team_norm', 'opponent_norm', 'season', 'gameweek', 'match_id',
                                      'away_season_goals', 'away_season_xg',
                                      'home_season_goals', 'home_season_xg']].copy()
        opp_ha_lookup2 = opp_ha_lookup2.rename(columns={
            'team_norm': 'opponent_norm', 'opponent_norm': 'team_norm'
        })
        # When our is_home=1, opponent is away, so use opponent's away_season_goals
        # When our is_home=0, opponent is home, so use opponent's home_season_goals
        team_match = team_match.merge(
            opp_ha_lookup2[['team_norm', 'opponent_norm', 'season', 'gameweek', 'match_id',
                            'away_season_goals', 'away_season_xg',
                            'home_season_goals', 'home_season_xg']],
            on=['team_norm', 'opponent_norm', 'season', 'gameweek', 'match_id'], how='left',
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
        _opp_cold_rows = team_match['_is_opp_cold'].fillna(False).astype(bool)
        team_match.loc[_opp_cold_rows, 'opp_ha_season_goals'] = team_match.loc[
            _opp_cold_rows, 'opp_season_goals_per_game']
        team_match.loc[_opp_cold_rows, 'opp_ha_season_xg'] = team_match.loc[
            _opp_cold_rows, 'opp_season_xg_per_game']
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
        # Estimate: team xGA times the opponent's blended goals/xG attack rate,
        # normalized by the league-average goals rate.
        league_avg_goals = 1.3  # fixed normalization; learned priors use prior seasons only
        if league_avg_goals < 0.5:
            league_avg_goals = 1.3  # safety fallback

        team_xga = team_match['ha_season_xga'].fillna(team_match['season_xga_per_game']).fillna(league_avg_goals)
        opp_goals = team_match['opp_ha_season_goals'].fillna(
            team_match['opp_season_goals_per_game']).fillna(league_avg_goals)
        opp_xg = team_match['opp_ha_season_xg'].fillna(
            team_match['opp_season_xg_per_game']).fillna(league_avg_goals)
        # Goals and xG contribute equally, so a short finishing drought cannot
        # imply a zero-strength attack while chance quality remains non-zero.
        opp_attack = (self.GOALS_ATTACK_WEIGHT * opp_goals
                      + (1.0 - self.GOALS_ATTACK_WEIGHT) * opp_xg)

        raw_prior_lambda = team_xga * (opp_attack / league_avg_goals)
        team_match['prior_lambda'] = raw_prior_lambda.clip(
            self.PRIOR_LAMBDA_MIN, self.PRIOR_LAMBDA_MAX)

        # Naive CS probability: direct Poisson anchor from prior_lambda
        team_match['naive_cs_prob'] = np.exp(-team_match['prior_lambda'])

        # --- Home advantage interactions ---
        team_match['home_x_team_xga_roll10'] = (team_match['is_home'].fillna(0)
                                                  * team_match['team_xga_roll10'].fillna(1.0))
        team_match['home_x_season_cs_rate'] = (team_match['is_home'].fillna(0)
                                                 * team_match['season_cs_rate'].fillna(0.25))

        # Drop temp columns
        team_match = team_match.drop(
            columns=['team_norm', 'opponent_norm', '_is_cold', '_is_opp_cold',
                     '_promoted_prior_weight', '_opp_promoted_prior_weight',
                     '_opp_current_season_goals', '_opp_current_season_xg'],
            errors='ignore'
        )

        return team_match

    def _prepare_X(self, df: pd.DataFrame):
        df = self.manager_features(df)
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
        prior = np.clip(
            prior,
            CleanSheetModel.PRIOR_LAMBDA_MIN,
            CleanSheetModel.PRIOR_LAMBDA_MAX,
        )
        # Replace NaN with log of league average (~1.38)
        prior = np.where(np.isnan(prior), 1.38, prior)
        return np.log(prior)

    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train on team-match data to predict goals conceded."""
        return self.fit_prepared(self.prepare_team_features(df), verbose)

    def fit_prepared(self, team_df: pd.DataFrame, verbose: bool = True):
        """Shared fitting path for final training, temporal CV, and evaluation."""
        team_df = team_df.dropna(subset=['goals_conceded']).copy()
        self.fit_manager_features(team_df)
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
