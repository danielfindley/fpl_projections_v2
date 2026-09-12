"""
Monte Carlo Simulation-Based Bonus Points Model

Instead of predicting bonus directly, this model:
1. Predicts baseline BPS (raw score from "boring" stats)
2. Uses existing model predictions (goals, assists, CS probability)
3. Runs Monte Carlo simulation to determine bonus from BPS rankings

This is more accurate because bonus points are a ranking-based competition.
"""

from ..features import ManagerFeatureMixin
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import requests


# BPS scoring rules (2025-26 season)
BPS_RULES = {
    # Major events (simulated)
    'goal': {
        'GK': 12, 'DEF': 12, 'MID': 18, 'FWD': 24
    },
    'assist': 9,
    'clean_sheet': {
        'GK': 12, 'DEF': 12, 'MID': 0, 'FWD': 0
    },
    # Penalties
    'goal_conceded': {
        'GK': -4, 'DEF': -4, 'MID': 0, 'FWD': 0
    },
    'yellow_card': -3,
    'red_card': -9,
    'own_goal': -6,
    'penalty_miss': -6,
    'penalty_save': 15,
}

# Caps for per90 stats to prevent inflation from low-minutes appearances
# These represent realistic maximum values a player could achieve in 90 minutes
PER90_CAPS = {
    'tackles_per90': 8.0,
    'clearances_per90': 12.0,
    'interceptions_per90': 6.0,
    'recoveries_per90': 15.0,
    'blocks_per90': 6.0,
    'key_passes_per90': 6.0,
    'shots_per90': 8.0,
    'goals_per90': 3.0,
    'assists_per90': 3.0,
    'xg_per90': 2.0,
    'xa_per90': 1.5,
}


def cap_per90_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap per90 stats at realistic maximum values.
    
    This prevents inflated stats from players who played very few minutes
    (e.g., 2 tackles in 10 mins = 18 tackles/90, which is unrealistic).
    """
    df = df.copy()
    
    for stat_base, cap_value in PER90_CAPS.items():
        # Find all columns matching this stat pattern (e.g., tackles_per90_roll5, tackles_per90_roll3)
        matching_cols = [col for col in df.columns if stat_base in col]
        for col in matching_cols:
            if col in df.columns:
                df[col] = df[col].clip(upper=cap_value)
    
    return df


def normalize_team_name(name: str) -> str:
    """
    Normalize team name for consistent match grouping.
    
    Handles variations like:
    - 'brighton__hove_albion' vs 'Brighton'
    - 'manchester_city' vs 'Man City'
    - 'manchester_united' vs 'Man Utd'
    """
    if pd.isna(name):
        return ''
    
    name = str(name).lower().strip()
    # Replace underscores with spaces and normalize whitespace
    name = name.replace('_', ' ').replace('  ', ' ').strip()
    
    # Canonical mappings - map all variations to a standard short name
    mappings = {
        # Full names to short
        'brighton and hove albion': 'brighton',
        'brighton hove albion': 'brighton',
        'brighton  hove albion': 'brighton',
        'brighton & hove albion': 'brighton',
        'manchester city': 'man city',
        'manchester united': 'man utd',
        'tottenham hotspur': 'spurs',
        'tottenham': 'spurs',
        'wolverhampton wanderers': 'wolves',
        'wolverhampton': 'wolves',
        'nottingham forest': 'forest',
        'nottham forest': 'forest',
        "nott'm forest": 'forest',
        'newcastle united': 'newcastle',
        'west ham united': 'west ham',
        'crystal palace': 'palace',
        'aston villa': 'villa',
        'leeds united': 'leeds',
        'leicester city': 'leicester',
        # Short variations
        'man city': 'man city',
        'man utd': 'man utd',
    }
    
    # Check exact matches first
    if name in mappings:
        return mappings[name]
    
    # Check partial matches
    for full_name, short_name in mappings.items():
        if full_name in name:
            return short_name
    
    # Default: return first word (handles most cases like 'arsenal', 'chelsea', etc.)
    # But keep two-word names that aren't in mappings
    words = name.split()
    if len(words) == 1:
        return words[0]
    elif len(words) == 2 and words[0] in ['west', 'man', 'aston', 'crystal', 'leeds', 'leicester']:
        return name  # Keep compound names
    else:
        return words[0]  # Just first word for long names


def get_fpl_availability():
    """Fetch FPL player availability data from API."""
    try:
        response = requests.get(
            "https://fantasy.premierleague.com/api/bootstrap-static/",
            timeout=10
        )
        data = response.json()
        
        availability = {}
        for player in data['elements']:
            # Use web_name (display name) as key
            name = player['web_name'].lower()
            availability[name] = {
                'chance_of_playing': player.get('chance_of_playing_next_round'),
                'status': player.get('status', 'a'),  # a=available, i=injured, s=suspended, etc.
                'news': player.get('news', ''),
            }
            # Also add by full name
            full_name = f"{player['first_name']} {player['second_name']}".lower()
            availability[full_name] = availability[name]
        
        return availability
    except Exception as e:
        print(f"Warning: Could not fetch FPL availability: {e}")
        return {}


class BaselineBPSModel(ManagerFeatureMixin):
    """
    Predicts baseline BPS score from "boring" stats.
    
    This excludes goals, assists, and clean sheets - those are simulated separately.
    Baseline BPS comes from: passes, tackles, recoveries, saves, etc.
    """
    
    FEATURES = [
        # Passing/possession stats (primary contributors to baseline BPS)
        'key_passes_per90_roll5',
        'key_passes_per90_roll3',
        
        # Defensive stats (contribute to BPS)
        'tackles_per90_roll5',
        'tackles_per90_roll3',
        'interceptions_per90_roll5',
        'interceptions_per90_roll3',
        'clearances_per90_roll5',
        'blocks_per90_roll5',
        'recoveries_per90_roll5',
        
        # Shots (contribute to BPS even if not scored)
        'shots_per90_roll5',
        'shots_per90_roll3',
        
        # Recent scoring form (indicates attacking involvement)
        'goals_per90_roll5',
        'goals_per90_roll3',
        'assists_per90_roll5',
        'assists_per90_roll3',
        'xg_per90_roll5',
        'xa_per90_roll5',
        
        # LIFETIME PLAYER PROFILE
        'lifetime_goals_per90',
        'lifetime_assists_per90',
        'lifetime_xg_per90',
        'lifetime_key_passes_per90',
        'lifetime_defcon_per90',
        'lifetime_tackles_per90',
        'lifetime_interceptions_per90',
        'lifetime_minutes',
        
        # Position indicators
        'is_fwd',
        'is_mid',
        'is_def',
        'is_gk',
        
        # Match context
        'is_home',
        
        # Team/opponent context
        'team_goals_roll5',
        'team_xg_roll5',
        
        # Minutes
        'minutes_roll5',

        # Manager embeddings (8-dim PCA over rolling-20-prior manager stats)
        'manager_emb_0', 'manager_emb_1', 'manager_emb_2', 'manager_emb_3',
        'manager_emb_4', 'manager_emb_5', 'manager_emb_6', 'manager_emb_7',
    ]

    TARGET = 'baseline_bps'
    
    def __init__(self, **xgb_params):
        default_params = {
            'n_estimators': 150,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBRegressor(**default_params)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._features_used = []
    
    def _compute_baseline_bps(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute baseline BPS by subtracting major event BPS from total BPS.
        
        baseline_bps = bps - (goals * goal_bps) - (assists * 9) - (cs * cs_bps) + penalties
        """
        df = df.copy()
        
        # Get position for position-dependent BPS
        fallback_pos = df.get('position', pd.Series(2, index=df.index)).map(
            {0: 'GK', 1: 'DEF', 2: 'MID', 3: 'FWD'}).fillna('MID')
        fpl_pos = df.get('fpl_position', fallback_pos)
        
        # Calculate BPS from major events
        goal_bps = fpl_pos.map(lambda p: BPS_RULES['goal'].get(p, 18))
        cs_bps = fpl_pos.map(lambda p: BPS_RULES['clean_sheet'].get(p, 0))
        
        goals = df['goals'].fillna(0) if 'goals' in df.columns else pd.Series(0, index=df.index)
        assists = df['assists'].fillna(0) if 'assists' in df.columns else pd.Series(0, index=df.index)
        
        # Clean sheet: 1 if opponent_goals == 0 and player played 60+ mins
        opponent_goals = df.get('_bps_goals_against', df.get('opponent_goals',
            df.get('goals_conceded', pd.Series(1., index=df.index)))).fillna(1)
        minutes = df['minutes'].fillna(0) if 'minutes' in df.columns else pd.Series(60, index=df.index)
        clean_sheet = ((opponent_goals == 0) & (minutes >= 60)).astype(int)
        
        # Total BPS from major events
        conceded_bps = fpl_pos.map(lambda p: BPS_RULES['goal_conceded'].get(p, 0))
        yellow = df.get('yellow_cards', pd.Series(0., index=df.index)).fillna(0)
        red = df.get('red_cards', pd.Series(0., index=df.index)).fillna(0)
        major_event_bps = (goals * goal_bps + assists * BPS_RULES['assist'] +
                           clean_sheet * cs_bps + opponent_goals * conceded_bps * (minutes >= 60) +
                           yellow * BPS_RULES['yellow_card'] + red * BPS_RULES['red_card'])
        
        # Baseline = total - major events
        total_bps = df['bps'].fillna(0) if 'bps' in df.columns else pd.Series(0, index=df.index)
        baseline = total_bps - major_event_bps
        
        # Floor at 0
        return np.maximum(baseline, 0)
    
    def _estimate_baseline_bps(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate baseline BPS from stats when actual BPS data is not available.

        Uses official FPL BPS coefficients:
        - Playing 60+ mins: 6, playing 1-59 mins: 3
        - Key passes: 1 per
        - Tackles won: 2 per
        - Interceptions: 1 per
        - Recoveries: 1 per
        - Clearances: 1 per
        - Blocks: 1 per
        - Shots on target: 2 per (approximated as shots * 0.4)
        """
        mins = df['minutes'].fillna(0)
        mins_per90 = mins / 90

        # Base for playing: 6 if 60+ mins, 3 if 1-59 mins
        baseline = np.where(mins >= 60, 6.0, np.where(mins >= 1, 3.0, 0.0))

        # Key passes (1 BPS each)
        if 'key_passes_per90_roll5' in df.columns:
            baseline += df['key_passes_per90_roll5'].fillna(0) * mins_per90 * 1

        # Tackles won (2 BPS each)
        if 'tackles_per90_roll5' in df.columns:
            baseline += df['tackles_per90_roll5'].fillna(0) * mins_per90 * 2

        # Interceptions (1 BPS each)
        if 'interceptions_per90_roll5' in df.columns:
            baseline += df['interceptions_per90_roll5'].fillna(0) * mins_per90 * 1

        # Recoveries (1 BPS each)
        if 'recoveries_per90_roll5' in df.columns:
            baseline += df['recoveries_per90_roll5'].fillna(0) * mins_per90 * 1

        # Clearances (1 BPS each)
        if 'clearances_per90_roll5' in df.columns:
            baseline += df['clearances_per90_roll5'].fillna(0) * mins_per90 * 1

        # Blocks (1 BPS each)
        if 'blocks_per90_roll5' in df.columns:
            baseline += df['blocks_per90_roll5'].fillna(0) * mins_per90 * 1

        # Shots on target approximation (2 BPS each, ~40% of shots are on target)
        if 'shots_per90_roll5' in df.columns:
            baseline += df['shots_per90_roll5'].fillna(0) * mins_per90 * 0.4 * 2

        return pd.Series(baseline, index=df.index)
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train the baseline BPS model."""
        df = df.copy()
        
        # Build full-match conceded counts before filtering to the 60+ training subset.
        if {'match_id', 'team', 'opponent', 'goals'} <= set(df):
            stats = df.assign(_team=df['team'].map(normalize_team_name),
                              _og=df.get('own_goal', pd.Series(0., index=df.index)).fillna(0))
            totals = stats.groupby(['match_id', '_team'])[['goals', '_og']].sum()
            opposing = pd.MultiIndex.from_arrays([df['match_id'], df['opponent'].map(normalize_team_name)])
            own = pd.MultiIndex.from_arrays([df['match_id'], df['team'].map(normalize_team_name)])
            df['_bps_goals_against'] = (totals['goals'].reindex(opposing).fillna(0).to_numpy() +
                                        totals['_og'].reindex(own).fillna(0).to_numpy())
        # Only train on players who played 60+ mins
        played_mask = (df['minutes'] >= 60) if 'minutes' in df.columns else pd.Series(True, index=df.index)
        df = df[played_mask].copy()
        
        # Cap per90 stats to prevent learning from inflated values
        df = cap_per90_stats(df)
        
        # Compute target: baseline BPS
        if 'bps' in df.columns:
            df['baseline_bps'] = self._compute_baseline_bps(df)
            self._has_bps_data = True
            if verbose:
                print("  Using actual BPS data for training")
        else:
            df['baseline_bps'] = self._estimate_baseline_bps(df)
            self._has_bps_data = False
            if verbose:
                print("  Estimating baseline BPS from stats (no actual BPS data)")
        
        # Get available features
        available_features = [f for f in self.FEATURES if f in df.columns]
        
        for feat in available_features:
            df[feat] = df[feat].fillna(0)
        
        self.fit_manager_features(df)
        X = self.manager_features(df)[available_features].fillna(0).astype(float)
        # Inference scales the baseline by sampled minutes / 90: learn a rate.
        y = df['baseline_bps'].fillna(0) / (df['minutes'] / 90).clip(lower=1 / 90)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Sample weights by minutes
        sample_weights = df['minutes'].values.copy()
        sample_weights = sample_weights / sample_weights.mean()
        
        if verbose:
            print(f"Training BaselineBPSModel on {len(X)} samples...")
            print(f"  Features used: {len(available_features)}")
            print(f"  Avg baseline BPS: {y.mean():.1f}")
        
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        self.is_fitted = True
        self._features_used = available_features
        
        if verbose:
            y_pred = self.model.predict(X_scaled)
            print(f"  MAE: {mean_absolute_error(y, y_pred):.2f}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict baseline BPS."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        df = df.copy()
        
        for feat in self._features_used:
            if feat not in df.columns:
                df[feat] = 0
        
        X = self.manager_features(df)[self._features_used].fillna(0).astype(float)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        
        return np.maximum(preds, 0)
    
    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        return pd.DataFrame({
            'feature': self._features_used,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)


class BonusModel:
    """
    Monte Carlo simulation-based bonus model.
    
    Uses a BaselineBPSModel to predict baseline BPS, then simulates
    goals/assists/CS to determine bonus from BPS rankings within each match.
    """
    
    TARGET = 'bonus'
    FEATURES = BaselineBPSModel.FEATURES.copy()
    
    def __init__(self, n_simulations: int = 5000, **xgb_params):
        self.n_simulations = n_simulations
        self.baseline_model = BaselineBPSModel(**xgb_params)
        self.is_fitted = False
        self.fpl_availability = {}
        self._last_simulations = None
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train the baseline BPS model."""
        if verbose:
            print(f"Training BonusModel (Monte Carlo, {self.n_simulations} sims)...")
        
        self.baseline_model.fit(df, verbose=verbose)
        self.is_fitted = True
        self._features_used = self.baseline_model._features_used
        
        # Live availability is supplied by the forecast, never fetched during fitting.
        return self
    
    def _get_player_availability(self, player_name: str) -> float:
        """Get availability probability for a player (0-1)."""
        if not self.fpl_availability:
            return 1.0
        
        if pd.isna(player_name):
            return 1.0
        
        name_lower = str(player_name).lower()
        
        # Try exact match
        if name_lower in self.fpl_availability:
            info = self.fpl_availability[name_lower]
            chance = info.get('chance_of_playing')
            status = info.get('status', 'a')
            
            # If injured/suspended with 0% chance, return 0
            if status in ['i', 's', 'u'] and chance == 0:
                return 0.0
            
            if chance is not None:
                return chance / 100.0
            return 1.0
        
        # Try last name only
        parts = str(player_name).split()
        if len(parts) > 1:
            last_name = parts[-1].lower()
            if last_name in self.fpl_availability:
                info = self.fpl_availability[last_name]
                chance = info.get('chance_of_playing')
                status = info.get('status', 'a')
                
                if status in ['i', 's', 'u'] and chance == 0:
                    return 0.0
                
                if chance is not None:
                    return chance / 100.0
        
        return 1.0
    
    @staticmethod
    def award_bonus(bps, playing):
        """Competition ranks: two tied first receive 3,3; next receives 1."""
        bps, playing = np.asarray(bps), np.asarray(playing, dtype=bool)
        ranks = np.zeros(bps.shape, dtype=int)
        for i in range(bps.shape[1]):
            ranks[:, i] = ((bps > bps[:, i, None]) & playing).sum(axis=1)
        return np.where(playing, np.maximum(3 - ranks, 0), 0)

    def simulate(self, df, probability=None, state_minutes=None, *,
                 n_simulations=None, seed=42, defcon_r=None):
        """Shared match events for weekly points, bonus, charts and five-week runs.

        Team goal totals are sampled once. Scorers/assisters are allocated from
        minutes-adjusted player intensities, leaving a residual bucket for players
        outside the forecast pool/no assist. If player intensities exceed the team
        total, they are proportionally reconciled. One scorer and at most one
        different assister are credited per goal. Player role draws are presently
        independent; this is not an eleven-player lineup simulator.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        frame = cap_per90_stats(df).reset_index(drop=True)
        n, ns = len(frame), int(n_simulations or self.n_simulations)
        rng = np.random.default_rng(seed)
        def vector(column, default=0):
            return pd.to_numeric(frame.get(column, pd.Series(default, index=frame.index)),
                                 errors='coerce').fillna(default).to_numpy(dtype=float)
        base_minutes = np.clip(vector('pred_minutes', 60), 1, 90)
        if probability is None:
            appear = np.clip(vector('pred_appear_prob', 1), 0, 1)
            p60 = np.clip(vector('pred_60_prob_cond', 1), 0, 1)
            probability = np.column_stack([1 - appear, appear * (1 - p60), appear * p60])
            state_minutes = np.column_stack([
                np.zeros(n), np.minimum(base_minutes, 59), np.maximum(base_minutes, 60)])
        probability = np.asarray(probability, dtype=float)
        if probability.ndim != 2 or probability.shape[0] != n:
            raise ValueError("Role probabilities must have one row per fixture-player")
        if not np.isfinite(probability).all() or (probability < 0).any():
            raise ValueError("Role probabilities must be finite and non-negative")
        if not np.allclose(probability.sum(axis=1), 1):
            raise ValueError("Role probabilities must sum to one")
        state_minutes = np.asarray(state_minutes, dtype=float)
        if state_minutes.ndim == 1:
            state_minutes = np.broadcast_to(state_minutes, probability.shape)
        if state_minutes.shape != probability.shape or not np.isfinite(state_minutes).all():
            raise ValueError("State minutes must match the role probabilities")
        cumulative = probability.cumsum(axis=1)
        draws = rng.random((ns, n))
        state = (draws[:, :, None] > cumulative[None, :, :]).sum(axis=2)
        state = np.minimum(state, probability.shape[1] - 1)
        minutes = np.take_along_axis(np.broadcast_to(state_minutes, (ns, *state_minutes.shape)),
                                     state[:, :, None], axis=2)[:, :, 0].clip(0, 90)
        playing, sixty = minutes > 0, minutes >= 60
        scale = minutes / base_minutes[None, :]
        goals, assists = np.zeros((ns, n), dtype=int), np.zeros((ns, n), dtype=int)
        against, team_goals = np.zeros((ns, n), dtype=int), np.zeros((ns, n), dtype=int)
        team = frame.get('team', pd.Series('team', index=frame.index)).map(normalize_team_name).to_numpy()
        opponent = frame.get('opponent', pd.Series('opponent', index=frame.index)).map(normalize_team_name).to_numpy()
        # Even legacy callers without match_id are separated by season/GW/pair.
        keys = []
        for i, row in frame.iterrows():
            fixture = row.get('match_id')
            if pd.isna(fixture):
                fixture = '|'.join(sorted([team[i], opponent[i]]))
            keys.append((str(row.get('season', '')), str(row.get('gameweek', '')), str(fixture)))
        groups = {}
        for i, key in enumerate(keys):
            groups.setdefault(key, []).append(i)
        goal_mean = np.maximum(vector('pred_exp_goals'), 0)
        assist_mean = np.maximum(vector('pred_exp_assists'), 0)
        team_mean = np.maximum(vector('pred_team_goals', 1.3), .001)
        ga_mean = np.maximum(vector('pred_goals_against', 1.3), .001)

        for indices in groups.values():
            indices = np.asarray(indices)
            sides = sorted(set(team[indices]) | set(opponent[indices]))
            scores = {}
            for side in sides:
                attackers = indices[team[indices] == side]
                defenders = indices[opponent[indices] == side]
                lam = float(np.mean(team_mean[attackers])) if len(attackers) else float(np.mean(ga_mean[defenders]))
                score = rng.poisson(lam, ns)
                scores[side] = score
                if not len(attackers):
                    continue
                team_goals[:, attackers] = score[:, None]
                scorer_weights = goal_mean[attackers][None, :] * scale[:, attackers] / lam
                scorer_weights /= np.maximum(scorer_weights.sum(axis=1, keepdims=True), 1)
                assist_weights = assist_mean[attackers][None, :] * scale[:, attackers] / lam
                for event in range(int(score.max(initial=0))):
                    active = np.flatnonzero(score > event)
                    cdf = scorer_weights[active].cumsum(axis=1)
                    scorer = (rng.random(len(active))[:, None] > cdf).sum(axis=1)
                    credited = scorer < len(attackers)
                    goals[active[credited], attackers[scorer[credited]]] += 1
                    weights = assist_weights[active].copy()
                    # An unmodeled scorer can still have a modeled assister.
                    weights[np.flatnonzero(credited), scorer[credited]] = 0
                    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1)
                    assister = (rng.random(len(active))[:, None] > weights.cumsum(axis=1)).sum(axis=1)
                    credited_assist = assister < len(attackers)
                    assists[active[credited_assist], attackers[assister[credited_assist]]] += 1
            for i in indices:
                against[:, i] = scores[opponent[i]]

        cs = ((against == 0) & sixty).astype(int)
        yellows = rng.binomial(1, np.clip(vector('pred_yellow_prob')[None, :] * scale, 0, 1))
        reds = rng.binomial(1, np.clip(vector('pred_red_prob')[None, :] * scale, 0, 1))
        saves = rng.poisson(np.maximum(vector('pred_exp_saves')[None, :] * scale, 0))
        mu = np.maximum(vector('pred_exp_defcon')[None, :] * scale, 0)
        defcon = (rng.negative_binomial(defcon_r, defcon_r / (defcon_r + mu))
                  if defcon_r is not None and defcon_r > 0 else rng.poisson(mu))
        positions = frame.get('fpl_position', pd.Series('MID', index=frame.index)).to_numpy()
        # minutes_roll5 is a MEAN, not cumulative exposure.
        games = np.minimum(vector('lifetime_appearances', 5), 5)
        reliability = np.clip(vector('minutes_roll5', 60) * games / 300., .3, 1.)
        baseline = self.baseline_model.predict(frame) * reliability
        goal_bps = np.array([BPS_RULES['goal'].get(p, 18) for p in positions])
        cs_bps = np.array([BPS_RULES['clean_sheet'].get(p, 0) for p in positions])
        gc_bps = np.array([BPS_RULES['goal_conceded'].get(p, 0) for p in positions])
        bps = np.rint(baseline[None, :] * minutes / 90 +
                      goals * goal_bps + assists * BPS_RULES['assist'] +
                      cs * cs_bps + against * gc_bps * sixty +
                      yellows * BPS_RULES['yellow_card'] + reds * BPS_RULES['red_card'])
        bonus = np.zeros((ns, n), dtype=int)
        for indices in groups.values():
            bonus[:, indices] = self.award_bonus(bps[:, indices], playing[:, indices])
        self._last_simulations = {
            'minutes': minutes, 'goals': goals, 'assists': assists, 'cs': cs,
            'goals_against': against, 'team_goals': team_goals, 'yellows': yellows,
            'reds': reds, 'saves': saves, 'defcon': defcon, 'bonus': bonus, 'bps': bps,
        }
        return self._last_simulations

    def predict(self, df, pred_goals=None, pred_assists=None, pred_cs_prob=None,
                pred_minutes=None, fpl_positions=None, pred_yellow_prob=None):
        """Compatibility interface; new callers pass role distributions to simulate."""
        frame = df.copy()
        for column, value in [('pred_exp_goals', pred_goals), ('pred_exp_assists', pred_assists),
                              ('pred_cs_prob', pred_cs_prob), ('pred_minutes', pred_minutes),
                              ('fpl_position', fpl_positions), ('pred_yellow_prob', pred_yellow_prob)]:
            if value is not None:
                frame[column] = value
        return self.simulate(frame)['bonus'].mean(axis=0)

    def get_last_simulations(self) -> dict:
        """Return per-simulation arrays from the last predict() call.

        Returns dict with keys:
            goals:   (n_sims, n_players) sampled goals
            assists: (n_sims, n_players) sampled assists
            cs:      (n_sims, n_players) binary clean sheet
            yellows: (n_sims, n_players) binary yellow card
            bonus:   (n_sims, n_players) awarded bonus 0-3
        """
        if self._last_simulations is None:
            raise ValueError("No simulations available. Call predict() first.")
        return self._last_simulations

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance from baseline model."""
        return self.baseline_model.feature_importance()


def score_simulations(frame, simulations, rules):
    """Score the existing event draws once, using the pipeline's configured rules."""
    minutes = simulations['minutes']
    positions = frame['fpl_position'].to_numpy()
    sixty = minutes >= 60
    goal_values = np.array([rules['goal'].get(p, 5) for p in positions])
    cs_values = np.array([rules['clean_sheet'].get(p, 0) for p in positions])
    gc_values = np.array([rules['goals_conceded_2'].get(p, 0) for p in positions])
    thresholds = np.where(positions == 'DEF', 10, 12)
    components = {
        'exp_appearance_pts': np.where(sixty, rules['appearance_60'],
                                      np.where(minutes > 0, rules['appearance_1'], 0)),
        'exp_goals_pts': simulations['goals'] * goal_values,
        'exp_assists_pts': simulations['assists'] * rules['assist'],
        'exp_cs_pts': simulations['cs'] * cs_values,
        'exp_conceded_penalty': (simulations['goals_against'] // 2) * gc_values * sixty,
        'exp_saves_pts': (simulations['saves'] // 3) * (positions == 'GK') * rules['saves_per_3'],
        'exp_defcon_pts': ((simulations['defcon'] >= thresholds) * sixty *
                           np.isin(positions, ['DEF', 'MID']) * rules['defcon']),
        'exp_bonus_pts': simulations['bonus'],
        'exp_yellow_pts': simulations['yellows'] * rules['yellow_card'],
        'exp_red_pts': simulations['reds'] * rules['red_card'],
    }
    simulations['total_points'] = sum(components.values())
    # Keep the components in the archive too: charts never need to resample.
    simulations.update(components)
    return components
