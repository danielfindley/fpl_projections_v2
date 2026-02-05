"""
Monte Carlo Simulation-Based Bonus Points Model

Instead of predicting bonus directly, this model:
1. Predicts baseline BPS (raw score from "boring" stats)
2. Uses existing model predictions (goals, assists, CS probability)
3. Runs Monte Carlo simulation to determine bonus from BPS rankings

This is more accurate because bonus points are a ranking-based competition.
"""

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


class BaselineBPSModel:
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
        fpl_pos = df.get('fpl_position', pd.Series(['MID'] * len(df), index=df.index))
        
        # Calculate BPS from major events
        goal_bps = fpl_pos.map(lambda p: BPS_RULES['goal'].get(p, 18))
        cs_bps = fpl_pos.map(lambda p: BPS_RULES['clean_sheet'].get(p, 0))
        
        goals = df['goals'].fillna(0) if 'goals' in df.columns else pd.Series(0, index=df.index)
        assists = df['assists'].fillna(0) if 'assists' in df.columns else pd.Series(0, index=df.index)
        
        # Clean sheet: 1 if opponent_goals == 0 and player played 60+ mins
        opponent_goals = df['opponent_goals'].fillna(1) if 'opponent_goals' in df.columns else pd.Series(1, index=df.index)
        minutes = df['minutes'].fillna(0) if 'minutes' in df.columns else pd.Series(60, index=df.index)
        clean_sheet = ((opponent_goals == 0) & (minutes >= 60)).astype(int)
        
        # Total BPS from major events
        major_event_bps = (goals * goal_bps) + (assists * 9) + (clean_sheet * cs_bps)
        
        # Baseline = total - major events
        total_bps = df['bps'].fillna(0) if 'bps' in df.columns else pd.Series(0, index=df.index)
        baseline = total_bps - major_event_bps
        
        # Floor at 0
        return np.maximum(baseline, 0)
    
    def _estimate_baseline_bps(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate baseline BPS from stats when actual BPS data is not available.
        """
        baseline = np.zeros(len(df))
        
        # Base for playing 60+ mins
        baseline += 6
        
        # Key passes
        if 'key_passes_per90_roll5' in df.columns:
            key_passes = df['key_passes_per90_roll5'].fillna(0) * (df['minutes'].fillna(0) / 90)
            baseline += key_passes * 2
        
        # Tackles
        if 'tackles_per90_roll5' in df.columns:
            tackles = df['tackles_per90_roll5'].fillna(0) * (df['minutes'].fillna(0) / 90)
            baseline += tackles * 2
        
        # Interceptions
        if 'interceptions_per90_roll5' in df.columns:
            ints = df['interceptions_per90_roll5'].fillna(0) * (df['minutes'].fillna(0) / 90)
            baseline += ints * 3
        
        # Recoveries
        if 'recoveries_per90_roll5' in df.columns:
            recoveries = df['recoveries_per90_roll5'].fillna(0) * (df['minutes'].fillna(0) / 90)
            baseline += recoveries * 1
        
        return pd.Series(baseline, index=df.index)
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train the baseline BPS model."""
        df = df.copy()
        
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
        
        X = df[available_features].fillna(0).astype(float)
        y = df['baseline_bps'].fillna(0)
        
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
        
        X = df[self._features_used].fillna(0).astype(float)
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
    
    def __init__(self, n_simulations: int = 1000, **xgb_params):
        self.n_simulations = n_simulations
        self.baseline_model = BaselineBPSModel(**xgb_params)
        self.is_fitted = False
        self.fpl_availability = {}
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train the baseline BPS model."""
        if verbose:
            print(f"Training BonusModel (Monte Carlo, {self.n_simulations} sims)...")
        
        self.baseline_model.fit(df, verbose=verbose)
        self.is_fitted = True
        self._features_used = self.baseline_model._features_used
        
        # Fetch FPL availability
        self.fpl_availability = get_fpl_availability()
        if verbose and self.fpl_availability:
            print(f"  Loaded FPL availability for {len(self.fpl_availability)} players")
        
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
    
    def predict(self, df: pd.DataFrame, pred_goals=None, pred_assists=None, 
                pred_cs_prob=None, pred_minutes=None, fpl_positions=None) -> np.ndarray:
        """
        Predict expected bonus using Monte Carlo simulation.
        
        Players with low predicted minutes or low FPL availability are excluded
        from bonus competition.
        
        Includes fixes for per90 stat inflation:
        1. Caps per90 stats at realistic maximums
        2. Applies reliability weighting based on historical minutes
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        df = df.copy()
        n = len(df)
        
        # Fix #1: Cap per90 stats to prevent inflation from low-minutes appearances
        df = cap_per90_stats(df)
        
        # Refresh FPL availability
        if not self.fpl_availability:
            self.fpl_availability = get_fpl_availability()
        
        # Get predictions from df if not provided
        if pred_goals is None:
            pred_goals = df.get('pred_exp_goals', np.zeros(n))
        if pred_assists is None:
            pred_assists = df.get('pred_exp_assists', np.zeros(n))
        if pred_cs_prob is None:
            pred_cs_prob = df.get('pred_cs_prob', np.full(n, 0.25))
        if pred_minutes is None:
            pred_minutes = df.get('pred_minutes', np.full(n, 60))
        
        pred_goals = np.array(pred_goals)
        pred_assists = np.array(pred_assists)
        pred_cs_prob = np.array(pred_cs_prob)
        pred_minutes = np.array(pred_minutes)
        
        # Get positions
        if fpl_positions is None:
            fpl_positions = df.get('fpl_position', pd.Series(['MID'] * n)).values
        
        # Get FPL availability for each player
        player_names = df.get('player_name', pd.Series([''] * n))
        availability = np.array([self._get_player_availability(name) for name in player_names])
        
        # Baseline BPS prediction
        baseline_bps = self.baseline_model.predict(df)
        
        # Fix #2: Apply reliability weighting based on historical minutes
        # Players with fewer rolling minutes have less reliable per90 stats
        # 300 minutes (5 full games) = full reliability, less = reduced confidence
        minutes_roll5 = df.get('minutes_roll5', pd.Series(np.full(n, 300))).fillna(300).values
        reliability = np.clip(minutes_roll5 / 300.0, 0.3, 1.0)
        baseline_bps = baseline_bps * reliability
        
        # Scale by predicted minutes for this match
        baseline_bps = baseline_bps * np.clip(pred_minutes / 90, 0, 1.0)
        
        # Zero out BPS for players unlikely to play
        # Players with <10 predicted minutes or 0% availability don't compete
        playing_mask = (pred_minutes >= 10) & (availability > 0)
        baseline_bps = np.where(playing_mask, baseline_bps, 0)
        
        # BPS values by position
        goal_bps = np.array([BPS_RULES['goal'].get(p, 18) for p in fpl_positions])
        cs_bps = np.array([BPS_RULES['clean_sheet'].get(p, 0) for p in fpl_positions])
        mins_60_mask = (pred_minutes >= 60).astype(float)
        
        # Match grouping using proper normalization
        if 'team' in df.columns and 'opponent' in df.columns:
            match_groups = df.apply(
                lambda r: '_vs_'.join(sorted([
                    normalize_team_name(r.get('team', '')), 
                    normalize_team_name(r.get('opponent', ''))
                ])),
                axis=1
            ).values
        else:
            match_groups = np.array(['match'] * n)
        
        # Monte Carlo simulation
        n_sims = self.n_simulations
        
        # Sample goals, assists, clean sheets for all simulations
        all_goals = np.random.poisson(np.maximum(pred_goals, 0), (n_sims, n))
        all_assists = np.random.poisson(np.maximum(pred_assists, 0), (n_sims, n))
        all_cs = ((np.random.random((n_sims, n)) < pred_cs_prob) * mins_60_mask).astype(int)
        
        # Apply playing mask - non-playing players get 0 events
        all_goals = all_goals * playing_mask
        all_assists = all_assists * playing_mask
        all_cs = all_cs * playing_mask
        
        # Calculate total BPS for each simulation
        all_bps = (
            baseline_bps +
            all_goals * goal_bps +
            all_assists * BPS_RULES['assist'] +
            all_cs * cs_bps
        ) * playing_mask
        
        # Compute bonus per match
        total_bonus = np.zeros(n)
        unique_matches = np.unique(match_groups)
        match_to_indices = {m: np.where(match_groups == m)[0] for m in unique_matches}
        
        for match_id, match_idx in match_to_indices.items():
            if len(match_idx) == 0:
                continue
            
            # Get BPS for this match across all simulations: (n_sims, n_match)
            match_bps = all_bps[:, match_idx]
            
            # Initialize match bonus
            match_bonus = np.zeros((n_sims, len(match_idx)))
            
            # For each simulation, rank players and award bonus
            for sim in range(n_sims):
                bps_sim = match_bps[sim]
                
                # Only consider players who are actually playing
                playing_in_match = bps_sim > 0
                if not np.any(playing_in_match):
                    continue
                
                sorted_idx = np.argsort(-bps_sim)
                sorted_bps = bps_sim[sorted_idx]
                
                # Award 3, 2, 1 handling ties
                bonus_to_award = [3, 2, 1]
                bonus_idx = 0
                i = 0
                
                while i < len(sorted_idx) and bonus_idx < 3:
                    if sorted_bps[i] <= 0:  # Skip non-playing players
                        i += 1
                        continue
                    
                    # Find all tied at this BPS level
                    tied_mask = bps_sim == sorted_bps[i]
                    n_tied = np.sum(tied_mask)
                    
                    # Award current bonus to all tied players
                    match_bonus[sim, tied_mask] = bonus_to_award[bonus_idx]
                    
                    # Move to next bonus level
                    bonus_idx += 1
                    i += n_tied
            
            total_bonus[match_idx] = match_bonus.mean(axis=0)
        
        return total_bonus
    
    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance from baseline model."""
        return self.baseline_model.feature_importance()

        if not self.is_fitted:

            raise ValueError("Model not fitted")

        

        n = len(df)

        

        # Get predictions from df if not provided

        if pred_goals is None:

            pred_goals = df.get('pred_exp_goals', np.zeros(n))

        if pred_assists is None:

            pred_assists = df.get('pred_exp_assists', np.zeros(n))

        if pred_cs_prob is None:

            pred_cs_prob = df.get('pred_cs_prob', np.full(n, 0.25))

        if pred_minutes is None:

            pred_minutes = df.get('pred_minutes', np.full(n, 60))

        

        pred_goals = np.array(pred_goals)

        pred_assists = np.array(pred_assists)

        pred_cs_prob = np.array(pred_cs_prob)

        pred_minutes = np.array(pred_minutes)

        

        # Get positions

        if fpl_positions is None:

            fpl_positions = df.get('fpl_position', pd.Series(['MID'] * n)).values

        

        # Baseline BPS (scaled by minutes)

        X = self._prepare_X(df)

        X_scaled = self.scaler.transform(X)

        baseline_bps = self.model.predict(X_scaled)

        baseline_bps = baseline_bps * np.clip(pred_minutes / 90, 0, 1.0)

        

        # Zero out BPS for players with very low predicted minutes

        baseline_bps = np.where(pred_minutes < 10, 0, baseline_bps)

        

        # BPS values by position

        goal_bps = np.array([BPS_GOAL.get(p, 18) for p in fpl_positions])

        cs_bps = np.array([BPS_CS.get(p, 0) for p in fpl_positions])

        mins_mask = (pred_minutes >= 60).astype(float)

        playing_mask = (pred_minutes >= 10).astype(float)  # Must play to get bonus

        

        # Monte Carlo simulation (only for players predicted to play)

        all_goals = np.random.poisson(np.maximum(pred_goals, 0), (self.n_simulations, n)) * playing_mask

        all_assists = np.random.poisson(np.maximum(pred_assists, 0), (self.n_simulations, n)) * playing_mask

        all_cs = ((np.random.random((self.n_simulations, n)) < pred_cs_prob) * mins_mask).astype(int)

        

        all_bps = (baseline_bps + all_goals * goal_bps + all_assists * BPS_ASSIST + all_cs * cs_bps) * playing_mask

        

        # Match grouping

        def normalize_team(name):

            if pd.isna(name):

                return ''

            return str(name).lower().replace('_', ' ').replace("'", "").strip()[:10]

        

        if 'team' in df.columns and 'opponent' in df.columns:

            match_groups = df.apply(

                lambda r: '_'.join(sorted([normalize_team(r.get('team', '')), 

                                           normalize_team(r.get('opponent', ''))])),

                axis=1

            ).values

        else:

            match_groups = np.array(['match'] * n)

        

        # Compute bonus per simulation

        total_bonus = np.zeros(n)

        unique_matches = np.unique(match_groups)

        

        for match_id in unique_matches:

            match_idx = np.where(match_groups == match_id)[0]

            if len(match_idx) == 0:

                continue

            

            match_bps = all_bps[:, match_idx]

            match_bonus = np.zeros((self.n_simulations, len(match_idx)))

            

            for sim in range(self.n_simulations):

                bps_sim = match_bps[sim]

                sorted_idx = np.argsort(-bps_sim)

                

                # Award 3, 2, 1 to top 3 (handling ties)

                if len(sorted_idx) >= 1:

                    match_bonus[sim, sorted_idx[0]] = 3

                if len(sorted_idx) >= 2 and bps_sim[sorted_idx[1]] < bps_sim[sorted_idx[0]]:

                    match_bonus[sim, sorted_idx[1]] = 2

                if len(sorted_idx) >= 3 and bps_sim[sorted_idx[2]] < bps_sim[sorted_idx[1]]:

                    match_bonus[sim, sorted_idx[2]] = 1

            

            total_bonus[match_idx] = match_bonus.mean(axis=0)

        

        return total_bonus

    

    def feature_importance(self) -> pd.DataFrame:

        if not self.is_fitted:

            raise ValueError("Model not fitted")

        return pd.DataFrame({

            'feature': self.FEATURES,

            'importance': self.model.feature_importances_

        }).sort_values('importance', ascending=False)

