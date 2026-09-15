import unittest

import numpy as np
import pandas as pd
import pytest

from src.models.clean_sheet import CleanSheetModel


def _team_row(team, opponent, season, gameweek, is_home, goals, xg,
              key_passes=8, shots_on_target=4):
    return {
        'team': team,
        'opponent': opponent,
        'season': season,
        'gameweek': gameweek,
        'is_home': is_home,
        'goals': goals,
        'own_goal': 0,
        'xg': xg,
        'key_passes': key_passes,
        'shots_on_target': shots_on_target,
        'tackles': 10,
        'interceptions': 8,
        'clearances': 12,
        'blocks': 3,
        'saves': 2,
        'xgot_faced': xg,
        'accurate_passes': 350,
        'touches': 550,
    }


class CleanSheetFeatureTests(unittest.TestCase):
    def _history(self):
        rows = []
        # Opening baseline season: Alpha and Beta are established clubs.
        for gw in range(1, 5):
            rows += [
                _team_row('Alpha', 'Beta', '2020/2021', gw, gw % 2, 2, 1.8),
                _team_row('Beta', 'Alpha', '2020/2021', gw, 1 - gw % 2, 1, 1.1),
            ]

        # Cohort is promoted in 2021/22 and supplies the next promoted prior.
        for gw in range(1, 5):
            rows += [
                _team_row('Alpha', 'Cohort', '2021/2022', gw, gw % 2, 2, 1.8),
                _team_row('Cohort', 'Alpha', '2021/2022', gw, 1 - gw % 2,
                          1, 1.0, key_passes=6, shots_on_target=3),
            ]

        # Newcomer has two volatile results; its target-GW features should still
        # use Cohort's prior rather than these one/two-match samples.
        for gw, newcomer_xg in [(1, 0.4), (2, 0.2)]:
            rows += [
                _team_row('Alpha', 'Newcomer', '2022/2023', gw, gw % 2,
                          2, 1.8),
                _team_row('Newcomer', 'Alpha', '2022/2023', gw, 1 - gw % 2,
                          0, newcomer_xg, key_passes=2, shots_on_target=1),
            ]
        return pd.DataFrame(rows)

    def test_future_row_advances_form_and_uses_prior_promoted_cohort(self):
        history = self._history()
        fixtures = pd.DataFrame([
            {'team': 'Alpha', 'opponent': 'Newcomer', 'season': '2022/2023',
             'gameweek': 3, 'is_home': 1},
            {'team': 'Newcomer', 'opponent': 'Alpha', 'season': '2022/2023',
             'gameweek': 3, 'is_home': 0},
        ])

        features = CleanSheetModel().prepare_team_features(
            history, prediction_fixtures=fixtures
        )
        target = features[features['_is_prediction']]
        self.assertEqual(len(target), 2)

        alpha = target[target['team'] == 'Alpha'].iloc[0]
        # Target row includes GW2. Reusing the GW2 pre-match row would return 0.4.
        self.assertAlmostEqual(alpha['team_xga_roll1'], 0.2)

        # After two matches the ramp is 60% prior / 40% current evidence. The
        # earlier promoted cohort scored exactly 1.0 goal/xG per match.
        self.assertAlmostEqual(alpha['opp_season_goals_per_game'], 0.6)
        self.assertAlmostEqual(alpha['opp_season_xg_per_game'], 0.72)
        self.assertAlmostEqual(alpha['opp_ha_season_goals'], 0.6)
        self.assertAlmostEqual(alpha['opp_ha_season_xg'], 0.72)
        self.assertAlmostEqual(alpha['opp_xg_roll1'], 0.68)
        self.assertAlmostEqual(alpha['opp_xg_roll2'], 0.72)
        self.assertAlmostEqual(alpha['opp_key_passes_roll1'], 4.4)
        self.assertAlmostEqual(alpha['opp_key_passes_roll2'], 4.4)
        self.assertAlmostEqual(alpha['opp_shots_ot_roll1'], 2.2)
        self.assertAlmostEqual(alpha['opp_shots_ot_roll2'], 2.2)
        # Wider windows do not yet have enough observations, so they stay at
        # the promoted prior rather than falling through to league average.
        for window in [3, 5, 7, 10]:
            self.assertAlmostEqual(alpha[f'opp_xg_roll{window}'], 1.0)
            self.assertAlmostEqual(alpha[f'opp_key_passes_roll{window}'], 6.0)
            self.assertAlmostEqual(alpha[f'opp_shots_ot_roll{window}'], 3.0)

        model_features = target[CleanSheetModel.FEATURES].astype(float).to_numpy()
        self.assertTrue(np.isfinite(model_features).all())

    def test_goals_conceded_join_uses_both_fixture_sides(self):
        history = self._history()
        features = CleanSheetModel().prepare_team_features(history)
        actual = features[
            (features['season'] == '2022/2023')
            & (features['gameweek'] == 2)
        ]
        alpha = actual[actual['team'] == 'Alpha'].iloc[0]
        newcomer = actual[actual['team'] == 'Newcomer'].iloc[0]
        self.assertEqual(alpha['goals_conceded'], 0)
        self.assertEqual(newcomer['goals_conceded'], 2)

    def test_promoted_attack_transitions_over_five_matches_without_zero_lambda(self):
        history = self._history()

        # Give the promoted club a third goalless match with meaningful xG. The
        # GW4 prediction is the boundary that previously dropped the prior and
        # collapsed prior_lambda to zero.
        history = pd.concat([
            history,
            pd.DataFrame([
                _team_row('Alpha', 'Newcomer', '2022/2023', 3, 0, 2, 1.8),
                _team_row('Newcomer', 'Alpha', '2022/2023', 3, 1,
                          0, 1.2, key_passes=7, shots_on_target=3),
            ]),
        ], ignore_index=True)
        fixtures = pd.DataFrame([
            {'team': 'Alpha', 'opponent': 'Newcomer', 'season': '2022/2023',
             'gameweek': 4, 'is_home': 0},
            {'team': 'Newcomer', 'opponent': 'Alpha', 'season': '2022/2023',
             'gameweek': 4, 'is_home': 1},
        ])

        features = CleanSheetModel().prepare_team_features(
            history, prediction_fixtures=fixtures
        )
        alpha = features[
            features['_is_prediction'] & features['team'].eq('Alpha')
        ].iloc[0]

        # Three completed matches leave 40% promoted prior / 60% current form.
        self.assertAlmostEqual(alpha['opp_season_goals_per_game'], 0.4)
        self.assertAlmostEqual(alpha['opp_season_xg_per_game'], 0.76)
        self.assertAlmostEqual(alpha['opp_ha_season_goals'], 0.4)
        self.assertAlmostEqual(alpha['opp_ha_season_xg'], 0.76)
        self.assertGreaterEqual(
            alpha['prior_lambda'], CleanSheetModel.PRIOR_LAMBDA_MIN
        )
        self.assertLess(alpha['naive_cs_prob'], 1.0)

        # The lambda anchor is exactly the value seen by the model's base margin.
        base_lambda = np.exp(CleanSheetModel._get_base_margin(
            pd.DataFrame([alpha])
        ))[0]
        self.assertAlmostEqual(base_lambda, alpha['prior_lambda'])

    def test_promoted_prior_is_fully_removed_after_five_matches(self):
        history = self._history()
        extra_matches = [
            (3, 1, 1.2),
            (4, 0, 0.8),
            (5, 1, 1.0),
        ]
        rows = []
        for gw, newcomer_is_home, newcomer_xg in extra_matches:
            rows += [
                _team_row('Alpha', 'Newcomer', '2022/2023', gw,
                          1 - newcomer_is_home, 2, 1.8),
                _team_row('Newcomer', 'Alpha', '2022/2023', gw,
                          newcomer_is_home, 0, newcomer_xg,
                          key_passes=7, shots_on_target=3),
            ]
        history = pd.concat([history, pd.DataFrame(rows)], ignore_index=True)
        fixtures = pd.DataFrame([
            {'team': 'Alpha', 'opponent': 'Newcomer', 'season': '2022/2023',
             'gameweek': 6, 'is_home': 0},
            {'team': 'Newcomer', 'opponent': 'Alpha', 'season': '2022/2023',
             'gameweek': 6, 'is_home': 1},
        ])

        features = CleanSheetModel().prepare_team_features(
            history, prediction_fixtures=fixtures
        )
        alpha = features[
            features['_is_prediction'] & features['team'].eq('Alpha')
        ].iloc[0]

        # At five completed matches, the promoted prior has zero weight. The
        # opponent identity is Newcomer's actual 0 goals and 0.72 xG per game.
        self.assertAlmostEqual(alpha['opp_season_goals_per_game'], 0.0)
        self.assertAlmostEqual(alpha['opp_season_xg_per_game'], 0.72)
        self.assertGreaterEqual(
            alpha['prior_lambda'], CleanSheetModel.PRIOR_LAMBDA_MIN
        )


if __name__ == '__main__':
    unittest.main()



def _dated_history(weeks=9):
    rows = []
    for gw in range(1, weeks + 1):
        for team, opponent, home, offset in [('Alpha', 'Beta', 1, 0), ('Beta', 'Alpha', 0, 2)]:
            for position in (0, 2):
                row = _team_row(team, opponent, '2024/2025', gw, home,
                                int(position == 2 and gw % 3 == 0), .2 + position / 4)
                row.update(player_id=offset + position // 2, player_name=f'{team} {position}',
                           position=position, fpl_position='GK' if position == 0 else 'MID',
                           minutes=90 if position == 0 or gw % 2 else 20,
                           match_id=gw, match_date=pd.Timestamp('2024-08-01', tz='UTC') + pd.Timedelta(days=gw * 7),
                           assists=int(position == 2 and gw % 4 == 0), shots=2, xa=.1,
                           recoveries=3, fouls_committed=1, yellow_cards=int(gw % 3 == 0),
                           red_cards=0, bonus=0)
                rows.append(row)
    return pd.DataFrame(rows)


def test_rolling_per90_pools_minutes_exposure_and_remains_deadline_safe():
    from src.features import chronological_frame, prior_exposure_rate
    frame = pd.DataFrame({
        'player_id': [1, 1, 1],
        'season': ['2024/2025'] * 3,
        'gameweek': [1, 2, 3],
        'minutes': [6, 90, 1],
        'tackles': [1, 2, 999],
        'match_date': pd.to_datetime(['2024-08-01', '2024-08-08', '2024-08-15'], utc=True),
    })
    frame = chronological_frame(frame)
    result = prior_exposure_rate(frame, 'player_id', 'tackles', window=5)
    assert result.iloc[2] == pytest.approx(90 * 3 / 96)

    changed = frame.copy()
    changed.loc[2, 'tackles'] = 1_000_000
    changed_result = prior_exposure_rate(changed, 'player_id', 'tackles', window=5)
    assert changed_result.iloc[2] == result.iloc[2]


def test_defcon_prefers_fpl_position_then_falls_back_to_fotmob(tmp_path):
    from src.features import compute_rolling_features
    raw = _dated_history(3)
    outfield = raw['position'].eq(2)
    raw.loc[outfield, ['tackles', 'interceptions', 'clearances', 'blocks', 'recoveries']] = [1, 2, 3, 4, 20]
    # Both are FotMob midfielders. The authoritative FPL DEF uses CBIT, while
    # the row without an FPL classification falls back to FotMob MID/CBIRT.
    raw.loc[raw['player_id'].eq(1), 'fpl_position'] = 'DEF'
    raw.loc[raw['player_id'].eq(3), 'fpl_position'] = pd.NA
    raw.attrs['data_dir'] = str(tmp_path)
    features = compute_rolling_features(raw, verbose=False)

    fantasy_def = features[features['player_id'].eq(1)]
    fantasy_mid = features[features['player_id'].eq(3)]
    assert fantasy_def['is_mid'].eq(1).all()
    assert fantasy_def['defcon_is_def'].eq(1).all()
    assert fantasy_def['defcon_position_source'].eq('FPL').all()
    assert fantasy_def['defcon'].eq(10).all()
    assert fantasy_def['defcon_threshold'].eq(10).all()
    assert fantasy_mid['defcon_is_mid'].eq(1).all()
    assert fantasy_mid['defcon_position_source'].eq('FotMob').all()
    assert fantasy_mid['defcon'].eq(30).all()
    assert fantasy_mid['defcon_threshold'].eq(12).all()


def test_defcon_probability_uses_fpl_position_and_excludes_other_roles():
    from scipy.stats import poisson
    from src.models.defcon import DefconModel
    model = DefconModel(n_estimators=1)
    model.predict = lambda frame: np.full(len(frame), 11.0)
    frame = pd.DataFrame({
        'defcon_position': ['DEF', 'MID', 'FWD', pd.NA],
        # Deliberately contradictory FotMob flags must have no effect.
        'is_def': [0, 1, 1, 1],
    })
    probability = model.predict_threshold_prob(frame)
    assert probability[0] == pytest.approx(1 - poisson.cdf(9, 11))
    assert probability[1] == pytest.approx(1 - poisson.cdf(11, 11))
    np.testing.assert_array_equal(probability[2:], [0.0, 0.0])


def test_shared_scoring_handles_missing_fpl_positions_and_uses_defcon_fallback():
    from src.models.bonus import score_simulations
    from src.pipeline import FPL_POINTS
    frame = pd.DataFrame({
        'fpl_position': [pd.NA, pd.NA, 'FWD', pd.NA],
        'defcon_position': ['DEF', 'MID', 'FWD', pd.NA],
    })
    zeros = np.zeros((1, len(frame)), dtype=int)
    simulations = {
        'minutes': np.full_like(zeros, 90),
        'goals': zeros.copy(),
        'assists': zeros.copy(),
        'cs': zeros.copy(),
        'goals_against': zeros.copy(),
        'saves': zeros.copy(),
        'defcon': np.array([[11, 12, 99, 99]]),
        'bonus': zeros.copy(),
        'yellows': zeros.copy(),
        'reds': zeros.copy(),
    }
    scored = score_simulations(frame, simulations, FPL_POINTS)
    np.testing.assert_array_equal(scored['exp_defcon_pts'], [[2, 2, 0, 0]])


def test_deadline_features_do_not_see_first_dgw_result_or_postponed_result(tmp_path):
    from src.features import chronological_frame, prior_stat, compute_rolling_features
    frame = _dated_history(5)
    # GW2 is delayed beyond GW4; GW3 has two fixtures whose features must agree.
    frame.loc[frame['gameweek'].eq(2), 'match_date'] = pd.Timestamp('2024-09-15', tz='UTC')
    extra = frame[frame['gameweek'].eq(3)].copy()
    extra['match_id'] = 33
    extra['match_date'] += pd.Timedelta(days=2)
    frame = pd.concat([frame, extra], ignore_index=True)
    frame.attrs['data_dir'] = str(tmp_path)
    features = compute_rolling_features(frame, verbose=False)
    selected = features[features['player_id'].eq(1) & features['gameweek'].eq(3)]
    assert len(selected) == 2
    assert selected['goals_last1'].eq(0).all()  # only GW1 has completed
    assert selected['current_season_apps'].eq(1).all()
    assert selected['last_minutes'].eq(90).all()

    changed = frame.copy()
    changed.loc[changed['gameweek'].ge(3), ['goals', 'xg', 'minutes']] = [8, 7., 1]
    changed.loc[changed['gameweek'].ge(3), [
        'accurate_passes', 'passes_attempted', 'saves_inside_box',
        'penalty_goals',
    ]] = [999, 1000, 20, 4]
    changed.attrs['data_dir'] = str(tmp_path)
    changed_features = compute_rolling_features(changed, verbose=False)
    columns = [c for c in features if '_roll' in c or c.startswith('lifetime_')]
    before = features[features['gameweek'].eq(3)].sort_values(['player_id', 'match_id'])
    after = changed_features[changed_features['gameweek'].eq(3)].sort_values(['player_id', 'match_id'])
    np.testing.assert_allclose(before[columns], after[columns], equal_nan=True)


def test_deadline_splits_keep_matches_whole_and_purge_late_results():
    from src.features import chronological_frame, deadline_splits
    frame = chronological_frame(_dated_history())
    frame.loc[frame['match_id'].eq(1), 'result_time'] = pd.Timestamp('2025-01-01', tz='UTC')
    folds = list(deadline_splits(frame, n_splits=3))
    assert len(folds) == 3
    for ti, vi in folds:
        train, valid = frame.iloc[ti], frame.iloc[vi]
        assert set(train['match_id']).isdisjoint(valid['match_id'])
        assert (train['result_time'] < valid['forecast_time'].min()).all()
        for match in valid['match_id'].unique():
            assert valid['match_id'].eq(match).sum() == frame['match_id'].eq(match).sum()
        assert not train['match_id'].eq(1).any()


def test_fixture_grid_retains_terminal_absence_but_not_blank_week():
    from src.features import build_appearance_grid
    frame = _dated_history(5)
    frame = frame[~frame['gameweek'].eq(3)]  # club has no match that GW
    frame = frame[~(frame['player_id'].eq(1) & frame['gameweek'].ge(4))]
    grid = build_appearance_grid(frame, verbose=False)
    player = grid[grid['player_id'].eq(1)]
    assert not player['gameweek'].eq(3).any()
    assert set(player['gameweek']) == {2, 4, 5}
    assert player.loc[player['gameweek'].ge(4), 'appeared'].eq(0).all()


def test_manager_basis_is_fitted_on_training_only_and_reused():
    from src.features import ManagerFeatureMixin, MANAGER_EMB_COLS
    rng = np.random.default_rng(8)
    columns = [f'manager_raw_{i}' for i in range(10)]
    train = pd.DataFrame(rng.normal(size=(30, 10)), columns=columns)
    train['manager_prior_games'] = 20
    model = ManagerFeatureMixin().fit_manager_features(train)
    original = model.manager_features(train)[MANAGER_EMB_COLS].to_numpy()
    scaler_mean = model.manager_basis[1].mean_.copy()
    future = train.copy()
    future[columns] += 10000
    model.manager_features(future)
    np.testing.assert_array_equal(scaler_mean, model.manager_basis[1].mean_)
    np.testing.assert_allclose(original, model.manager_features(train)[MANAGER_EMB_COLS])


def test_shared_clean_sheet_fit_matches_production_wrapper(tmp_path):
    from src.features import compute_rolling_features
    from src.pipeline import FPLPipeline
    raw = _dated_history()
    raw.attrs['data_dir'] = str(tmp_path)
    frame = compute_rolling_features(raw, verbose=False)
    params = {'n_estimators': 4, 'max_depth': 2, 'n_jobs': 1,
              'selected_features': ['prior_lambda', 'is_home']}
    teams = CleanSheetModel().prepare_team_features(frame)
    production = CleanSheetModel(**params).fit(frame, verbose=False)
    evaluation = FPLPipeline._fit_model('clean_sheet', teams, params)
    np.testing.assert_allclose(production.predict_goals_against(teams),
                               evaluation.predict_goals_against(teams))
    assert evaluation.selected_features == params['selected_features']


def test_oof_cold_start_never_uses_actual_minutes_and_fold_fits_are_past(monkeypatch):
    from src.pipeline import FPLPipeline
    from src.features import chronological_frame
    frame = chronological_frame(_dated_history())
    frame['minutes_roll5'] = 42.
    seen = []
    class Probe:
        def predict(self, valid):
            assert seen[-1] < valid['forecast_time'].min()
            return np.full(len(valid), 37.)
    def fit(name, training, params):
        seen.append(training['result_time'].max())
        return Probe()
    monkeypatch.setattr(FPLPipeline, '_fit_model', staticmethod(fit))
    result = FPLPipeline()._generate_oof_minutes(frame, {})
    assert seen
    assert set(result) == {42., 37.}
    assert not np.isin(result, frame['minutes'].unique()).any()


def test_train_predict_holdout_use_shared_wrappers_without_production_files(tmp_path, monkeypatch):
    from src.features import compute_rolling_features, chronological_frame
    from src.pipeline import FPLPipeline
    import src.pipeline as pipeline_module
    raw = _dated_history(9)
    raw.attrs['data_dir'] = str(tmp_path)
    pipeline = FPLPipeline(str(tmp_path), n_sims=40)
    pipeline.raw_df = chronological_frame(raw)
    pipeline.df = compute_rolling_features(raw, verbose=False)
    pipeline.current_season_players = set(raw['player_id'])
    pipeline.tuned_params = {
        name: {'n_estimators': 3, 'max_depth': 2, 'n_jobs': 1}
        for name in ('minutes', 'goals', 'assists', 'clean_sheet', 'defcon', 'saves')}
    pipeline.tuned_params['minutes']['appear_params'] = {'n_estimators': 3, 'n_jobs': 1}
    pipeline.train(verbose=False)
    assert not np.array_equal(pipeline.df['pred_minutes'], pipeline.df['minutes'])
    assert pipeline.models['minutes'].appear_is_fitted

    monkeypatch.setattr(pipeline_module, 'get_fpl_positions', lambda: {})
    monkeypatch.setattr(pipeline_module, 'get_fpl_availability', lambda: {})
    monkeypatch.setattr(pipeline_module, 'get_fpl_current_squads', lambda: ({}, {}))
    monkeypatch.setattr(pipeline, '_get_gw_fixtures', lambda *args: pd.DataFrame([{
        'home_team': 'Alpha', 'away_team': 'Beta', 'match_id': 99,
        'match_date': pd.Timestamp('2024-10-12', tz='UTC')}]))
    forecast = pipeline.predict(10, '2024/2025', verbose=False)
    assert len(forecast) == 4
    assert forecast['match_id'].nunique() == 1
    np.testing.assert_allclose(forecast['exp_total_pts'],
                               pipeline.last_simulations['total_points'].mean(axis=0))
    assert forecast['exp_total_pts_uncond'].equals(forecast['exp_total_pts'])
    frame = pipeline.df
    metrics = pipeline._evaluate_on_test_set(
        ['minutes', 'clean_sheet', 'goals', 'assists', 'defcon', 'saves'],
        frame[frame['gameweek'] <= 6], frame[frame['gameweek'] > 6], verbose=False)
    assert np.isfinite(metrics['_fpl_points_mae']['mae_ex_bonus'])
    assert all(np.isfinite(metrics[name]['primary']) for name in
               ('minutes', 'clean_sheet', 'goals', 'assists', 'defcon', 'saves'))


def test_fast_tuning_keeps_oof_dependencies_causal(tmp_path, monkeypatch):
    from src.features import compute_rolling_features
    from src.pipeline import FPLPipeline
    raw = _dated_history(7)
    raw.attrs['data_dir'] = str(tmp_path)
    frame = compute_rolling_features(raw, verbose=False)
    pipeline = FPLPipeline(str(tmp_path), n_sims=10)
    original = FPLPipeline._fit_model
    training_deadlines = []
    def fit(name, rows, params=None):
        training_deadlines.append(rows['forecast_time'].max())
        return original(name, rows, {**(params or {}), 'n_estimators': 2, 'n_jobs': 1})
    monkeypatch.setattr(FPLPipeline, '_fit_model', staticmethod(fit))
    params, scores = pipeline._tune_in_process(['clean_sheet'], 1, False, frame)
    assert len(set(training_deadlines)) > 1
    assert max(training_deadlines) < frame['forecast_time'].max()
    assert np.isfinite(scores['clean_sheet'])
    assert params['clean_sheet']['selected_features']


def test_spawned_tuning_matches_in_process(tmp_path):
    from src.pipeline import FPLPipeline
    raw = _dated_history(5)
    in_process = FPLPipeline(str(tmp_path))
    spawned = FPLPipeline(str(tmp_path))
    expected_params, expected_scores = in_process._tune_in_process(['clean_sheet'], 1, False, raw)
    actual_params, actual_scores = spawned._tune_with_subprocess(['clean_sheet'], 1, False, raw)
    assert expected_params == actual_params
    np.testing.assert_allclose(expected_scores['clean_sheet'], actual_scores['clean_sheet'])


def test_manager_missing_historical_cache_never_uses_future_coach(tmp_path):
    from src.features import add_manager_embeddings, chronological_frame
    raw = chronological_frame(_dated_history(6))
    metadata = raw[['match_id', 'match_date']].drop_duplicates()
    (tmp_path / 'matches').mkdir()
    metadata.to_csv(tmp_path / 'matches' / 'match_details.csv', index=False)
    managers = pd.DataFrame({
        'match_id': [1, 3, 4, 5, 6], 'home_team': ['Alpha'] * 5, 'away_team': ['Beta'] * 5,
        'home_manager_id': [1, 1, 2, 2, 2], 'home_manager_name': ['old', 'old', 'new', 'new', 'new'],
        'away_manager_id': [3] * 5, 'away_manager_name': ['other'] * 5,
        'home_formation': ['4-3-3'] * 5, 'away_formation': ['4-4-2'] * 5})
    managers.to_csv(tmp_path / 'match_managers.csv', index=False)
    initial = add_manager_embeddings(raw, str(tmp_path), verbose=False)
    altered = raw.copy()
    altered.loc[altered['gameweek'].ge(3), ['goals', 'minutes']] = [7, 1]
    revised = add_manager_embeddings(altered, str(tmp_path), verbose=False)
    cols = [c for c in initial if c.startswith('manager_raw_')] + ['manager_prior_games']
    first = initial[initial['gameweek'].eq(2)].sort_values('player_id')
    second = revised[revised['gameweek'].eq(2)].sort_values('player_id')
    np.testing.assert_allclose(first[cols], second[cols])
    assert first['manager_prior_games'].eq(1).all()


def test_explicit_registration_includes_predebut_and_excludes_departure():
    from src.features import build_appearance_grid
    raw = _dated_history(5)
    raw = raw[~(raw['player_id'].eq(1) & raw['gameweek'].eq(1))].copy()
    raw.attrs['roster_spells'] = pd.DataFrame([{
        'player_id': 1, 'team': 'Alpha', 'season': '2024/2025',
        'start_date': '2024-08-01', 'end_date': '2024-08-30'}])
    grid = build_appearance_grid(raw, verbose=False)
    assert set(grid['gameweek']) == {1, 2, 3, 4}
    assert grid.loc[grid['gameweek'].eq(1), 'appeared'].eq(0).all()
    assert not grid['gameweek'].eq(5).any()
