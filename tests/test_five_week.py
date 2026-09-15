import numpy as np
import pandas as pd

from src.five_week import (
    WeekForecast,
    apply_live_availability,
    build_horizon_training_frame,
    current_durability,
)
from src.five_week_viz import build_report_payload, render_five_week_html


def _history():
    return pd.DataFrame({
        "player_id": [7] * 6,
        "season": ["2024/2025"] * 6,
        "gameweek": [1, 2, 3, 4, 5, 6],
        "minutes": [90, 0, 20, 70, 88, 0],
        "is_gk": [0] * 6,
        "is_def": [0] * 6,
        "is_mid": [1] * 6,
        "is_fwd": [0] * 6,
    })


def test_direct_horizon_target_is_future_week_and_features_are_shifted():
    frame, _ = build_horizon_training_frame(_history())
    row = frame[(frame["gameweek"] == 2) & (frame["horizon"] == 3)].iloc[0]

    assert row["target_gameweek"] == 4
    assert row["target_minutes"] == 70
    assert row["role_state"] == 2
    assert row["last_minutes"] == 90


def test_current_durability_includes_latest_observed_week():
    _, grid = build_horizon_training_frame(_history())
    durability = current_durability(grid).iloc[0]

    assert durability["career_appear_rate"] == 4 / 6
    assert durability["appear_rate_roll20"] == 4 / 6


def test_live_zero_chance_recovers_across_horizons():
    probability = np.array([[0.1, 0.2, 0.3, 0.4]])
    h1 = apply_live_availability(probability, pd.Series([0]), horizon=1)
    h5 = apply_live_availability(probability, pd.Series([0]), horizon=5)

    np.testing.assert_allclose(h1, [[1.0, 0.0, 0.0, 0.0]], atol=1e-8)
    assert h5[0, 0] < 0.2
    np.testing.assert_allclose(h5.sum(axis=1), [1.0])


def test_report_is_standalone_sortable_and_has_distribution_modal(tmp_path):
    forecasts = {}
    for gameweek in range(3, 8):
        frame = pd.DataFrame([{
            "player_id": 7,
            "player_name": "Example Player",
            "team": "Example FC",
            "fpl_position": "MID",
            "opponent": "Opponent FC",
            "is_home": gameweek % 2,
            "pred_appear_prob": 0.9,
            "pred_minutes_uncond": 72,
            "pred_minutes": 80,
            "pred_exp_goals": 0.2,
            "pred_exp_assists": 0.15,
            "pred_cs_prob": 0.3,
            "role_prob_0": 0.1,
            "role_prob_1": 0.1,
            "role_prob_2": 0.2,
            "role_prob_3": 0.6,
        }])
        forecasts[gameweek] = WeekForecast(frame, {0: np.array([0, 2, 4, 8, 10], dtype=float)})

    metrics = {
        "selected": "durability",
        "durability": {"multiclass_log_loss": 0.9},
    }
    payload = build_report_payload(forecasts, "2026/2027", metrics)
    output = render_five_week_html(payload, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert "fixtureFilter" in html
    assert "data-scope=\"average\"" in html
    assert "distributionModal" in html
    assert "Example Player" in html
    assert payload["players"][0]["five_week_total"] == 24.0



def _match_forecast():
    return pd.DataFrame({
        "player_id": range(4), "player_name": ["A forward", "A defender", "B forward", "B defender"],
        "team": ["Alpha", "Alpha", "Beta", "Beta"], "opponent": ["Beta", "Beta", "Alpha", "Alpha"],
        "match_id": [1] * 4, "season": ["2024/2025"] * 4, "gameweek": [1] * 4,
        "is_home": [1, 1, 0, 0],
        "fpl_position": ["FWD", "DEF", "FWD", "DEF"],
        "pred_minutes": [90.] * 4, "pred_appear_prob": [1.] * 4,
        "pred_team_goals": [1.5, 1.5, .8, .8], "pred_goals_against": [.8, .8, 1.5, 1.5],
        "pred_cs_prob": [.45, .45, .25, .25],
        "pred_exp_goals": [1.5, 0, .8, 0], "pred_exp_assists": [1., 1., .5, .5],
        "pred_exp_defcon": [3., 12., 3., 12.], "pred_yellow_prob": [.1] * 4,
        "pred_red_prob": [.01] * 4, "pred_exp_saves": [0.] * 4, "minutes_roll5": [90.] * 4,
    })


def _bonus_model(n=4000):
    from src.models.bonus import BonusModel
    model = BonusModel(n_simulations=n)
    class Baseline:
        def predict(self, frame):
            return np.full(len(frame), 15.)
    model.baseline_model = Baseline()
    model.is_fitted = True
    return model


def test_bonus_ties_skip_occupied_ranks_and_absent_players():
    from src.models.bonus import BonusModel
    bps = np.array([[30, 30, 20, 10], [30, 20, 20, 10], [30, 30, 30, 20], [-1, -2, -3, 100]])
    playing = np.ones(bps.shape, dtype=bool)
    playing[-1, -1] = False
    np.testing.assert_array_equal(BonusModel.award_bonus(bps, playing),
                                  [[3, 3, 1, 0], [3, 2, 2, 0], [3, 3, 3, 0], [3, 2, 1, 0]])


def test_match_events_conserve_goals_share_clean_sheets_and_reproduce():
    frame, model = _match_forecast(), _bonus_model()
    probability = np.tile([0., 0., 1.], (4, 1))
    minutes = np.tile([0., 20., 90.], (4, 1))
    draws = model.simulate(frame, probability, minutes, seed=7, defcon_r=4)
    for a, b in [(0, 1), (2, 3)]:
        np.testing.assert_array_equal(draws['goals_against'][:, a], draws['goals_against'][:, b])
        np.testing.assert_array_equal(draws['cs'][:, a], draws['cs'][:, b])
        np.testing.assert_array_equal(draws['goals'][:, [a, b]].sum(axis=1), draws['team_goals'][:, a])
        assert (draws['assists'][:, [a, b]].sum(axis=1) <= draws['team_goals'][:, a]).all()
        # All team's goals belong to its only scorer: that player cannot self-assist.
        assert not draws['assists'][:, a].any()
    np.testing.assert_array_equal(draws['goals_against'][:, 0], draws['team_goals'][:, 2])
    assert (draws['cs'][draws['goals_against'] > 0] == 0).all()
    again = model.simulate(frame, probability, minutes, seed=7, defcon_r=4)
    for key in draws:
        np.testing.assert_array_equal(draws[key], again[key])


def test_absent_players_have_zero_events_and_points():
    from src.models.bonus import score_simulations
    from src.pipeline import FPL_POINTS
    frame, model = _match_forecast(), _bonus_model(200)
    probability = np.tile([1., 0., 0.], (4, 1))
    draws = model.simulate(frame, probability, [0., 20., 90.])
    score_simulations(frame, draws, FPL_POINTS)
    for key in ('minutes', 'goals', 'assists', 'cs', 'yellows', 'reds', 'saves', 'defcon', 'bonus', 'total_points'):
        assert not draws[key].any()


def test_minutes_mixture_has_no_sixty_minute_expectation_cliff():
    from src.models.minutes import MinutesModel
    from src.pipeline import FPLPipeline
    from types import SimpleNamespace
    model = MinutesModel()
    model.is_fitted = True
    model.classifier = SimpleNamespace(predict_proba=lambda frame: np.full(len(frame), .5))
    model.starter_model = SimpleNamespace(predict=lambda frame: np.full(len(frame), 90.))
    model.sub_model = SimpleNamespace(predict=lambda frame: np.full(len(frame), 20.))
    frame = _match_forecast()
    frame['pred_minutes'] = model.predict(frame)
    assert frame['pred_minutes'].eq(55.).all()
    probability, states = model.predict_distribution(frame)
    pipeline = FPLPipeline(n_sims=12000)
    pipeline.models = {'bonus': _bonus_model(12000), 'defcon': SimpleNamespace(dispersion_r=4.)}
    scored, draws = pipeline._simulate_points(frame, probability, states)
    np.testing.assert_allclose(scored['exp_appearance_pts'], 1.5, atol=.02)
    assert (scored.loc[[1, 3], 'exp_cs_pts'] > 0).all()
    np.testing.assert_allclose(scored['exp_total_pts'], draws['total_points'].mean(axis=0))
    probability, _ = model.predict_distribution(frame, lineup_p_start=np.ones(4), lineup_weight=1.)
    np.testing.assert_allclose(probability[:, 2], .9)  # start != 60+


def test_chart_consumes_saved_totals_and_dgw_sum_without_resampling(tmp_path):
    from src.viz import _build_player_data, generate_distribution_html
    frame = _match_forecast().iloc[:2].copy()
    frame['player_id'] = 1
    frame['player_name'] = 'Same player'
    frame['fpl_position'] = 'DEF'
    frame['fixture_num'] = [1, 2]
    frame['_sim_idx'] = [0, 1]
    totals = np.array([[0., 2.], [2., 4.], [8., 0.], [3., 2.]])
    prediction = frame.iloc[:1].copy()
    prediction['exp_total_pts'] = totals.sum(axis=1).mean()
    sims = {key: np.zeros((4, 2)) for key in ('goals', 'assists', 'cs', 'bonus')}
    sims['total_points'] = totals
    state = np.random.get_state()
    players, distributions = _build_player_data(prediction, sims, 10, frame)
    np.testing.assert_array_equal(distributions[0], totals.sum(axis=1))
    assert players[0]['exp_pts'] == prediction['exp_total_pts'].iloc[0]
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    output = tmp_path / 'distributions.html'
    generate_distribution_html(prediction, sims, output_path=output,
                               gameweek=4, predictions_per_fixture=frame,
                               squad={'sentinel': 'must be ignored'})
    html = output.read_text(encoding='utf-8')
    assert '4 Monte Carlo simulations' in html
    assert '__N_SIMS__' not in html
    assert 'including non-appearances' in html
    assert 'P10-P90 range' in html
    assert 'const stats = `E[pts]=' not in html
    assert '.text(stats)' not in html
    assert '90% CI' not in html
    assert 'Optimal Squad' not in html
    assert '__SQUAD__' not in html
    assert 'sq-wrap' not in html


def test_distribution_support_panels_show_samples_review_and_fixtures():
    from src.viz import (
        _build_fixture_forecasts_html,
        _build_last_gw_html,
        _build_metrics_html,
    )

    frame = _match_forecast()
    fixture_html = _build_fixture_forecasts_html(frame, gameweek=2)
    assert 'GW2 Fixture Forecasts' in fixture_html
    assert 'Alpha' in fixture_html and 'Beta' in fixture_html
    assert '1.50 &ndash; 0.80' in fixture_html
    assert 'CS 45%' in fixture_html and 'CS 25%' in fixture_html
    assert 'before player-minute risk' in fixture_html

    metrics = {
        'sections': [{
            'title': 'Sub-Models (Holdout Test Set)',
            'rows': [
                {'model': 'Minutes', 'metric': 'Huber Loss', 'score': '121.1772'},
                {'model': 'Goals', 'metric': 'Poisson Dev', 'score': '0.4000'},
            ],
        }],
        'calibration': None,
        'sample_info': {
            'evaluation': {
                'train_player_rows': 66797,
                'train_team_sides': 4616,
                'train_span': '2020/2021 through 2026/2027 GW3',
                'test_player_rows': 307,
                'test_team_sides': 20,
                'test_span': '2026/2027 GW4',
            },
            'models': {
                'goals': {'train': 66797, 'test': 307, 'unit': 'player apps'},
                'minutes': {
                    'train': 66797, 'test': 307, 'unit': 'played apps',
                    'detail': 'appearance grid: 106,087 (42,802 DNP)',
                },
            },
            'final_training': {
                'player_rows': 67104,
                'team_sides': 4636,
                'span': '2020/2021 through 2026/2027 GW4',
                'appearance_grid_rows': 106474,
                'appearance_grid_nonappearances': 42901,
            },
        },
    }
    metrics_html = _build_metrics_html(
        metrics, predictions=frame, predictions_per_fixture=frame, gameweek=2)
    assert 'Evaluation &amp; Training Samples' in metrics_html
    assert '66,797 &rarr; 307' in metrics_html
    assert 'appearance grid: 106,087 (42,802 DNP)' in metrics_html
    assert '67,104 appearances' in metrics_html
    assert '106,474 eligible player-fixture rows' in metrics_html
    assert '4 current players across 1 fixtures' in metrics_html

    review_html = _build_last_gw_html({
        'gameweek': 1,
        'rows': [{
            'player_name': 'Previous Star', 'team': 'Alpha', 'position': 'MID',
            'predicted': 6.2, 'actual': 8.0, 'minutes': 90,
        }],
        'mean_predicted': 6.2,
        'mean_actual': 8.0,
    })
    assert 'GW1 Top 10' in review_html
    assert 'Previous Star' in review_html


def test_five_week_calls_same_simulator():
    from src.five_week import FiveWeekForecaster
    from src.pipeline import FPLPipeline
    from types import SimpleNamespace
    frame = _match_forecast()
    probability = np.tile([.1, .1, .2, .6], (4, 1))
    states = np.array([0., 20., 70., 88.])
    pipeline = FPLPipeline(n_sims=100)
    pipeline.models = {'bonus': _bonus_model(100), 'defcon': SimpleNamespace(dispersion_r=4.)}
    forecaster = object.__new__(FiveWeekForecaster)
    forecaster.pipeline, forecaster.n_sims, forecaster.seed = pipeline, 100, 42
    forecaster.role_model = SimpleNamespace(state_minutes=states)
    horizon_draws = forecaster._simulate(frame, probability, 5)
    _, weekly_draws = pipeline._simulate_points(frame, probability, states,
                                                n_simulations=100, seed=42 + 5 * 1009)
    for i in range(4):
        np.testing.assert_array_equal(horizon_draws[i], weekly_draws['total_points'][:, i])


def test_saved_dgw_arrays_sum_by_player_id_without_fixture_frame():
    from src.viz import _build_player_data
    frame = _match_forecast().iloc[:1].copy()
    frame['exp_total_pts'] = 4.
    draws = np.array([[0., 2.], [2., 4.]])
    sims = {key: np.zeros((2, 2)) for key in ('goals', 'assists', 'cs', 'bonus')}
    sims.update(total_points=draws, player_names=[frame.iloc[0]['player_name']] * 2,
                player_ids=[frame.iloc[0]['player_id']] * 2)
    players, totals = _build_player_data(frame, sims, 10)
    np.testing.assert_array_equal(totals[0], [2., 6.])
    assert players[0]['exp_pts'] == 4.


def test_optimizer_does_not_discount_new_points_twice():
    from src.optimizer import optimize_squad
    rows = []
    for position, count in [('GK', 2), ('DEF', 5), ('MID', 5), ('FWD', 3)]:
        for i in range(count):
            rows.append({'player_id': len(rows), 'player_name': f'{position}{i}',
                         'fpl_position': position, 'team': f'Club{len(rows) // 3}',
                         'price': 4., 'pred_appear_prob': .5,
                         'exp_total_pts': 4., 'exp_total_pts_uncond': 4.})
    frame = pd.DataFrame(rows)
    result = optimize_squad(frame, verbose=False)
    assert result['squad']['exp_pts_uncond'].eq(4.).all()
    legacy = optimize_squad(frame.drop(columns='exp_total_pts_uncond'), verbose=False)
    assert legacy['squad']['exp_pts_uncond'].eq(2.).all()


def test_lineup_probability_is_joint_and_availability_is_applied_once():
    from src.models.minutes import MinutesModel
    from types import SimpleNamespace
    model = MinutesModel()
    model.is_fitted = True
    model.classifier = SimpleNamespace(predict_proba=lambda frame: np.full(len(frame), .5))
    model.starter_model = SimpleNamespace(predict=lambda frame: np.full(len(frame), 90.))
    model.sub_model = SimpleNamespace(predict=lambda frame: np.full(len(frame), 20.))
    frame = _match_forecast()
    probability, _ = model.predict_distribution(
        frame, appear_prob=np.full(4, .2), lineup_p_start=np.ones(4),
        lineup_weight=1., availability=np.array([1., .5, 0., 1.]))
    np.testing.assert_allclose(probability[0], [0., .1, .9])
    np.testing.assert_allclose(probability[1], [.5, .05, .45])
    np.testing.assert_allclose(probability[2], [1., 0., 0.])
    np.testing.assert_allclose(probability.sum(axis=1), 1.)


def test_bonus_baseline_removes_shared_penalties_before_simulation():
    from src.models.bonus import BaselineBPSModel
    frame = pd.DataFrame({
        'position': [1], 'minutes': [90], 'goals': [1], 'assists': [0],
        'opponent_goals': [2], 'yellow_cards': [1], 'red_cards': [0], 'bps': [10.]})
    # DEF goal +12, conceded -8, yellow -3: events contribute +1, baseline is 9.
    np.testing.assert_allclose(BaselineBPSModel()._compute_baseline_bps(frame), [9.])


def test_bonus_baseline_uses_2026_penalty_and_goalkeeper_save_rules():
    from src.models.bonus import BaselineBPSModel
    frame = pd.DataFrame({
        'season': ['2026/2027', '2026/2027'],
        'fpl_position': ['FWD', 'GK'],
        'minutes': [90, 90], 'goals': [1, 0], 'penalty_goals': [1, 0],
        'assists': [0, 0], 'opponent_goals': [1, 1],
        'yellow_cards': [0, 0], 'red_cards': [0, 0],
        'saves': [0, 4], 'saves_inside_box': [0, 3],
        'saved_penalties': [0, 0], 'bps': [8., 5.],
    })
    # Penalty goal is +12 for a forward. GK events are -4 conceded +8 saves
    # +3 inside-box saves, so both residual baselines are allowed to be -4/-2.
    np.testing.assert_allclose(
        BaselineBPSModel()._compute_baseline_bps(frame), [-4., -2.])


def test_bonus_simulation_scores_penalties_and_goalkeeper_saves_in_bps():
    frame, model = _match_forecast(), _bonus_model(300)
    frame['season'] = '2026/2027'
    frame['penalty_goal_share_roll10'] = [1., 0., 0., 0.]
    frame['inside_box_save_share_roll5'] = [0., 1., 0., 0.]
    frame.loc[0, 'fpl_position'] = 'FWD'
    frame.loc[1, 'fpl_position'] = 'GK'
    frame.loc[1, 'pred_exp_saves'] = 5.
    probability = np.tile([0., 0., 1.], (4, 1))
    states = np.tile([0., 20., 90.], (4, 1))

    draws = model.simulate(frame, probability, states, seed=19)

    np.testing.assert_array_equal(draws['penalty_goals'][:, 0], draws['goals'][:, 0])
    np.testing.assert_array_equal(draws['saves_inside_box'][:, 1], draws['saves'][:, 1])
    np.testing.assert_array_equal(draws['save_bps'][:, 1], 3 * draws['saves'][:, 1])
