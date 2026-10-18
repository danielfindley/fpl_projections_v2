import numpy as np
import pandas as pd

from src.models.sequence import TOKEN_COLUMNS, EventBlend, build_history


def test_history_only_uses_results_known_at_the_deadline():
    t = pd.Timestamp('2026-01-01', tz='UTC')
    pool = pd.DataFrame({
        'player_id': [1, 1, 1, 1],
        'team': ['a', 'a', 'b', 'b'],
        'minutes': [90, 45, 90, 90],
        'goals': [1, 2, 3, 4],
        # Match 3 kicks off before the deadline but finishes after it; match 4 is
        # the row itself. Neither may appear in the row's history.
        'result_time': [t, t + pd.Timedelta(days=7), t + pd.Timedelta(days=13, hours=23),
                        t + pd.Timedelta(days=21)],
    })
    rows = pd.DataFrame({'player_id': [1, 2], 'team': ['b', 'b'],
                         'forecast_time': [t + pd.Timedelta(days=13, hours=22)] * 2})

    tokens, mask = build_history(pool, rows, history=3)

    goals = TOKEN_COLUMNS.index('goals')
    assert mask[0].tolist() == [True, False, False]
    assert tokens[0, 1:, goals].tolist() == [1, 2]
    # Most recent match last; neither prior match was for the current club.
    assert tokens[0, -1, -2] < tokens[0, -2, -2]
    assert tokens[0, 1:, -1].tolist() == [0, 0]
    # Unknown player: fully padded, all zeros.
    assert mask[1].all() and not tokens[1].any()


def test_event_blend_mixes_neural_rate_times_minutes_with_xgboost():
    from types import SimpleNamespace
    rows = pd.DataFrame({'pred_minutes': [90., 45., 90.],
                         'defcon_position': ['DEF', 'MID', 'FWD'],
                         'defcon': [12, 4, 0], 'minutes': [90, 45, 90]})
    heads = pd.DataFrame({'goals_per90': [.6, .6, .6], 'defcon_per90': [10., 10., 10.],
                          'saves_per90': [20., 3., 3.]}, index=rows.index)
    player = SimpleNamespace(predict_heads=lambda df: heads.loc[df.index])
    base = SimpleNamespace(TARGET='goals', predict=lambda df: np.full(len(df), .2),
                           predict_per90=lambda df: np.full(len(df), 1.),
                           feature_importance=lambda: 'xgb importance')

    goals = EventBlend(player, base, 'goals', .5, rows)
    # Neural expected goals = rate x pred_minutes / 90; blended halfway with XGB.
    np.testing.assert_allclose(goals.predict(rows), [.4, .25, .4])
    assert goals.TARGET == 'goals' and goals.feature_importance() == 'xgb importance'

    defcon = EventBlend(player, SimpleNamespace(**{**vars(base), 'TARGET': 'defcon'}), 'defcon', 1., rows)
    # Only resolved DEF/MID rows carry a defcon expectation.
    np.testing.assert_allclose(defcon.predict(rows), [10., 5., 0.])
    assert defcon.dispersion_r is None or defcon.dispersion_r > 0

    saves = EventBlend(player, base, 'saves', .5, rows)
    # The neural per-90 side is capped at 12, as SavesModel caps its own.
    np.testing.assert_allclose(saves.predict_per90(rows), [6.5, 2., 2.])
    np.testing.assert_allclose(saves.predict_expected_saves(rows, [90, 45, 0]), [6.5, 1., 0.])
