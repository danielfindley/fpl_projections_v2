import unittest

import numpy as np
import pandas as pd

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
