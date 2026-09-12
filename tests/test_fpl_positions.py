import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data_loader import (
    fetch_fpl_actual_points, get_fpl_positions, map_fpl_position,
    merge_fpl_card_data,
)


class FPLPositionMatchingTests(unittest.TestCase):
    @staticmethod
    def _bootstrap():
        return {
            "elements": [
                {
                    "first_name": "Victor",
                    "second_name": "Munoz",
                    "web_name": "Munoz",
                    "element_type": 3,
                },
                {
                    "first_name": "Daniel",
                    "second_name": "Muñoz Mejía",
                    "web_name": "Muñoz",
                    "element_type": 2,
                },
            ]
        }

    @patch("requests.get")
    def test_shared_surname_does_not_cross_assign_position(self, mock_get):
        response = Mock()
        response.json.return_value = self._bootstrap()
        mock_get.return_value = response

        positions = get_fpl_positions()

        self.assertEqual(positions["victor munoz"], "MID")
        self.assertNotIn("munoz", positions)
        self.assertEqual(
            map_fpl_position(3, "Víctor Muñoz", positions),
            "MID",
        )

    @patch("requests.get")
    def test_compound_fpl_surname_matches_shorter_fotmob_full_name(self, mock_get):
        response = Mock()
        response.json.return_value = self._bootstrap()
        mock_get.return_value = response

        positions = get_fpl_positions()

        self.assertEqual(
            map_fpl_position(2, "Daniel Muñoz", positions),
            "DEF",
        )


    def test_strict_mode_never_falls_back_to_fotmob_role(self):
        self.assertTrue(pd.isna(map_fpl_position(
            1, "Unknown Player", {}, fallback_to_fotmob=False)))
        self.assertEqual(map_fpl_position(1, "Unknown Player", {}), "DEF")

    @patch("requests.get", side_effect=OSError("offline"))
    def test_actual_points_uses_cache_when_fpl_api_is_offline(self, _mock_get):
        cached = pd.DataFrame([{
            'player_name': 'Anan Khalaili', 'web_name': 'Khalaili',
            'team': 'Crystal Palace', 'season': '2026/2027', 'gameweek': 1,
            'fpl_position': 'DEF',
        }])
        with TemporaryDirectory() as directory:
            cached.to_csv(Path(directory) / 'fpl_actual_points.csv', index=False)
            result = fetch_fpl_actual_points(cache_dir=directory, verbose=False)
        self.assertEqual(result.iloc[0]['fpl_position'], 'DEF')

    @patch("src.data_loader.fetch_fpl_actual_points")
    def test_cached_position_applies_to_every_row_in_its_season(self, fetch):
        fetch.return_value = pd.DataFrame([{
            'player_name': 'Anan Khalaili', 'web_name': 'Khalaili',
            'team': 'Crystal Palace', 'season': '2026/2027', 'gameweek': 1,
            'fpl_position': 'DEF', 'yellow_cards': 0, 'red_cards': 0,
            'bonus': 0, 'actual_total_points': 1,
        }])
        raw = pd.DataFrame({
            'player_name': ['Anan Khalaili', 'Anan Khalaili'],
            'team': ['Crystal Palace', 'Crystal Palace'],
            'season': ['2026/2027', '2026/2027'],
            'gameweek': [1, 2],
        })
        merged = merge_fpl_card_data(raw, verbose=False)
        self.assertEqual(merged['fpl_position'].tolist(), ['DEF', 'DEF'])


if __name__ == "__main__":
    unittest.main()
