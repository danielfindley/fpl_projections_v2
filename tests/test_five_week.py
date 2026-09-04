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
