"""Interactive HTML distribution visualization for FPL predictions.

Generates a standalone HTML file with a D3.js ridge plot showing
Monte Carlo points distributions for top outfield players.
"""

import json
import re
from copy import deepcopy
from html import escape
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from pathlib import Path


def _sample_count(value):
    """Format a sample count without turning a missing value into zero."""
    return f'{int(value):,}' if value is not None else '&mdash;'


def _build_sample_html(sample_info, predictions=None,
                       predictions_per_fixture=None, gameweek=None):
    """Explain the temporal holdout and the separate final-production fit."""
    if not sample_info:
        return ''

    evaluation = sample_info.get('evaluation') or {}
    final = sample_info.get('final_training') or {}

    prediction_players = len(predictions) if predictions is not None else final.get('prediction_players')
    if predictions_per_fixture is not None and len(predictions_per_fixture):
        if 'match_id' in predictions_per_fixture:
            prediction_fixtures = predictions_per_fixture['match_id'].nunique()
        else:
            prediction_fixtures = final.get('prediction_fixtures')
    else:
        prediction_fixtures = final.get('prediction_fixtures')

    season = None
    if predictions is not None and 'season' in predictions:
        seasons = predictions['season'].dropna().astype(str)
        if len(seasons):
            season = seasons.iloc[0]
    target = final.get('target') or (
        f'{season} GW{gameweek}' if season and gameweek else f'GW{gameweek}' if gameweek else 'the forecast gameweek'
    )

    evaluation_text = (
        f'<strong>Held-out evaluation:</strong> fit on '
        f'{_sample_count(evaluation.get("train_player_rows"))} player-match appearances '
        f'({_sample_count(evaluation.get("train_team_sides"))} team-sides), '
        f'{escape(str(evaluation.get("train_span", "historical data")))}. '
        f'Scored on {_sample_count(evaluation.get("test_player_rows"))} players who appeared in '
        f'{escape(str(evaluation.get("test_span", "the holdout")))} '
        f'({_sample_count(evaluation.get("test_team_sides"))} team-sides). '
        f'The holdout outcomes were never available to the evaluation fit.'
    )
    final_text = (
        f'<strong>Final {escape(str(target))} forecast:</strong> refit on all '
        f'{_sample_count(final.get("player_rows"))} appearances '
        f'({_sample_count(final.get("team_sides"))} team-sides), '
        f'{escape(str(final.get("span", "including the completed holdout")))}.'
    )
    if final.get('appearance_grid_rows') is not None:
        final_text += (
            f' The appearance classifier uses '
            f'{_sample_count(final.get("appearance_grid_rows"))} eligible player-fixture rows'
        )
        if final.get('appearance_grid_nonappearances') is not None:
            final_text += (
                f' ({_sample_count(final.get("appearance_grid_nonappearances"))} non-appearances)'
            )
        final_text += '.'
    if prediction_players is not None:
        final_text += f' Predictions cover {_sample_count(prediction_players)} current players'
        if prediction_fixtures is not None:
            final_text += f' across {_sample_count(prediction_fixtures)} fixtures'
        final_text += '.'

    return f'''
<div style="margin:18px 0 0;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 14px">
  <h3 style="color:#e0e0e0;text-align:center;margin:0 0 8px;font-size:14px;font-weight:600">Evaluation &amp; Training Samples</h3>
  <p style="font-size:11px;line-height:1.55;color:#8b949e;margin:0 0 7px">{evaluation_text}</p>
  <p style="font-size:11px;line-height:1.55;color:#8b949e;margin:0">{final_text}</p>
</div>'''


def _build_metrics_html(metrics, predictions=None, predictions_per_fixture=None,
                        gameweek=None):
    """Build HTML for sub-model + overall-points metrics + calibration plot.

    Accepts either the legacy list-of-rows format (single section, no
    calibration) or the new dict ``{'sections': [...], 'calibration': [...]}``.
    """
    if not metrics:
        return ''

    if isinstance(metrics, list):
        sections = [{'title': 'Model Accuracy (Holdout Test Set)', 'rows': metrics}]
        calibration = None
        sample_info = None
    else:
        sections = metrics.get('sections') or []
        calibration = metrics.get('calibration')
        sample_info = metrics.get('sample_info')

    parts = ['<div style="max-width:620px;margin:32px auto 16px;padding:0 16px">']
    if sample_info:
        parts.append(_build_sample_html(
            sample_info, predictions, predictions_per_fixture, gameweek))
    model_samples = (sample_info or {}).get('models') or {}
    for section in sections:
        rows = section.get('rows') or []
        if not rows:
            continue
        show_model = any(r.get('model') for r in rows)
        show_sample = show_model and bool(model_samples)
        body = ''
        for r in rows:
            if show_model:
                if show_sample:
                    sample = model_samples.get(str(r.get('model', '')).lower(), {})
                    unit = escape(str(sample.get('unit', 'rows')))
                    detail = sample.get('detail')
                    detail_html = (
                        f'<br><span style="color:#666;font-size:10px">'
                        f'{escape(str(detail))}</span>' if detail else ''
                    )
                    body += (
                        f'<tr><td style="padding:6px 8px">{r.get("model","")}</td>'
                        f'<td style="padding:6px 8px">{r["metric"]}'
                        f'<br><span style="color:#e6edf3;font-weight:600;font-variant-numeric:tabular-nums">{r["score"]}</span></td>'
                        f'<td style="padding:6px 8px;text-align:right;'
                        f'font-variant-numeric:tabular-nums">'
                        f'<span style="white-space:nowrap">{_sample_count(sample.get("train"))} '
                        f'&rarr; {_sample_count(sample.get("test"))}</span>'
                        f'<br><span style="color:#666;font-size:10px">{unit}</span>'
                        f'{detail_html}'
                        f'</td>'
                        f'</tr>\n'
                    )
                else:
                    body += (
                        f'<tr><td style="padding:6px 12px">{r.get("model","")}</td>'
                        f'<td style="padding:6px 12px">{r["metric"]}</td>'
                        f'<td style="padding:6px 12px;text-align:right;font-variant-numeric:tabular-nums">{r["score"]}</td></tr>\n'
                    )
            else:
                body += (
                    f'<tr><td style="padding:6px 12px">{r["metric"]}</td>'
                    f'<td style="padding:6px 12px;text-align:right;font-variant-numeric:tabular-nums">{r["score"]}</td></tr>\n'
                )
        if show_model:
            if show_sample:
                columns = '<colgroup><col style="width:24%"><col style="width:38%"><col style="width:38%"></colgroup>'
                head = (
                    '<th style="padding:8px;text-align:left">Model</th>'
                    '<th style="padding:8px;text-align:left">Test metric</th>'
                    '<th style="padding:8px;text-align:right;white-space:nowrap">Train &rarr; test</th>'
                )
            else:
                columns = '<colgroup><col style="width:34%"><col style="width:38%"><col style="width:28%"></colgroup>'
                head = (
                    '<th style="padding:8px 12px;text-align:left">Model</th>'
                    '<th style="padding:8px 12px;text-align:left">Metric</th>'
                    '<th style="padding:8px 12px;text-align:right">Test</th>'
                )
        else:
            columns = '<colgroup><col style="width:72%"><col style="width:28%"></colgroup>'
            head = (
                '<th style="padding:8px 12px;text-align:left">Metric</th>'
                '<th style="padding:8px 12px;text-align:right">Test</th>'
            )
        section_note = ''
        if section.get('note'):
            section_note = (
                f'<p style="color:#777;text-align:center;margin:-4px 0 9px;'
                f'font-size:11px;line-height:1.4">'
                f'{escape(str(section["note"]))}</p>'
            )
        parts.append(f'''
<div style="margin:18px 0 0">
  <h3 style="color:#e0e0e0;text-align:center;margin-bottom:10px;font-size:14px;font-weight:600">{section["title"]}</h3>
{section_note}
  <table style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:13px;color:#ccc;background:#1a1a2e;border-radius:8px;overflow:hidden">
    {columns}
    <thead><tr style="background:#16213e;color:#7fdbca">{head}</tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>''')

    if calibration:
        calibration_note = None
        if sample_info:
            evaluation = sample_info.get('evaluation') or {}
            calibration_note = (
                f'{_sample_count(evaluation.get("test_player_rows"))} played-player rows from '
                f'{escape(str(evaluation.get("test_span", "the holdout")))}; '
                'the same sample used for overall FPL-points evaluation.'
            )
        parts.append(_build_calibration_svg(calibration, calibration_note))

    parts.append('</div>')
    return '\n'.join(parts)



def _build_last_gw_html(last_gw):
    """Build the 'last gameweek' table: top predicted players vs what they scored."""
    if not last_gw or not last_gw.get('rows'):
        return ''

    body = ''
    for r in last_gw['rows']:
        actual = r.get('actual')
        pred = r['predicted']
        if actual is None:
            actual_cell = '<span style="color:#666">&mdash;</span>'
        else:
            # Green when the player beat the projection, red when he missed it.
            delta = actual - pred
            color = '#3fb950' if delta >= 0 else '#f85149'
            actual_cell = (
                f'<span style="color:{color};font-weight:600">{actual:.0f}</span>'
                f'<br><span style="color:{color};font-size:10px">{delta:+.1f}</span>'
            )
        mins = r.get('minutes')
        mins_cell = '&mdash;' if mins is None else f'{mins:.0f}'
        pos = r.get('position') or ''
        dot = _POS_COLOR.get(pos, '#888')
        body += (
            f'<tr>'
            f'<td style="padding:6px 12px">'
            f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{dot};margin-right:7px"></span>'
            f'{r["player_name"]}<br><span style="color:#777;font-size:10px;margin-left:14px">{r.get("team","")}</span></td>'
            f'<td style="padding:6px 8px;text-align:right;font-variant-numeric:tabular-nums">{mins_cell}</td>'
            f'<td style="padding:6px 8px;text-align:right;font-variant-numeric:tabular-nums;color:#999">{pred:.1f}</td>'
            f'<td style="padding:6px 8px;text-align:right;font-variant-numeric:tabular-nums">{actual_cell}</td>'
            f'</tr>' + chr(10)
        )

    foot = ''
    if last_gw.get('mean_actual') is not None:
        foot = (
            f'<tfoot><tr style="background:#16213e;color:#7fdbca;font-weight:600">'
            f'<td style="padding:7px 12px" colspan="2">Mean</td>'
            f'<td style="padding:7px 8px;text-align:right;font-variant-numeric:tabular-nums">{last_gw["mean_predicted"]:.1f}</td>'
            f'<td style="padding:7px 8px;text-align:right;font-variant-numeric:tabular-nums">{last_gw["mean_actual"]:.1f}</td>'
            f'</tr></tfoot>'
        )

    return f'''
<div style="max-width:620px;margin:32px auto 16px;padding:0 16px">
  <h3 style="color:#e0e0e0;text-align:center;margin-bottom:4px;font-size:14px;font-weight:600">
    GW{last_gw["gameweek"]} Top 10 &mdash; Predicted vs Actual</h3>
  <p style="color:#777;text-align:center;margin:0 0 10px;font-size:11px">
    Last week&rsquo;s highest projected players and what they returned</p>
  <table style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:13px;color:#ccc;background:#1a1a2e;border-radius:8px;overflow:hidden">
    <colgroup><col style="width:50%"><col style="width:13%"><col style="width:15%"><col style="width:22%"></colgroup>
    <thead><tr style="background:#16213e;color:#7fdbca">
      <th style="padding:8px 12px;text-align:left">Player</th>
      <th style="padding:8px;text-align:right">Min</th>
      <th style="padding:8px;text-align:right">Pred</th>
      <th style="padding:8px;text-align:right">Actual<br><span style="font-size:9px;font-weight:400">(+/-)</span></th>
    </tr></thead>
    <tbody>{body}</tbody>
    {foot}
  </table>
</div>'''


def _build_fixture_forecasts_html(predictions_per_fixture, gameweek=None):
    """Build expected-score and team clean-sheet forecasts for each fixture."""
    if predictions_per_fixture is None or not len(predictions_per_fixture):
        return ''

    required = {'team', 'opponent', 'is_home', 'pred_team_goals', 'pred_cs_prob'}
    if not required.issubset(predictions_per_fixture.columns):
        return ''

    frame = predictions_per_fixture.copy()
    frame['_home_flag'] = pd.to_numeric(frame['is_home'], errors='coerce')
    if 'match_id' in frame:
        grouped = frame.groupby('match_id', sort=False, dropna=False)
    else:
        frame['_fixture_key'] = frame.apply(
            lambda r: '||'.join(sorted([str(r['team']), str(r['opponent'])])), axis=1)
        grouped = frame.groupby('_fixture_key', sort=False, dropna=False)

    fixtures = []
    for _, group in grouped:
        home = group[group['_home_flag'].eq(1)]
        away = group[group['_home_flag'].eq(0)]
        if home.empty or away.empty:
            continue

        def _summary(rows):
            team = str(rows['team'].dropna().iloc[0])
            goals = pd.to_numeric(rows['pred_team_goals'], errors='coerce').dropna()
            cs = pd.to_numeric(rows['pred_cs_prob'], errors='coerce').dropna()
            return {
                'team': team,
                'goals': float(goals.median()) if len(goals) else None,
                'cs': float(cs.median()) if len(cs) else None,
            }

        kickoff = pd.to_datetime(group.get('match_date'), errors='coerce', utc=True).min() \
            if 'match_date' in group else pd.NaT
        fixtures.append({'home': _summary(home), 'away': _summary(away), 'kickoff': kickoff})

    if not fixtures:
        return ''
    fixtures.sort(key=lambda f: f['kickoff'] if pd.notna(f['kickoff']) else pd.Timestamp.max.tz_localize('UTC'))

    rows = []
    for fixture in fixtures:
        home, away = fixture['home'], fixture['away']
        home_goals = '&mdash;' if home['goals'] is None else f'{home["goals"]:.2f}'
        away_goals = '&mdash;' if away['goals'] is None else f'{away["goals"]:.2f}'
        home_cs = '&mdash;' if home['cs'] is None else f'{100 * home["cs"]:.0f}%'
        away_cs = '&mdash;' if away['cs'] is None else f'{100 * away["cs"]:.0f}%'
        rows.append(f'''
<div style="display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);gap:8px;align-items:center;padding:9px 10px;border-top:1px solid #21262d">
  <div style="min-width:0;text-align:right"><div style="font-size:12px;color:#c9d1d9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{escape(home['team'])}</div><div style="font-size:10px;color:#8b949e">CS {home_cs}</div></div>
  <div style="font-size:15px;font-weight:700;color:#7fdbca;font-variant-numeric:tabular-nums;white-space:nowrap">{home_goals} &ndash; {away_goals}</div>
  <div style="min-width:0"><div style="font-size:12px;color:#c9d1d9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{escape(away['team'])}</div><div style="font-size:10px;color:#8b949e">CS {away_cs}</div></div>
</div>''')

    label = f'GW{gameweek} ' if gameweek else ''
    return f'''
<div style="max-width:620px;margin:32px auto 16px;padding:0 16px">
  <h3 style="color:#e0e0e0;text-align:center;margin-bottom:4px;font-size:14px;font-weight:600">{label}Fixture Forecasts</h3>
  <p style="color:#777;text-align:center;margin:0 0 10px;font-size:11px;line-height:1.4">Expected score uses each team&rsquo;s mean model goals. CS is the team probability before player-minute risk.</p>
  <div style="background:#1a1a2e;border-radius:8px;overflow:hidden">{''.join(rows)}</div>
</div>'''


# Dot colours: DEF/MID/FWD match the ridge-plot legend so the two views agree.
_POS_COLOR = {'GK': '#a371f7', 'DEF': '#3fb950', 'MID': '#58a6ff', 'FWD': '#d29922'}


def _build_calibration_svg(calibration, sample_note=None):
    """Render predicted-vs-actual calibration as overlaid line plots.

    One line per series (predicted mean, actual mean) across pred-points
    buckets. Well-calibrated => the two lines overlap.
    """
    if not calibration:
        return ''

    n = len(calibration)
    W = 580
    H = 280
    pad_l, pad_r = 40, 14
    pad_t, pad_b = 30, 60
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b
    # Center each bucket on its own slot
    if n > 1:
        slot = plot_w / (n - 1)
        x_at = lambda i: pad_l + i * slot
    else:
        x_at = lambda i: pad_l + plot_w / 2

    max_val = max(max(b['pred'], b['actual']) for b in calibration)
    y_max = max(max_val * 1.15, 4.0)

    def y_to(v):
        return pad_t + plot_h - (v / y_max) * plot_h

    grid = []
    for i in range(6):
        v = i * y_max / 5
        grid.append(
            f'<line x1="{pad_l}" x2="{W-pad_r}" y1="{y_to(v):.1f}" y2="{y_to(v):.1f}" '
            f'stroke="#21262d" stroke-dasharray="2,3"/>'
        )
        grid.append(
            f'<text x="{pad_l-6}" y="{y_to(v)+4:.1f}" text-anchor="end" font-size="10" fill="#8b949e">{v:.1f}</text>'
        )

    pred_pts = [(x_at(i), y_to(b['pred'])) for i, b in enumerate(calibration)]
    actual_pts = [(x_at(i), y_to(b['actual'])) for i, b in enumerate(calibration)]

    def _polyline(pts, color):
        d = ' '.join(f'{px:.1f},{py:.1f}' for px, py in pts)
        return (
            f'<polyline points="{d}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def _dots(pts, color):
        return ''.join(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{color}" stroke="#1a1a2e" stroke-width="1"/>'
            for px, py in pts
        )

    pred_line = _polyline(pred_pts, '#58a6ff') + _dots(pred_pts, '#58a6ff')
    actual_line = _polyline(actual_pts, '#3fb950') + _dots(actual_pts, '#3fb950')

    labels = []
    for i, b in enumerate(calibration):
        x = x_at(i)
        labels.append(
            f'<text x="{x:.1f}" y="{H - pad_b + 16}" text-anchor="middle" font-size="10" fill="#c9d1d9">{b["bucket"]}</text>'
        )
        labels.append(
            f'<text x="{x:.1f}" y="{H - pad_b + 30}" text-anchor="middle" font-size="9" fill="#484f58">n={b["n"]:,}</text>'
        )

    legend = (
        f'<line x1="{W/2 - 110}" x2="{W/2 - 90}" y1="11" y2="11" stroke="#58a6ff" stroke-width="2"/>'
        f'<circle cx="{W/2 - 100}" cy="11" r="3.5" fill="#58a6ff" stroke="#1a1a2e" stroke-width="1"/>'
        f'<text x="{W/2 - 84}" y="15" font-size="11" fill="#c9d1d9">Predicted (mean)</text>'
        f'<line x1="{W/2 + 22}" x2="{W/2 + 42}" y1="11" y2="11" stroke="#3fb950" stroke-width="2"/>'
        f'<circle cx="{W/2 + 32}" cy="11" r="3.5" fill="#3fb950" stroke="#1a1a2e" stroke-width="1"/>'
        f'<text x="{W/2 + 48}" y="15" font-size="11" fill="#c9d1d9">Actual (mean)</text>'
    )

    footnote = sample_note or (
        'Each bucket aggregates test-set rows whose predicted FPL points (inc-bonus) '
        'fall in that range. Well-calibrated &rArr; the two lines overlap.'
    )
    return f'''
<div style="margin:24px 0 0">
  <h3 style="color:#e0e0e0;text-align:center;margin-bottom:8px;font-size:14px;font-weight:600">Points Calibration: Predicted vs Actual by Bucket</h3>
  <div style="background:#1a1a2e;border-radius:8px;padding:12px">
    <svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block" preserveAspectRatio="xMidYMid meet">
      {legend}
      {''.join(grid)}
      {pred_line}
      {actual_line}
      {''.join(labels)}
      <text x="{W/2}" y="{H-4}" text-anchor="middle" font-size="11" fill="#8b949e">Predicted points bucket</text>
      <text x="14" y="{pad_t + plot_h/2}" text-anchor="middle" font-size="11" fill="#8b949e" transform="rotate(-90 14 {pad_t + plot_h/2})">Mean points</text>
    </svg>
    <p style="font-size:11px;color:#8b949e;text-align:center;margin-top:6px">{footnote}</p>
  </div>
</div>'''


FPL_GOAL_PTS = {'GK': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}


def _build_player_data(predictions, simulations, top_n, predictions_per_fixture=None):
    """Build simulation-based player data for both desktop and mobile viz.

    Returns (players_data, all_totals_flat) where players_data is a list of
    dicts with KDE curves, stats, and cumulative probabilities.

    If predictions_per_fixture is provided (pre-DGW-aggregation DataFrame with
    a '_sim_idx' column), per-sim totals are built fixture-by-fixture and
    summed per player, so DGW distributions match their aggregated exp_pts.
    """
    rng = np.random.default_rng(42)

    top = predictions[predictions['fpl_position'] != 'GK'].nlargest(top_n, 'exp_total_pts')

    sim_goals = simulations['goals']
    sim_assists = simulations['assists']
    sim_cs = simulations['cs']
    sim_bonus = simulations['bonus']
    N_SIMS = sim_goals.shape[0]

    use_per_fixture = (
        predictions_per_fixture is not None
        and '_sim_idx' in predictions_per_fixture.columns
    )

    if use_per_fixture:
        key_col = 'player_id' if 'player_id' in predictions_per_fixture.columns else 'player_name'
        pf_by_player = {}
        for _, pf_row in predictions_per_fixture.iterrows():
            pf_by_player.setdefault(pf_row[key_col], []).append(pf_row)
    else:
        sim_names = simulations['player_names']
        name_to_indices = {}
        for idx, name in enumerate(sim_names):
            name_to_indices.setdefault(name, []).append(idx)
        id_to_indices = {}
        for idx, player_id in enumerate(simulations.get('player_ids', [])):
            id_to_indices.setdefault(player_id, []).append(idx)

    players_data = []
    all_totals_flat = []

    for _, row in top.iterrows():
        pos = row['fpl_position']
        player_name = row['player_name']

        if use_per_fixture:
            player_key = row.get(key_col, player_name)
            fixture_rows = pf_by_player.get(player_key)
            if not fixture_rows:
                print(f"WARNING: {player_name} not found in per-fixture predictions, skipping")
                continue
        else:
            indices = id_to_indices.get(row.get('player_id'), name_to_indices.get(player_name))
            if not indices:
                print(f"WARNING: {player_name} not found in simulation arrays, skipping")
                continue
            fixture_rows = []
            for pidx in indices:
                fx = row.copy()
                fx['_sim_idx'] = pidx
                fixture_rows.append(fx)

        total = np.zeros(N_SIMS)
        for fx in fixture_rows:
            pidx = int(fx['_sim_idx'])
            if use_per_fixture:
                pidx = int(fx['_sim_idx'])
                mins = fx.get('pred_minutes', 0)
                ga = max(fx.get('pred_goals_against', 1.2), 0.01)
                defcon_p = fx.get('pred_defcon_prob', 0)
                yellow_p = fx.get('pred_yellow_prob', 0)
                red_p = fx.get('pred_red_prob', 0)
            else:
                mins = row.get('pred_minutes', 0)
                ga = max(row.get('pred_goals_against', 1.2), 0.01)
                defcon_p = row.get('pred_defcon_prob', 0)
                yellow_p = row.get('pred_yellow_prob', 0)
                red_p = row.get('pred_red_prob', 0)

            if 'total_points' in simulations:
                total += simulations['total_points'][:, pidx]
                continue

            # Legacy archives did not save scored draws.
            goals = sim_goals[:, pidx]
            assists = sim_assists[:, pidx]
            cs = sim_cs[:, pidx]
            bonus_pts = sim_bonus[:, pidx]

            app = np.full(N_SIMS, 2 if mins >= 60 else (1 if mins >= 1 else 0))
            goal_pts = goals * FPL_GOAL_PTS.get(pos, 5)
            assist_pts = assists * 3

            cs_pts = np.zeros(N_SIMS)
            if mins >= 60 and pos in ('DEF', 'MID'):
                cs_pts = cs * {'DEF': 4, 'MID': 1}.get(pos, 0)

            conceded_pen = np.zeros(N_SIMS)
            if mins >= 60 and pos == 'DEF':
                conceded_pen = -(rng.poisson(ga, N_SIMS) // 2)

            defcon_pts = np.zeros(N_SIMS)
            if mins >= 60 and pos in ('DEF', 'MID'):
                defcon_pts = rng.binomial(1, np.clip(defcon_p, 0, 1), N_SIMS) * 2

            yellow = -rng.binomial(1, np.clip(yellow_p, 0, 1), N_SIMS)
            red = -3 * rng.binomial(1, np.clip(red_p, 0, 1), N_SIMS)

            fixture_total = (
                app + goal_pts + assist_pts + cs_pts
                + conceded_pen + defcon_pts + bonus_pts + yellow + red
            )
            total = total + fixture_total

        all_totals_flat.append(total)

        # KDE curve
        try:
            kde = gaussian_kde(total, bw_method=0.35)
            x_grid = np.linspace(total.min() - 1, min(total.max() + 1, 30), 200)
            y_grid = kde(x_grid)
        except Exception:
            x_grid = np.linspace(0, 20, 200)
            y_grid = np.zeros(200)

        # Cumulative P(>=X)
        max_pts = int(total.max()) + 1
        cum_prob = {}
        for x in range(int(total.min()), max_pts + 1):
            cum_prob[str(x)] = round(float((total >= x).mean() * 100), 1)

        # Tooltip metrics: aggregate across fixtures.
        # Counts (goals/assists/minutes/bonus) sum; probabilities (CS/defcon) use
        # P(>=1 across fixtures) = 1 - prod(1 - p_i) so they stay in [0, 100%].
        # For SGW players this reduces to the single fixture's value (identical to old behavior).
        if use_per_fixture:
            agg_goals = sum(float(fx.get('pred_exp_goals', 0) or 0) for fx in fixture_rows)
            agg_assists = sum(float(fx.get('pred_exp_assists', 0) or 0) for fx in fixture_rows)
            agg_minutes = sum(float(fx.get('pred_minutes', 0) or 0) for fx in fixture_rows)
            agg_bonus = sum(float(fx.get('pred_bonus', 0) or 0) for fx in fixture_rows)
            _cs_miss = 1.0
            _dc_miss = 1.0
            for fx in fixture_rows:
                _cs_miss *= (1.0 - float(fx.get('pred_cs_prob', 0) or 0))
                _dc_miss *= (1.0 - float(fx.get('pred_defcon_prob', 0) or 0))
            agg_cs = (1.0 - _cs_miss) * 100
            agg_defcon = (1.0 - _dc_miss) * 100
        else:
            agg_goals = float(row.get('pred_exp_goals', 0) or 0)
            agg_assists = float(row.get('pred_exp_assists', 0) or 0)
            agg_minutes = float(row.get('pred_minutes', 0) or 0)
            agg_cs = float(row.get('pred_cs_prob', 0) or 0) * 100
            agg_defcon = float(row.get('pred_defcon_prob', 0) or 0) * 100
            agg_bonus = float(row.get('pred_bonus', 0) or 0)

        if 'total_points' in simulations:
            idx = [int(fx['_sim_idx']) for fx in fixture_rows]
            agg_goals = float(simulations['goals'][:, idx].sum(axis=1).mean())
            agg_assists = float(simulations['assists'][:, idx].sum(axis=1).mean())
            agg_bonus = float(simulations['bonus'][:, idx].sum(axis=1).mean())
            agg_cs = float((simulations['cs'][:, idx].sum(axis=1) > 0).mean() * 100)
            if 'minutes' in simulations:
                agg_minutes = float(simulations['minutes'][:, idx].sum(axis=1).mean())
            if 'exp_defcon_pts' in simulations:
                agg_defcon = float((simulations['exp_defcon_pts'][:, idx].sum(axis=1) > 0).mean() * 100)

        players_data.append({
            'name': player_name,
            'position': pos,
            'team': str(row.get('team', '')),
            'opponent': str(row.get('opponent', '?')),
            'is_home': bool(row.get('is_home', False)),
            'exp_pts': round(float(total.mean() if 'total_points' in simulations else row['exp_total_pts']), 2),
            'pred_goals': round(agg_goals, 2),
            'pred_assists': round(agg_assists, 2),
            'pred_minutes': round(agg_minutes, 1),
            'pred_cs': round(agg_cs, 1),
            'pred_defcon': round(agg_defcon, 1),
            'pred_bonus': round(agg_bonus, 2),
            'median': round(float(np.median(total)), 1),
            'p10': round(float(np.percentile(total, 10)), 1),
            'p90': round(float(np.percentile(total, 90)), 1),
            'p_10plus': round(float((total >= 10).mean() * 100), 1),
            'p_15plus': round(float((total >= 15).mean() * 100), 1),
            'kde_x': [round(float(v), 2) for v in x_grid],
            'kde_y': [round(float(v), 5) for v in y_grid],
            'cum_prob': cum_prob,
        })

    return players_data, all_totals_flat


def _extract_for_template(html):
    """Extract (style, body_inner, inline_script) from a standalone HTML template.

    Strips the d3 CDN script tag and rewrites ``const DATA = /*__DATA__*/null;``
    to read from the shared ``window.DATA`` so the outer wrapper can define data
    once. Preserves the ``<!--__METRICS__-->`` placeholder.
    """
    style_m = re.search(r'<style>(.*?)</style>', html, re.DOTALL)
    style = style_m.group(1) if style_m else ''

    body_m = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL)
    body = body_m.group(1) if body_m else ''

    body = re.sub(
        r'\s*<script\s+src="https://d3js\.org/d3\.v7\.min\.js"></script>\s*',
        '\n',
        body,
    )

    script_m = re.search(r'<script>(.*?)</script>', body, re.DOTALL)
    script = script_m.group(1) if script_m else ''
    if script_m:
        body = body[:script_m.start()] + body[script_m.end():]

    script = re.sub(
        r'const\s+DATA\s*=\s*/\*__DATA__\*/\s*null\s*;?',
        'const DATA = window.DATA;',
        script,
    )

    return style.strip(), body.strip(), script.strip()


def generate_distribution_html(
    predictions,
    simulations,
    output_path='distributions.html',
    top_n=200,
    gameweek=None,
    metrics=None,
    predictions_per_fixture=None,
    squad=None,
    last_gw_review=None,
):
    """Generate a self-contained responsive HTML file.

    Emits one file containing both the desktop D3 ridge plot and the
    mobile card layout as inert ``<template>`` elements. At load time an
    outer script picks the right one based on viewport width (<=768px
    activates mobile). ``squad`` is accepted for archived callers but is no
    longer rendered on the distributions page.
    """
    players_data, all_totals_flat = _build_player_data(predictions, simulations, top_n, predictions_per_fixture)

    all_flat = np.concatenate(all_totals_flat)
    x_min = float(all_flat.min() - 1)
    x_max = float(min(all_flat.max() + 1, 25))

    gw_label = f'GW{gameweek}' if gameweek else ''

    data = {
        'players': players_data,
        'x_min': x_min,
        'x_max': x_max,
        'gw_label': gw_label,
    }

    d_style, d_body, d_script = _extract_for_template(_HTML_TEMPLATE)
    m_style, m_body, m_script = _extract_for_template(_MOBILE_TEMPLATE)

    metrics = deepcopy(metrics)
    html = _RESPONSIVE_TEMPLATE
    html = html.replace('__DESKTOP_STYLE__', d_style)
    html = html.replace('__DESKTOP_BODY__', d_body)
    html = html.replace('__DESKTOP_SCRIPT__', d_script)
    html = html.replace('__MOBILE_STYLE__', m_style)
    html = html.replace('__MOBILE_BODY__', m_body)
    html = html.replace('__MOBILE_SCRIPT__', m_script)
    html = html.replace('/*__DATA__*/null', json.dumps(data))
    html = html.replace('__N_SIMS__', f'{len(all_totals_flat[0]):,}')
    html = html.replace('<!--__LASTGW__-->', _build_last_gw_html(last_gw_review))
    html = html.replace('<!--__METRICS__-->', _build_metrics_html(
        metrics, predictions, predictions_per_fixture, gameweek))
    html = html.replace('<!--__FIXTURES__-->', _build_fixture_forecasts_html(
        predictions_per_fixture, gameweek))

    Path(output_path).write_text(html, encoding='utf-8')
    print(f"Distribution visualization saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# HTML template with embedded D3.js visualization
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FPL Points Distribution</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  background:#0d1117;color:#c9d1d9;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  display:flex;justify-content:center;padding:20px 10px;
}
.container{width:100%;max-width:1100px;min-width:0}
header{text-align:center;margin-bottom:18px}
h1{font-size:22px;font-weight:700;color:#e6edf3;margin-bottom:4px}
.subtitle{font-size:13px;color:#8b949e}
.controls{
  display:flex;justify-content:center;gap:24px;align-items:center;
  margin-bottom:14px;flex-wrap:wrap;
}
.filter-group{display:flex;gap:6px}
.filter-btn{
  background:#21262d;color:#c9d1d9;border:1px solid #30363d;
  padding:5px 14px;border-radius:6px;font-size:12px;cursor:pointer;
  transition:all .15s;
}
.filter-btn:hover{border-color:#58a6ff;color:#58a6ff}
.filter-btn.active{background:#1f6feb;border-color:#1f6feb;color:#fff}
.sort-group select{
  background:#21262d;color:#c9d1d9;border:1px solid #30363d;
  padding:5px 10px;border-radius:6px;font-size:12px;cursor:pointer;
}
#chart{overflow-x:auto}
#chart svg{display:block;margin:0 auto}
.tooltip{
  position:fixed;pointer-events:none;z-index:100;
  background:#161b22;border:1px solid #30363d;border-radius:8px;
  padding:12px 16px;box-shadow:0 8px 24px rgba(0,0,0,.4);
  font-size:12px;line-height:1.6;max-width:280px;
  display:none;
}
.tt-name{font-weight:700;font-size:14px;color:#e6edf3;margin-bottom:2px}
.tt-meta{font-size:11px;color:#8b949e;margin-bottom:8px}
.tt-row{display:flex;justify-content:space-between;gap:16px}
.tt-row .label{color:#8b949e}
.tt-row .value{color:#e6edf3;font-weight:600}
.tt-divider{border-top:1px solid #21262d;margin:6px 0}
.tt-cursor{margin-top:6px;padding:6px 8px;background:#1c2128;border-radius:4px;
  text-align:center;font-size:13px;color:#f0883e;font-weight:600}
.legend{
  display:flex;justify-content:center;gap:20px;margin-top:12px;font-size:11px;color:#8b949e
}
.legend-item{display:flex;align-items:center;gap:5px}
.legend-dot{width:10px;height:10px;border-radius:50%}
.methodology{
  margin-bottom:16px;background:#161b22;border:1px solid #21262d;
  border-radius:8px;padding:0;overflow:hidden;
}
.methodology summary{
  padding:10px 16px;cursor:pointer;font-size:13px;font-weight:600;
  color:#58a6ff;list-style:none;
}
.methodology summary::-webkit-details-marker{display:none}
.methodology summary::before{content:'▸ ';font-size:11px}
.methodology[open] summary::before{content:'▾ '}
.method-body{
  padding:4px 16px 14px;font-size:12px;line-height:1.7;color:#8b949e;
}
.method-body p{margin-bottom:8px}
.method-body strong{color:#c9d1d9}
.method-body ul{margin:4px 0 8px 18px}
.method-body li{margin-bottom:3px}
/* The review and metrics column sits beside the ridge plot on desktop. */
.side-col{
  display:flex;flex-direction:column;align-items:stretch;
  flex:0 0 340px;min-width:0;margin-left:18px;
}
.side-col > div[style]{max-width:100% !important;margin-left:0 !important;margin-right:0 !important;overflow-x:auto}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1 id="title">FPL Points Distribution</h1>
    <p class="subtitle">Top Outfield Players &mdash; Monte Carlo Simulation (correlated bonus)</p>
  </header>
  <details class="methodology">
    <summary>Methodology &amp; How to Read This</summary>
    <div class="method-body">
      <p>Each curve shows the <strong>probability distribution of FPL points</strong> a player could score in the upcoming gameweek, estimated from <strong>__N_SIMS__ Monte Carlo simulations</strong>.</p>
      <p><strong>How the simulations work:</strong> Each trial samples a player's appearance and minutes, then shares one match score across goals, assists, clean sheets, and goals-conceded penalties. Goals and assists are allocated to players; saves, cards, defensive contributions, and BPS-based bonus are scored within the same trial. Expected points and threshold probabilities come from these same simulated totals, including non-appearances. The resulting __N_SIMS__ totals are smoothed into the KDE curves you see below.</p>
      <p><strong>How to read the plot:</strong></p>
      <ul>
        <li>The <strong>width/spread</strong> of a curve shows outcome variance &mdash; wider means less predictable.</li>
        <li>The <span style="color:#f85149">red dashed line</span> marks the <strong>expected value E[pts]</strong>.</li>
        <li><strong>Hover</strong> over any curve to see detailed stats and the probability of scoring &ge; any point threshold.</li>
        <li>Use the <strong>filters</strong> to isolate positions and the <strong>sort</strong> dropdown to reorder by different metrics.</li>
      </ul>
      <p>Model inputs include rolling xG, xGA, fixture difficulty, home/away, and historical per-90 stats. All predictions are for a single gameweek.</p>
    </div>
  </details>
  <div class="controls">
    <div class="filter-group">
      <button class="filter-btn active" data-pos="ALL">All</button>
      <button class="filter-btn" data-pos="DEF">DEF</button>
      <button class="filter-btn" data-pos="MID">MID</button>
      <button class="filter-btn" data-pos="FWD">FWD</button>
    </div>
    <div class="sort-group">
      <select id="sort-select">
        <option value="exp_pts">Sort: E[pts]</option>
        <option value="pred_goals">Sort: Prj Goals</option>
        <option value="pred_assists">Sort: Prj Assists</option>
        <option value="pred_cs">Sort: CS %</option>
        <option value="pred_defcon">Sort: Defcon %</option>
        <option value="p_10plus">Sort: Upside P(10+)</option>
        <option value="median">Sort: Median</option>
        <option value="p90">Sort: Ceiling (P90)</option>
      </select>
    </div>
  </div>
  <div id="chart"></div>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#3fb950"></div> DEF</div>
    <div class="legend-item"><div class="legend-dot" style="background:#58a6ff"></div> MID</div>
    <div class="legend-item"><div class="legend-dot" style="background:#d29922"></div> FWD</div>
    <div class="legend-item" style="gap:3px">
      <svg width="20" height="10"><line x1="0" y1="5" x2="20" y2="5"
        stroke="#f85149" stroke-width="1.5" stroke-dasharray="3,3"/></svg> E[pts]
    </div>
  </div>
</div>
<div class="tooltip" id="tooltip"></div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const DATA = /*__DATA__*/null;

// --- Configuration ---
const ROW_HEIGHT = 50;
const CURVE_HEIGHT = 70;
const MARGIN = {top: 35, right: 30, bottom: 45, left: 20};
const POS_COLOR = {DEF:'#3fb950', MID:'#58a6ff', FWD:'#d29922'};

// Title
if (DATA.gw_label) {
  document.getElementById('title').textContent =
    DATA.gw_label + ' Points Distribution';
}

// State
let activePos = 'ALL';
let sortKey = 'exp_pts';

// Filter buttons
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activePos = btn.dataset.pos;
    render();
  });
});

// Sort select
document.getElementById('sort-select').addEventListener('change', e => {
  sortKey = e.target.value;
  render();
});

const tooltip = document.getElementById('tooltip');

function getPlayers() {
  let p = DATA.players;
  if (activePos !== 'ALL') p = p.filter(d => d.position === activePos);
  return [...p].sort((a, b) => b[sortKey] - a[sortKey]);
}

function render() {
  const players = getPlayers();
  const W = Math.min(document.querySelector('.container').clientWidth || window.innerWidth - 20, 1100);
  const H = MARGIN.top + players.length * ROW_HEIGHT + MARGIN.bottom;

  d3.select('#chart').selectAll('*').remove();

  const svg = d3.select('#chart')
    .append('svg')
    .attr('width', W)
    .attr('height', H);

  const x = d3.scaleLinear()
    .domain([DATA.x_min, DATA.x_max])
    .range([MARGIN.left + 10, W - MARGIN.right]);

  // Grid lines
  const ticks = x.ticks(15);
  svg.append('g').selectAll('line').data(ticks).join('line')
    .attr('x1', d => x(d)).attr('x2', d => x(d))
    .attr('y1', MARGIN.top - 10).attr('y2', H - MARGIN.bottom)
    .attr('stroke', '#21262d').attr('stroke-dasharray', '2,3');

  // X axis
  svg.append('g')
    .attr('transform', `translate(0,${H - MARGIN.bottom})`)
    .call(d3.axisBottom(x).ticks(15).tickSize(4))
    .call(g => g.select('.domain').attr('stroke', '#30363d'))
    .call(g => g.selectAll('.tick line').attr('stroke', '#30363d'))
    .call(g => g.selectAll('.tick text').attr('fill', '#8b949e').attr('font-size', '11px'));

  // X label
  svg.append('text')
    .attr('x', W / 2).attr('y', H - 6)
    .attr('text-anchor', 'middle')
    .attr('fill', '#8b949e').attr('font-size', '12px')
    .text('Total FPL Points');

  // --- Draw players (bottom-to-top so top curves render last / on top) ---
  const groups = [];
  for (let ri = players.length - 1; ri >= 0; ri--) {
    const player = players[ri];
    const baseY = MARGIN.top + ri * ROW_HEIGHT + ROW_HEIGHT;
    const color = POS_COLOR[player.position] || '#aaa';
    const maxKde = player.kde_y.reduce((a, b) => a > b ? a : b, 1e-9);

    const g = svg.append('g').attr('class', 'player-row');

    // Paired data for area/line
    const pts = player.kde_x.map((xv, j) => ({
      px: x(xv),
      py: baseY - (player.kde_y[j] / maxKde) * CURVE_HEIGHT
    }));

    // Fill
    const areaGen = d3.area()
      .x(d => d.px).y0(baseY).y1(d => d.py).curve(d3.curveBasis);
    g.append('path').datum(pts).attr('d', areaGen)
      .attr('fill', color).attr('fill-opacity', 0.4)
      .attr('class', 'area');

    // Outline
    const lineGen = d3.line().x(d => d.px).y(d => d.py).curve(d3.curveBasis);
    g.append('path').datum(pts).attr('d', lineGen)
      .attr('stroke', color).attr('stroke-width', 1.5).attr('fill', 'none')
      .attr('class', 'outline');

    // E[pts] dashed line
    g.append('line')
      .attr('x1', x(player.exp_pts)).attr('x2', x(player.exp_pts))
      .attr('y1', baseY).attr('y2', baseY - CURVE_HEIGHT * 0.85)
      .attr('stroke', '#f85149').attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3').attr('opacity', 0.6);

    // Player label
    g.append('text')
      .attr('x', MARGIN.left + 12)
      .attr('y', baseY - ROW_HEIGHT + 12)
      .attr('fill', '#e6edf3').attr('font-size', '11px').attr('font-weight', 600)
      .text(player.name);

    // Position + opponent sub-label
    const sub = player.position + ' vs ' + player.opponent + (player.is_home ? ' (H)' : ' (A)');
    g.append('text')
      .attr('x', MARGIN.left + 12)
      .attr('y', baseY - ROW_HEIGHT + 26)
      .attr('fill', color).attr('font-size', '10px')
      .text(sub);

    // Hover rect (covers the row band only)
    g.append('rect')
      .attr('x', MARGIN.left).attr('y', baseY - ROW_HEIGHT)
      .attr('width', W - MARGIN.left - MARGIN.right)
      .attr('height', ROW_HEIGHT)
      .attr('fill', 'transparent').style('cursor', 'crosshair')
      .datum(player)
      .on('mouseenter', function() {
        g.select('.area').attr('fill-opacity', 0.7);
        g.select('.outline').attr('stroke-width', 2.5);
      })
      .on('mouseleave', function() {
        g.select('.area').attr('fill-opacity', 0.4);
        g.select('.outline').attr('stroke-width', 1.5);
        tooltip.style.display = 'none';
      })
      .on('mousemove', function(event) {
        const [mx] = d3.pointer(event, svg.node());
        const ptVal = Math.round(x.invert(mx));
        const pAbove = player.cum_prob[String(ptVal)] || 0;

        tooltip.innerHTML = `
          <div class="tt-name">${player.name}</div>
          <div class="tt-meta">${player.position} &middot; ${player.team} &middot; vs ${player.opponent}${player.is_home ? ' (H)' : ''}</div>
          <div class="tt-row"><span class="label">Prj Goals</span><span class="value">${player.pred_goals}</span></div>
          <div class="tt-row"><span class="label">Prj Assists</span><span class="value">${player.pred_assists}</span></div>
          <div class="tt-row"><span class="label">Prj Minutes</span><span class="value">${player.pred_minutes}</span></div>
          <div class="tt-row"><span class="label">CS %</span><span class="value">${player.pred_cs}%</span></div>
          <div class="tt-row"><span class="label">Defcon %</span><span class="value">${player.pred_defcon}%</span></div>
          <div class="tt-row"><span class="label">Prj Bonus</span><span class="value">${player.pred_bonus}</span></div>
          <div class="tt-divider"></div>
          <div class="tt-row"><span class="label">E[pts]</span><span class="value">${player.exp_pts}</span></div>
          <div class="tt-row"><span class="label">Median</span><span class="value">${player.median}</span></div>
          <div class="tt-row"><span class="label">P10-P90 range</span><span class="value">${player.p10} &ndash; ${player.p90}</span></div>
          <div class="tt-row"><span class="label">P(10+)</span><span class="value">${player.p_10plus}%</span></div>
          <div class="tt-row"><span class="label">P(15+)</span><span class="value">${player.p_15plus}%</span></div>
          <div class="tt-divider"></div>
          <div class="tt-cursor">P(&ge;${ptVal}) = ${pAbove}%</div>
        `;
        tooltip.style.display = 'block';

        // Position tooltip
        const tx = event.clientX + 18;
        const ty = event.clientY - 10;
        const tw = tooltip.offsetWidth;
        const th = tooltip.offsetHeight;
        tooltip.style.left = (tx + tw > window.innerWidth ? event.clientX - tw - 12 : tx) + 'px';
        tooltip.style.top = (ty + th > window.innerHeight ? event.clientY - th - 8 : ty) + 'px';
      });

    groups.push({g, player, baseY});
  }

  // Crosshair line
  const crosshair = svg.append('line')
    .attr('y1', MARGIN.top - 10).attr('y2', H - MARGIN.bottom)
    .attr('stroke', '#c9d1d9').attr('stroke-width', 0.5)
    .attr('stroke-dasharray', '4,4').attr('opacity', 0)
    .attr('pointer-events', 'none');

  const crossLabel = svg.append('text')
    .attr('fill', '#c9d1d9').attr('font-size', '10px')
    .attr('text-anchor', 'middle').attr('opacity', 0)
    .attr('pointer-events', 'none');

  svg.on('mousemove', function(event) {
    const [mx] = d3.pointer(event);
    if (mx > MARGIN.left && mx < W - MARGIN.right) {
      crosshair.attr('x1', mx).attr('x2', mx).attr('opacity', 0.4);
      const val = x.invert(mx);
      crossLabel.attr('x', mx).attr('y', MARGIN.top - 14)
        .text(val.toFixed(1) + ' pts').attr('opacity', 0.7);
    }
  }).on('mouseleave', function() {
    crosshair.attr('opacity', 0);
    crossLabel.attr('opacity', 0);
  });
}

// Initial render — wait for layout to be ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', render);
} else {
  render();
}
window.addEventListener('resize', render);
</script>
<div class="side-col">
<!--__LASTGW__-->
<!--__METRICS__-->
<!--__FIXTURES__-->
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Mobile HTML template — card-based layout
# ---------------------------------------------------------------------------

_MOBILE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>FPL Points Distribution</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  background:#0d1117;color:#c9d1d9;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  padding:12px;-webkit-tap-highlight-color:transparent;
  display:flex;justify-content:center;
}
.page{width:100%;max-width:560px;min-width:0}
.page > div[style]{width:100% !important;max-width:100% !important;overflow:hidden}
.page table{table-layout:fixed}
.page th,.page td{min-width:0;overflow-wrap:anywhere}
.header{text-align:center;margin-bottom:14px}
.header h1{font-size:18px;font-weight:700;color:#e6edf3}
.header p{font-size:12px;color:#8b949e;margin-top:2px}

.controls{
  display:flex;gap:8px;align-items:center;margin-bottom:14px;
  flex-wrap:wrap;justify-content:center;
}
.pill{
  background:#21262d;color:#c9d1d9;border:1px solid #30363d;
  padding:8px 16px;border-radius:20px;font-size:13px;font-weight:500;
  cursor:pointer;transition:all .15s;-webkit-user-select:none;user-select:none;
}
.pill:active{transform:scale(0.95)}
.pill.active{background:#1f6feb;border-color:#1f6feb;color:#fff}
.sort-sel{
  background:#21262d;color:#c9d1d9;border:1px solid #30363d;
  padding:8px 12px;border-radius:20px;font-size:13px;
  -webkit-appearance:none;appearance:none;
}

.cards{display:flex;flex-direction:column;gap:10px}

.card{
  background:#161b22;border:1px solid #21262d;border-radius:12px;
  overflow:hidden;transition:border-color .2s;
}
.card.expanded{border-color:#30363d}
.card-top{padding:12px 14px 0;display:flex;align-items:flex-start;gap:10px}
.rank{
  background:#21262d;color:#8b949e;font-size:12px;font-weight:700;
  width:26px;height:26px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
}
.info{flex:1;min-width:0}
.player-name{font-size:15px;font-weight:700;color:#e6edf3;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.meta{font-size:12px;color:#8b949e;margin-top:1px;display:flex;align-items:center;gap:6px}
.pos-badge{
  font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;
  text-transform:uppercase;letter-spacing:.5px;
}
.pos-DEF{background:rgba(63,185,80,.15);color:#3fb950}
.pos-MID{background:rgba(88,166,255,.15);color:#58a6ff}
.pos-FWD{background:rgba(210,153,34,.15);color:#d29922}
.pts-badge{
  background:#1c2128;border:1px solid #30363d;border-radius:8px;
  padding:4px 10px;font-size:16px;font-weight:700;color:#e6edf3;
  flex-shrink:0;text-align:center;line-height:1.2;
}
.pts-badge small{font-size:9px;color:#8b949e;font-weight:400;display:block}

.chart-area{padding:6px 14px;height:65px}
.chart-area svg{width:100%;height:100%;display:block}

.stats-row{
  display:grid;grid-template-columns:repeat(4,1fr);
  padding:0 14px 12px;gap:4px;
}
.stat{text-align:center}
.stat .val{font-size:14px;font-weight:700;color:#e6edf3}
.stat .lbl{font-size:10px;color:#8b949e}

.expanded-section{
  max-height:0;overflow:hidden;transition:max-height .3s ease;
}
.card.expanded .expanded-section{max-height:300px}
.prob-grid{
  display:grid;grid-template-columns:repeat(3,1fr);gap:6px;
  padding:0 14px 14px;
}
.prob-cell{
  background:#1c2128;border-radius:8px;padding:8px 6px;text-align:center;
}
.prob-cell .threshold{font-size:11px;color:#8b949e}
.prob-cell .pct{font-size:16px;font-weight:700;color:#e6edf3}
.prob-cell .bar{
  height:3px;border-radius:2px;background:#21262d;margin-top:4px;overflow:hidden;
}
.prob-cell .bar-fill{height:100%;border-radius:2px;transition:width .3s}

.tap-hint{text-align:center;font-size:11px;color:#484f58;padding:4px 14px 12px}
</style>
</head>
<body>
<div class="page">
<div class="header">
  <h1 id="title">FPL Points Distribution</h1>
  <p>Monte Carlo Simulation &middot; Tap card for details</p>
</div>
<!--__LASTGW__-->
<!--__METRICS__-->
<div class="controls">
  <div class="pill active" data-pos="ALL">All</div>
  <div class="pill" data-pos="DEF">DEF</div>
  <div class="pill" data-pos="MID">MID</div>
  <div class="pill" data-pos="FWD">FWD</div>
  <select class="sort-sel" id="sort-select">
    <option value="exp_pts">Sort: E[pts]</option>
    <option value="pred_goals">Sort: Prj Goals</option>
    <option value="pred_assists">Sort: Prj Assists</option>
    <option value="pred_cs">Sort: CS %</option>
    <option value="pred_defcon">Sort: Defcon %</option>
    <option value="p_10plus">Sort: P(10+)</option>
    <option value="median">Sort: Median</option>
    <option value="p90">Sort: Ceiling</option>
  </select>
</div>
<div class="cards" id="cards"></div>
<!--__FIXTURES__-->
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const DATA = /*__DATA__*/null;

const POS_COLOR = {DEF:'#3fb950', MID:'#58a6ff', FWD:'#d29922'};
const PROB_THRESHOLDS = [2, 5, 8, 10, 12, 15];

if (DATA.gw_label) {
  document.getElementById('title').textContent = DATA.gw_label + ' Points Distribution';
}

let activePos = 'ALL';
let sortKey = 'exp_pts';

document.querySelectorAll('.pill').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.pill').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activePos = btn.dataset.pos;
    render();
  });
});
document.getElementById('sort-select').addEventListener('change', e => {
  sortKey = e.target.value;
  render();
});

function getPlayers() {
  let p = DATA.players;
  if (activePos !== 'ALL') p = p.filter(d => d.position === activePos);
  return [...p].sort((a, b) => b[sortKey] - a[sortKey]);
}

function render() {
  const players = getPlayers();
  const container = document.getElementById('cards');
  container.innerHTML = '';

  players.forEach((player, i) => {
    const color = POS_COLOR[player.position] || '#aaa';
    const venue = player.is_home ? '(H)' : '(A)';

    // --- Build card HTML ---
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <div class="card-top">
        <div class="rank">${i + 1}</div>
        <div class="info">
          <div class="player-name">${player.name}</div>
          <div class="meta">
            <span class="pos-badge pos-${player.position}">${player.position}</span>
            ${player.team} vs ${player.opponent} ${venue}
          </div>
        </div>
        <div class="pts-badge">${player.exp_pts}<small>E[pts]</small></div>
      </div>
      <div class="chart-area"></div>
      <div class="stats-row" style="grid-template-columns:repeat(3,1fr)">
        <div class="stat"><div class="val">${player.pred_goals}</div><div class="lbl">Prj Goals</div></div>
        <div class="stat"><div class="val">${player.pred_assists}</div><div class="lbl">Prj Assists</div></div>
        <div class="stat"><div class="val">${player.pred_minutes}</div><div class="lbl">Prj Min</div></div>
      </div>
      <div class="stats-row" style="grid-template-columns:repeat(3,1fr)">
        <div class="stat"><div class="val">${player.pred_cs}%</div><div class="lbl">CS %</div></div>
        <div class="stat"><div class="val">${player.pred_defcon}%</div><div class="lbl">Defcon %</div></div>
        <div class="stat"><div class="val">${player.pred_bonus}</div><div class="lbl">Prj Bonus</div></div>
      </div>
      <div class="stats-row">
        <div class="stat"><div class="val">${player.median}</div><div class="lbl">Median</div></div>
        <div class="stat"><div class="val">${player.p10}&ndash;${player.p90}</div><div class="lbl">P10-P90 range</div></div>
        <div class="stat"><div class="val">${player.p_10plus}%</div><div class="lbl">P(10+)</div></div>
        <div class="stat"><div class="val">${player.p_15plus}%</div><div class="lbl">P(15+)</div></div>
      </div>
      <div class="expanded-section">
        <div class="prob-grid"></div>
      </div>
      <div class="tap-hint">Tap for probability breakdown</div>
    `;

    // --- Tap to expand ---
    card.addEventListener('click', () => {
      const wasExpanded = card.classList.contains('expanded');
      // Collapse all
      container.querySelectorAll('.card.expanded').forEach(c => c.classList.remove('expanded'));
      if (!wasExpanded) card.classList.add('expanded');
    });

    container.appendChild(card);

    // --- Draw mini KDE chart ---
    const chartEl = card.querySelector('.chart-area');
    const cW = 400, cH = 55;
    const svg = d3.select(chartEl).append('svg')
      .attr('viewBox', `0 0 ${cW} ${cH}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    const x = d3.scaleLinear().domain([DATA.x_min, DATA.x_max]).range([0, cW]);
    const maxY = player.kde_y.reduce((a, b) => a > b ? a : b, 1e-9);
    const y = d3.scaleLinear().domain([0, maxY]).range([cH - 8, 4]);

    // Axis ticks
    [0, 5, 10, 15, 20].filter(v => v >= DATA.x_min && v <= DATA.x_max).forEach(v => {
      svg.append('line')
        .attr('x1', x(v)).attr('x2', x(v))
        .attr('y1', 0).attr('y2', cH - 8)
        .attr('stroke', '#21262d').attr('stroke-dasharray', '2,2');
      svg.append('text')
        .attr('x', x(v)).attr('y', cH)
        .attr('text-anchor', 'middle').attr('font-size', '8px').attr('fill', '#484f58')
        .text(v);
    });

    // KDE area
    const pts = player.kde_x.map((xv, j) => [x(xv), y(player.kde_y[j])]);
    const area = d3.area()
      .x(d => d[0]).y0(cH - 8).y1(d => d[1]).curve(d3.curveBasis);
    const line = d3.line().x(d => d[0]).y(d => d[1]).curve(d3.curveBasis);

    svg.append('path').datum(pts).attr('d', area)
      .attr('fill', color).attr('fill-opacity', 0.35);
    svg.append('path').datum(pts).attr('d', line)
      .attr('stroke', color).attr('stroke-width', 1.5).attr('fill', 'none');

    // E[pts] line
    svg.append('line')
      .attr('x1', x(player.exp_pts)).attr('x2', x(player.exp_pts))
      .attr('y1', 4).attr('y2', cH - 8)
      .attr('stroke', '#f85149').attr('stroke-width', 1).attr('stroke-dasharray', '3,3')
      .attr('opacity', 0.7);

    // --- Probability grid (expanded section) ---
    const probGrid = card.querySelector('.prob-grid');
    PROB_THRESHOLDS.forEach(thr => {
      const pct = player.cum_prob[String(thr)] || 0;
      const barColor = pct > 30 ? '#3fb950' : pct > 10 ? '#d29922' : '#f85149';
      const cell = document.createElement('div');
      cell.className = 'prob-cell';
      cell.innerHTML = `
        <div class="threshold">&ge;${thr} pts</div>
        <div class="pct">${pct}%</div>
        <div class="bar"><div class="bar-fill" style="width:${pct}%;background:${barColor}"></div></div>
      `;
      probGrid.appendChild(cell);
    });
  });
}

render();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Responsive wrapper — hosts desktop + mobile as inert <template> elements
# and activates the correct one based on viewport width.
# ---------------------------------------------------------------------------

_RESPONSIVE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FPL Points Distribution</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
</style>
</head>
<body>
<template id="desktop-tmpl">
<style>
__DESKTOP_STYLE__
</style>
__DESKTOP_BODY__
<script>
__DESKTOP_SCRIPT__
</script>
</template>
<template id="mobile-tmpl">
<style>
__MOBILE_STYLE__
</style>
__MOBILE_BODY__
<script>
__MOBILE_SCRIPT__
</script>
</template>
<script>
window.DATA = /*__DATA__*/null;
function activate(id) {
  const tmpl = document.getElementById(id);
  if (!tmpl) return;
  const frag = tmpl.content.cloneNode(true);
  frag.querySelectorAll('script').forEach(old => {
    const s = document.createElement('script');
    for (const attr of old.attributes) s.setAttribute(attr.name, attr.value);
    s.textContent = old.textContent;
    old.parentNode.replaceChild(s, old);
  });
  document.body.appendChild(frag);
}
activate(window.matchMedia('(max-width: 768px)').matches ? 'mobile-tmpl' : 'desktop-tmpl');
</script>
</body>
</html>
"""
