from src.viz import _build_metrics_html


def test_clean_sheet_display_name_resolves_team_side_sample_counts():
    metrics = {
        'sections': [{
            'title': 'Sub-Models (Holdout Test Set)',
            'rows': [{'model': 'Clean Sheet', 'metric': 'Poisson Dev', 'score': '1.1098'}],
        }],
        'sample_info': {
            'models': {'clean_sheet': {'train': 4454, 'test': 200, 'unit': 'team-sides'}},
        },
    }

    html = _build_metrics_html(metrics)

    assert '4,454 &rarr; 200' in html
    assert 'team-sides' in html
