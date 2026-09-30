"""Standalone HTML report for the isolated five-gameweek forecast."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .five_week import ROLE_COLORS, ROLE_NAMES, WeekForecast


def _histogram(samples: np.ndarray, scale: float = 1.0) -> List[Dict]:
    rounded = np.rint(np.asarray(samples, dtype=float)).astype(int)
    values, counts = np.unique(rounded, return_counts=True)
    return [
        {"value": round(float(value) / scale, 2), "probability": round(float(count / counts.sum()), 5)}
        for value, count in zip(values, counts)
    ]


def _distribution(samples: np.ndarray, scale: float = 1.0,
                  upside_threshold: float = 10.0) -> Dict:
    samples = np.asarray(samples, dtype=float) / scale
    return {
        "mean": round(float(samples.mean()), 3),
        "p10": round(float(np.percentile(samples, 10)), 2),
        "p25": round(float(np.percentile(samples, 25)), 2),
        "median": round(float(np.percentile(samples, 50)), 2),
        "p75": round(float(np.percentile(samples, 75)), 2),
        "p90": round(float(np.percentile(samples, 90)), 2),
        "upside": round(float(np.mean(samples >= upside_threshold)), 4),
        "upside_label": f"P({upside_threshold:g}+)",
        "histogram": _histogram(samples * scale, scale=scale),
    }


def build_report_payload(
    forecasts: Dict[int, WeekForecast],
    season: str,
    model_metrics: Dict,
) -> Dict:
    """Combine fixture rows and simulations into one player-level report payload."""
    gameweeks = sorted(forecasts)
    n_sims = next(
        (
            len(samples)
            for forecast in forecasts.values()
            for samples in forecast.simulations.values()
        ),
        0,
    )
    players: Dict[str, Dict] = {}
    total_samples: Dict[str, np.ndarray] = {}

    for gameweek in gameweeks:
        forecast = forecasts[gameweek]
        frame = forecast.predictions.reset_index(drop=True)
        if frame.empty:
            continue
        for player_id, indices in frame.groupby("player_id", sort=False).groups.items():
            idxs = list(indices)
            rows = frame.loc[idxs]
            first = rows.iloc[0]
            key = str(player_id)
            player = players.setdefault(key, {
                "id": key,
                "name": str(first.get("player_name", "")),
                "team": str(first.get("team", "")),
                "position": str(first.get("fpl_position", "MID")),
                "weeks": {},
            })
            # A DGW has multiple fixture rows. Summing samples with the same index
            # produces the gameweek total while keeping a single table column.
            sims = np.zeros(n_sims, dtype=float)
            for idx in idxs:
                sims += forecast.simulations[int(idx)]

            fixture_parts = []
            fixture_ids = []
            for _, row in rows.iterrows():
                opponent = str(row.get("opponent", ""))
                home = int(row.get("is_home", 0)) == 1
                fixture_parts.append(f"vs {opponent}" if home else f"@ {opponent}")
                teams = sorted([str(row.get("team", "")), opponent])
                fixture_ids.append(f"gw{gameweek}:{'|'.join(teams)}")

            role_prob = [float(rows[f"role_prob_{i}"].mean()) for i in range(4)]
            week_data = {
                "gameweek": int(gameweek),
                "fixture": " / ".join(fixture_parts),
                "fixture_ids": fixture_ids,
                "xpts": round(float(sims.mean()), 3),
                "appearance": round(float(1.0 - np.prod(1.0 - rows["pred_appear_prob"])), 4),
                "minutes": round(float(rows["pred_minutes_uncond"].sum()), 1),
                "minutes_if_playing": round(float(rows["pred_minutes"].mean()), 1),
                "xg": round(float(rows["pred_exp_goals"].sum()), 3),
                "xa": round(float(rows["pred_exp_assists"].sum()), 3),
                "clean_sheet": round(float(rows["pred_cs_prob"].mean()), 4),
                "roles": [round(value, 4) for value in role_prob],
                "distribution": _distribution(sims, upside_threshold=10),
            }
            player["weeks"][str(gameweek)] = week_data
            total_samples.setdefault(key, np.zeros(n_sims, dtype=float))
            total_samples[key] += sims

    for key, player in players.items():
        sims = total_samples[key]
        player["total"] = _distribution(sims, upside_threshold=30)
        player["average"] = _distribution(sims, scale=max(len(gameweeks), 1), upside_threshold=6)
        player["five_week_total"] = player["total"]["mean"]
        player["average_xpts"] = player["average"]["mean"]

    fixture_options = {}
    for player in players.values():
        for week in player["weeks"].values():
            for fixture_id in week["fixture_ids"]:
                _, teams = fixture_id.split(":", 1)
                fixture_options[fixture_id] = f"GW{week['gameweek']} · {teams.replace('|', ' v ')}"

    return {
        "meta": {
            "season": season,
            "gameweeks": gameweeks,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_simulations": n_sims,
            "model_metrics": model_metrics,
            "role_names": list(ROLE_NAMES),
            "role_colors": list(ROLE_COLORS),
        },
        "fixtures": [
            {"id": key, "label": fixture_options[key]}
            for key in sorted(fixture_options, key=lambda item: (int(item.split(":")[0][2:]), item))
        ],
        "players": sorted(players.values(), key=lambda player: player["average_xpts"], reverse=True),
    }


def render_five_week_html(payload: Dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = _HTML.replace("__REPORT_DATA__", data_json)
    output_path.write_text(html, encoding="utf-8")
    return output_path


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FPL 5-Gameweek Forecast</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#c9d1d9;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;padding:20px 10px 40px}
.shell{width:100%;max-width:1280px;margin:0 auto}
.hero{text-align:center;margin-bottom:18px}
.hero h1{font-size:22px;font-weight:700;color:#e6edf3;margin-bottom:4px}
.hero p{font-size:13px;color:#8b949e;max-width:720px;margin:0 auto}
.badge{display:inline-block;margin-top:8px;font-size:11px;color:#8b949e;background:#161b22;border:1px solid #21262d;border-radius:6px;padding:3px 10px}
.summary{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:14px}
.stat{padding:10px 14px;background:#161b22;border:1px solid #21262d;border-radius:8px}
.stat span{display:block;color:#8b949e;font-size:11px}
.stat strong{display:block;font-size:16px;color:#e6edf3;margin-top:2px}
.toolbar{display:grid;grid-template-columns:minmax(200px,1.5fr) repeat(3,minmax(140px,.7fr));gap:8px;margin-bottom:10px}
.control{width:100%;height:32px;border-radius:6px;border:1px solid #30363d;background:#21262d;color:#c9d1d9;padding:0 10px;font:inherit;font-size:12px;outline:0}
.control:focus{border-color:#58a6ff}
.table-wrap{overflow:auto;border:1px solid #21262d;border-radius:8px;background:#0d1117;max-height:74vh}
table{width:100%;min-width:1180px;border-collapse:separate;border-spacing:0}
th{position:sticky;top:0;z-index:3;background:#161b22;color:#8b949e;text-align:left;font-size:11px;font-weight:600;border-bottom:1px solid #30363d;padding:0}
th button{all:unset;box-sizing:border-box;cursor:pointer;width:100%;padding:10px}
th button:hover{color:#58a6ff}
th.sorted button{color:#58a6ff}
td{padding:9px 10px;border-bottom:1px solid #21262d;vertical-align:middle}
tbody tr:hover td{background:#161b22}
th:first-child,td:first-child{position:sticky;left:0;z-index:2;background:#0d1117}
th:first-child{z-index:4;background:#161b22}
tbody tr:hover td:first-child{background:#161b22}
.player{display:flex;align-items:center;gap:10px;min-width:180px}
.pos{display:grid;place-items:center;width:32px;height:22px;border-radius:4px;background:#21262d;color:#c9d1d9;font-size:10px;font-weight:700}
.name{font-weight:600;color:#e6edf3}
.team{font-size:11px;color:#8b949e}
.gwcell{min-width:140px}
.score{border:0;background:none;color:#e6edf3;font:700 16px/1 inherit;padding:0;cursor:pointer;border-bottom:1px dashed #484f58}
.score:hover{color:#58a6ff;border-color:#58a6ff}
.fixture{font-size:11px;color:#8b949e;margin-top:5px;max-width:140px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.subline{font-size:10px;color:#6e7681;margin-top:1px}
.totals{background:#0f141b}
.total-score{color:#f0883e}
.empty{color:#484f58}
.no-results{text-align:center;color:#8b949e;padding:40px!important}
.legend{display:flex;justify-content:space-between;gap:16px;color:#8b949e;font-size:11px;margin-top:10px}
.legend strong{color:#c9d1d9}
dialog{width:min(700px,calc(100% - 24px));margin:auto;border:1px solid #30363d;border-radius:8px;background:#161b22;color:#c9d1d9;box-shadow:0 8px 24px rgba(0,0,0,.4)}
dialog::backdrop{background:rgba(1,4,9,.75)}
.modal-head{display:flex;justify-content:space-between;gap:16px;padding:14px 18px;border-bottom:1px solid #21262d}
.modal-head h2{font-size:16px;color:#e6edf3}
.modal-head p{color:#8b949e;font-size:12px;margin-top:2px}
.close{border:1px solid #30363d;border-radius:6px;background:#21262d;color:#c9d1d9;width:30px;height:30px;font-size:18px;cursor:pointer}
.close:hover{border-color:#58a6ff;color:#58a6ff}
.modal-body{padding:16px 18px 18px}
.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
.metric{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:8px 10px}
.metric span{display:block;color:#8b949e;font-size:10px}
.metric strong{font-size:15px;color:#e6edf3}
.chart{height:240px;margin:16px 0 6px}
.chart svg{width:100%;height:100%;overflow:visible}
.axis{stroke:#30363d;stroke-width:1}
.bar{fill:#58a6ff;opacity:.85}
.bar:hover{fill:#f0883e;opacity:1}
.chart text{fill:#8b949e;font-size:10px}
.roles{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:14px}
.role{border-radius:6px;padding:8px;background:#0d1117;border:1px solid #21262d}
.role i{display:block;height:3px;border-radius:3px;margin-bottom:6px}
.role span{color:#8b949e;font-size:10px}
.role strong{display:block;color:#e6edf3}
.method{margin-top:16px;padding-top:12px;border-top:1px solid #21262d;color:#8b949e;font-size:11px}
@media(max-width:820px){.summary{grid-template-columns:repeat(2,1fr)}.toolbar{grid-template-columns:1fr 1fr}.toolbar input{grid-column:1/-1}.metrics,.roles{grid-template-columns:repeat(2,1fr)}.legend{display:block}.legend span{display:block;margin-top:4px}}
</style>
</head>
<body>
<main class="shell">
  <header class="hero">
    <h1>FPL 5-Gameweek Forecast</h1>
    <p>Fixture-level projections with playing-time uncertainty carried forward. Click any points number to see its simulated distribution.</p>
    <div class="badge" id="generatedBadge"></div>
  </header>
  <section class="summary" id="summary"></section>
  <section class="toolbar" aria-label="Table filters">
    <input class="control" id="search" type="search" placeholder="Search player or team…">
    <select class="control" id="teamFilter"><option value="">All teams</option></select>
    <select class="control" id="positionFilter"><option value="">All positions</option><option>GK</option><option>DEF</option><option>MID</option><option>FWD</option></select>
    <select class="control" id="fixtureFilter"><option value="">All fixtures</option></select>
  </section>
  <div class="table-wrap">
    <table><thead><tr id="headRow"><th data-key="name"><button>Player</button></th></tr></thead><tbody id="body"></tbody></table>
  </div>
  <div class="legend"><span><strong id="visibleCount"></strong> players shown · Default sort: average xPts</span><span>Pts are unconditional: a zero-minute outcome is included in every distribution.</span></div>
</main>
<dialog id="distributionModal">
  <div class="modal-head"><div><h2 id="modalTitle"></h2><p id="modalSubtitle"></p></div><button class="close" id="modalClose" aria-label="Close">×</button></div>
  <div class="modal-body"><div class="metrics" id="modalMetrics"></div><div class="chart" id="chart"></div><div class="roles" id="roles"></div><div class="method">Each bar is the probability of that points outcome across the Monte Carlo simulations. The role model separately learns absent, substitute, 60–79 minute, and 80+ minute outcomes for GW+1 through GW+5.</div></div>
</dialog>
<script id="reportData" type="application/json">__REPORT_DATA__</script>
<script>
const report=JSON.parse(document.getElementById('reportData').textContent);const gws=report.meta.gameweeks;const byId=new Map(report.players.map(p=>[p.id,p]));let sortKey='avg',sortDir='desc';
const fmt=n=>Number(n).toFixed(2);const pct=n=>`${Math.round(Number(n)*100)}%`;const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
document.getElementById('generatedBadge').textContent=`${report.meta.season} · ${report.meta.n_simulations.toLocaleString()} simulations`;
const selected=report.meta.model_metrics.selected||'direct horizon';const score=report.meta.model_metrics[selected]?.multiclass_log_loss;
document.getElementById('summary').innerHTML=[['Window',`GW${gws[0]}–GW${gws.at(-1)}`],['Players',report.players.length],['Minutes model',selected==='durability'?'Durability selected':'Baseline selected'],['Holdout log loss',score==null?'—':Number(score).toFixed(3)]].map(([a,b])=>`<div class="stat"><span>${a}</span><strong>${b}</strong></div>`).join('');
const head=document.getElementById('headRow');for(const gw of gws)head.insertAdjacentHTML('beforeend',`<th data-key="gw:${gw}"><button>GW${gw}</button></th>`);head.insertAdjacentHTML('beforeend','<th data-key="total"><button>5GW total</button></th><th data-key="avg"><button>Average</button></th>');
const teams=[...new Set(report.players.map(p=>p.team))].sort();document.getElementById('teamFilter').insertAdjacentHTML('beforeend',teams.map(t=>`<option value="${esc(t)}">${esc(t)}</option>`).join(''));
document.getElementById('fixtureFilter').insertAdjacentHTML('beforeend',report.fixtures.map(f=>`<option value="${esc(f.id)}">${esc(f.label)}</option>`).join(''));
function valueFor(p,key){if(key==='name')return p.name.toLowerCase();if(key==='total')return p.five_week_total;if(key==='avg')return p.average_xpts;if(key.startsWith('gw:'))return p.weeks[key.slice(3)]?.xpts??-999;return 0}
function render(){const q=document.getElementById('search').value.trim().toLowerCase(),team=document.getElementById('teamFilter').value,pos=document.getElementById('positionFilter').value,fixture=document.getElementById('fixtureFilter').value;let rows=report.players.filter(p=>(!q||`${p.name} ${p.team}`.toLowerCase().includes(q))&&(!team||p.team===team)&&(!pos||p.position===pos)&&(!fixture||Object.values(p.weeks).some(w=>w.fixture_ids.includes(fixture))));rows.sort((a,b)=>{const av=valueFor(a,sortKey),bv=valueFor(b,sortKey),d=typeof av==='string'?av.localeCompare(bv):av-bv;return sortDir==='asc'?d:-d});document.querySelectorAll('th').forEach(th=>th.classList.toggle('sorted',th.dataset.key===sortKey));const body=document.getElementById('body');if(!rows.length){body.innerHTML=`<tr><td class="no-results" colspan="${gws.length+3}">No players match these filters.</td></tr>`}else body.innerHTML=rows.map(p=>{const weeks=gws.map(gw=>{const w=p.weeks[String(gw)];return w?`<td class="gwcell"><button class="score" data-player="${esc(p.id)}" data-scope="gw:${gw}">${fmt(w.xpts)}</button><div class="fixture" title="${esc(w.fixture)}">${esc(w.fixture)}</div><div class="subline">${pct(w.appearance)} play · ${Math.round(w.minutes)} xMin</div></td>`:'<td class="empty">—</td>'}).join('');return `<tr><td><div class="player"><span class="pos">${esc(p.position)}</span><div><div class="name">${esc(p.name)}</div><div class="team">${esc(p.team)}</div></div></div></td>${weeks}<td class="totals"><button class="score total-score" data-player="${esc(p.id)}" data-scope="total">${fmt(p.five_week_total)}</button><div class="subline">P30+ ${pct(p.total.upside)}</div></td><td class="totals"><button class="score total-score" data-player="${esc(p.id)}" data-scope="average">${fmt(p.average_xpts)}</button><div class="subline">per GW</div></td></tr>`}).join('');document.getElementById('visibleCount').textContent=rows.length}
document.querySelectorAll('th button').forEach(btn=>btn.addEventListener('click',()=>{const key=btn.parentElement.dataset.key;if(sortKey===key)sortDir=sortDir==='desc'?'asc':'desc';else{sortKey=key;sortDir=key==='name'?'asc':'desc'}render()}));document.querySelectorAll('.control').forEach(el=>el.addEventListener(el.tagName==='INPUT'?'input':'change',render));
function metric(label,value){return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`}
function makeChart(d){const h=d.histogram;if(!h.length)return '';const W=650,H=235,pad={l:34,r:8,t:10,b:28},max=Math.max(...h.map(x=>x.probability),.01),bw=(W-pad.l-pad.r)/h.length;let bars=h.map((x,i)=>{const bh=(H-pad.t-pad.b)*x.probability/max,xx=pad.l+i*bw,yy=H-pad.b-bh;return `<rect class="bar" x="${xx+1}" y="${yy}" width="${Math.max(bw-2,1)}" height="${bh}"><title>${x.value} pts: ${pct(x.probability)}</title></rect>`}).join('');const step=Math.max(1,Math.ceil(h.length/9));let labels=h.map((x,i)=>i%step===0?`<text x="${pad.l+(i+.5)*bw}" y="${H-8}" text-anchor="middle">${x.value}</text>`:'').join('');return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Points probability distribution"><line class="axis" x1="${pad.l}" y1="${H-pad.b}" x2="${W-pad.r}" y2="${H-pad.b}"/>${bars}${labels}<text x="4" y="18">${pct(max)}</text></svg>`}
function openDistribution(playerId,scope){const p=byId.get(playerId);let d,subtitle,roles=null;if(scope.startsWith('gw:')){const gw=scope.slice(3),w=p.weeks[gw];d=w.distribution;subtitle=`GW${gw} · ${w.fixture} · ${pct(w.appearance)} chance to play`;roles=w.roles}else if(scope==='total'){d=p.total;subtitle=`GW${gws[0]}–GW${gws.at(-1)} total`}else{d=p.average;subtitle=`Average per gameweek · GW${gws[0]}–GW${gws.at(-1)}`};document.getElementById('modalTitle').textContent=p.name;document.getElementById('modalSubtitle').textContent=subtitle;document.getElementById('modalMetrics').innerHTML=metric('Mean',fmt(d.mean))+metric('Median',fmt(d.median))+metric('P10',fmt(d.p10))+metric('P90',fmt(d.p90))+metric(d.upside_label,pct(d.upside));document.getElementById('chart').innerHTML=makeChart(d);document.getElementById('roles').innerHTML=roles?roles.map((v,i)=>`<div class="role"><i style="background:${report.meta.role_colors[i]}"></i><span>${report.meta.role_names[i]}</span><strong>${pct(v)}</strong></div>`).join(''):'';document.getElementById('distributionModal').showModal()}
document.getElementById('body').addEventListener('click',e=>{const btn=e.target.closest('.score');if(btn)openDistribution(btn.dataset.player,btn.dataset.scope)});document.getElementById('modalClose').addEventListener('click',()=>document.getElementById('distributionModal').close());document.getElementById('distributionModal').addEventListener('click',e=>{if(e.target===e.currentTarget)e.currentTarget.close()});render();
</script>
</body></html>'''
