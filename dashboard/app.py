from flask import Flask, render_template_string
import csv, os, re
from datetime import datetime

app = Flask(__name__)

RESULTS_CSV = '/opt/lbm/Lb-Python/results/param_study_results.csv'
STUDY_LOG   = '/opt/lbm/study.log'
TOTAL_RUNS  = 18

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<title>LBM Parameter Study</title>
<style>
:root{--bg:#0f1117;--surface:#1a1d27;--border:#2a2d3e;--green:#22c55e;--red:#ef4444;--yellow:#f59e0b;--blue:#3b82f6;--gray:#6b7280;--text:#e2e8f0}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Courier New',monospace;padding:1.5rem}
h1{font-size:1.1rem;font-weight:700;letter-spacing:.15em;color:var(--blue);margin-bottom:1.5rem;text-transform:uppercase}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin-bottom:1.5rem}
.card{background:var(--surface);border:1px solid var(--border);padding:1rem 1.25rem}
.card-label{font-size:.65rem;color:var(--gray);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.4rem}
.card-value{font-size:1.6rem;font-weight:700}
.green{color:var(--green)}.red{color:var(--red)}.yellow{color:var(--yellow)}.blue{color:var(--blue)}
.bar{background:var(--border);height:8px;margin-bottom:1.5rem}
.bar-fill{height:8px;background:var(--blue)}
.current{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--yellow);padding:1rem 1.25rem;margin-bottom:1.5rem;font-size:.82rem}
.current .lbl{color:var(--yellow);font-weight:700;font-size:.95rem;margin-bottom:.4rem}
.ibar{background:var(--border);height:5px;margin-top:.5rem}
.ibar-fill{height:5px;background:var(--yellow)}
table{width:100%;border-collapse:collapse;font-size:.78rem}
th{background:var(--surface);color:var(--gray);text-transform:uppercase;font-size:.65rem;letter-spacing:.08em;padding:.5rem .75rem;text-align:left;border-bottom:1px solid var(--border)}
td{padding:.45rem .75rem;border-bottom:1px solid var(--border)}
tr:hover td{background:var(--surface)}
.b{display:inline-block;padding:.15rem .5rem;font-size:.7rem;font-weight:700}
.b-done{background:#14532d;color:var(--green)}
.b-failed{background:#450a0a;color:var(--red)}
.b-unstable{background:#451a03;color:var(--yellow)}
.b-stalled{background:#1e1b4b;color:#818cf8}
.footer{font-size:.65rem;color:var(--gray);margin-top:1.5rem}
</style>
</head>
<body>
<h1>&#9927; LBM Parameter Study &mdash; Free Surface Shan-Chen D2Q9</h1>
<div class="grid">
  <div class="card"><div class="card-label">Total Runs</div><div class="card-value blue">{{total}}</div></div>
  <div class="card"><div class="card-label">Completed</div><div class="card-value green">{{completed}}</div></div>
  <div class="card"><div class="card-label">Done</div><div class="card-value green">{{done_count}}</div></div>
  <div class="card"><div class="card-label">Failed/Unstable</div><div class="card-value red">{{failed_count}}</div></div>
  <div class="card"><div class="card-label">Remaining</div><div class="card-value yellow">{{remaining}}</div></div>
</div>
<div class="bar"><div class="bar-fill" style="width:{{overall_pct}}%"></div></div>
{% if current %}
<div class="current">
  <div class="lbl">&#9654; Run {{current.run_index}}/{{total}} &mdash; {{current.label}} &nbsp; {{current.pct}}%</div>
  <div>iter={{current.iter}} &nbsp;|&nbsp; overall={{completed}}/{{total}} runs done</div>
  <div class="ibar"><div class="ibar-fill" style="width:{{current.pct}}%"></div></div>
</div>
{% else %}
<div class="current"><div class="lbl">&mdash; No active run</div></div>
{% endif %}
<table>
<thead><tr><th>#</th><th>Label</th><th>Result</th><th>tau_f</th><th>Kf</th><th>theta</th><th>vf_W</th><th>sigma</th><th>capMult</th><th>add_st</th></tr></thead>
<tbody>
{% for r in results %}
<tr>
  <td style="color:var(--gray)">{{loop.index}}</td>
  <td>{{r.label}}</td>
  <td><span class="b b-{{r.result.lower()}}">{{r.result}}</span></td>
  <td>{{r.tau_f}}</td><td>{{r.Kf}}</td><td>{{r.vf_theta}}</td>
  <td>{{r.vf_W}}</td><td>{{r.vf_sigma}}</td><td>{{r.vf_capMult}}</td><td>{{r.add_st}}</td>
</tr>
{% endfor %}
</tbody>
</table>
<div class="footer">Auto-refreshes every 30s &nbsp;|&nbsp; {{timestamp}}</div>
</body></html>"""


def read_csv():
    if not os.path.exists(RESULTS_CSV):
        return []
    with open(RESULTS_CSV, 'r') as f:
        return list(csv.DictReader(f))


def get_current_run():
    if not os.path.exists(STUDY_LOG):
        return None
    with open(STUDY_LOG, 'r', errors='replace') as f:
        lines = f.readlines()
    label, run_index, iter_val, pct = None, None, 0, 0.0
    for line in reversed(lines):
        m = re.search(r'Run (\d+)/\d+ (\S+): iter=(\d+) \(([0-9.]+)%\)', line)
        if m:
            run_index = int(m.group(1))
            label     = m.group(2)
            iter_val  = int(m.group(3))
            pct       = float(m.group(4))
            break
        m2 = re.search(r'Run (\d+)/\d+ — (\S+)', line)
        if m2 and not label:
            run_index = int(m2.group(1))
            label     = m2.group(2)
    if not label:
        return None
    return {"label": label, "run_index": run_index, "iter": iter_val, "pct": round(pct, 1)}


@app.route('/')
def index():
    results      = read_csv()
    current      = get_current_run()
    completed    = len(results)
    done_count   = sum(1 for r in results if r['result'] == 'DONE')
    failed_count = sum(1 for r in results if r['result'] in ('FAILED', 'UNSTABLE', 'STALLED'))
    overall_pct  = round(completed / TOTAL_RUNS * 100, 1)
    return render_template_string(HTML,
        results=results, current=current, total=TOTAL_RUNS,
        completed=completed, done_count=done_count, failed_count=failed_count,
        remaining=TOTAL_RUNS - completed, overall_pct=overall_pct,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=False)
