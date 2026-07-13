"""
Parameter study orchestrator for free_surface_SchanChen_nondim_D2Q9_08_Zhang_Cap02.py

Phase 1 (this file, as configured): sweeps the cartesian product of
tau_g x tau_f (60 combos), holding vf_theta fixed at FIXED_VF_THETA. Once
those results are in, phase 2 (a separate vf_theta-only sweep, locking in
whichever tau_g/tau_f phase 1 picks) can reuse this same structure.

Launches the sim once per combination via CLI overrides
(--tau_g/--tau_f/--vf_theta). Each run is aborted early if a NaN/Inf/crash is
detected in its log (no point burning iterations on a blown-up sim), and the
sweep continues with the next combination regardless. After every run, a CSV
row and an HTML summary (embedding the run's final phi snapshot) are
written/refreshed, so partial progress is inspectable even if the sweep is
interrupted.

Usage: python param_study.py    (run this file directly, no CLI args needed)
"""
import csv
import glob
import os
import re
import subprocess
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
SIM_SCRIPT = os.path.join(SCRIPT_DIR, 'free_surface_SchanChen_nondim_D2Q9_08_Zhang_Cap02.py')
IMAGES_ROOT = os.path.join(SCRIPT_DIR, 'FreesurfaceImages')
PHI_RESULTS_FILE = os.path.join(SCRIPT_DIR, 'phi_results.txt')
RUNS_DIR = os.path.join(SCRIPT_DIR, 'param_study_runs')
RESULTS_CSV = os.path.join(SCRIPT_DIR, 'param_study_phase1_tau_results.csv')
REPORT_HTML = os.path.join(SCRIPT_DIR, 'param_study_phase1_tau_report.html')

# Must match ACTIVE_CASE / DEFAULT_D_ND / CAPILLARY_PROOF.Kf in the sim script,
# since the images_dir folder name is derived from these.
ACTIVE_CASE = 'proof_capillary'
NODES = 300
KF_BASELINE = 0.002

TAU_G_RANGE = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
TAU_F_RANGE = [1.00, 1.10, 1.20, 1.30, 1.50, 2.00]
FIXED_VF_THETA = 60.0  # held constant for phase 1; phase 2 sweeps this instead

POLL_SECONDS = 30
STALL_LIMIT = 5  # 5 x 30s = 2.5 min with no progress -> kill as stalled


def make_param_sets():
    sets = []
    n = 0
    for tau_g in TAU_G_RANGE:
        for tau_f in TAU_F_RANGE:
            n += 1
            sets.append(dict(
                run_index=n,
                tau_g=float(tau_g),
                tau_f=float(tau_f),
                vf_theta=float(FIXED_VF_THETA),
                label=f"run{n:03d}_taug{tau_g}_tauf{tau_f}",
            ))
    return sets


def images_dir_for(tau_g, tau_f, vf_theta):
    """Predict the output folder the sim will create for these params
    (mirrors PARAMETER_STUB construction in the sim script)."""
    script_filename = os.path.basename(SIM_SCRIPT).rsplit('.py', 1)[0]
    stub = (
        f"__{ACTIVE_CASE}__nodes_{NODES}__tau_f_{tau_f}"
        f"__tau_g_{tau_g}__Kf_{KF_BASELINE}__Theta_{vf_theta}"
    )
    return os.path.join(IMAGES_ROOT, script_filename + stub)


def build_cmd(params):
    return [
        PYTHON, SIM_SCRIPT,
        '--tau_g', str(params['tau_g']),
        '--tau_f', str(params['tau_f']),
        '--vf_theta', str(params['vf_theta']),
    ]


def get_progress(text):
    for line in reversed(text.splitlines()):
        if 'Simulation Execution' in line and 'iteration:' in line:
            try:
                return int(line.split('iteration:')[1].split(';')[0].strip())
            except (IndexError, ValueError):
                pass
    return 0


def has_crashed(text):
    return 'Traceback (most recent call last)' in text


def last_sigma_ratio(text):
    matches = re.findall(r'\[eq49 check\].*?ratio=([\d.eE+-]+)', text)
    return float(matches[-1]) if matches else None


def compute_meniscus_rise(phi_results_path):
    """Reads phi_results.txt (written every iteration by the sim, so this is
    the final iteration's field once the process has exited) and measures
    the phi=0.5 interface height at the left/right edges relative to the
    centerline (x=Xn//2). Positive = edges sit higher than center (meniscus
    rises at the walls); negative = edges sit lower (falls)."""
    if not os.path.exists(phi_results_path):
        return None

    with open(phi_results_path, 'r') as f:
        f.readline()  # header: "phi_min phi_max"
        data = np.loadtxt(f)

    if data.ndim != 2:
        return None

    full = data[::-1, :].T  # full[x, y], y=0 bottom .. y=Yn+1 top
    x_ext, y_ext = full.shape
    y_idx = np.arange(y_ext)

    def interface_y(col):
        # Find the actual phi=0.5 crossing via a sign change, not "first index
        # >= 0.5" -- that only works if the profile rises with y. Here phi
        # falls with y (liquid near y=0, gas near y=Yn+1), so argmax-of-bool
        # would just return index 0 every time (already True) and never find
        # the real transition.
        sign = np.sign(col - 0.5)
        changes = np.where(np.diff(sign) != 0)[0]
        if len(changes) == 0:
            return None
        idx = int(changes[0])
        y0, y1 = y_idx[idx], y_idx[idx + 1]
        p0, p1 = col[idx], col[idx + 1]
        if p1 == p0:
            return float(y0)
        return float(y0 + (0.5 - p0) * (y1 - y0) / (p1 - p0))

    y_center = interface_y(full[x_ext // 2, :])
    y_left = interface_y(full[1, :])
    y_right = interface_y(full[x_ext - 2, :])

    if y_center is None or y_left is None or y_right is None:
        return None

    return (y_left - y_center + y_right - y_center) / 2.0


def latest_snapshot(images_dir):
    if not os.path.isdir(images_dir):
        return None
    hits = sorted(glob.glob(os.path.join(images_dir, 'phi_snapshot_iter_*.png')))
    return hits[-1] if hits else None


def run_one(params, total):
    label = params['label']
    run_log_path = os.path.join(RUNS_DIR, f'{label}.log')
    os.makedirs(RUNS_DIR, exist_ok=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"[ORCHESTRATOR] Run {params['run_index']}/{total} — {label}", flush=True)
    print(f"[ORCHESTRATOR] tau_g={params['tau_g']} tau_f={params['tau_f']} "
          f"vf_theta={params['vf_theta']}", flush=True)

    cmd = build_cmd(params)
    with open(run_log_path, 'w') as log_f:
        proc = subprocess.Popen(cmd, cwd=SCRIPT_DIR, stdout=log_f, stderr=subprocess.STDOUT)

    last_progress = 0
    stall_count = 0
    result = 'UNKNOWN'

    while True:
        time.sleep(POLL_SECONDS)

        with open(run_log_path, 'r', errors='replace') as f:
            text = f.read()

        if has_crashed(text):
            print(f"[ORCHESTRATOR] Crash/instability detected — killing", flush=True)
            proc.kill()
            proc.wait()
            result = 'UNSTABLE'
            break

        if proc.poll() is not None:
            result = 'DONE' if proc.returncode == 0 else 'FAILED'
            break

        progress = get_progress(text)
        print(f"[ORCHESTRATOR] {label}: iter={progress}", flush=True)

        if progress == last_progress:
            stall_count += 1
            if stall_count >= STALL_LIMIT:
                print(f"[ORCHESTRATOR] Stalled at iter={progress} — killing", flush=True)
                proc.kill()
                proc.wait()
                result = 'STALLED'
                break
        else:
            stall_count = 0
        last_progress = progress

    with open(run_log_path, 'r', errors='replace') as f:
        final_text = f.read()

    sigma_ratio = last_sigma_ratio(final_text)
    meniscus_rise = compute_meniscus_rise(PHI_RESULTS_FILE) if result == 'DONE' else None

    run_dir = os.path.join(RUNS_DIR, label)
    os.makedirs(run_dir, exist_ok=True)
    image_rel_path = None
    src_img = latest_snapshot(images_dir_for(params['tau_g'], params['tau_f'], params['vf_theta']))
    if src_img:
        dst_img = os.path.join(run_dir, 'final_snapshot.png')
        with open(src_img, 'rb') as sf, open(dst_img, 'wb') as df:
            df.write(sf.read())
        image_rel_path = os.path.relpath(dst_img, SCRIPT_DIR)

    print(f"[ORCHESTRATOR] {label}: FINISHED — {result}  "
          f"meniscus_rise={meniscus_rise}  sigma_ratio={sigma_ratio}", flush=True)

    return dict(
        result=result,
        sigma_ratio=sigma_ratio,
        meniscus_rise=meniscus_rise,
        image_path=image_rel_path,
    )


def get_completed_labels():
    if not os.path.exists(RESULTS_CSV):
        return set()
    with open(RESULTS_CSV, 'r', newline='') as f:
        return {row['label'] for row in csv.DictReader(f)}


FIELDNAMES = ['run', 'label', 'tau_g', 'tau_f', 'vf_theta',
              'meniscus_rise', 'sigma_ratio', 'result', 'image_path']


def write_csv_row(row):
    existing = []
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'r', newline='') as f:
            existing = list(csv.DictReader(f))

    updated = False
    for i, r in enumerate(existing):
        if r['label'] == row['label']:
            existing[i] = row
            updated = True
            break
    if not updated:
        existing.append(row)

    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(existing)

    return existing


def write_html_report(rows):
    def fmt(v, digits=4):
        if v in (None, ''):
            return 'n/a'
        try:
            return f'{float(v):.{digits}f}'
        except ValueError:
            return str(v)

    parts = [
        '<html><head><meta charset="utf-8"><title>Param Study: tau_g x tau_f x vf_theta</title>',
        '<style>',
        'body{font-family:sans-serif;font-size:13px;} table{border-collapse:collapse;width:100%;}',
        'th,td{border:1px solid #ccc;padding:4px 8px;text-align:center;vertical-align:middle;}',
        'th{background:#eee;position:sticky;top:0;} img{max-height:120px;}',
        '.ok{color:green;font-weight:bold;} .fail{color:#b00;font-weight:bold;}',
        '</style></head><body>',
        '<h2>Parameter study: tau_g x tau_f x vf_theta</h2>',
        '<table>',
        '<tr><th>run</th><th>tau_g</th><th>tau_f</th><th>vf_theta</th>'
        '<th>meniscus rise/fall</th><th>sigma check (ratio)</th>'
        '<th>result</th><th>snapshot</th></tr>',
    ]

    for r in rows:
        ok = r['result'] == 'DONE'
        css = 'ok' if ok else 'fail'
        label_txt = 'SUCCESS' if ok else f"FAIL ({r['result']})"
        img_html = (f'<img src="{r["image_path"]}">' if r.get('image_path') else 'n/a')
        parts.append(
            f"<tr><td>{r['run']}</td><td>{r['tau_g']}</td><td>{r['tau_f']}</td>"
            f"<td>{r['vf_theta']}</td><td>{fmt(r['meniscus_rise'])}</td>"
            f"<td>{fmt(r['sigma_ratio'])}</td>"
            f"<td class=\"{css}\">{label_txt}</td><td>{img_html}</td></tr>"
        )

    parts.append('</table></body></html>')

    with open(REPORT_HTML, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))


def main():
    param_sets = make_param_sets()
    total = len(param_sets)
    completed = get_completed_labels()
    print(f"[ORCHESTRATOR] Parameter study starting: {total} runs "
          f"({len(completed)} already done)", flush=True)

    for params in param_sets:
        label = params['label']
        if label in completed:
            print(f"[ORCHESTRATOR] Skipping {label} — already in CSV", flush=True)
            continue

        outcome = run_one(params, total)
        row = dict(
            run=params['run_index'], label=label,
            tau_g=params['tau_g'], tau_f=params['tau_f'], vf_theta=params['vf_theta'],
            meniscus_rise=outcome['meniscus_rise'], sigma_ratio=outcome['sigma_ratio'],
            result=outcome['result'], image_path=outcome['image_path'],
        )
        rows = write_csv_row(row)
        write_html_report(rows)
        print(f"[ORCHESTRATOR] Progress: {params['run_index']}/{total} complete", flush=True)

    print(f"[ORCHESTRATOR] ALL DONE — {total} runs complete. "
          f"See {RESULTS_CSV} and {REPORT_HTML}", flush=True)


if __name__ == '__main__':
    main()
