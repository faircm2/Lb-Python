import subprocess
import os
import time
import csv

PYTHON     = '/opt/lbm/venv/bin/python'
SCRIPT     = '/opt/lbm/Lb-Python/free_surface_SchanChen_nondim_D2Q9_08_Cap110.py'
WORK_DIR   = '/opt/lbm/Lb-Python'
LOG        = '/opt/lbm/run.log'
RESULTS_CSV= '/opt/lbm/Lb-Python/results/param_study_results.csv'
RUNS_DIR = '/opt/lbm/runs'

BASELINE = dict(
    nodes=200, tau_f=1.5, Kf=0.002,
    xi=6.0, vf_W=4, vf_sigma=0.01,
    vf_theta=60.0, vf_capMult=100.0,
    add_st=1, add_bf=1
)

def make_param_sets():
    sets = []

    # 1. vf_theta — KEY variable
    for v in [30, 60, 90, 120, 150]:
        p = BASELINE.copy(); p['vf_theta'] = v
        p['_label'] = f'theta_{v}'
        sets.append(p)

    # 2. Kf
    for v in [0.001, 0.005, 0.01]:
        p = BASELINE.copy(); p['Kf'] = v
        p['_label'] = f'Kf_{v}'
        sets.append(p)

    # 3. tau_f
    for v in [0.7, 1.0, 2.0]:
        p = BASELINE.copy(); p['tau_f'] = v
        p['_label'] = f'tau_f_{v}'
        sets.append(p)

    # 4. vf_W
    for v in [3, 6]:
        p = BASELINE.copy(); p['vf_W'] = v
        p['_label'] = f'vf_W_{v}'
        sets.append(p)

    # 5. vf_sigma
    for v in [0.005, 0.05]:
        p = BASELINE.copy(); p['vf_sigma'] = v
        p['_label'] = f'vf_sigma_{v}'
        sets.append(p)

    # 6. vf_capMult
    for v in [1, 10]:
        p = BASELINE.copy(); p['vf_capMult'] = v
        p['_label'] = f'capMult_{v}'
        sets.append(p)

    # 7. Zhang force off
    p = BASELINE.copy(); p['add_st'] = 0
    p['_label'] = 'zhang_off'
    sets.append(p)

    return sets


def build_cmd(params):
    cmd = [PYTHON, SCRIPT]
    for k, v in params.items():
        if k.startswith('_'):
            continue
        cmd += [f'--{k}', str(v)]
    return cmd


def get_completed_labels():
    if not os.path.exists(RESULTS_CSV):
        return set()
    with open(RESULTS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        return {row['label'] for row in reader}    


def check_log(log_path):
    """Returns: 'running', 'unstable', 'done'"""
    if not os.path.exists(log_path):
        return 'running'
    with open(log_path, 'r', errors='replace') as f:
        content = f.read()
    if 'NaN' in content or 'Traceback' in content:
        return 'unstable'
    if '100.0 %' in content:
        return 'done'
    # Early stop: still running after 3000 iters check
    return 'running'


def get_progress(log_path):
    """Extract current iteration from log."""
    if not os.path.exists(log_path):
        return 0
    with open(log_path, 'r', errors='replace') as f:
        lines = f.readlines()
    for line in reversed(lines):
        if 'Simulation Execution' in line:
            try:
                return int(line.split('iteration:')[1].split(';')[0].strip())
            except:
                pass
    return 0


def run_one(params, run_index, total):
    label    = params.get('_label', f'run_{run_index}')
    run_log  = os.path.join(RUNS_DIR, f'{label}.log')
    os.makedirs(RUNS_DIR, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"[ORCHESTRATOR] Run {run_index}/{total} — {label}", flush=True)
    print(f"[ORCHESTRATOR] Params: { {k:v for k,v in params.items() if not k.startswith('_')} }", flush=True)
    print(f"[ORCHESTRATOR] Log: {run_log}", flush=True)
    print(f"{'='*60}", flush=True)

    open(run_log, 'w').close()
    open(LOG, 'w').close()

    cmd = build_cmd(params)
    with open(run_log, 'w') as log_f:
        proc = subprocess.Popen(cmd, cwd=WORK_DIR, stdout=log_f, stderr=log_f)

    last_progress = 0
    stall_count   = 0
    result        = 'UNKNOWN'

    while True:
        time.sleep(30)

        if proc.poll() is not None:
            status = check_log(run_log)
            result = 'DONE' if status == 'done' else 'FAILED'
            break

        status = check_log(run_log)

        if status == 'unstable':
            print(f"[ORCHESTRATOR] Instability detected — killing", flush=True)
            proc.kill()
            result = 'UNSTABLE'
            break

        progress = get_progress(run_log)
        print(f"[ORCHESTRATOR] Run {run_index}/{total} {label}: iter={progress} ({100*progress/12001:.1f}%)  overall={run_index-1}/{total} runs done", flush=True)

        if progress >= 3000:
            if progress == last_progress:
                stall_count += 1
                if stall_count >= 3:
                    print(f"[ORCHESTRATOR] Stalled — killing", flush=True)
                    proc.kill()
                    result = 'STALLED'
                    break
            else:
                stall_count = 0
            last_progress = progress

    print(f"[ORCHESTRATOR] Run {run_index}/{total} {label}: FINISHED — {result}", flush=True)
    return result


def write_csv_row(label, params, result):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['label','result'] + 
                                [k for k in BASELINE.keys()])
        if write_header:
            writer.writeheader()
        row = {'label': label, 'result': result}
        row.update({k: params.get(k, BASELINE[k]) for k in BASELINE.keys()})
        writer.writerow(row)


def push_csv():
    """Commit and push results CSV to GitHub."""
    subprocess.run(['git', 'add', 'results/param_study_results.csv'],
                   cwd=WORK_DIR)
    subprocess.run(['git', 'commit', '-m', 'param_study: update results CSV'],
                   cwd=WORK_DIR)
    subprocess.run(['git', 'pull', '--rebase'], cwd=WORK_DIR)
    subprocess.run(['git', 'push'], cwd=WORK_DIR)


def main():
    param_sets = make_param_sets()
    total = len(param_sets)
    completed = get_completed_labels()
    print(f"[ORCHESTRATOR] Parameter study starting: {total} runs ({len(completed)} already done)", flush=True)

    for i, params in enumerate(param_sets, 1):
        label = params.get('_label', f'run_{i}')
        if label in completed:
            print(f"[ORCHESTRATOR] Skipping {label} — already in CSV", flush=True)
            continue
        result = run_one(params, i, total)
        write_csv_row(label, params, result)
        push_csv()
        print(f"[ORCHESTRATOR] Overall progress: {i}/{total} runs complete ({100*i/total:.0f}%)", flush=True)
        time.sleep(5)

    print(f"[ORCHESTRATOR] ALL DONE — {total} runs complete", flush=True)


if __name__ == '__main__':
    main()