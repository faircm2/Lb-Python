"""
Superimpose exact meniscus profiles (from interface_profile.csv, written
directly from real phi data -- not image-decoded) for a fixed tau_g across
all tau_f values. Produces one overlay PNG per tau_g in the sweep.

Usage: python overlay_menisci_exact.py
Reads param_study_phase1_tau_results.csv + param_study_runs/<label>/interface_profile.csv
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(SCRIPT_DIR, 'param_study_phase1_tau_results.csv')
RUNS_DIR = os.path.join(SCRIPT_DIR, 'param_study_runs')
OUT_DIR = os.path.join(SCRIPT_DIR, 'menisci_overlays_exact')


def load_profile(label):
    path = os.path.join(RUNS_DIR, label, 'interface_profile.csv')
    if not os.path.exists(path):
        return None, None
    xs, ys = [], []
    with open(path, 'r', newline='') as f:
        for row in csv.DictReader(f):
            xs.append(int(row['x']))
            ys.append(float(row['interface_y']) if row['interface_y'] != '' else float('nan'))
    return xs, ys


def main():
    with open(RESULTS_CSV, 'r', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['result'] == 'DONE']

    tau_g_vals = sorted({float(r['tau_g']) for r in rows})
    os.makedirs(OUT_DIR, exist_ok=True)
    cmap = matplotlib.colormaps['viridis']

    for tau_g in tau_g_vals:
        group = sorted([r for r in rows if float(r['tau_g']) == tau_g],
                        key=lambda r: float(r['tau_f']))
        if not group:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))
        for i, r in enumerate(group):
            xs, ys = load_profile(r['label'])
            if xs is None:
                print(f"  skipping {r['label']} -- no interface_profile.csv")
                continue
            color = cmap(i / max(1, len(group) - 1))
            ax.plot(xs, ys, label=f"tau_f={r['tau_f']}", color=color, linewidth=1.8)

        ax.set_xlabel('x-index')
        ax.set_ylabel('interface y-index (phi=0.5 crossing)')
        ax.set_title(f'Superimposed meniscus profiles, tau_g={tau_g} (exact)')
        ax.set_ylim(140, 160)
        ax.legend(title='tau_f')
        ax.grid(alpha=0.3)

        out_path = os.path.join(OUT_DIR, f'menisci_overlay_taug{tau_g}.png')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print('Saved:', out_path)


if __name__ == '__main__':
    main()
