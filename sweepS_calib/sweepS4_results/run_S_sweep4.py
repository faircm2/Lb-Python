"""
Fourth wettability calibration sweep, for bwUniCluster: long runs on the range S = 0.50 ... 0.65 where the first sweeps still looked like a meniscus.

Sweep 3 showed that the rise at S = 0.65 was still growing at iteration 4500 (29.7, 33.0, 35.3 cells at 1500, 3000, 4500), so the 3000-iteration values of
the first sweep were not settled, and that from S = 0.66 on the meniscus runs away up the wall.  Here every run goes to 30,000 iterations with phi saved
at 3000, 6000, 12,000, 18,000, 24,000 and 29,999, so convergence.csv shows where (and whether) the rise settles and whether S = 0.62 ... 0.65 run away later.

  S = 0.50, 0.55, 0.60, 0.62, 0.64, 0.65     one single-core Slurm job each (Slurm job array), started by submit_S_sweep4.sh

Everything else is taken from run_S_sweep.py (same Cap102 copies, rise -> angle through the circular arc relation).
    python run_S_sweep4.py prepare     writes and compiles the copies and sweepS_results4/runs.txt (one run name per line)
    python run_S_sweep4.py analyse     calibration.csv / calibration.png (last dump) and convergence.csv / convergence.png (every dump)
"""
import csv
import os
import sys

import numpy as np

import run_S_sweep as base

base.S_VALUES = [0.50, 0.55, 0.60, 0.62, 0.64, 0.65]
base.ITERATIONS = 30000
base.DUMP_ITERATIONS = (3000, 6000, 12000, 18000, 24000, 29999)
base.OUT_DIR = "sweepS_results4"


def prepare():
    os.makedirs(base.OUT_DIR, exist_ok=True)
    names = []
    for S in base.S_VALUES:
        name, fname, theta = base.make_copy(S)
        base.subprocess.check_call([sys.executable, "-m", "py_compile", fname])
        names.append(name)
    with open(base.OUT_DIR + "/runs.txt", "w") as fh:
        fh.write("\n".join(names) + "\n")
    print("prepared %d copies" % len(names))


def analyse():
    runs = [(S, "sweepS_S%.2f" % S, round(180.0 * (1.0 - S), 4)) for S in base.S_VALUES]
    base.analyse(runs)
    rows = []
    for S, name, theta in runs:
        for it in base.DUMP_ITERATIONS:
            f = "%s/%s_it%d.npy" % (base.OUT_DIR, name, it)
            if not os.path.exists(f):
                rows.append((S, it, float("nan"), float("nan")))
                continue
            p = np.load(f)
            centre = base.interface_height(p[200])
            rise = np.nanmean([base.interface_height(p[1]) - centre, base.interface_height(p[400]) - centre])
            rows.append((S, it, rise, base.angle_of_rise(rise)))
    with open(base.OUT_DIR + "/convergence.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["S", "iteration", "rise_cells", "theta_effective_deg"])
        for r in rows:
            w.writerow(["%.2f" % r[0], r[1], "%.2f" % r[2], "%.1f" % r[3]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for S in base.S_VALUES:
            sel = [r for r in rows if r[0] == S]
            ax.plot([r[1] for r in sel], [r[2] for r in sel], "o-", label="S = %.2f" % S)
        ax.set_xlabel("iteration")
        ax.set_ylabel("meniscus rise, wall minus centre (cells)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(base.OUT_DIR + "/convergence.png", dpi=150)
    except Exception as e:           # plotting must never lose the csv
        print("plot failed:", e)
    for r in rows:
        print("S %.2f  iteration %5d  rise %.1f cells  effective %.1f deg" % (r[0], r[1], r[2], r[3]))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "analyse"):
        raise SystemExit("usage: python run_S_sweep4.py prepare | analyse")
    prepare() if sys.argv[1] == "prepare" else analyse()
