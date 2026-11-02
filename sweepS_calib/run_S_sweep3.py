"""
Third wettability calibration sweep: zooms into the jump of the effective angle between S = 0.65 (71 deg) and S = 0.70 (44 deg) and checks whether
the 3000-iteration rise of run_S_sweep.py was settled.

  S = 0.65, 0.66, 0.67, 0.68, 0.69, 0.70   (0.65 and 0.70 are repeated as references)
  6000 iterations instead of 3000, phi saved at 1500, 3000, 4500 and 5999

Everything else is taken from run_S_sweep.py (same Cap102 copies, 4 at a time, rise -> angle through the circular arc relation).
If a run's effective angle at 4500 and 5999 differs by more than about 1 deg it had not settled, and the jump of the first sweep is partly a transient.
Results in sweepS_results3/:
  calibration.csv / calibration.png      effective angle at the last dump (5999)
  convergence.csv / convergence.png      effective angle of every S at every saved iteration
The copies of the template are named sweepS_S<S>.py, as in the first sweep, so S = 0.65 and 0.70 overwrite the copies of that sweep (same code, only
the iteration count differs; the results of the first sweep stay in sweepS_results/).
Start with:   python3 run_S_sweep3.py          (dry run that only writes and compiles the copies:  python3 run_S_sweep3.py --dry)
"""
import csv
import os
import sys

import numpy as np

import run_S_sweep as base

base.S_VALUES = [0.65, 0.66, 0.67, 0.68, 0.69, 0.70]
base.ITERATIONS = 6000
base.DUMP_ITERATIONS = (1500, 3000, 4500, 5999)
base.OUT_DIR = "sweepS_results3"


def convergence():
    rows = []
    for S in base.S_VALUES:
        name = "sweepS_S%.2f" % S
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
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for it in base.DUMP_ITERATIONS:
            sel = [r for r in rows if r[1] == it]
            ax.plot([r[0] for r in sel], [r[3] for r in sel], "o-", label="iteration %d" % it)
        ax.set_xlabel("wall wettability S")
        ax.set_ylabel("effective contact angle (deg)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(base.OUT_DIR + "/convergence.png", dpi=150)
    except Exception as e:           # plotting must never lose the csv
        print("plot failed:", e)
    for r in rows:
        print("S %.2f  iteration %5d  rise %.1f cells  effective %.1f deg" % (r[0], r[1], r[2], r[3]))


if __name__ == "__main__":
    base.main()
    if "--dry" not in sys.argv:
        convergence()
