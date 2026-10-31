"""
Second wettability calibration sweep: the S values that run_S_sweep.py did not cover.

  S = 0.10, 0.25, 0.40, 0.45   non-wetting side (nominal angle above 90 deg, the interface dips at the wall)
  S = 0.85, 0.95, 1.00         strongly wetting side (nominal angle below 30 deg)

Everything else is taken from run_S_sweep.py (same Cap102 copies, 3000 iterations, 4 at a time).
Because the interface dips at the wall for angles above 90 deg, the conversion rise -> angle is the signed form
        rise = (W/2) * (1 - sin(theta)) / cos(theta)       (positive: climbs the wall, negative: dips; theta = 90 deg gives 0)
The script waits until run_S_sweep.py has finished, so the two sweeps never run at the same time.
Results: sweepS_results2/calibration.csv, and calibration_all.csv / calibration_all.png with both sweeps together.
Start with:   python3 run_S_sweep2.py          (dry run:  python3 run_S_sweep2.py --dry)
"""
import csv
import os
import subprocess
import sys
import time

import numpy as np

import run_S_sweep as base

base.S_VALUES = [0.10, 0.25, 0.40, 0.45, 0.85, 0.95, 1.00]
base.OUT_DIR = "sweepS_results2"


def rise_of_angle(theta_deg):
    t = np.radians(theta_deg)
    c = np.cos(t)
    if abs(c) < 1e-9:
        return 0.0
    return (base.W / 2.0) * (1.0 - np.sin(t)) / c


def angle_of_rise(rise):
    lo, hi = 0.1, 179.9         # rise_of_angle decreases from +W/2/... near 0 deg to -W/2 at 180 deg
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if rise_of_angle(mid) > rise:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


base.rise_of_angle = rise_of_angle
base.angle_of_rise = angle_of_rise


def read_csv(path):
    rows = []
    if os.path.exists(path):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                rows.append((float(r["S"]), float(r["theta_nominal_deg"]), float(r["theta_effective_deg"])))
    return rows


def combine():
    rows = sorted(read_csv("sweepS_results/calibration.csv") + read_csv("sweepS_results2/calibration.csv"))
    with open("sweepS_results2/calibration_all.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["S", "theta_nominal_deg", "theta_effective_deg"])
        for r in rows:
            w.writerow(["%.2f" % r[0], "%.1f" % r[1], "%.1f" % r[2]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot([r[0] for r in rows], [r[2] for r in rows], "o-", label="effective angle from meniscus rise")
        ax.plot([r[0] for r in rows], [r[1] for r in rows], "k--", label="script mapping  theta = 180 (1 - S)")
        ax.set_xlabel("wall wettability S")
        ax.set_ylabel("contact angle (deg)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("sweepS_results2/calibration_all.png", dpi=150)
    except Exception as e:
        print("plot failed:", e)


if __name__ == "__main__":
    if "--dry" not in sys.argv:
        while subprocess.call("pgrep -f 'run_S_sweep[.]py'", shell=True, stdout=subprocess.DEVNULL) == 0:
            time.sleep(30)
    base.main()
    if "--dry" not in sys.argv:
        combine()
