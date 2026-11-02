"""
Calibration of the Inamuro wall wettability S against the contact angle that the 2D capillary sim (Cap102) really produces.

For each S the script
  1. makes a copy of free_surface_SchanChen_nondim_D2Q9_08_Cap102.py in which the nominal angle is set so that S = 1 - theta/180 (the script's own mapping),
     the run is cut to 3000 iterations and phi is saved at a few iterations,
  2. runs the copies, at most N_PARALLEL at the same time,
  3. measures the meniscus rise (wall minus centre) from the saved phi and converts it to an effective contact angle through the circular arc relation
        rise = (W/2) / cos(theta) * (1 - sin(theta)),       W = channel width in cells
     (Bond number is about 0.2, so gravity changes the rise by less than 0.5 cells).

Results: sweepS_results/calibration.csv and sweepS_results/calibration.png (S, nominal angle, rise at both walls, effective angle).
Start with:   python3 run_S_sweep.py          (dry run that only writes and compiles the copies:  python3 run_S_sweep.py --dry)
"""
import csv
import os
import subprocess
import sys
import time

import numpy as np

TEMPLATE = "free_surface_SchanChen_nondim_D2Q9_08_Cap102.py"
S_VALUES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]
ITERATIONS = 3000
DUMP_ITERATIONS = (1500, 2000, 2500, 2999)
N_PARALLEL = 4
OUT_DIR = "sweepS_results"
W = 400.0                  # channel width in cells (DEFAULT_D_ND of the template)
PHI_MID = 0.0535           # (phi*_G + phi*_L) / 2


def make_copy(S):
    name = "sweepS_S%.2f" % S
    text = open(TEMPLATE, encoding="utf-8").read().replace("\r\n", "\n")

    def replace_once(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise SystemExit("template changed, cannot find exactly once: %r (found %d)" % (old, text.count(old)))
        text = text.replace(old, new)

    theta = round(180.0 * (1.0 - S), 4)
    replace_once("    vf_theta = 60.0, #60\n", "    vf_theta = %r,\n" % theta)
    replace_once("CORE_TOTAL_ITERATIONS = 5000", "CORE_TOTAL_ITERATIONS = %d" % ITERATIONS)
    hook = ("while iteration < fc.TOTAL_ITERATIONS:\n"
            "    if iteration in %r:\n"
            "        np.save('%s/%s_it%%d.npy' %% iteration, _phi)\n" % (DUMP_ITERATIONS, OUT_DIR, name))
    replace_once("while iteration < fc.TOTAL_ITERATIONS:\n", hook)
    fname = name + ".py"
    open(fname, "w", encoding="utf-8").write(text)
    return name, fname, theta


def interface_height(col):
    """y (in cells, fractional) where phi falls through PHI_MID going up the column."""
    c = col[1:-1]
    i = np.where((c[:-1] >= PHI_MID) & (c[1:] < PHI_MID))[0]
    if len(i) == 0:
        return float("nan")
    i = i[-1]
    return i + 1 + (c[i] - PHI_MID) / (c[i] - c[i + 1])


def rise_of_angle(theta_deg):
    t = np.radians(theta_deg)
    return (W / 2.0) / np.cos(t) * (1.0 - np.sin(t))


def angle_of_rise(rise):
    lo, hi = 0.0, 89.9          # rise_of_angle decreases from large (theta 0) to 0 (theta 90)
    if rise <= 0:
        return 90.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if rise_of_angle(mid) > rise:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def analyse(runs):
    rows = []
    for S, name, theta in runs:
        last = None
        for it in sorted(DUMP_ITERATIONS, reverse=True):
            f = "%s/%s_it%d.npy" % (OUT_DIR, name, it)
            if os.path.exists(f):
                last = (it, np.load(f))
                break
        if last is None:
            rows.append((S, theta, float("nan"), float("nan"), float("nan"), float("nan")))
            continue
        it, p = last
        centre = interface_height(p[200])
        rise_l = interface_height(p[1]) - centre
        rise_r = interface_height(p[400]) - centre
        rise = np.nanmean([rise_l, rise_r])
        rows.append((S, theta, it, rise_l, rise_r, angle_of_rise(rise)))
    with open(OUT_DIR + "/calibration.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["S", "theta_nominal_deg", "last_dump_iteration", "rise_left_cells", "rise_right_cells", "theta_effective_deg"])
        for r in rows:
            w.writerow(["%.2f" % r[0], "%.1f" % r[1], r[2], "%.2f" % r[3], "%.2f" % r[4], "%.1f" % r[5]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot([r[0] for r in rows], [r[5] for r in rows], "o-", label="effective angle from meniscus rise")
        ax.plot([r[0] for r in rows], [r[1] for r in rows], "k--", label="script mapping  theta = 180 (1 - S)")
        ax.set_xlabel("wall wettability S")
        ax.set_ylabel("contact angle (deg)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR + "/calibration.png", dpi=150)
    except Exception as e:           # plotting must never lose the csv
        print("plot failed:", e)
    for r in rows:
        print("S %.2f  nominal %.1f deg  rise %.1f / %.1f cells  effective %.1f deg" % (r[0], r[1], r[3], r[4], r[5]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = []
    for S in S_VALUES:
        name, fname, theta = make_copy(S)
        subprocess.check_call([sys.executable, "-m", "py_compile", fname])
        runs.append((S, name, theta))
    if "--dry" in sys.argv:
        print("dry run: wrote and compiled %d copies" % len(runs))
        return
    running = []
    todo = list(runs)
    while todo or running:
        running = [(p, r) for p, r in running if p.poll() is None]
        while todo and len(running) < N_PARALLEL:
            S, name, theta = todo.pop(0)
            log = open("%s/%s.log" % (OUT_DIR, name), "w")
            p = subprocess.Popen(["nice", "-n", "5", sys.executable, name + ".py"], stdout=log, stderr=subprocess.STDOUT)
            running.append((p, (S, name, theta)))
            print(time.strftime("%H:%M:%S"), "started", name, "nominal", theta, flush=True)
        time.sleep(15)
    analyse(runs)
    print(time.strftime("%H:%M:%S"), "sweep finished", flush=True)


if __name__ == "__main__":
    main()
