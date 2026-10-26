"""
Test of the 2D Inamuro capillary sim (Cap102) in a 100 x 100 cell box.

Why: in the 400 cell box the slowest capillary wave has a period of about 72,000 iterations and is practically undamped, so the meniscus
does not settle in a run you can afford.  In a 100 cell box the period is about 9000 iterations and the viscous decay time about 3000
iterations, so the meniscus should settle within 10,000 to 20,000 iterations (the runs are also 16 times cheaper).

Three copies of Cap102 are made and run in parallel, each with 20,000 iterations and phi saved every 500:
    nominal wall angle 90 deg (S = 0.50, flat interface expected),  60 deg (S = 0.667),  30 deg (S = 0.833).
Afterwards the interface profile h(x) - h(centre) and the wall rise are compared with the Young-Laplace solution for the same angle
(circular arc with gravity, solved numerically, sigma_lat = 0.81 as measured with Eq. 16).
Results: cap102_N100_results/summary.csv, profiles.png, rise_vs_iteration.png
Start with:   python3 run_cap102_N100.py          (dry run that only writes and compiles the copies:  python3 run_cap102_N100.py --dry)
"""
import csv
import os
import subprocess
import sys
import time

import numpy as np

TEMPLATE = "free_surface_SchanChen_nondim_D2Q9_08_Cap102.py"
N = 100
ITERATIONS = 20000
DUMP_EVERY = 500
THETAS = [90.0, 60.0, 30.0]
OUT_DIR = "cap102_N100_results"
PHI_MID = 0.0535
SIGMA_LAT = 0.81           # Eq. (16) value measured in the 400 cell runs
DRHO = 49.0


def make_copy(theta):
    name = "cap102_N100_th%d" % theta
    text = open(TEMPLATE, encoding="utf-8").read().replace("\r\n", "\n")

    def replace_once(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise SystemExit("template changed, cannot find exactly once: %r (found %d)" % (old, text.count(old)))
        text = text.replace(old, new)

    replace_once("DEFAULT_D_ND = 400", "DEFAULT_D_ND = %d" % N)
    replace_once("    vf_theta = 60.0, #60\n", "    vf_theta = %r,\n" % theta)
    replace_once("    vf_xi = 8, #4, previously was 6 -> (400/300)² * 6 = 8\n", "    vf_xi = 4,\n")
    replace_once("CORE_TOTAL_ITERATIONS = 5000", "CORE_TOTAL_ITERATIONS = %d" % ITERATIONS)
    hook = ("while iteration < fc.TOTAL_ITERATIONS:\n"
            "    if iteration %% %d == 0:\n"
            "        np.save('%s/%s_it%%d.npy' %% iteration, _phi)\n" % (DUMP_EVERY, OUT_DIR, name))
    replace_once("while iteration < fc.TOTAL_ITERATIONS:\n", hook)
    fname = name + ".py"
    open(fname, "w", encoding="utf-8").write(text)
    return name, fname


def interface_height(col):
    c = col[1:-1]
    i = np.where((c[:-1] >= PHI_MID) & (c[1:] < PHI_MID))[0]
    if len(i) == 0:
        return float("nan")
    i = i[-1]
    return i + 1 + (c[i] - PHI_MID) / (c[i] - c[i + 1])


def g_lattice():
    D, nu, n_tau = 1e-3, 2.9e-6, 0.6
    dx = D / N
    dt = (1.0 / 3.0) * (n_tau - 0.5) * dx ** 2 / nu
    return 9.81 * dt ** 2 / dx


def young_laplace(theta_deg, ybar):
    """Interface y(x) for x in [0, N/2] (centre to wall) between two walls a distance N apart; fixed mean level ybar."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import fsolve
    a = DRHO * g_lattice() / SIGMA_LAT
    W = float(N)
    cot = 1.0 / np.tan(np.radians(theta_deg)) if abs(theta_deg - 90.0) > 1e-6 else 0.0

    def shape(yc, y0):
        f = lambda x, s: [s[1], (1 + s[1] ** 2) ** 1.5 * a * (s[0] - y0)]
        sol = solve_ivp(f, [0, W / 2], [yc, 0.0], dense_output=True, rtol=1e-10, atol=1e-12)
        x = np.linspace(0, W / 2, 401)
        y, yp = sol.sol(x)
        return x, y, yp

    def res(p):
        x, y, yp = shape(*p)
        return [yp[-1] - cot, np.trapezoid(y, x) / (W / 2) - ybar]

    yc0 = ybar - 0.1 * W * cot
    p = fsolve(res, [yc0, yc0 - 1.0], xtol=1e-10)
    return shape(*p)[:2]


def analyse(names):
    os.makedirs(OUT_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    fig_p, axp = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
    fig_r, axr = plt.subplots(figsize=(6, 4))
    for k, (theta, name) in enumerate(names):
        its = sorted(int(f.split("_it")[1][:-4]) for f in os.listdir(OUT_DIR) if f.startswith(name + "_it") and f.endswith(".npy"))
        if not its:
            continue
        first = np.load("%s/%s_it%d.npy" % (OUT_DIR, name, its[0]))
        ybar = interface_height(first[N // 2])
        try:
            xt, yt = young_laplace(theta, ybar)
            theory_rise = float(yt[-1] - yt[0])
        except Exception as e:
            print("theory failed for", theta, e)
            xt = yt = None
            theory_rise = float("nan")
        rise_t = []
        for it in its:
            p = np.load("%s/%s_it%d.npy" % (OUT_DIR, name, it))
            hs = np.array([interface_height(p[x]) for x in range(1, N + 1)])
            centre = np.nanmean(hs[N // 2 - 3:N // 2 + 3])
            rise_t.append(0.5 * ((hs[0] - centre) + (hs[-1] - centre)))
            rows.append((theta, it, "%.2f" % hs[0], "%.2f" % centre, "%.2f" % hs[-1], "%.2f" % theory_rise))
        ax = axp[0, k]
        for it in (its[len(its) // 4], its[len(its) // 2], its[-1]):
            p = np.load("%s/%s_it%d.npy" % (OUT_DIR, name, it))
            hs = np.array([interface_height(p[x]) for x in range(1, N + 1)])
            ax.plot(np.arange(1, N + 1), hs - np.nanmean(hs[N // 2 - 3:N // 2 + 3]), label="iteration %d" % it)
        if xt is not None:
            ax.plot(N / 2 + xt, yt - yt[0], "k--", label="Young-Laplace")
            ax.plot(N / 2 - xt, yt - yt[0], "k--")
        ax.set_title("nominal %d deg" % theta)
        ax.set_xlabel("x")
        ax.set_ylabel("interface height - centre")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        axr.plot(its, rise_t, label="nominal %d deg" % theta)
        if np.isfinite(theory_rise):
            axr.axhline(theory_rise, ls=":", color=axr.lines[-1].get_color())
    axr.set_xlabel("iteration")
    axr.set_ylabel("rise at wall - centre (cells)")
    axr.legend()
    axr.grid(alpha=0.3)
    fig_p.tight_layout()
    fig_p.savefig(OUT_DIR + "/profiles.png", dpi=130)
    fig_r.tight_layout()
    fig_r.savefig(OUT_DIR + "/rise_vs_iteration.png", dpi=130)
    with open(OUT_DIR + "/summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["theta_nominal", "iteration", "h_wall_left", "h_centre", "h_wall_right", "rise_theory_cells"])
        w.writerows(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    names = []
    for th in THETAS:
        name, fname = make_copy(th)
        subprocess.check_call([sys.executable, "-m", "py_compile", fname])
        names.append((th, name))
    if "--dry" in sys.argv:
        print("dry run: wrote and compiled %d copies" % len(names))
        return
    procs = []
    for th, name in names:
        log = open("%s/%s.log" % (OUT_DIR, name), "w")
        procs.append(subprocess.Popen(["nice", "-n", "5", sys.executable, name + ".py"], stdout=log, stderr=subprocess.STDOUT))
        print(time.strftime("%H:%M:%S"), "started", name, flush=True)
    for p in procs:
        p.wait()
    analyse(names)
    print(time.strftime("%H:%M:%S"), "finished", flush=True)


if __name__ == "__main__":
    main()
