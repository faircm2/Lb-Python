"""
Comparison plots for the parameter study (sigma, theta, tau_f series and the closed-x twin).

Expected layout: one sub-folder per job name, each holding the plane dumps the simulation writes
    <study_dir>/<job_name>/plane_yz_iter_XXXXXX.npz        (or <job_name>/planes/...)
The job names are the --job-name values of parm_study_v2__tau_f__sigma__theta.sh (plus 'validation'
for job 8414074, the periodic twin of closed_x). Runs that are missing are skipped.

usage:   python compare_runs_claude.py <study_dir> [out_dir]
output:  out_dir/series_sigma.png, series_theta.png, series_tau_f.png, twin_closed_x.png, summary.csv
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CS = np.sqrt(1 / 3)
D_PHYS, N_CELLS, NU_PHYS = 1e-3, 384, 2.9e-6            # same constants as the simulation
DX = D_PHYS / N_CELLS
PHI_FILL = 38                                           # initial liquid height in cells (10 % of 384)

# job name: sigma, tau_f, theta, alpha, periodic_x
RUNS = {
    "baseline":   (0.01,  1.0,  60, 90, 1),
    "theta_75":   (0.01,  1.0,  75, 90, 1),
    "theta_90":   (0.01,  1.0,  90, 90, 1),
    "theta_105":  (0.01,  1.0, 105, 90, 1),
    "theta_120":  (0.01,  1.0, 120, 90, 1),
    "sig_0.001":  (0.001, 1.0,  60, 90, 1),
    "sig_0.1":    (0.1,   1.0,  60, 90, 1),
    "sig_1":      (1.0,   1.0,  60, 90, 1),
    "tau_0.6":    (0.01,  0.6,  60, 90, 1),
    "tau_0.8":    (0.01,  0.8,  60, 90, 1),
    "tau_1.25":   (0.01,  1.25, 60, 90, 1),
    "closed_x":   (0.1,   1.0,  60, 45, 0),
    "validation": (0.1,   1.0,  60, 45, 1),
}

SERIES = {   # name: (title, job names in plotting order, label function)
    "sigma": ("sigma series  (tau_f 1.0, theta 60, alpha 90)", ["sig_0.001", "baseline", "sig_0.1", "sig_1"],
              lambda p: "sigma=%g" % p["sigma"]),
    "theta": ("theta series  (sigma 0.01, tau_f 1.0, alpha 90)", ["baseline", "theta_75", "theta_90", "theta_105", "theta_120"],
              lambda p: "theta=%g" % p["theta"]),
    "tau_f": ("tau_f series  (sigma 0.01, theta 60, alpha 90)", ["tau_0.6", "tau_0.8", "baseline", "tau_1.25"],
              lambda p: "tau_f=%g" % p["tau_f"]),
    "twin_closed_x": ("closed x walls vs periodic x  (sigma 0.1, tau_f 1.0, theta 60, alpha 45)", ["validation", "closed_x"],
                      lambda p: "x periodic" if p["periodic"] else "x closed (walls)"),
}


def params(job):
    s, t, th, a, per = RUNS[job]
    nu = (t - 0.5) / 3
    return dict(sigma=s, tau_f=t, theta=th, alpha=a, periodic=per, dt=nu * DX**2 / NU_PHYS)


def load_run(folder, dt):
    files = sorted(glob.glob(os.path.join(folder, "plane_yz_iter_*.npz")) +
                   glob.glob(os.path.join(folder, "planes", "plane_yz_iter_*.npz")))
    if not files:
        return None
    its, rows, profiles = [], [], []
    for f in files:
        d = np.load(f)
        phi, u = d["phi"], d["u"]
        its.append(int(re.search(r"iter_(\d+)", f).group(1)))
        h = phi.sum(axis=1)                              # liquid height per y (cells)
        speed = np.sqrt((u ** 2).sum(axis=0))
        liq = phi > 0.5
        rows.append((0.5 * (h[0] + h[-1]), h[h.size // 2 - 1:h.size // 2 + 1].mean(), h[0] - h[-1],
                     speed[liq].max() / CS if liq.any() else 0.0, phi.sum()))
        profiles.append(h)
    its = np.array(its)
    o = np.argsort(its)
    r = np.array(rows)[o]
    return dict(it=its[o], t=its[o] * dt * 1e3, wall=r[:, 0], centre=r[:, 1], lr=r[:, 2], mach=r[:, 3],
                mass=r[:, 4] / r[0, 4], prof=[profiles[i] for i in o])


def plot_series(name, study, out_dir, data):
    title, jobs, label = SERIES[name]
    jobs = [j for j in jobs if j in data]
    if not jobs:
        return
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(jobs))) if len(jobs) > 2 else ["tab:blue", "tab:red"]
    sigma_series = name == "sigma"
    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))
    fig.suptitle("Parameter study: " + title, fontsize=13)
    t_common = min(data[j]["t"][-1] for j in jobs)
    for j, c in zip(jobs, cols):
        d, p = data[j], params(j)
        lab = label(p)
        x = d["t"] * (p["sigma"] if sigma_series else 1.0)
        ax[0, 0].plot(x, d["wall"], color=c, lw=2, label=lab)
        ax[0, 1].plot(x, d["centre"], color=c, lw=2, label=lab)
        ax[0, 2].plot(d["t"], d["mach"], color=c, lw=1.8, label=lab)
        ax[1, 0].plot(d["t"], d["mass"], color=c, lw=1.8, label=lab)
        ax[1, 1].plot(d["t"], d["lr"], color=c, lw=1.8, label=lab)
        k = int(np.abs(d["t"] - t_common).argmin())
        h = d["prof"][k]
        ax[1, 2].plot(np.arange(h.size) + 0.5, h, color=c, lw=2, label="%s  (t=%.2f ms)" % (lab, d["t"][k]))
    xl = "sigma * t   (ms)" if sigma_series else "physical time   (ms)"
    for a, ttl, yl in ((ax[0, 0], "liquid height at the walls", "cells"), (ax[0, 1], "liquid height at the channel centre", "cells")):
        a.set_title(ttl)
        a.set_ylabel(yl)
        a.set_xlabel(xl)
        a.axhline(PHI_FILL, color="gray", ls=":", lw=1)
    ax[0, 2].set_title("max Mach (liquid)")
    ax[0, 2].set_xlabel("physical time   (ms)")
    ax[1, 0].set_title("total phi / initial  (mass conservation)")
    ax[1, 0].set_xlabel("physical time   (ms)")
    ax[1, 1].set_title("left - right wall height  (should be 0)")
    ax[1, 1].set_ylabel("cells")
    ax[1, 1].set_xlabel("physical time   (ms)")
    ax[1, 2].set_title("liquid height profile at the common time t = %.2f ms" % t_common)
    ax[1, 2].set_xlabel("y  (cells)")
    ax[1, 2].set_ylabel("liquid height  (cells)")
    ax[1, 2].axhline(PHI_FILL, color="gray", ls=":", lw=1)
    for a in ax.ravel():
        a.grid(alpha=0.3)
        a.legend(fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(out_dir, "series_%s.png" % name if not name.startswith("twin") else name + ".png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def main(study, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data = {}
    for job in RUNS:
        d = load_run(os.path.join(study, job), params(job)["dt"])
        if d is None:
            print("missing:", job)
        else:
            data[job] = d
    for name in SERIES:
        plot_series(name, study, out_dir, data)
    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        f.write("run,sigma,tau_f,theta,alpha,periodic_x,last_iter,t_ms,wall,centre,lr_asym,max_Ma_liq,mass_change\n")
        print("\n%-11s %7s %6s %5s %5s %3s | %8s %7s | %6s %6s %6s %9s %9s" %
              ("run", "sigma", "tau_f", "theta", "alpha", "per", "iter", "t(ms)", "wall", "centre", "L-R", "Ma(liq)", "mass chg"))
        for job, d in data.items():
            p = params(job)
            print("%-11s %7g %6g %5g %5g %3d | %8d %7.2f | %6.1f %6.1f %6.1f %9.4f %+9.2e" %
                  (job, p["sigma"], p["tau_f"], p["theta"], p["alpha"], p["periodic"], d["it"][-1], d["t"][-1],
                   d["wall"][-1], d["centre"][-1], d["lr"][-1], d["mach"][-1], d["mass"][-1] - 1))
            f.write("%s,%g,%g,%g,%g,%d,%d,%.4f,%.2f,%.2f,%.2f,%.5f,%.3e\n" %
                    (job, p["sigma"], p["tau_f"], p["theta"], p["alpha"], p["periodic"], d["it"][-1], d["t"][-1],
                     d["wall"][-1], d["centre"][-1], d["lr"][-1], d["mach"][-1], d["mass"][-1] - 1))
    print("\nwrote", os.path.join(out_dir, "summary.csv"))


if __name__ == "__main__":
    study = sys.argv[1]
    main(study, sys.argv[2] if len(sys.argv) > 2 else os.path.join(study, "compare"))
