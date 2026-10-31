"""
Bubble growth on the floor against iteration, for the parameter studies of the 3D channel model (Cap07 runs).
Run on the cluster in /work/classic/fr_jn194-lbm/scaling (numpy and matplotlib are in the lbm env):

    /home/fr/fr_jn194/.conda/envs/lbm/bin/python studies_vs_iter_system.py

Common to all runs: W = 4, tau_g = 0.65, rho_L = 10, rho_G = 0.01, nu = 2.9e-6, D = 1.0 mm, periodic x, 2048 ranks.
Per run the bubble growth g = 1 - min(phi) on the floor xf-layer (first fluid layer next to the floor, y = 3..190, mid-channel plane)
is taken from every plane_yz dump (iteration 0 is skipped: g = 0 there).  A run that ended in NaN is drawn up to its last good dump,
and its legend entry says so.  Runs whose folder does not exist (yet) are skipped, unfinished runs are drawn as far as they have got.
Studies:  sigma (tau_f 1.0, theta 60) | theta (sigma 0.01, tau_f 1.0) | tau_f (sigma 0.01, theta 60) | alpha (sigma 0.01, tau_f 1.0, theta 60)
Writes sigma_vs_iter.png, theta_vs_iter.png, tau_f_vs_iter.png, alpha_vs_iter.png (only if alpha runs exist) and studies_vs_iter.csv.
"""
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["legend.fancybox"] = False      # square legend frames
import numpy as np

BASE = "*Cap07*tau_f_%s__tau_g_0.65*Theta_%s_alpha%s_sigma%s_periodicx_x%d"

# study -> (title, file name, [(label, tau_f, theta, alpha, sigma, multiple)])
STUDIES = {
    "sigma": ("σ study  (τ_f = 1.0, θ = 60°)", "sigma_vs_iter.png",
              [("σ = 0.001", "1.0", "60.0", "90", "0.001", 3), ("σ = 0.01", "1.0", "60.0", "90", "0.01", 3),
               ("σ = 0.1", "1.0", "60.0", "90", "0.1", 3), ("σ = 1", "1.0", "60.0", "90", "1", 2)]),
    "theta": ("θ study  (σ = 0.01, τ_f = 1.0)", "theta_vs_iter.png",
              [("θ = 60°", "1.0", "60.0", "90", "0.01", 3), ("θ = 75°", "1.0", "75.0", "90", "0.01", 3),
               ("θ = 90°", "1.0", "90.0", "90", "0.01", 3), ("θ = 105°", "1.0", "105.0", "90", "0.01", 3),
               ("θ = 120°", "1.0", "120.0", "90", "0.01", 3)]),
    "tau_f": ("τ_f study  (σ = 0.01, θ = 60°)", "tau_f_vs_iter.png",
              [("τ_f = 0.6", "0.6", "60.0", "90", "0.01", 3), ("τ_f = 0.8", "0.8", "60.0", "90", "0.01", 5),
               ("τ_f = 1.0", "1.0", "60.0", "90", "0.01", 3), ("τ_f = 1.25", "1.25", "60.0", "90", "0.01", 2)]),
    "alpha": ("α study  (σ = 0.01, τ_f = 1.0, θ = 60°)", "alpha_vs_iter.png",
              [("α = 45°", "1.0", "60.0", "45", "0.01", 3), ("α = 65°", "1.0", "60.0", "65", "0.01", 3),
               ("α = 75°", "1.0", "60.0", "75", "0.01", 3), ("α = 85°", "1.0", "60.0", "85", "0.01", 3),
               ("α = 90°", "1.0", "60.0", "90", "0.01", 3)]),
}
FLOOR_EPS = 1e-9          # g is plotted on a log axis; 1 - phi can be exactly 0


def say(text):
    print(text.encode("ascii", "replace").decode())


def find_folder(pattern):
    hits = sorted(d for d in glob.glob("**/" + pattern, recursive=True) if os.path.isdir(d))
    return hits[0] if hits else None


def series(folder):
    its, g = [], []
    for f in sorted(glob.glob(os.path.join(folder, "plane_yz_iter_*.npz"))):
        it = int(re.search(r"iter_(\d+)", f).group(1))
        if it == 0:
            continue
        p = np.load(f)["phi"]
        its.append(it)
        g.append(1.0 - float(p[3:190, 0].min()))
    return np.array(its), np.array(g)


rows = []
for key, (title, fname, runs) in STUDIES.items():
    data = []
    for label, tf, th, al, sg, mult in runs:
        folder = find_folder(BASE % (tf, th, al, sg, mult))
        if folder is None:
            say("%s: no folder found" % label)
            continue
        its, g = series(folder)
        if len(its) == 0:
            say("%s: no dumps in %s" % (label, folder))
            continue
        if np.isnan(g).any():
            first_nan = int(its[np.where(np.isnan(g))[0][0]])
            say("%s: NaN from iteration %d" % (label, first_nan))
            label = "%s  (collapsed, NaN from it. %d)" % (label, first_nan)
            ok = ~np.isnan(g)
            its, g = its[ok], g[ok]
            if len(its) == 0:
                continue
        data.append((label, its, g))
        for it, v in zip(its, g):
            rows.append((key, label, int(it), "%.6e" % v))
    if not data:
        continue
    fig, a = plt.subplots(figsize=(6.2, 4.4))
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, max(len(data), 2)))
    for c, (label, its, g) in zip(cols, data):
        a.plot(its, np.maximum(g, FLOOR_EPS), color=c, lw=1.5, label=label)
    a.set_yscale("log")
    a.set_xlabel("iteration")
    a.set_ylabel("Bubble growth 1-min(φ) on floor xf-layer")
    a.set_title(title, fontsize=10)
    a.grid(alpha=0.25)
    a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=170)
    plt.close(fig)
    say("wrote %s" % fname)
with open("studies_vs_iter.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["study", "run", "iteration", "bubble_growth_1_minus_minphi"])
    w.writerows(rows)
say("wrote studies_vs_iter.csv")
