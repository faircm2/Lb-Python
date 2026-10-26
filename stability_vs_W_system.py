"""
Stability against interface width W, from the plane_yz dumps of the sigma 0.5 runs.
Run on the cluster in /work/classic/fr_jn194-lbm/scaling (numpy, scipy and matplotlib are in the lbm env):

    /home/fr/fr_jn194/.conda/envs/lbm/bin/python stability_vs_W_system.py

Per run: the bubble growth g = 1 - min(phi) on the floor xf-layer (the first fluid layer next to the floor, y = 3..190, mid-channel plane) at every dump,
    final bubble growth     g at the last dump,
    bubble growth rate      change of g per 1,000 iterations, slope of a straight-line fit over the dumps from iteration 6,000 on
                            (above 0: still growing, 0 or below: levelled off),
    bubbles                 first iteration with a gas pocket enclosed by liquid, or none.
Series: W = 4 to 13 from the Cap06 runs, W = 14 to 17 from Cap07 runs, plus Cap07 reruns of W = 7 and W = 8 (drawn as open markers / dashed lines).
Writes stability_vs_W.png and stability_vs_W.csv (the csv also holds the relative rate, the slope of ln(g) per 1,000 iterations).
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
from scipy import ndimage as ndi

# W -> folder name pattern.  W = 4 is the Cap06 run; W = 5..13 are Cap06 runs; W = 14..17 are Cap07 runs (Cap07 reproduces Cap06 exactly).
RUNS = {4: "*Cap06*sigma0.5_periodicx"}
RUNS.update({W: "*Cap06*sigma0.5_W%d_periodicx" % W for W in range(5, 14)})
RUNS.update({W: "*Cap07*sigma0.5_W%d_periodicx" % W for W in range(14, 18)})
RERUNS = {W: "*Cap07*sigma0.5_W%d_periodicx" % W for W in (7, 8)}      # Cap07 reruns of the two scattered points
FIT_FROM = 6000


def find_folder(pattern):
    hits = sorted(d for d in glob.glob("**/" + pattern, recursive=True) if os.path.isdir(d))
    return hits[0] if hits else None


def has_bubble(p):
    lab, n = ndi.label(p < 0.5)
    top = set(np.unique(lab[:, -1])) - {0}
    return any(k not in top for k in range(1, n + 1))


def series(folder):
    its, mins, first_bubble = [], [], None
    for f in sorted(glob.glob(os.path.join(folder, "plane_yz_iter_*.npz"))):
        it = int(re.search(r"iter_(\d+)", f).group(1))
        p = np.load(f)["phi"]
        its.append(it)
        mins.append(float(p[3:190, 0].min()))
        if first_bubble is None and has_bubble(p):
            first_bubble = it
    return np.array(its), np.array(mins), first_bubble


rows, data = [], {}      # rows: (W, last_it, final, rate, first_bubble, rel_rate, tag)
for tag, runs in (("main", RUNS), ("rerun", RERUNS)):
    for W in sorted(runs):
        pat = runs[W]
        folder = find_folder(pat)
        if folder is None:
            print("W = %2d (%s): no folder found for %s" % (W, tag, pat))
            continue
        its, mins, fb = series(folder)
        d = np.maximum(1.0 - mins, 1e-6)      # phi = 1.00001 at the first dumps would give a negative value
        sel = its >= FIT_FROM
        rate = float(np.polyfit(its[sel], d[sel], 1)[0]) * 1000.0 if sel.sum() >= 3 else float("nan")                 # change of g per 1000 iterations
        rel_rate = float(np.polyfit(its[sel], np.log(d[sel]), 1)[0]) * 1000.0 if sel.sum() >= 3 else float("nan")     # slope of ln(g) per 1000 iterations
        data[(W, tag)] = (its, d)
        rows.append((W, int(its[-1]), float(d[-1]), rate, fb, rel_rate, tag))
        print("W = %2d (%-5s): last dump %6d  final bubble growth %.5f  bubble growth rate %+.5f per 1000 it  (relative %+.3f)  bubbles %s" % (W, tag, its[-1], d[-1], rate, rel_rate, fb if fb is not None else "none"))

with open("stability_vs_W.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["W", "series", "last_iteration", "final_bubble_growth_1_minus_minphi", "bubble_growth_per_1000_it", "first_bubble_iteration", "relative_growth_ln_per_1000_it"])
    for r in rows:
        w.writerow([r[0], r[6], r[1], "%.6f" % r[2], "%.6f" % r[3], "" if r[4] is None else r[4], "%.4f" % r[5]])

main = [r for r in rows if r[6] == "main"]
rerun = [r for r in rows if r[6] == "rerun"]
ws = [r[0] for r in main]
cols = plt.cm.viridis(np.linspace(0.05, 0.9, max(len(ws), 2)))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

for c, W in zip(cols, ws):
    its, d = data[(W, "main")]
    ax[0].plot(its, d, color=c, label="W = %d" % W, lw=1.4)
for r in rerun:
    its, d = data[(r[0], "rerun")]
    ax[0].plot(its, d, color="black", ls="--", lw=1.0, label="W = %d rerun" % r[0])
ax[0].set_yscale("log")
ax[0].set_xlabel("iteration")
ax[0].set_ylabel("1-min(φ) on floor xf-layer")
ax[0].set_title("Bubble growth on floor xf-layer vs iter;  σ=0,5", fontsize=10)
ax[0].axvline(FIT_FROM, color="grey", ls=":", lw=1)
ax[0].legend(frameon=False, fontsize=7, ncol=2)

ax[1].plot(ws, [r[2] for r in main], "o-", color="#1f4e79")
ax[1].plot([r[0] for r in rerun], [r[2] for r in rerun], "o", mfc="none", mec="black", ms=9, label="rerun (Cap07)")
ax[1].set_yscale("log")
ax[1].set_xlabel("interface thickness W")
ax[1].set_ylabel("Final bubble growth on floor xf-layer")
ax[1].set_title("Bubble growth vs. interface thickness;  σ=0,5", fontsize=10)
ax[1].set_xticks(ws)
if rerun:
    ax[1].legend(frameon=False, fontsize=8)

rates = [r[3] for r in main]
ax[2].axhline(0, color="grey", lw=1)
ax[2].bar(ws, rates, width=0.7, color=["#b03a2e" if x > 0 else "#2e7d32" for x in rates])
ax[2].plot([r[0] for r in rerun], [r[3] for r in rerun], "o", mfc="none", mec="black", ms=9, label="rerun (Cap07)")
for r in main:
    if r[4] is not None:
        ax[2].annotate("bubbles from it. %d" % r[4], (r[0], r[3]), ha="center", va="bottom", fontsize=8)
ax[2].set_xlabel("interface thickness W")
ax[2].set_ylabel("Bubble growth/1000 iter on floor xf-layer")
ax[2].set_title("Bubble growth rate vs. interface thickness;  σ=0,5", fontsize=10)
ax[2].set_xticks(ws)
if rerun:
    ax[2].legend(frameon=False, fontsize=8)

for a in ax:
    a.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("stability_vs_W.png", dpi=150)
print("wrote stability_vs_W.png and stability_vs_W.csv")
