"""Meniscus height against the 0.4 mm channel depth for the theta range runs (sigma_lat 0.4, tau_f 0.55, alpha 90, periodic x, W 10, fill 51).
Reads surface_height.npz of each run (unpacked from surfh.tgz into ./unpacked).  Row = one dump: column 0 is the iteration, columns 1..384 the film
surface height in cells for every y across the channel width (floor at 0).  dx = 1.2 mm / 384 = 3.125 um, dt = (1/3)(tau_f - 1/2) dx^2 / nu.
Writes wall_height_vs_time.png, profile_across_width.png and meniscus_theta_range.csv."""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["legend.fancybox"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DX = 1.2e-3 / 384 * 1e6                                    # um per cell
DT = (1 / 3) * (0.55 - 0.5) * (1.2e-3 / 384) ** 2 / 1e-6   # s per iteration
H_CHANNEL = 400.0                                          # um

TAIL = "_W10_nu1e-06_D1.2_fill51_periodicx"
# label, pattern (tau_f 0.55 Cap07 folders), colour, linestyle
# sigma 0.4 theta series: colour runs from blue (small theta) to red (90 deg)
THETA_RUNS = [
    (62.5, "θ = 62.5°  (24,001 it.)", "_x2"),
    (65.0, "θ = 65°  (LONG, cancelled at 38,574 it.)", "_x12"),
    (70.0, "θ = 70°  (24,001 it.)", "_x2"),
    (75.0, "θ = 75°  (24,001 it.)", "_x2"),
    (80.0, "θ = 80°  (24,001 it.)", "_x2"),
    (85.0, "θ = 85°  (24,001 it.)", "_x2"),
    (88.0, "θ = 88°  (24,001 it.)", "_x2"),
    (90.0, "θ = 90°  (24,001 it.)", "_x2"),
]
_cols = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(THETA_RUNS)))
RUNS = [(lab, "*Theta_%.1f_alpha90_sigma0.4" % th + TAIL + suf, _cols[i], "-") for i, (th, lab, suf) in enumerate(THETA_RUNS)]
RUNS += [
    ("θ = 65°, σ = 0.3  (EXP_s0.3)", "*Theta_65.0_alpha90_sigma0.3" + TAIL + "_x2", "0.45", "--"),
    ("θ = 65°, σ = 0.5  (12,000 it.)", "*Theta_65.0_alpha90_sigma0.5" + TAIL, "tab:orange", "--"),
]


def load(pattern):
    hits = glob.glob(os.path.join(HERE, "unpacked", "*Cap07*tau_f_0.55*" + pattern, "surface_height.npz"))
    if len(hits) != 1:
        raise SystemExit("expected exactly one folder for %s, found %d" % (pattern, len(hits)))
    return np.load(hits[0])["SurfaceHeight"]


figs, ax = [], []
for _ in range(2):
    f, a = plt.subplots(figsize=(6.4, 4.8))
    figs.append(f)
    ax.append(a)
rows = []
for label, pattern, col, ls in RUNS:
    s = load(pattern)
    it = s[:, 0]
    h = s[:, 1:] * DX                                   # um, shape [dump, y]
    wall = 0.5 * (h[:, 0] + h[:, -1])
    peak = h.max(axis=1)
    t = it * DT * 1e3
    ax[0].plot(t, wall, color=col, ls=ls, lw=1.4, label=label)
    ax[1].plot(np.arange(h.shape[1]) * DX / 1000, h[-1], color=col, ls=ls, lw=1.4, label=label)
    i = int(np.argmax(wall))
    rows.append((label, int(it[-1]), "%.2f" % (it[-1] * DT * 1e3), "%.0f" % wall.max(), "%.2f" % t[i], "%.0f" % wall[-1], "%.0f" % peak.max()))
for a in ax:
    a.axhline(H_CHANNEL, color="k", lw=1.2)
    a.grid(alpha=0.25)
ax[0].text(0.02, H_CHANNEL + 6, "channel depth 400 µm", fontsize=8, transform=ax[0].get_yaxis_transform())
ax[0].set_xlabel("time (ms)")
ax[0].set_ylabel("film height at the side walls (µm)")
ax[0].set_title("wall height against time", fontsize=10)
ax[0].set_ylim(0, 440)
ax[0].legend(frameon=False, fontsize=7, loc="lower right", ncol=2)
ax[1].set_xlabel("y, channel width (mm)")
ax[1].set_ylabel("film height (µm)")
ax[1].set_title("film height across the width at the last dump", fontsize=10)
ax[1].set_ylim(0, 440)
ax[1].legend(frameon=False, fontsize=7, loc="lower center", ncol=2)
for f, name in zip(figs, ("wall_height_vs_time.png", "profile_across_width.png")):
    f.tight_layout()
    f.savefig(os.path.join(HERE, name), dpi=170)

with open(os.path.join(HERE, "meniscus_theta_range.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["run", "last_iteration", "last_time_ms", "max_wall_height_um", "time_of_max_ms", "wall_height_last_um", "max_height_anywhere_um"])
    w.writerows(rows)
for r in rows:
    print(" | ".join(str(x) for x in r).encode("ascii", "replace").decode())
