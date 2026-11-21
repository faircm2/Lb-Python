"""Meniscus rise against time for the parameter-sensitivity runs (studies 3, 4, 6, 7, 8).

usage: python analyse_meniscus.py <runs folder> [<out folder>]
  <runs folder> holds one folder per run: <tag>/<script+stub>/surface_height.npz  (tag like s3_tauf_1.5)

Meniscus rise R at the mid-x plane, from surface_height.npz (row = iteration, then the free-surface height of
every y column, in cells):
    R = mean(h at the two walls, 3 outer columns each) - median(h in the central 10 % of the channel)
Time axis in physical units (dt follows tau_f: dt = Cs^2 (tau_f - 0.5) dx^2 / nu).

Outputs: meniscus_summary.csv (R at the end of every run), meniscus_rise_vs_time.png (one panel per varied parameter).
"""
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CS2 = 1.0 / 3.0
D_ND = 384
NU_DEFAULT = 2.9e-6
D_DEFAULT = 1e-3


def rise(h):
    ny = h.shape[1]
    wall = 0.5 * (h[:, :3].mean(1) + h[:, -3:].mean(1))
    cen = np.median(h[:, ny // 2 - ny // 20: ny // 2 + ny // 20], axis=1)
    return wall - cen


def parse_stub(name):
    """tau_f, nu, D from the run folder name (defaults of the script when the name leaves them out)"""
    tf = float(re.search(r"__tau_f_([0-9.]+)__", name).group(1))
    m = re.search(r"_nu([0-9.e+-]+?)(?:_[A-Za-z]|$)", name)
    nu = float(m.group(1)) if m else NU_DEFAULT
    m = re.search(r"_D([0-9.]+?)(?:_[A-Za-z]|$)", name)
    d = float(m.group(1)) * 1e-3 if m else D_DEFAULT
    return tf, nu, d


# tag prefix -> (name, symbol, default value)
PARAMS = [
    ("s3", "Flow relaxation time", "τ_f", 1.0),
    ("s4", "Interface relaxation time", "τ_g", 0.65),
    ("s6", "Interface thickness (cells)", "W", 4.0),
    ("s7", "Wall contact angle (°)", "θ", 60.0),
    ("s8", "Surface tension (lattice units)", "σ", 0.1),
]
COLORS = ["purple", "red", "orange", "lime", "aqua"]
DEFAULT_TEXT ="τ_f = 1.0,  τ_g = 0.65,  W = 4,  θ = 60°,  σ = 0.1"


def tag_value(tag):
    return float(tag.split("_")[-1])


def main():
    runs = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else runs
    os.makedirs(out, exist_ok=True)
    curves = {}
    rows = []
    for f in sorted(glob.glob(os.path.join(runs, "*", "*", "surface_height.npz"))):
        tag = os.path.basename(os.path.dirname(os.path.dirname(f)))
        stub = os.path.basename(os.path.dirname(f))
        tf, nu, d = parse_stub(stub)
        dt = CS2 * (tf - 0.5) * (d / D_ND) ** 2 / nu
        sh = np.load(f)["SurfaceHeight"]
        n = sh[:, 0]
        r = rise(sh[:, 1:])
        curves[tag] = (n * dt * 1e3, r)
        rows.append((tag, int(n[-1]), n[-1] * dt * 1e3, r[-1]))

    with open(os.path.join(out, "meniscus_summary.csv"), "w") as fh:
        fh.write("tag,iterations,t_end_ms,R_end_cells\n")
        for r_ in rows:
            fh.write("%s,%d,%.3f,%.3f\n" % r_)
    print("%-16s %10s %10s %10s" % ("tag", "iterations", "t_end[ms]", "R_end"))
    for r_ in rows:
        print("%-16s %10d %10.3f %10.3f" % r_)

    present = [p for p in PARAMS if any(t.startswith(p[0] + "_") for t in curves)]
    ymax = 1.05 * max(r.max() for _, r in curves.values())
    tmax = max(t.max() for t, _ in curves.values())
    ncol = 2
    nrow = int(np.ceil((len(present) + 1) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.4 * nrow), sharey=True)
    axs = axs.ravel()
    cmap = plt.get_cmap("viridis")
    for ax, (g, name, sym, default) in zip(axs, present):
        tags = sorted([t for t in curves if t.startswith(g + "_")], key=tag_value)
        order = sorted({tag_value(t) for t in tags} | {default})
        for i, t in enumerate(tags):
            v = tag_value(t)
            ax.plot(*curves[t], color=COLORS[i % len(COLORS)], lw=2, label="%s = %g" % (sym, v))
        ax.plot(*curves["s0_base"], color="black", lw=2.8, label="%s = %g  (default)" % (sym, default))
        ax.set_title(name + "  " + sym, fontsize=11)
        ax.set_xlim(0, tmax * 1.02)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("time (ms)")
        ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
        ax.grid(alpha=0.25)
        ax.set_box_aspect(1)
    for i in range(0, len(axs), ncol):
        axs[i].set_ylabel("meniscus rise R (cells)")
    guide = axs[len(present)]
    guide.axis("off")
    guide.text(0.0, 1.0,
               "R = surface height at the walls minus\n"
               "surface height in the middle of the channel\n"
               "(mid-x plane, lattice cells).\n\n"
               "Every curve is one simulation, all run for\n"
               "the same physical time (18.7 ms), one setting\n"
               "varied at a time.\n\n"
               "Black = default setting:\n" + DEFAULT_TEXT,
               va="top", ha="left", fontsize=10.5, linespacing=1.35, transform=guide.transAxes)
    for ax in axs[len(present) + 1:]:
        ax.axis("off")
    fig.tight_layout(h_pad=4.0)
    fig.savefig(os.path.join(out, "meniscus_rise_vs_time.png"), dpi=130)
    print("wrote", os.path.join(out, "meniscus_summary.csv"), "and meniscus_rise_vs_time.png")


if __name__ == "__main__":
    main()
