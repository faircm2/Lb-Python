"""All terms against iteration, one column per interface width W.  Shows which term dominates and when it blows up.
Row 1: |term| at one cell (W=3 failing cell x154 y10 z29 for W=3 and W=7; W=4: x131 y27 z13, largest |u| in the last row with box max|u| < 1)
Row 2: max over the fluid cells of the dump box of |term| (cell independent; solid cells excluded)
Colours = equation group:  eq 28 u=A+B+C (reds), eq 27 p (greens), eq 20/21 forces (blues), Allen-Cahn (oranges/browns), state rho (black, dotted).
Rows end where the box max|u| first exceeds 100 (W=3, W=4); W=7 is shown over its whole dump window (its 2215-2254, no failure).
Data: W=3 terms_W3c (job 8820586), W=4 terms_W4b (job 8815366), W=7 terms_W7 (job 8840757).
Run on the PC in herringbone\\run_3d:  python xs1_terms_by_w.py       ->  xs1_terms_by_w.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

np.seterr(all="ignore")
os.environ["TP"] = "4.9"                                              # train period; must be set before the import
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from herringbone_idealised import Herringbone, herringbone_mask
SOLID = herringbone_mask(736, 64, 64, Herringbone())[1:737, 1:65, 1:65]      # index = cell number - 1


def load(folder, parts):
    d = [np.load(os.path.join(HERE, folder, f)) for f, _ in parts]
    axis = parts[0][1]
    return d[0]["it"], {k: np.concatenate([p[k] for p in d], axis=axis) for k in d[0].files if k != "it"}


CASES = [
    ("W=3  (job 8820586)", (148, 5, 22), (154, 10, 29), *load("terms_W3c", [("terms_x148_y5_z22.npz", -1), ("terms_x148_y5_z32.npz", -1)])),
    ("W=4  (job 8815366)", (125, 22, 9), None, *load("terms_W4b", [("terms_x125_y22_z9.npz", -2), ("terms_x125_y32_z9.npz", -2)])),
    ("W=7  (job 8840757, no failure)", (148, 5, 22), (154, 10, 29), *load("terms_W7", [("terms_x148_y5_z22.npz", -1), ("terms_x148_y5_z32.npz", -1)])),
]
TERMS = [                                                             # key, label, colour, linestyle
    ("u", "|u|", "#7f0000", "-"), ("A", "A = J/rho", "#d7301f", "-"), ("B", "B = Fs dt/(2rho)", "#fc8d59", "-"), ("C", "C = g/(2rho)", "#fdbb84", "-"),
    ("p", "p", "#006d2c", "-"), ("sum_f", "sum f", "#41ab5d", "--"), ("p_corr", "dt/2 u.grad(rho Cs2)", "#a1d99b", ":"),
    ("Fs", "Fs", "#08306b", "-"), ("G", "G (body)", "#2171b5", "--"), ("Fi_mom_force", "Fi mom, force", "#4292c6", "-."), ("Fi_mom_grad", "Fi mom, grad(rho Cs2)", "#9ecae1", ":"),
    ("grad_rhoCs2", "grad(rho Cs2)", "#6a51a3", "-"),
    ("ddt_phi", "dphi/dt", "#a63603", "-"), ("div_phi_u", "div(phi u)", "#e6550d", "--"), ("M_lap_phi", "M lap(phi)", "#fd8d3c", "-."), ("M_div_lambda_n", "M div(lambda n)", "#fdae6b", ":"),
    ("rho", "rho", "k", ":"),
]


def mag(a):
    return np.abs(a) if a.ndim == 4 else np.sqrt((a ** 2).sum(axis=1))


fig, axs = plt.subplots(2, len(CASES), figsize=(7 * len(CASES), 13), layout="constrained", sharey="row")
for col, (name, org, fixed, it, T) in enumerate(CASES):
    speed = mag(T["u"])
    smax = np.where(np.isfinite(speed.reshape(len(it), -1)).all(axis=1), np.nan_to_num(speed).reshape(len(it), -1).max(axis=1), np.inf)
    last = int(np.where(smax < 1.0)[0].max())
    stop = int(np.where(smax > 100)[0].min()) if (smax > 100).any() else len(it) - 1
    sl = slice(0, stop + 1)
    if fixed is None:
        i = np.unravel_index(np.nanargmax(speed[last]), speed[last].shape)
    else:
        i = tuple(c - o for c, o in zip(fixed, org))
    cell = tuple(int(o + k) for o, k in zip(org, i))
    bx, by, bz = speed.shape[1:]
    fluid = ~SOLID[org[0] - 1:org[0] - 1 + bx, org[1] - 1:org[1] - 1 + by, org[2] - 1:org[2] - 1 + bz]
    top = {}
    for row in (0, 1):
        ax = axs[row, col]
        for key, lab, colr, ls in TERMS:
            m = mag(T[key])[sl]
            v = m[(slice(None),) + i] if row == 0 else np.nanmax(np.where(fluid, m, np.nan), axis=(1, 2, 3))
            if row == 1:
                top[key] = np.nanmax(v[-6:])
            ax.plot(it[sl], np.where(v > 1e-14, v, np.nan), color=colr, ls=ls, lw=1.8, label=lab)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.axhline(1 / np.sqrt(3), color="k", lw=0.6, ls=(0, (1, 4)))
        if row == 0:
            ax.set_title(f"{name}\ncell x{cell[0]} y{cell[1]} z{cell[2]}", fontsize=10)
        else:
            ax.set_title("max over the fluid cells of the box", fontsize=10)
            ax.set_xlabel("iteration")
    order = sorted(top, key=lambda k: -top[k])[:4]
    print(f"{name}: shown its {it[0]}..{it[stop]}; largest box-max terms over its last 6 rows:", ", ".join(f"{k} {top[k]:.2e}" for k in order))
axs[0, 0].legend(fontsize=7, ncol=2, loc="lower left")
fig.suptitle("Terms of the equations vs iteration for W = 3, 4, 7 (dotted line: Cs)")
out = os.path.join(HERE, "xs1_terms_by_w.png")
fig.savefig(out, dpi=100)
print("wrote", out)
