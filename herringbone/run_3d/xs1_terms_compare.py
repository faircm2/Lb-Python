"""Contribution of the terms of the equations at the failing cell of the W=3 and the W=4 run, one column per run (W=7: healthy run, same box and window as W=3, shown to its last row), same rows.
Rows (all |term|, log scale):
  1 state            phi, rho, |u| (dashed: Cs = 0.577)
  2 eq 28            u = A + B + C   with A = J/rho (momentum moment of f), B = Fs dt/(2 rho) (capillary), C = body force/(2 rho)
  3 eq 27            p = sum f + dt/2 u.grad(rho Cs2)
  4 eq 20,21         force terms of the f equation: Fs, G (body force), Fi moments (force part, grad(rho Cs2) part), grad(rho Cs2)
  5 Allen-Cahn       dphi/dt, div(phi u), M lap(phi), M div(lambda n)
Cell = (W=7: the W=3 failing cell x154 y10 z29)  largest |u| in the last row where the box maximum |u| is still < 1 (cell is therefore the one that starts the failure); the rows shown end where the box max|u| first exceeds 100.
Data:  W=3 job 8820586 terms_W3c (box x148-160 y5-17 z22-33, its 2215-2254);  W=4 job 8815366 terms_W4b (box x125-138 y22-33 z9-20, its 10170-10269).
Run on the PC in herringbone\\run_3d:  python xs1_terms_compare.py       ->  xs1_terms_compare.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CS = 1 / np.sqrt(3)


def load(folder, origin, parts):
    """parts: list of (file, concat axis). Returns it, dict of arrays (time first), origin."""
    d = [np.load(os.path.join(HERE, folder, f)) for f, _ in parts]
    axis = parts[0][1]
    T = {k: np.concatenate([p[k] for p in d], axis=axis) for k in d[0].files if k != "it"}
    return d[0]["it"], T, origin


CASES = [
    ("W=3 (job 8820586)", *load("terms_W3c", (148, 5, 22), [("terms_x148_y5_z22.npz", -1), ("terms_x148_y5_z32.npz", -1)])),
    ("W=4 (job 8815366)", *load("terms_W4b", (125, 22, 9), [("terms_x125_y22_z9.npz", -2), ("terms_x125_y32_z9.npz", -2)])),
    ("W=7 (job 8840757, no failure)", *load("terms_W7", (148, 5, 22), [("terms_x148_y5_z22.npz", -1), ("terms_x148_y5_z32.npz", -1)])),
]


def mag(a):
    return np.abs(a) if a.ndim == 4 else np.sqrt((a ** 2).sum(axis=1))


ROWS = [
    ("state", [("phi", "phi"), ("rho", "rho"), ("u", "|u|")]),
    ("eq 28:  u = A + B + C", [("A", "A = J/rho"), ("B", "B = Fs dt/(2 rho)"), ("C", "C = g/(2 rho)"), ("u", "|u|")]),
    ("eq 27:  p", [("sum_f", "sum f"), ("p_corr", "dt/2 u.grad(rho Cs2)"), ("p", "p")]),
    ("eq 20, 21:  forces in the f equation", [("Fs", "Fs"), ("G", "G (body force)"), ("Fi_mom_force", "Fi moment, force part"), ("Fi_mom_grad", "Fi moment, grad(rho Cs2) part"), ("grad_rhoCs2", "grad(rho Cs2)")]),
    ("Allen-Cahn:  dphi/dt + div(phi u) = div[M(grad phi - lambda n)]", [("ddt_phi", "dphi/dt"), ("div_phi_u", "div(phi u)"), ("M_lap_phi", "M lap(phi)"), ("M_div_lambda_n", "M div(lambda n)")]),
]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, axs = plt.subplots(len(ROWS), len(CASES), figsize=(7.5 * len(CASES), 4.2 * len(ROWS)), layout="constrained", squeeze=False, sharey="row")
for col, (name, it, T, org) in enumerate(CASES):
    speed = mag(T["u"]).reshape(len(it), -1)
    smax = np.where(np.isfinite(speed).all(axis=1), np.nan_to_num(speed).max(axis=1), np.inf)      # a row with any non-finite value counts as failed
    last = int(np.where(smax < 1.0)[0].max())                          # last row with box max|u| < 1
    stop = int(np.where(smax > 100)[0].min()) if (smax > 100).any() else len(it) - 1
    sl = slice(0, stop + 1)
    i = np.unravel_index(np.nanargmax(mag(T["u"])[last]), mag(T["u"])[last].shape)
    if name.startswith("W=7"):                                          # healthy run: show the cell where W=3 fails (x154 y10 z29), not its own fastest cell
        i = tuple(c - o for c, o in zip((154, 10, 29), org))
    cell = tuple(int(o + k) for o, k in zip(org, i))
    print(f"{name}: rows {it[0]}..{it[-1]}, cell {cell} (largest |u| = {mag(T['u'])[last][i]:.3f} at it {it[last]}), shown to it {it[stop]}")
    for row, (title, keys) in enumerate(ROWS):
        ax = axs[row, col]
        for j, (k, lab) in enumerate(keys):
            v = mag(T[k])[sl][(slice(None),) + i]
            ax.plot(it[sl], np.where(v > 1e-14, v, np.nan), label=lab, color=COLORS[j], lw=1.6)      # exact zeros (phi = 0, ...) are gaps on the log scale
        if row in (0, 1):
            ax.axhline(CS, color="k", ls="--", lw=0.8)
        ax.axvline(it[last], color="grey", ls=":", lw=1)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        ax.set_title((f"{name}: cell x{cell[0]} y{cell[1]} z{cell[2]}\n" if row == 0 else "") + title, fontsize=10)
        if row == len(ROWS) - 1:
            ax.set_xlabel("iteration")
fig.suptitle("|term| at the failing cell (dotted: last row with box max|u| < 1;  dashed: Cs)")
out = os.path.join(HERE, "xs1_terms_compare.png")
fig.savefig(out, dpi=100)
print("wrote", out)
