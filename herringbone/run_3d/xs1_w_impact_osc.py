"""Impact of the interface width W on the gas: growth of the period-10 oscillation, W=3 against W=4, at the same iterations.
Envelope = over the gas cells (phi < 0.05 in all 10 rows of the window) of  max - min over the last 10 iterations,  for p and for u_y,
box x148-160 y5-17 z22-31 (z22 files).   W=3: terms_W3d (its 1650-1809, job 8840770), terms_W3 (its 2150-2229, job 8792532);
W=4: terms_W4 (its 2150-2229, job 8800242).  Pass further folders as  label:folder  to add runs (e.g. W=7 once its dump is here).
Run on the PC in herringbone\\run_3d:  python xs1_w_impact_osc.py       ->  xs1_w_impact_osc.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
W = 10
SERIES = [("W=3", "terms_W3d"), ("W=3", "terms_W3"), ("W=4", "terms_W4")]
SERIES += [tuple(a.split(":", 1)) for a in sys.argv[1:]]          # extra runs as  label:folder
COLOR = {}
fig, axs = plt.subplots(2, 1, figsize=(10, 11), layout="constrained")
for label, folder in SERIES:
    d = np.load(os.path.join(HERE, folder, "terms_x148_y5_z22.npz"))
    it, phi, p, uy = d["it"], d["phi"], d["p"], d["u"][:, 1]
    ok = np.isfinite(p).reshape(len(it), -1).all(1) & (np.nan_to_num(np.abs(d["u"])).reshape(len(it), -1).max(1) < 1.0)
    n = len(it) if ok.all() else int(np.argmin(ok))
    xs, env_p, env_u = [], [], []
    for k in range(W, n):
        sl = slice(k - W, k)
        g = (phi[sl] < 0.05).all(0)
        if g.sum() == 0:
            continue
        xs.append(it[k])
        env_p.append((p[sl].max(0) - p[sl].min(0))[g].max())
        env_u.append((uy[sl].max(0) - uy[sl].min(0))[g].max())
    col = COLOR.setdefault(label, plt.rcParams["axes.prop_cycle"].by_key()["color"][len(COLOR)])
    print(f"{label} {folder}: its {xs[0]}..{xs[-1]}  p-osc {env_p[0]:.2e} -> {env_p[-1]:.2e}  u_y-osc {env_u[0]:.2e} -> {env_u[-1]:.2e}")
    for ax, y in zip(axs, (env_p, env_u)):
        ax.plot(xs, y, color=col, lw=2, label=f"{label}  ({folder})")
for ax, t in zip(axs, ("p", "u_y")):
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(f"oscillation amplitude of {t} (max - min over 10 its), max over gas cells")
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle("Gas oscillation vs interface width W (same box, sigma 0.1, theta 60)")
out = os.path.join(HERE, "xs1_w_impact_osc.png")
fig.savefig(out, dpi=110)
print("wrote", out)
