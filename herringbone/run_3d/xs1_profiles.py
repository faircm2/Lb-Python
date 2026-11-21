"""Streamwise profiles h(x) and Q(x) of the xs1.0 runs at several iterations (where does the film amplitude grow?).

Data: xs1_pack/fill14/ (4983) and xs1_late/fill14/, xs1_late/fill9/ with fields3d_iter_XXXXXX.npz.
Usage: python xs1_profiles.py    ->  xs1_profiles.png and a text table of h(x) amplitude per region
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TP"] = "4.9"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from herringbone_idealised import Herringbone, herringbone_mask

XN, YN, ZN = 736, 64, 64
D, DX = 1.2e-3, 1.2e-3 / YN
solid = herringbone_mask(XN, YN, ZN, Herringbone())[1:XN + 1, 1:YN + 1, 1:ZN + 1]
x_mm = (np.arange(XN) + 0.5) * DX * 1e3
TRAINS = ((53, 260), (315, 522))                                  # cell numbers of the groove blocks
REGIONS = {"offset 0-52": (0, 53), "train 1": (53, 260), "gap": (260, 315), "train 2": (315, 522), "run-out": (522, XN)}

RUNS = {  # name: (nu, tau_f, [(iteration, file)])
    "fill 14, nu 2.9e-6": (2.9e-6, 1.0, [(4983, "xs1_pack/fill14/fields3d_iter_004983.npz"), (8400, "xs1_late/fill14/fields3d_iter_008400.npz"),
                                          (10080, "xs1_late/fill14/fields3d_iter_010080.npz"), (12000, "xs1_late/fill14/fields3d_iter_012000.npz")]),
    "fill 9, nu 1e-6": (1e-6, 0.65, [(8400, "xs1_late/fill9/fields3d_iter_008400.npz")]),
}


def profiles(fn, nu, tau):
    d = np.load(os.path.join(HERE, fn))
    phi, u = d["phi"].astype(float), d["u"].astype(float)
    q = DX ** 3 / ((1 / 3) * (tau - 0.5) * DX ** 2 / nu) * 6e7           # lattice flux -> ml/min
    return phi.sum(axis=2).mean(axis=1), (phi * np.where(solid, 0.0, u[0]) * ~solid).sum(axis=(1, 2)) * q


fig, axes = plt.subplots(2, len(RUNS), figsize=(15, 7), sharex=True, layout="constrained")
print(f"{'run':<20s} {'iter':>5s} | " + " | ".join(f"{r:^21s}" for r in REGIONS) + "      (h min..max, cells)")
for col, (name, (nu, tau, files)) in enumerate(RUNS.items()):
    for (it, fn), c in zip(files, plt.cm.viridis(np.linspace(0, 0.9, len(files)))):
        h, q = profiles(fn, nu, tau)
        axes[0, col].plot(x_mm, h, color=c, label=f"iter {it}")
        axes[1, col].plot(x_mm, q, color=c, label=f"iter {it}")
        print(f"{name:<20s} {it:5d} | " + " | ".join(f"{h[a:b].min():8.1f}..{h[a:b].max():<8.1f}   " for a, b in REGIONS.values()))
    axes[0, col].set(title=name, ylabel="liquid column height h (cells)")
    axes[1, col].set(xlabel="x (mm)", ylabel="Q through the x-plane (ml/min)")
    axes[1, col].axhline(1.0, color="k", lw=0.6, ls=":")
    for ax in axes[:, col]:
        for a, b in TRAINS:
            ax.axvspan(a * DX * 1e3, b * DX * 1e3, color="0.85", zorder=0)
        ax.legend(fontsize=8)
fig.suptitle("xs1.0 streamwise profiles (grey = groove trains)")
out = os.path.join(HERE, "xs1_profiles.png")
fig.savefig(out, dpi=110)
print("wrote", out)
