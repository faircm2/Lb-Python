"""Period-10 oscillation, W=3 (job 8792532) vs W=4 (job 8800242), same box x148-160, y5-17, z22-32, iterations 2150-2229.
Envelope = half of (max - min) of the field over a sliding window of 10 iterations (one period), maximum over the gas cells of the box
(phi<0.05 in the first row), for p, rho, u_y; plus the same at the cell x154 y9 z29 and the dominant period there.
Run on the PC in herringbone\\run_3d:  python xs1_w3_vs_w4.py terms_W3 terms_W4      -> xs1_w3_vs_w4.png
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

folders = sys.argv[1:3] if len(sys.argv) > 2 else ["terms_W3", "terms_W4"]
labels = {folders[0]: "W=3", folders[1]: "W=4"}
CELL = (6, 4, 7)                                        # x154 y9 z29, box-local
WIN = 10


def envelope(a, mask):
    """a (T, x, y, z) -> (T-WIN+1,) max over mask cells of half the peak-to-peak over a sliding window"""
    out = []
    for k in range(WIN - 1, a.shape[0]):
        w = a[k - WIN + 1:k + 1]
        out.append((0.5 * (w.max(axis=0) - w.min(axis=0)))[mask].max())
    return np.array(out)


data = {}
for f in folders:
    T = np.load(f"{f}/terms_x148_y5_z22.npz")
    it = T["it"]
    gas = T["phi"][0] < 0.05
    fields = {"p": T["p"], "rho": T["rho"], "u_y": T["u"][:, 1], "phi": T["phi"]}
    data[f] = dict(it=it, gas=gas, env={k: envelope(v, gas) for k, v in fields.items()}, cell={k: v[(slice(None),) + CELL] for k, v in fields.items()},
                   umax=np.sqrt((T["u"] ** 2).sum(axis=1))[:, gas].max(axis=1))
    s = fields["p"][:40][(slice(None),) + CELL]
    s = s - s.mean()
    P = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    print(f"{labels[f]}: box gas cells {gas.sum()}, dominant period of p at the cell (its 2150-2189): {1 / np.fft.rfftfreq(len(s))[P[1:].argmax() + 1]:.1f} iterations")

print(f"\nenvelope (max over box gas cells), sampled every 10 iterations; last column = last / first")
for name in ("p", "rho", "u_y", "phi"):
    for f in folders:
        e = data[f]["env"][name]
        print(f"{name:4s} {labels[f]}: " + " ".join(f"{v:.3e}" for v in e[::10]) + f"   x{e[-1] / e[0]:.2f}")
for f in folders:
    um = data[f]["umax"]
    print(f"max gas |u| {labels[f]}: " + " ".join(f"{v:.3f}" for v in um[::10]) + f"   last {um[-1]:.3f}")

fig, axs = plt.subplots(2, 3, figsize=(17, 11))
for col, name in enumerate(("p", "rho", "u_y")):
    ax = axs[0, col]
    for f in folders:
        d = data[f]
        ax.plot(d["it"][WIN - 1:], d["env"][name], label=labels[f])
    ax.set_yscale("log"); ax.set_title(f"envelope of {name} oscillation (max over box gas cells)"); ax.set_xlabel("iteration"); ax.legend(); ax.grid(alpha=0.3); ax.set_box_aspect(1)
    ax = axs[1, col]
    for f in folders:
        d = data[f]
        ax.plot(d["it"], d["cell"][name], label=labels[f])
    ax.set_title(f"{name} at the cell x154 y9 z29"); ax.set_xlabel("iteration"); ax.legend(); ax.grid(alpha=0.3); ax.set_box_aspect(1)
fig.tight_layout()
fig.savefig("xs1_w3_vs_w4.png", dpi=100)
print("\nwrote xs1_w3_vs_w4.png")
