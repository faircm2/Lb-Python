"""Structure of the growing oscillation at the failing gas cell: space-time diagrams of u_y, rho, p along x, y and z through the cell
(box x148-159, y5-16, z22-31, iterations 2150-2229), plus the dominant period / wavelength.
Run on the PC in herringbone\\run_3d:  python xs1_osc_structure.py terms_W3      -> xs1_osc_structure.png
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else "terms_W3"
T = np.load(f"{folder}/terms_x148_y5_z22.npz")
it, u, rho, p, phi = T["it"], T["u"], T["rho"], T["p"], T["phi"]
bx, by, bz = rho.shape[1:]
ix, iy, iz = 6, 4, 7                                   # x154 y9 z29, the cell of the budget
fields = {"u_y": u[:, 1], "rho": rho, "p": p}
lines = {"x": (slice(None), iy, iz), "y": (ix, slice(None), iz), "z": (ix, iy, slice(None))}

fig, axs = plt.subplots(3, 3, figsize=(16, 16))
for r, (fname, F) in enumerate(fields.items()):
    for col, (ax_name, sl) in enumerate(lines.items()):
        D = F[(slice(None),) + sl]                      # (time, space)
        D = D - D.mean(axis=0, keepdims=True)           # remove the static profile so the oscillation shows
        v = np.abs(D).max()
        im = axs[r, col].imshow(D.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
                                extent=[it[0], it[-1], -0.5, D.shape[1] - 0.5])
        axs[r, col].set_title(f"{fname} - time mean, along {ax_name} through x154 y9 z29")
        axs[r, col].set_xlabel("iteration")
        axs[r, col].set_ylabel(f"{ax_name} cell index in box")
        axs[r, col].set_box_aspect(1)
        fig.colorbar(im, ax=axs[r, col], fraction=0.046)
fig.tight_layout()
fig.savefig("xs1_osc_structure.png", dpi=100)

# dominant period at the cell (first 40 rows, before the amplitude is large) and spatial wavelength along each axis
for name, F in fields.items():
    s = F[:40, ix, iy, iz] - F[:40, ix, iy, iz].mean()
    f = np.fft.rfftfreq(len(s), 1.0)
    P = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    k = P[1:].argmax() + 1
    print(f"{name}: dominant period at the cell, its 2150-2189: {1 / f[k]:.1f} iterations")
for name, F in fields.items():
    for ax_name, sl in lines.items():
        row = F[(slice(60),) + sl].std(axis=0)           # amplitude of the oscillation along the line, rows 0-59
        print(f"{name:4s} oscillation amplitude (std over its 2150-2209) along {ax_name}: " + " ".join(f"{a:.4f}" for a in row))
print("wrote xs1_osc_structure.png")
