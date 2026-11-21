"""Term dump of the W=4 failure (job 8815366): which terms of the equations change rapidly before the NaN?
Reads terms_x125_y22_z9.npz (y22-31) and terms_x125_y32_z9.npz (y32-33) of box x125-138, y22-33, z9-20, iterations 10170-10269 (NaN from 10254).
Run on the PC in herringbone\\run_3d:  python xs1_w4b_terms.py terms_W4b
Writes xs1_w4b_cell.png (every term vs iteration at the cell with the largest |u|), xs1_w4b_growth.png and prints the growth table.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else "terms_W4b"
X0, Y0, Z0 = 125, 22, 9
parts = [np.load(f"{folder}/terms_x125_y{y}_z9.npz") for y in (22, 32)]
it = parts[0]["it"]
T = {k: np.concatenate([p[k] for p in parts], axis=-2) for k in parts[0].files if k != "it"}      # y axis is second to last
ok = np.isfinite(T["phi"]).reshape(len(it), -1).all(1) & np.isfinite(T["u"]).reshape(len(it), -1).all(1)
n = int(np.argmin(ok)) if not ok.all() else len(it)                                                # first non-finite row
it = it[:n]
T = {k: v[:n] for k, v in T.items()}
print("rows used:", it[0], "..", it[-1], " box shape:", T["phi"].shape[1:])


def mag(a):
    return np.abs(a) if a.ndim == 4 else np.sqrt((a ** 2).sum(axis=1))


speed = mag(T["u"])
print("max |u| over the box, every 8th row:")
for k in range(0, n, 8):
    i = np.unravel_index(np.argmax(speed[k]), speed[k].shape)
    print(f"  it {it[k]}: {speed[k].max():.3f} at x{X0 + i[0]} y{Y0 + i[1]} z{Z0 + i[2]} phi {T['phi'][k][i]:.3f}")
i = np.unravel_index(np.argmax(speed[-1]), speed[-1].shape)
print(f"  it {it[-1]} (last finite): {speed[-1].max():.3f} at x{X0 + i[0]} y{Y0 + i[1]} z{Z0 + i[2]} phi {T['phi'][-1][i]:.3f}")

ix, iy, iz = i                                                                                       # cell with the largest |u| in the last finite row
print(f"cell: global x{X0 + ix} y{Y0 + iy} z{Z0 + iz}")
gas0 = T["phi"][0] < 0.05

GROUPS = {
    "state": ["phi", "rho", "tau_f", "u"],
    "eq 28  u = A + B + C": ["mom_f", "A", "B", "C", "u"],
    "eq 5,6  Fs = mu_phi grad(phi)": ["mu_bulk", "mu_lap", "mu_phi", "grad_phi", "Fs"],
    "eq 27  p": ["sum_f", "p_corr", "p"],
    "eq 20,21  Fi moments": ["Fi_sum_force", "Fi_sum_grad", "Fi_mom_force", "Fi_mom_grad", "G"],
    "grad(rho Cs2)": ["grad_rhoCs2"],
    "Allen-Cahn terms": ["ddt_phi", "div_phi_u", "M_lap_phi", "M_div_lambda_n"],
    "eq 19  Gi moment": ["dphiu_dt", "cs2_lambda_n", "Gi_mom", "lambda"],
    "phi, phi_old, raw_phi": ["phi", "phi_old", "raw_phi"],
}
fig, axs = plt.subplots(3, 3, figsize=(17, 17))
for ax, (title, keys) in zip(axs.ravel(), GROUPS.items()):
    for k in keys:
        ax.plot(it, np.maximum(mag(T[k])[:, ix, iy, iz], 1e-14), label=k)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("iteration")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_box_aspect(1)
fig.suptitle(f"|term| at cell x{X0 + ix} y{Y0 + iy} z{Z0 + iz} (largest |u| at last finite row), W=4, job 8815366")
fig.tight_layout()
fig.savefig("xs1_w4b_cell.png", dpi=100)

# growth of the max over the cells with |u| above 0.05 at the last finite row (the failing cluster), first row -> last finite row
cluster = speed[-1] > 0.05
print(f"\ncluster cells (|u|>0.05 at last finite row): {int(cluster.sum())}")
rows = []
with np.errstate(invalid="ignore", divide="ignore"):
    for k, a in T.items():
        m = mag(a)
        first, last = m[0][cluster].max(), m[-1][cluster].max()
        rows.append((k, first, last, last / max(first, 1e-14)))
rows.sort(key=lambda r: -r[3])
print(f"{'term':>16s} {'max|.| first':>13s} {'max|.| last':>13s} {'ratio':>10s}   (cluster cells, row {it[0]} -> row {it[-1]})")
for k, f, l, r in rows:
    print(f"{k:>16s} {f:13.3e} {l:13.3e} {r:10.2f}")
fig2, ax = plt.subplots(figsize=(8, 10))
ax.barh([r[0] for r in rows][::-1], np.log10([max(r[3], 1e-3) for r in rows][::-1]))
ax.set_xlabel("log10 ( max|term| last row / first row )")
ax.set_title("growth over the window, cells with |u|>0.05 at the end")
ax.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("xs1_w4b_growth.png", dpi=100)
print("\nwrote xs1_w4b_cell.png, xs1_w4b_growth.png")
