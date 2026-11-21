"""Term dump of the W=3 failure (job 8792532): which terms of the equations change rapidly?
Reads terms_x148_y5_z*.npz (box x148-160, y5-17, z22-32, iterations 2150-2230; z22 file = z22..31, z32 file = z32) from a folder.
Run on the PC in C:\\Dirk\\Neethling\\iems\\master\\PythonLBCourse\\herringbone\\run_3d:  python xs1_terms_plot.py terms_W3
Writes xs1_terms_cell.png (every term vs iteration at the gas cell with the largest |u|),
xs1_terms_growth.png (growth of max|term| over the gas cells of the box, first -> last row) and prints the growth table.
"""
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else "terms_W3"
files = sorted(glob.glob(f"{folder}/terms_x148_y5_z*.npz"), key=lambda f: int(f.rsplit("_z", 1)[1].split(".")[0]))
parts = [np.load(f) for f in files]
it = parts[0]["it"]
T = {k: np.concatenate([p[k] for p in parts], axis=-1) for k in parts[0].files if k != "it"}   # z axis is last
print("files:", files, "\nrows:", it[0], "..", it[-1], "  box shape:", T["phi"].shape[1:])


def mag(a):
    """(T,...) scalar stays, (T,3,...) vector -> euclidean norm."""
    return np.abs(a) if a.ndim == 4 else np.sqrt((a ** 2).sum(axis=1))


# --- the cell: gas (phi<0.05 in the first row) with the largest |u| at any time ---------------------------------------
speed = mag(T["u"])
gas0 = T["phi"][0] < 0.05
with np.errstate(invalid="ignore"):
    peak = np.where(gas0, np.nanmax(speed, axis=0), -1.0)
ix, iy, iz = np.unravel_index(np.nanargmax(peak), peak.shape)
print(f"cell (box-local {ix},{iy},{iz}) = global x{148 + ix} y{5 + iy} z{22 + iz}: max |u| {peak[ix, iy, iz]:.3f}")

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
fig.suptitle(f"|term| at gas cell x{148 + ix} y{5 + iy} z{22 + iz} (largest |u|), W=3, job 8792532")
fig.tight_layout()
fig.savefig("xs1_terms_cell.png", dpi=110)

# --- growth of the max over all gas cells of the box -----------------------------------------------------------------
rows = []
with np.errstate(invalid="ignore", divide="ignore"):
    for k, a in T.items():
        m = mag(a)
        first, last = np.nanmax(m[0][gas0]), np.nanmax(m[-1][gas0])
        rows.append((k, first, last, last / max(first, 1e-14)))
rows.sort(key=lambda r: -r[3])
print(f"\n{'term':>16s} {'max|.| first':>13s} {'max|.| last':>13s} {'ratio':>10s}   (gas cells of the box, row {it[0]} -> row {it[-1]})")
for k, f, l, r in rows:
    print(f"{k:>16s} {f:13.3e} {l:13.3e} {r:10.2f}")
fig2, ax = plt.subplots(figsize=(8, 10))
ax.barh([r[0] for r in rows][::-1], np.log10([max(r[3], 1e-3) for r in rows][::-1]))
ax.set_xlabel("log10 ( max|term| last row / first row )")
ax.set_title("growth over the window, gas cells of the box")
ax.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("xs1_terms_growth.png", dpi=110)
print("\nwrote xs1_terms_cell.png, xs1_terms_growth.png")
