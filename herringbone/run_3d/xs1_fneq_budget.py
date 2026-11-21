"""Where does the momentum density J of the failing gas cell come from?  Splits the streamed J into equilibrium and non-equilibrium parts.
J_new(x) = sum_i c_i f*_i(x - c_i) / Cs2   with   f*_i = feq_i + (f*_i - feq_i)  (post-collision), and
feq_i = E_i (p + rho c.u + 1.5 rho (c.u)^2 - 0.5 rho u^2)   (Zhang eq 18)   split into
   J_p  : from E_i * p                (pressure part)
   J_1  : from E_i * rho c.u          (transported rho*u)
   J_2  : from E_i * rho(1.5(c.u)^2 - 0.5u^2)   (convective part)
   J_n  : from f*_i - feq_i           (non-equilibrium part, collision (1-omega) and forcing included)
feq uses u, rho, p of the previous iteration's end state (terms row k-1), as the collision of iteration k does.
Check line: J_p + J_1 + J_2 + J_n must equal mom_f of terms row k at the cell.
Needs terms_x148_y5_z22.npz and ftrace_x148_y5_z22.npz in the folder.  Run on the PC in herringbone\\run_3d:  python xs1_fneq_budget.py terms_W3
Writes xs1_fneq_budget.png.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else "terms_W3"
Cs2 = 1.0 / 3.0
c = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
              [1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1]])
E = np.array([2 / 9] + [1 / 9] * 6 + [1 / 72] * 8)

T = np.load(f"{folder}/terms_x148_y5_z22.npz")
F = np.load(f"{folder}/ftrace_x148_y5_z22.npz")
it = T["it"]
u_all, rho_all, p_all, phi_all, mom_all = T["u"], T["rho"], T["p"], T["phi"], T["mom_f"]
f_star = F["f_star"]                      # (T, 15, bx+2, by+2, bz+2) post-collision, box extended by 1
assert (F["it_f"] == it).all(), "ftrace and terms rows differ"
nT = len(it)
bx, by, bz = rho_all.shape[1:]
print("rows", it[0], "..", it[-1], " box", (bx, by, bz))

# cell: gas (phi<0.05 in row 0), interior of the box, largest |u| at any time
speed = np.sqrt((u_all ** 2).sum(axis=1))
peak = np.where(phi_all[0] < 0.05, speed.max(axis=0), -1.0)
mask = np.zeros_like(peak, dtype=bool)
mask[1:-1, 1:-1, 1:-1] = True
peak = np.where(mask, peak, -1.0)
ix, iy, iz0 = np.unravel_index(peak.argmax(), peak.shape)
comp = int(np.abs(u_all[-1, :, ix, iy, iz0]).argmax())
print(f"cell box-local {ix},{iy},{iz0} = global x{148 + ix} y{5 + iy} z{22 + iz0}, component {'xyz'[comp]}")


def pieces(k):
    """k >= 1.  Returns dict part -> J (3, bx, by, bz) streamed into every box cell at iteration k."""
    u, rho, p = u_all[k - 1], rho_all[k - 1], p_all[k - 1]
    usq = (u ** 2).sum(axis=0)
    fs = f_star[k][:, 1:-1, 1:-1, 1:-1]
    out = {n: np.zeros((3, bx, by, bz)) for n in ("p", "1", "2", "n")}
    for i in range(15):
        cu = c[i, 0] * u[0] + c[i, 1] * u[1] + c[i, 2] * u[2]
        ap, a1, a2 = E[i] * p, E[i] * rho * cu, E[i] * rho * (1.5 * cu ** 2 - 0.5 * usq)
        an = fs[i] - (ap + a1 + a2)
        for name, a in (("p", ap), ("1", a1), ("2", a2), ("n", an)):
            out[name] += c[i][:, None, None, None] * np.roll(a, shift=tuple(c[i]), axis=(0, 1, 2))[None] / Cs2
    return out


cells = [(ix, iy, z) for z in (iz0 - 1, iz0, iz0 + 1) if 1 <= z <= bz - 2]
ks = np.arange(1, nT)
res = {cell: {n: [] for n in ("p", "1", "2", "n", "tot", "mom")} for cell in cells}
for k in ks:
    P = pieces(k)
    for cell in cells:
        x, y, z = cell
        for n in ("p", "1", "2", "n"):
            res[cell][n].append(P[n][comp, x, y, z])
        res[cell]["tot"].append(sum(P[n][comp, x, y, z] for n in ("p", "1", "2", "n")))
        res[cell]["mom"].append(mom_all[k][comp, x, y, z])
for cell in cells:
    for n in res[cell]:
        res[cell][n] = np.array(res[cell][n])
    print(f"check cell {cell}: max |J_p+J_1+J_2+J_n - mom_f| = {np.abs(res[cell]['tot'] - res[cell]['mom']).max():.2e}   (mom_f scale {np.abs(res[cell]['mom']).max():.2e})")

main = (ix, iy, iz0)
print(f"\ncell {main}: J component {'xyz'[comp]} and its parts (J_p J_1 J_2 J_n) at selected iterations")
print(f"{'it':>5s} {'J':>10s} {'J_p':>10s} {'J_1':>10s} {'J_2':>10s} {'J_n':>10s} {'rho':>8s} {'u':>8s}")
for k in range(0, len(ks), 8):
    r = res[main]
    print(f"{it[ks[k]]:5d} {r['mom'][k]:10.5f} {r['p'][k]:10.5f} {r['1'][k]:10.5f} {r['2'][k]:10.5f} {r['n'][k]:10.5f} {rho_all[ks[k], ix, iy, iz0]:8.4f} {u_all[ks[k], comp, ix, iy, iz0]:8.4f}")

fig, axs = plt.subplots(len(cells), 2, figsize=(14, 6.5 * len(cells)))
axs = np.atleast_2d(axs)
for row, cell in enumerate(cells):
    r = res[cell]
    for col in (0, 1):
        ax = axs[row, col]
        for n, lab in (("mom", "J (mom_f)"), ("p", "J_p"), ("1", "J_1 (rho c.u)"), ("2", "J_2 (convective)"), ("n", "J_n (f*-feq)")):
            y = r[n] - (r[n][0] if col == 1 else 0.0)
            ax.plot(it[ks], y, label=lab, lw=2.5 if n == "mom" else 1.2, color="k" if n == "mom" else None)
        ax.set_title(f"cell x{148 + cell[0]} y{5 + cell[1]} z{22 + cell[2]}, J_{'xyz'[comp]}" + (" minus its value at the first row" if col else ""), fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_box_aspect(1)
fig.tight_layout()
fig.savefig("xs1_fneq_budget.png", dpi=110)
print("\nwrote xs1_fneq_budget.png")
