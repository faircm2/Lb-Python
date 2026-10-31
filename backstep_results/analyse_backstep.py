"""Backstep benchmark: time-averaged flow in the x-z mid-plane from the plane_xz dumps of the Cap08 backstep run.
Usage:  python analyse_backstep.py <folder with plane_xz_iter_*.npz> [first_iteration] [x_bs] [x_fs] [h_bs] [Xn] [Zn]
Writes <folder>/backstep_overview.png and prints the separation / reattachment numbers.
Units are lattice units; x and z count cells (z = 1 is the first fluid layer above the low floor)."""
import sys, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = sys.argv[1]
it0 = int(sys.argv[2]) if len(sys.argv) > 2 else 4500
x_bs, x_fs, h_bs = (int(a) for a in sys.argv[3:6]) if len(sys.argv) > 5 else (32, 96, 6)
Xn, Zn = (int(a) for a in sys.argv[6:8]) if len(sys.argv) > 7 else (128, 64)

files = [f for f in sorted(glob.glob(os.path.join(folder, "plane_xz_iter_*.npz"))) if int(f[-10:-4]) >= it0]
U = np.mean([np.load(f)["u"] for f in files], axis=0)       # (3, Xn, Zn), time average over the dumps
P = np.mean([np.load(f)["phi"] for f in files], axis=0)     # (Xn, Zn)
ux = U[0]
x = np.arange(1, Xn + 1); z = np.arange(1, Zn + 1)

solid = ((x < x_bs) | (x >= x_fs))[:, None] & (z <= h_bs)[None, :]
ux_f = np.where(solid, np.nan, ux)

def surface(phi):
    h = np.full(Xn, np.nan)
    for i in range(Xn):
        k = np.where(phi[i] > 0.5)[0]
        if len(k):
            k = k.max(); p0, p1 = phi[i, k], phi[i, min(k + 1, Zn - 1)]
            h[i] = k + 1 + (p0 - 0.5) / (p0 - p1 if p0 != p1 else 1.0)
    return h
h = surface(P)

wake = (x >= x_bs) & (x < x_fs)
print("%d dumps, iterations %s to %s" % (len(files), files[0][-10:-4], files[-1][-10:-4]))
for k in (0, 1, 2):
    s = np.sign(ux[wake, k]); ch = x[wake][1:][np.diff(s) != 0]
    print("z=%d above the wake floor: u_x changes sign at x = %s; smallest u_x = %.2e at x = %d" % (k + 1, list(ch), ux[wake, k].min(), x[wake][np.argmin(ux[wake, k])]))
uplate = float(np.nanmean(ux_f[x < x_bs - 2][:, h_bs:int(np.nanmin(h[x < x_bs - 2]))]))
print("mean speed in the film on the plateau: %.4f lattice units; film depth there %.1f cells; step height %d cells" % (uplate, np.nanmean(h[x < x_bs - 2]) - h_bs, h_bs))

fig, (a, b) = plt.subplots(2, 1, figsize=(13, 7.5), gridspec_kw=dict(height_ratios=[1.3, 1]))
zmax = int(np.nanmax(h)) + 4
m = a.pcolormesh(np.r_[x - 0.5, x[-1] + 0.5], np.r_[z - 0.5, z[-1] + 0.5][:zmax + 1], ux_f[:, :zmax].T * 1.0, cmap="viridis", shading="flat")
a.fill_between([0.5, x_bs - 0.5], 0.5, h_bs + 0.5, color="0.55"); a.fill_between([x_fs - 0.5, Xn + 0.5], 0.5, h_bs + 0.5, color="0.55")
a.plot(x, h, "w", lw=2); a.plot(x, h, "k", lw=0.8)
a.set_xlim(0.5, Xn + 0.5); a.set_ylim(0.5, zmax + 0.5); a.set_aspect("equal")
a.set_xlabel("x along the channel (cells)"); a.set_ylabel("z above the low floor (cells)")
a.set_title("Time-averaged streamwise velocity u_x in the mid-plane (grey = solid step, white line = free surface); flow goes left to right")
a.text(x_bs / 2, h_bs + 1.5, "plateau", ha="center", color="w", fontsize=9); a.text((x_bs + x_fs) / 2, 2.5, "wake (low floor)", ha="center", color="w", fontsize=9)
a.annotate("backstep", xy=(x_bs, h_bs), xytext=(x_bs - 8, h_bs + 5), color="w", fontsize=9, arrowprops=dict(arrowstyle="->", color="w"))
a.annotate("forward step", xy=(x_fs, h_bs), xytext=(x_fs + 3, h_bs + 5), color="w", fontsize=9, arrowprops=dict(arrowstyle="->", color="w"))
cb = fig.colorbar(m, ax=a, pad=0.01, fraction=0.025); cb.set_label("u_x (lattice units)")

for k, c in zip((0, 1, 2), ("#d62728", "#ff7f0e", "#1f77b4")):
    b.plot(x[wake], ux[wake, k] * 1e3, color=c, label="z = %d  (%d cell%s above the floor)" % (k + 1, k + 1, "" if k == 0 else "s"))
b.axhline(0, color="k", lw=0.8); b.axvline(x_bs, color="0.4", ls=":"); b.text(x_bs + 0.5, b.get_ylim()[1] * 0.93, "backstep", fontsize=9, color="0.3")
b.set_xlabel("x along the channel (cells)"); b.set_ylabel("u_x near the floor (lattice units x 10^-3)")
b.set_title("Flow right above the wake floor: below zero = flow runs backwards (recirculation)")
b.legend(fontsize=8, loc="upper right")
fig.tight_layout()
out = os.path.join(folder, "backstep_overview.png"); fig.savefig(out, dpi=120); print(out)
