"""W=3 (per-link q) run: is the gas-speed growth a layer?  For dumps 2150, 2190, 2220 and every z = 16..44 (above the floor surface, all fluid space):
number of gas cells (phi<0.05), mean phi over ALL cells of the slice, rms and max |u| over the gas cells.
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_w3_zprofile.py
"""
import glob
import numpy as np

folder = glob.glob("FreesurfaceImages/*_W3_xs1.0_*fill9.4_periodicx_hbs*")[0]
data = {}
for it in (2150, 2190, 2220):
    d = np.load(f"{folder}/fields3d_iter_{it:06d}.npz")
    data[it] = (d["phi"].astype(float), np.sqrt((d["u"].astype(float) ** 2).sum(0)))
hdr = "".join(f" | {it}: {'n_gas':>7s} {'<phi>':>7s} {'rms':>6s} {'max':>6s}" for it in data)
print(f"{'z':>3s}{hdr}")
for z in range(16, 45):
    row = f"{z:3d}"
    for it, (phi, sp) in data.items():
        p, s = phi[:, :, z], sp[:, :, z]
        g = p < 0.05
        n = int(g.sum())
        rms = float(np.sqrt((s[g] ** 2).mean())) if n else 0.0
        mx = float(s[g].max()) if n else 0.0
        row += f" | {it}: {n:7d} {p.mean():7.4f} {rms:6.3f} {mx:6.3f}"
    print(row)
