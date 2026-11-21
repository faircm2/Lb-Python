"""True q = |xf - xw| / |xf - xs| (Zhang eq 30) for every fluid-solid link of the herringbone mask, from the analytic groove geometry.
Mask as in the baseline run (XN=736, 64 x 64, TP=4.9, XS=1.0).  For each of the 14 directions d: fluid node xf (mask), solid node xs = xf + c_d (mask,
inside the box, i.e. not the outer walls), xw = crossing of the segment xf -> xs with the analytic solid region (bisection).  q = fraction of the link.
The solver uses q = 0.5 for all of them.  Per direction: number of links, mean/min/max q, share with |q-0.5| < 0.05; then a histogram of q for the
6 axis links and the 8 corner links.  Also the signed wall-position error along the link, (q-0.5)*|c| in cells.
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_qdist.py
"""
import os, sys
import numpy as np
os.environ["TP"] = "4.9"
sys.path.insert(0, "/work/classic/fr_jn194-lbm")
from herringbone.herringbone_idealised import Herringbone, herringbone_mask, is_solid

XN, YN, ZN = 736, 64, 64
hb = Herringbone(x_start=1.0)
dx = hb.width / YN
mask = herringbone_mask(XN, YN, ZN, hb)                       # (XN+2, YN+2, ZN+2), ghost layers included


def solid_at(i, j, k):
    """analytic solid test at (fractional) node coordinates i, j, k (cell units, same indexing as the mask)"""
    s = (i - 0.5) * dx
    t = (j - 0.5) * dx - YN * dx / 2
    h = (k - 0.5 - hb.z_floor) * dx
    return is_solid(s, t, h, hb.depth, hb.x_start, hb.n_trains, hb.train_period, hb.groove_angle_deg, hb.groove_width, hb.groove_pitch, hb.n_per_train)


rng = np.random.default_rng(1)
# sanity: the analytic function reproduces the mask at the node centres
ii, jj, kk = (rng.integers(0, n + 2, 200000) for n in (XN, YN, ZN))
print("mask vs analytic at node centres, mismatches:", int((solid_at(ii.astype(float), jj.astype(float), kk.astype(float)) != mask[ii, jj, kk]).sum()), "of 200000")

c = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
              [1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1]])
inner = np.zeros_like(mask)
inner[1:XN + 1, 1:YN + 1, 1:ZN + 1] = True
fluid = ~mask & inner
qs = {}
print(f"\n{'c_d':>12s} | {'links':>8s} | {'mean q':>7s} {'min q':>7s} {'max q':>7s} | {'|q-.5|<.05':>10s} | {'err mean':>8s} {'err min':>8s} {'err max':>8s} (cells along the link)")
for d, cd in enumerate(c):
    solid_nb = np.roll(mask & inner, shift=(-cd[0], -cd[1], -cd[2]), axis=(0, 1, 2))      # solid_nb[x] = (mask & inner)[x + c_d]
    ix, iy, iz = np.nonzero(fluid & solid_nb)
    n_all = len(ix)
    if n_all == 0:
        print(f"{str(tuple(int(a) for a in cd)):>12s} | {0:8d} | (no links)")
        qs[d] = np.zeros(0)
        continue
    if n_all > 300000:
        sel = rng.choice(n_all, 300000, replace=False)
        ix, iy, iz = ix[sel], iy[sel], iz[sel]
    lo = np.zeros(len(ix))
    hi = np.ones(len(ix))
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        sol = solid_at(ix + mid * cd[0], iy + mid * cd[1], iz + mid * cd[2])
        hi = np.where(sol, mid, hi)
        lo = np.where(sol, lo, mid)
    q = 0.5 * (lo + hi)
    qs[d] = q
    err = (q - 0.5) * np.linalg.norm(cd)
    print(f"{str(tuple(int(a) for a in cd)):>12s} | {n_all:8d} | {q.mean():7.3f} {q.min():7.3f} {q.max():7.3f} | {np.mean(np.abs(q - 0.5) < 0.05):10.3f} | {err.mean():8.3f} {err.min():8.3f} {err.max():8.3f}")

bins = np.linspace(0, 1, 11)
for name, ds in (("axis links (6)", range(0, 6)), ("corner links (8)", range(6, 14))):
    q = np.concatenate([qs[d] for d in ds])
    h, _ = np.histogram(q, bins=bins)
    print(f"\n{name}: q histogram, {len(q)} sampled links")
    print("  q in   " + " ".join(f"{bins[n]:.1f}-{bins[n + 1]:.1f}" for n in range(10)))
    print("  share  " + " ".join(f"{x / len(q):7.3f}" for x in h))
