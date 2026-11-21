"""Exact budget of the zeroth moment of f (sum_i f_i) at x=157 (onset cell) and x=158 (y=25, z=13), pass m = 10580..10597,
from the raw f_i traces (f_pre = before collision, f_star = after collision, before streaming) of the fill9.4 periodicx MULT=1 run.
    coll   = sum_i (f_star_i - f_pre_i)                      at the cell            (collision + forcing, local)
    stream = sum_i f_star_i(x - c_i) - sum_i f_star_i(x)     net inflow by streaming
    total  = coll + stream  =  sum f after streaming - sum f before collision
Estimates from the trace rows (u, rho, mom of pass m-1, central differences inside the box; Cs2 = 1/3):
    estC = Cs2 * u.grad(rho)                  (what the collision should add)
    estS = -Cs2 * div(rho*u + Fh),  Fh = rho*u - mom   (what streaming should remove)
First line: consistency check of f_pre against the stored f-moment (must be ~1e-15).
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_ftrace.py
"""
import glob
import numpy as np

folder = glob.glob("FreesurfaceImages/*fill9.4_periodicx_hbs*")[0]
F = np.load(glob.glob(folder + "/ftrace_x*.npz")[0])
T = np.load(glob.glob(folder + "/trace_x*.npz")[0])
it_f, f_pre, f_star, oe = F["it_f"], F["f_pre"], F["f_star"], F["origin_ext"]
it_t, rho, u, mom, o = T["it"], T["rho"], T["u"], T["mom"], T["origin"]
Cs2 = 1.0 / 3.0
c = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
              [1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1]])
print("origin_ext", oe, " origin", o, " f rows", len(it_f), it_f[0], "..", it_f[-1])
mstar = rho[:, None] * u + (rho[:, None] * u - mom)
j, k = 25 - o[1], 13 - o[2]
jf, kf = 25 - oe[1], 13 - oe[2]


def div(vec, n, i):
    return ((vec[n, 0, i + 1, j, k] - vec[n, 0, i - 1, j, k]) + (vec[n, 1, i, j + 1, k] - vec[n, 1, i, j - 1, k])
            + (vec[n, 2, i, j, k + 1] - vec[n, 2, i, j, k - 1])) / 2.0


def u_grad_rho(n, i):
    gr = np.array([(rho[n, i + 1, j, k] - rho[n, i - 1, j, k]) / 2.0, (rho[n, i, j + 1, k] - rho[n, i, j - 1, k]) / 2.0,
                   (rho[n, i, j, k + 1] - rho[n, i, j, k - 1]) / 2.0])
    return float(u[n, :, i, j, k] @ gr)


chk = 0.0
for m_idx, m in enumerate(it_f):
    nt = np.where(it_t == m - 1)[0]
    if len(nt) == 0:
        continue
    for x in (157, 158):
        mf = np.einsum('ia,i->a', c, f_pre[m_idx, :, x - oe[0], jf, kf]) / Cs2
        chk = max(chk, float(np.abs(mf - mom[nt[0], :, x - o[0], j, k]).max()))
print("consistency |sum c f_pre/Cs2 - mom(row m-1)| max =", chk)

hdr = f"{'m':>6s} |"
for x in (157, 158):
    hdr += f" x={x}: {'coll':>9s} {'stream':>9s} {'total':>9s} {'estC':>9s} {'estS':>9s} |"
print(hdr)
for m_idx, m in enumerate(it_f):
    if m < 10580 or m > 10597:
        continue
    nt = int(np.where(it_t == m - 1)[0][0])
    row = f"{m:6d} |"
    for x in (157, 158):
        ie = x - oe[0]
        coll = float((f_star[m_idx, :, ie, jf, kf] - f_pre[m_idx, :, ie, jf, kf]).sum())
        inflow = sum(f_star[m_idx, i, ie - c[i, 0], jf - c[i, 1], kf - c[i, 2]] for i in range(15))
        stream = float(inflow - f_star[m_idx, :, ie, jf, kf].sum())
        it_ = x - o[0]
        estC = Cs2 * u_grad_rho(nt, it_)
        estS = -Cs2 * div(mstar, nt, it_)
        row += f"        {coll:9.5f} {stream:9.5f} {coll + stream:9.5f} {estC:9.5f} {estS:9.5f} |"
    print(row)
