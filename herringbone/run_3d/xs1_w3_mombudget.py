"""W=3 trace run: momentum budget of the first diverging gas cells, passes m = 2212..2222.
J = sum_i c_i f_i / Cs2   (what the solver stores as 'mom').  For a cell, per pass m:
    J_pre   = J of f_pre   (before collision)
    coll    = J(f_star) - J(f_pre)                         (collision + forcing, local)
    stream  = J after streaming - J(f_star)                (net inflow by streaming)
    top     = direction i with the largest |c_i (f_star_i(x - c_i) - f_star_i(x))|/Cs2, its vector c_i, size, and rho / |u| of the source cell x - c_i
rho, u of the cell itself are from trace row m-1 (= state before collision in pass m).
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_w3_mombudget.py
"""
import glob
import numpy as np

folder = glob.glob("FreesurfaceImages/*_W3_xs1.0_*fill9.4_periodicx_hbs*")[0]
T = np.load(glob.glob(folder + "/trace_x*.npz")[0])
F = np.load(glob.glob(folder + "/ftrace_x*.npz")[0])
it_t, rho, u, o = T["it"], T["rho"], T["u"], T["origin"]
it_f, f_pre, f_star, oe = F["it_f"], F["f_pre"], F["f_star"], F["origin_ext"]
Cs2 = 1.0 / 3.0
c = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
              [1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1]])
v3 = lambda a: "(" + ",".join(f"{x:8.4f}" for x in a) + ")"
for cell in ((415, 51, 27), (415, 53, 28), (415, 52, 28)):
    e = tuple(cell[a] - oe[a] for a in range(3))        # index in the f trace
    b = tuple(cell[a] - o[a] for a in range(3))         # index in the trace box
    print(f"\ncell {cell}")
    print(f"{'m':>5s} | {'rho':>7s} {'|u|':>7s} | {'J_pre':^29s} | {'coll':^29s} | {'stream':^29s} | top stream source: c, size, source rho, source |u|")
    for m in range(2212, 2223):
        nf, nt = np.where(it_f == m)[0][0], np.where(it_t == m - 1)[0][0]
        fp, fs = f_pre[nf], f_star[nf]
        J_pre = np.einsum('ia,i->a', c, fp[(slice(None),) + e]) / Cs2
        J_star = np.einsum('ia,i->a', c, fs[(slice(None),) + e]) / Cs2
        S = np.array([c[i] * (fs[(i,) + tuple(e[a] - c[i][a] for a in range(3))] - fs[(i,) + e]) / Cs2 for i in range(15)])   # (15, 3)
        stream = S.sum(0)
        top = int(np.abs(S).sum(1).argmax())
        sb = tuple(b[a] - c[top][a] for a in range(3))
        su = np.sqrt((u[nt][(slice(None),) + sb] ** 2).sum())
        print(f"{m:5d} | {rho[nt][b]:7.4f} {np.sqrt((u[nt][(slice(None),) + b] ** 2).sum()):7.4f} | {v3(J_pre)} | {v3(J_star - J_pre)} | {v3(stream)} | "
              f"{tuple(int(x) for x in c[top])} {np.abs(S[top]).sum():.4f} {rho[nt][sb]:.4f} {su:.4f}")
