"""W=3 trace run: which cell of the box diverges first?  For each pass m = 2214..2226:
fastest cell of trace row m-1 (= state before collision in pass m), with phi, rho, |u|, |mom|, and the min/max of its 15 f_i
before collision (f_pre) and after collision (f_star), from the raw f traces.  Non-finite cells count as fastest.
Run on NEMO2 in /work/classic/fr_jn194-lbm: python xs1_w3_trace2.py
"""
import glob
import warnings
import numpy as np

warnings.filterwarnings("ignore")
folder = glob.glob("FreesurfaceImages/*_W3_xs1.0_*fill9.4_periodicx_hbs*")[0]
T = np.load(glob.glob(folder + "/trace_x*.npz")[0])
F = np.load(glob.glob(folder + "/ftrace_x*.npz")[0])
it_t, phi, rho, u, mom, o = T["it"], T["phi"], T["rho"], T["u"], T["mom"], T["origin"]
it_f, f_pre, f_star = F["it_f"], F["f_pre"], F["f_star"]
print(f"{'m':>5s} | {'cell (x,y,z)':<14s} {'phi':>9s} {'rho':>9s} {'|u|':>10s} {'|mom|':>10s} | {'f_pre min':>10s} {'f_pre max':>10s} | {'f_star min':>10s} {'f_star max':>10s}")
for m in range(2214, 2227):
    nt = np.where(it_t == m - 1)[0]
    nf = np.where(it_f == m)[0]
    if len(nt) == 0 or len(nf) == 0:
        continue
    nt, nf = nt[0], nf[0]
    sp = np.sqrt((u[nt] ** 2).sum(0))
    sp = np.where(np.isfinite(sp), sp, np.inf)
    i, j, k = (int(a) for a in np.unravel_index(sp.argmax(), sp.shape))
    mg = np.sqrt((mom[nt][:, i, j, k] ** 2).sum())
    a, b = f_pre[nf][:, i + 1, j + 1, k + 1], f_star[nf][:, i + 1, j + 1, k + 1]
    print(f"{m:5d} | {str((o[0] + i, o[1] + j, o[2] + k)):<14s} {phi[nt, i, j, k]:9.5f} {rho[nt, i, j, k]:9.5f} {sp[i, j, k]:10.4f} {mg:10.5f} | "
          f"{a.min():10.5f} {a.max():10.5f} | {b.min():10.5f} {b.max():10.5f}")
