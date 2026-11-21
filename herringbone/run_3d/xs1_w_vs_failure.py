"""Interface width W against the iteration at which the herringbone simulation fails.  Identical parameters in all runs (xs1.0 herringbone, sigma 0.1,
theta 60, tau_f 0.6, nu 1e-6, fill 9.4, 736 x 64 x 64 cells): W=3 job 8820586, W=4 job 8815366, W=7 job 8866565 (156013 iterations planned) and
the series W=5, 6, 8, 9, 10, 12, 14, 17 (array job 8873105, logs in logs/W_series, array task k -> W as in lbm_herringbone_W_series.sh).
From the 'iter N ... Ma=' lines of the logs (found anywhere in a line, the output of several ranks can interleave; the spacing of the lines is coarse):
  lower end = last log line with Ma < 0.5 ;   upper end = first log line with Ma > 1 or NaN
Dump-resolved first non-finite iteration where a term dump exists: W=3 2244, W=4 10254.
A run without any failure so far (W=14, still running when the logs were fetched) is drawn as an arrow at its last finite log line.
Run on the PC in herringbone\\run_3d:  python xs1_w_vs_failure.py    ->  xs1_w_vs_failure.png, xs1_w_vs_failure.csv
"""
import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(r"iter\s+(\d+)\s+t=\s*[\d.]+\s+ms\s+max\|u\|=(\S+)\s+Ma=(\S+)")
SERIES = os.path.join("logs", "W_series", "lbm_Wseries_8873105_{k}.out")
SERIES2 = os.path.join("logs", "W_series", "lbm_Wseries2_8879357_{k}.out")
RUNS = [(3, os.path.join("logs", "lbm_8820586.out"), 2244, "8820586"), (4, os.path.join("logs", "lbm_8815366.out"), 10254, "8815366"),
        (5, SERIES.format(k=0), None, "8873105_0"), (6, SERIES.format(k=1), None, "8873105_1"),
        (7, os.path.join("logs", "lbm_8866565.out"), None, "8866565"),
        (8, SERIES.format(k=2), None, "8873105_2"), (9, SERIES.format(k=3), None, "8873105_3"), (10, SERIES.format(k=4), None, "8873105_4"),
        (12, SERIES.format(k=5), None, "8873105_5"), (17, SERIES.format(k=7), None, "8873105_7"),
        (13, SERIES2.format(k=0), None, "8879357_0"), (14, os.path.join("logs", "W_series", "lbm_W14long_8878871.out"), None, "8878871, 624052 iterations"),
        (15, SERIES2.format(k=1), None, "8879357_1"), (16, SERIES2.format(k=2), None, "8879357_2")]
res = []
for W, name, dump_nan, job in RUNS:
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        print(f"W={W}: no log yet ({name})")
        continue
    e = [(int(a), float(c)) for a, b, c in PAT.findall(open(p, encoding="utf-8", errors="replace").read())]
    it, ma = np.array([i for i, _ in e]), np.array([m for _, m in e])
    bad = ~np.isfinite(ma) | (ma > 1.0)
    if bad.any():
        up = int(it[bad][0])
        lo = int(it[(it < up) & np.isfinite(ma) & (ma < 0.5)][-1])
        print(f"W={W:2d} (job {job}): failure between log lines {lo} and {up}" + (f"; dump-resolved first non-finite iteration {dump_nan}" if dump_nan else ""))
        res.append((W, lo, up, dump_nan, False))
    else:
        last = int(it[-1])
        print(f"W={W:2d} (job {job}): no failure up to iteration {last} (Ma {ma[-1]:.3f}) in the log")
        res.append((W, last, None, None, True))

# W=11 (job 8886923_0, amplitude monitor, sampled every 10 iterations): first non-finite max|u| at iteration 48740 (the log lines of that run are too coarse to show it)
res.append((11, 48740, 48740, 48740, False))
res.sort()
with open(os.path.join(HERE, "xs1_w_vs_failure.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["W", "last_finite_log_iteration", "first_nan_or_Ma_gt_1_log_iteration", "first_nonfinite_iteration_from_dumps", "no_failure_so_far"])
    w.writerows([(r[0], r[1], r[2] or "", r[3] or "", r[4]) for r in res])

# final bubble growth on the floor against W (sigma 0.5, Cap07 study), read by eye from the plot supplied by the user (approximate values)
BUBBLE = {4: 3.1e-2, 5: 1.4e-2, 6: 8.7e-3, 7: 4.4e-3, 8: 7.1e-3, 9: 5.8e-3, 10: 4.3e-3, 11: 4.9e-3, 12: 4.5e-3, 13: 3.6e-3, 14: 2.8e-3, 15: 2.5e-3, 16: 3.0e-3, 17: 3.9e-3}
fig, (ax, axb) = plt.subplots(2, 1, figsize=(10, 10.5), layout="constrained", sharex=True, height_ratios=[2.2, 1])
first = True
for W, lo, up, dn, survived in res:
    if survived:
        ax.annotate("", xy=(W, lo * 1.9), xytext=(W, lo), arrowprops=dict(arrowstyle="-|>", color="#2171b5", lw=3, mutation_scale=22))
        ax.plot(W, lo, "o", color="#2171b5", ms=7, label="no failure up to this iteration (run finished)" if first else None)
        ax.annotate(f">{lo}", (W, lo), textcoords="offset points", xytext=(0, -32 if W == 16 else -17), fontsize=9.5, ha="center")
        first = False
        continue
    ax.plot([W, W], [lo, up], color="#2171b5", lw=7, solid_capstyle="butt", alpha=0.45, label="failure interval from the log" if W == 3 else None)
    if dn:
        ax.plot(W, dn, "o", color="k", ms=8, label="first non-finite iteration (W=3, 4 from the dumps, W=11 from the amplitude monitor)" if W == 3 else None)
        ax.annotate(f"{dn}", (W, up), textcoords="offset points", xytext=(10, -20 if W == 11 else 4), fontsize=11)
    else:
        above = W in (5, 7, 9, 10, 12)                                        # label above or below the interval (keeps neighbouring labels apart)
        ax.annotate(f"{lo}–{up}", (W, up if above else lo), textcoords="offset points", xytext=(8, 8) if above else (8, -18), fontsize=10)
AMP_STOPS = [(13, 180000, "4.8 %"), (14, 180000, "2.1 %")]                    # jobs 8886923_1 and 8885554: the amplitude monitor stopped the run, batch amplitudes constant within the given spread
for k, (W, it, spread) in enumerate(AMP_STOPS):
    ax.plot(W, it, "*", color="#d94801", ms=15, zorder=5, label="amplitude monitor stopped the run at 180000: oscillation steady (spread 4.8 % at W=13, 2.1 % at W=14)" if k == 0 else None)
ax.set_yscale("log")
ws = [r[0] for r in res]
ax.set(xlabel="interface width W (cells)", ylabel="iteration at which the simulation fails", xticks=ws, xlim=(2.2, max(ws) + 1.6),
       title="Interface width against failure iteration, herringbone simulation\n(sigma 0.1, theta 60, tau_f 0.6, nu 1e-6, fill 9.4, same grid)")
ax.set_ylim(1000, 1.2e7)
ax.set_yticks([1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000])
ax.set_yticklabels(["1000", "2000", "5000", "10000", "20000", "50000", "100000", "200000", "500000", "1000000"])
ax.minorticks_off()
ax.grid(alpha=0.3, which="both")
ax.legend(loc="upper left", fontsize=10)
ax.set_xlabel("")
bw = sorted(BUBBLE)
axb.plot(bw, [BUBBLE[w] for w in bw], "o-", color="#c0392b", ms=6)
axb.set_yscale("log")
axb.set(xlabel="interface width W (cells)", ylabel="final bubble growth on the floor", ylim=(2e-3, 4e-2), xticks=sorted(set(ws) | set(bw)),
        title="Bubble growth against interface width (Cap07 study, sigma 0.5; values read by eye from the supplied plot)")
axb.set_yticks([3e-3, 1e-2, 3e-2]); axb.set_yticklabels(["3e-3", "1e-2", "3e-2"])
axb.minorticks_off()
axb.grid(alpha=0.3, which="both")
out = os.path.join(HERE, "xs1_w_vs_failure.png")
fig.savefig(out, dpi=110)
print("wrote", out)
