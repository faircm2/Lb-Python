"""Impact of the interface width W on the whole run: Mach number max|u|/Cs against iteration, from the job logs (line 'iter N ... max|u|= Ma= ...').
W=3: job 8820586,  W=4: job 8815366,  W=7: job 8840757.  Same geometry and parameters (xs1.0 herringbone, sigma 0.1, theta 60, tau_f 0.6, nu 1e-6, fill 9.4).
Run on the PC in herringbone\\run_3d:  python xs1_w_impact_ma.py       ->  xs1_w_impact_ma.png
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [("W=3", "lbm_8820586.out"), ("W=4", "lbm_8815366.out"), ("W=7", "lbm_8840757.out")]
PAT = re.compile(r"^iter\s+(\d+)\s+t=\s*\S+\s+ms\s+max\|u\|=(\S+)\s+Ma=(\S+)")
fig, axs = plt.subplots(1, 2, figsize=(15, 6.5), layout="constrained", sharey=True)
for label, name in RUNS:
    it, ma = [], []
    for line in open(os.path.join(HERE, "logs", name), encoding="utf-8", errors="replace"):
        m = PAT.match(line)
        if m:
            it.append(int(m.group(1))); ma.append(float(m.group(3)))
    it, ma = np.array(it), np.array(ma)
    ok = np.isfinite(ma)
    last = it[ok][-1]
    fail = it[~ok][0] if (~ok).any() else None
    print(f"{label} ({name}): {len(it)} log lines, last finite Ma {ma[ok][-1]:.3f} at it {last}, first non-finite log line at {fail}, last log line it {it[-1]}")
    for ax in axs:
        ax.plot(it[ok], ma[ok], marker="o", ms=3, lw=1.5, label=f"{label}" + (f"  (NaN after it {last})" if fail else "  (finished, no NaN)"))
        if fail:
            ax.axvline(last, color=ax.lines[-1].get_color(), ls=":", lw=1)
axs[0].set(xlabel="iteration", ylabel="Ma = max|u| / Cs", title="whole run")
axs[1].set(xlabel="iteration", xlim=(0, 2400), title="zoom on the W=3 failure")
for ax in axs:
    ax.set_yscale("log")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.grid(alpha=0.3)
axs[0].legend()
fig.suptitle("Effect of the interface width W on stability (dashed: Ma = 1, |u| = Cs)")
out = os.path.join(HERE, "xs1_w_impact_ma.png")
fig.savefig(out, dpi=110)
print("wrote", out)
