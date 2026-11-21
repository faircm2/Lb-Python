"""The raw oscillating signal max|u| of the W=14 amplitude run (job 8885554): every sample, one every 10 iterations, no filtering.
Run on the PC in herringbone\\run_3d\\u-oscillations:  python xs1_raw_signal.py [amplitude_history.npz]    ->  xs1_raw_signal.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "amplitude_history.npz")
S = np.load(path)["samples"]
it, v = S[:, 0], S[:, 1]
MS = 1828.266 / 156012                                                    # physical time per iteration in ms

fig, ax = plt.subplots(figsize=(16, 6), layout="constrained")
ax.plot(it, v, color="#2171b5", lw=0.5)
ax.set(xlim=(0, it.max()), xlabel="iteration", ylabel="max|u| (lattice units)",
       title=f"Raw signal: W=14, max |u(i)| over all cells, sampling rate f=0.1Hz, {len(v)} samples")
sec = ax.secondary_xaxis("top", functions=(lambda x: x * MS * 1e-3, lambda s: s / (MS * 1e-3)))
sec.set_xlabel("physical time (s)")
ax.grid(alpha=0.3)
out = os.path.join(HERE, "xs1_raw_signal.png")
fig.savefig(out, dpi=100)
print("wrote", out)
