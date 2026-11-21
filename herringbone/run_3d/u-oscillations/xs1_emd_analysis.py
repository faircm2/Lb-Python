"""Empirical mode decomposition (package emd 0.8.1) of the max|u| signal of the W=14 amplitude run (job 8885554), compared with the batch amplitudes of the monitor.
Signal after the ignored start-up (iteration >= 40000), one sample every 10 iterations.  Sifting splits the signal into intrinsic mode functions (IMFs), fastest first, plus a residual
(the slowest part).  For every IMF the instantaneous amplitude (Hilbert transform) and frequency are computed.
Comparison per batch of 20000 iterations: amplitude of the monitor (Hilbert envelope of the whole detrended batch) against the amplitude of the IMF that carries the 341.5-iteration oscillation.
Run on the PC in herringbone\\run_3d\\u-oscillations (needs  pip install emd pyyaml):  python xs1_emd_analysis.py [amplitude_history.npz]    ->  xs1_emd_analysis.png
"""
import os
import sys

import emd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "amplitude_history.npz")
d = np.load(path)
S, B = d["samples"], d["batches"]
STRIDE, BATCH, START = 10, 20000, 40000
FS = 1.0 / STRIDE                                                             # samples per iteration
m = S[:, 0] >= START
it, x = S[m, 0], S[m, 1]

imf = emd.sift.sift(x, max_imfs=8)                                            # (samples, n_imf + residual); the last column is the residual
n_imf = imf.shape[1]
IP, IF, IA = emd.spectra.frequency_transform(imf[:, :-1], FS, "hilbert")
print(f"{n_imf - 1} IMFs + residual from {len(x)} samples")
print(" IMF   mean period (it)   rms amplitude")
for k in range(n_imf - 1):
    per = 1 / np.average(IF[:, k], weights=IA[:, k])
    print(f"  {k + 1}    {per:10.1f}        {np.sqrt(np.mean(imf[:, k] ** 2)) * np.sqrt(2):.5f}")
print(f"  res   (slowest part)      range {imf[:, -1].min():.5f} .. {imf[:, -1].max():.5f}")
pers = np.array([1 / np.average(IF[:, k], weights=IA[:, k]) for k in range(n_imf - 1)])
kmain = int(np.argmin(np.abs(pers - 341.5)))                                  # the IMF of the 341.5-iteration oscillation
print(f"the oscillation of period 341.5 is IMF {kmain + 1} (mean period {pers[kmain]:.1f})")

# batch amplitudes: monitor (all components together) against the instantaneous amplitude of the IMF of the main oscillation
rows = []
for k, (a, b, lo, hi, amp, mean) in enumerate(B):
    if a < START:
        continue
    mb = (it >= a) & (it <= b)
    ia = IA[mb, kmain]
    cut = max(1, mb.sum() // 10)
    rows.append((k + 1, amp, ia[cut:-cut].max(), ia[cut:-cut].mean()))
rows = np.array(rows)
print(" batch  monitor amplitude  IMF-%d amplitude (max inside batch, mean)" % (kmain + 1))
for r in rows:
    print(f"  {int(r[0]):2d}      {r[1]:.5f}            {r[2]:.5f}   {r[3]:.5f}")
for name, col in (("monitor", 1), ("IMF max", 2), ("IMF mean", 3)):
    last = rows[-4:, col]
    print(f"spread of the last 4 batches, {name}: {(last.max() - last.min()) / last.mean() * 100:.1f} %")

fig, axs = plt.subplots(n_imf + 1, 1, figsize=(13, 2.0 * (n_imf + 1)), sharex=True, layout="constrained")
axs[0].plot(it, x, color="#2171b5", lw=0.4)
axs[0].set_ylabel("signal")
axs[0].set_title("Empirical mode decomposition of max|u| (W=14, iterations 40000-180000): IMFs from fast to slow, the last panel is the residual")
for k in range(n_imf - 1):
    ax = axs[k + 1]
    ax.plot(it, imf[:, k], color="#2171b5" if k != kmain else "#d62728", lw=0.5)
    ax.plot(it, IA[:, k], color="k", lw=1.0)
    ax.set_ylabel(f"IMF {k + 1}\n{pers[k]:.0f} it")
axs[-1].plot(it, imf[:, -1], color="#2ca02c", lw=1.5)
axs[-1].set_ylabel("residual")
axs[-1].set_xlabel("iteration")
for ax in axs:
    ax.grid(alpha=0.3)
fig.savefig(os.path.join(HERE, "xs1_emd_analysis.png"), dpi=90)

fig2, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8.5), layout="constrained")
a1.plot(it, imf[:, kmain], color="#2171b5", lw=0.5, label=f"IMF {kmain + 1} (period {pers[kmain]:.0f} iterations)")
a1.plot(it, IA[:, kmain], color="#d62728", lw=1.6, label="its instantaneous amplitude")
for b0 in range(START, int(it.max()), BATCH):
    a1.axvline(b0, color="0.7", lw=0.6)
a1.set(xlabel="iteration", ylabel="max|u| component", title=f"Main oscillation (IMF {kmain + 1}) and its amplitude; grey lines = batch limits")
a1.legend(loc="upper right")
a1.grid(alpha=0.3)
w = 0.38
xb = rows[:, 0]
b1 = a2.bar(xb - w / 2, rows[:, 1], w, color="#2171b5", label="Hilbert envelope of the whole detrended signal: maximum inside the batch")
b2 = a2.bar(xb + w / 2, rows[:, 2], w, color="#d62728", label=f"Hilbert envelope of EMD component {kmain + 1} (main oscillation) alone: maximum inside the batch")
for bars in (b1, b2):
    for r in bars:
        a2.text(r.get_x() + r.get_width() / 2, r.get_height(), f"{r.get_height():.5f}", ha="center", va="bottom", fontsize=8)
a2.set(xlabel="batch number", ylabel="amplitude (lattice units)", ylim=(0, rows[:, 1].max() * 1.2), title="Amplitude per batch, same statistic (maximum of the Hilbert envelope), two inputs: whole signal and main oscillation only")
a2.set_xticks(xb)
a2.legend(loc="upper right")
a2.grid(alpha=0.3, axis="y")
fig2.savefig(os.path.join(HERE, "xs1_emd_amplitude.png"), dpi=100)
print("wrote xs1_emd_analysis.png, xs1_emd_amplitude.png")
