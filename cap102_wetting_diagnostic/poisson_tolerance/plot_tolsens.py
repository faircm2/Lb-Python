import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TAGS = ['1e-3', '1e-4', '1e-5', '3e-6', '1e-6', '3e-7', '1e-7']
D = {}
for t in TAGS:
    try:
        a = np.loadtxt('out/tolsens_%s.csv' % t, delimiter=',')
    except Exception:
        continue
    if a.ndim == 2 and len(a) > 10:
        D[t] = a            # iteration, sweeps, max eps, wall s, div u rms
tags = [t for t in TAGS if t in D]
last = {t: int(D[t][-1, 0]) for t in tags}
print('last iteration reached:', last)
W1 = int(sys.argv[1]) if len(sys.argv) > 1 else min(last.values())
W0 = int(sys.argv[2]) if len(sys.argv) > 2 else 500
print('window', W0, W1)

tol, sw, ms, dv = [], [], [], []
for t in tags:
    a = D[t]
    m = (a[:, 0] >= W0) & (a[:, 0] <= W1)
    tol.append(float(t))
    sw.append(a[m, 1].mean())
    ms.append(1000 * (a[m, 3][-1] - a[m, 3][0]) / (a[m, 0][-1] - a[m, 0][0]))
    dv.append(np.median(a[m, 4]))
    print('tol %-5s  sweeps/iter %8.1f   ms/iter %8.1f   div u rms (median) %.2e   reached %d' % (t, sw[-1], ms[-1], dv[-1], last[t]))

fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
ax[0].loglog(tol, sw, 'o-')
ax[0].set_ylabel('Poisson sweeps per iteration (mean)')
ax[1].loglog(tol, ms, 'o-')
ax[1].set_ylabel('wall time per iteration (ms)')
ax[2].loglog(tol, dv, 'o-')
ax[2].set_ylabel('div u rms in the liquid bulk (median)')
for a_ in ax[:3]:
    a_.set_xlabel('Poisson tolerance epsilon_cutoff')
    a_.invert_xaxis()
    a_.grid(alpha=0.3, which='both')
ax[0].set_title('window iterations %d-%d' % (W0, W1))
for t in tags:
    a = D[t]
    k = 50
    s = np.convolve(a[:, 1], np.ones(k) / k, mode='valid')
    ax[3].semilogy(a[k - 1:, 0], s, label=t)
ax[3].set_xlabel('iteration')
ax[3].set_ylabel('sweeps per iteration (50-it mean)')
ax[3].grid(alpha=0.3, which='both')
ax[3].legend(title='tolerance', fontsize=8)
fig.tight_layout()
fig.savefig('out/tolsens.png', dpi=140)
print('saved out/tolsens.png')
