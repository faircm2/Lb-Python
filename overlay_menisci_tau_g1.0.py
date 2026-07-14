"""
Superimpose the decoded meniscus profiles for tau_g=1.0, all tau_f values,
from their final_snapshot.png images (phase 1 -- raw phi data no longer
available, so this uses approximate colormap decoding; see
decode_snapshot_profile.py). Zoomed to the interface region only.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from decode_snapshot_profile import decode_phi_profile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RUNS = [
    ('run055_taug1.0_tauf1.0', 1.0),
    ('run056_taug1.0_tauf1.1', 1.1),
    ('run057_taug1.0_tauf1.2', 1.2),
    ('run058_taug1.0_tauf1.3', 1.3),
    ('run059_taug1.0_tauf1.5', 1.5),
    ('run060_taug1.0_tauf2.0', 2.0),
]

fig, ax = plt.subplots(figsize=(9, 6))
cmap = matplotlib.colormaps['viridis']

for i, (label, tau_f) in enumerate(RUNS):
    png = os.path.join(SCRIPT_DIR, 'param_study_runs', label, 'final_snapshot.png')
    x, y = decode_phi_profile(png)
    color = cmap(i / (len(RUNS) - 1))
    ax.plot(x, y, label=f'tau_f={tau_f}', color=color, linewidth=1.8)

ax.set_xlabel('x-index')
ax.set_ylabel('interface y-index (phi=0.5 crossing)')
ax.set_title('Superimposed meniscus profiles, tau_g=1.0 (decoded from snapshots)')
ax.set_ylim(140, 160)
ax.legend(title='tau_f')
ax.grid(alpha=0.3)

out_path = os.path.join(SCRIPT_DIR, 'menisci_overlay_taug1.0.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print('Saved:', out_path)
