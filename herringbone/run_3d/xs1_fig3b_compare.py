"""Saliba et al., Fig. 3 (b) next to the simulation's Fig. 3 (b) analogue (xs1_fig3b.py).
Usage (PC, herringbone\run_3d):  python xs1_fig3b_compare.py 019222 ["W=7, job 8866565"]    ->  xs1_fig3b_compare_iterNNNNNN.png
Needs saliba_fig3b_reference.png (crop of his Fig. 3 (b)) and xs1_fig3b_iterNNNNNN.png (python xs1_fig3b.py fields3d_W7/fields3d_iter_NNNNNN.npz)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mi
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
it = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "W=7, job 8866565"
ref = mi.imread(os.path.join(HERE, "saliba_fig3b_reference.png"))
sim = mi.imread(os.path.join(HERE, f"xs1_fig3b_iter{it}.png"))
fig, axs = plt.subplots(1, 2, figsize=(20, 5.6), gridspec_kw=dict(width_ratios=[ref.shape[1] / ref.shape[0], sim.shape[1] / sim.shape[0]]), layout="constrained")
axs[0].imshow(ref)
axs[0].set_title("Saliba et al., Fig. 3 (b), 1 ml/min", fontsize=15)
axs[1].imshow(sim)
axs[1].set_title(f"Simulation ({label}, iteration {int(it)})", fontsize=15)
for a in axs:
    a.axis("off")
out = os.path.join(HERE, f"xs1_fig3b_compare_iter{it}.png")
fig.savefig(out, dpi=100, bbox_inches="tight", pad_inches=0.25)
print("wrote", out)
