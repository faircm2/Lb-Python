"""
Collage (2 columns x 3 rows) of the existing u_mag_layers_iter_<N>.png figures of one run.
Each tile is the complete figure of that iteration (six layers + its own colourbar), unchanged.

usage:  python make_umag_collage.py <run folder> [iter1 ... iter6]       (default 0 500 1285 4052 8400 12000)
output: <run folder>/collage_u_mag_layers.png
"""
import os
import sys

from PIL import Image

run = sys.argv[1]
its = [int(a) for a in sys.argv[2:]] or [0, 500, 1285, 4052, 8400, 12000]
tiles = [Image.open(os.path.join(run, "u_mag_layers_iter_%05d.png" % it)).convert("RGB") for it in its]
w, h = tiles[0].size
cols = 2
rows = (len(tiles) + cols - 1) // cols
canvas = Image.new("RGB", (cols * w, rows * h), "white")
for i, t in enumerate(tiles):
    canvas.paste(t, ((i % cols) * w, (i // cols) * h))
out = os.path.join(run, "collage_u_mag_layers.png")
canvas.save(out)
print("wrote", out, canvas.size)
