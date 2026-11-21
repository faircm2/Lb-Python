"""Streamline density of Saliba Fig. 4a (1 ml/min), measured from the image of the figure.
Calibration from the white frame: the two white side walls are 1.2 mm apart, the white floor line is z = 0, the walls are 0.4 mm high.
Coloured line pixels = pixels with saturation > S_MIN and value > V_MIN (white frame, text and the dark blue background are excluded).
Metrics (the same are computed for our streamlines in xs1_fig4.py):
  crossings per mm : number of separate coloured runs met along a horizontal line at height z (per mm of channel width), z = -0.1, 0.0, +0.1 mm
  length density   : length of the skeleton of the coloured pixels per mm^2 of the window y 0..1.2 mm, z -0.2..+0.4 mm
Usage:  python xs1_fig4_ref_density.py <path to the image of his Fig. 4a>
"""
import sys

import matplotlib.colors as mc
import matplotlib.image as mi
import numpy as np
from skimage.morphology import skeletonize

path = sys.argv[1]
img = mi.imread(path)[:, :, :3].astype(float)
if img.max() > 1.5:
    img /= 255.0
H, W, _ = img.shape
hsv = mc.rgb_to_hsv(img)
white = (img.min(axis=2) > 0.88)
print(f"{path}: {W} x {H} px")

# white frame of his Fig. 4a (found from the printout of the first run): side walls at columns 35 and 460, floor line at row 193, wall tops at row 62
X_L, X_R, ROW_FLOOR, ROW_TOP, ROW_BOTTOM = 35.0, 460.5, 193, 62, 241                 # ROW_BOTTOM: last row above the white margin of the image
PX_MM = (X_R - X_L) / 1.2                                                           # px per mm (1.2 mm between the side walls)
print(f"scale {PX_MM:.1f} px/mm; wall height {(ROW_FLOOR - ROW_TOP) / PX_MM:.2f} mm (the figure says 0.4 mm)")
colored = (hsv[:, :, 1] > 0.45) & (hsv[:, :, 2] > 0.45) & ~white
win = np.zeros_like(colored)
win[ROW_TOP:ROW_BOTTOM + 1, int(X_L) + 2:int(X_R) - 1] = True                         # inside the walls
m = colored & win
area = m.shape[1] and (int(X_R) - 1 - int(X_L) - 2) * (ROW_BOTTOM - ROW_TOP + 1) / PX_MM ** 2
sk = skeletonize(m)
# skeleton length: orthogonal neighbour pairs count 1, diagonal pairs sqrt(2)
orth = (sk[:, 1:] & sk[:, :-1]).sum() + (sk[1:, :] & sk[:-1, :]).sum()
diag = (sk[1:, 1:] & sk[:-1, :-1]).sum() + (sk[1:, :-1] & sk[:-1, 1:]).sum()
length_px = orth + np.sqrt(2) * diag
length_px_alt = sk.sum() * 1.1                                                      # rough alternative: 1.1 px per skeleton pixel
print(f"window area {area:.3f} mm^2, coloured pixels {int(m.sum())} ({m.sum() / win.sum() * 100:.1f} % of the window)")
print(f"skeleton length {length_px / PX_MM:.2f} mm (alt {length_px_alt / PX_MM:.2f} mm)  ->  length density {length_px / PX_MM / area:.1f} mm per mm^2 (alt {length_px_alt / PX_MM / area:.1f})")
for z in (-0.06, 0.04, 0.10):
    r = int(round(ROW_FLOOR - z * PX_MM))
    runs = [(m[r + d, :].astype(int)) for d in (-1, 0, 1)]
    line = np.maximum.reduce(runs)                                                  # 3-pixel thick probe line
    n = int(((line[1:] == 1) & (line[:-1] == 0)).sum())
    print(f"z = {z:+.1f} mm: {n} separate coloured runs along the probe line = {n / 1.2:.0f} crossings per mm of width")
