"""
Approximate reconstruction of the phi interface profile from a rendered
final_snapshot.png (RdBu imshow, origin='lower'), for cases where the raw
phi field is no longer available. Used only for visual overlays -- not a
precise measurement (colormap decoding has inherent anti-aliasing error).
"""
import numpy as np
from PIL import Image
import matplotlib


def find_heatmap_bbox(rgb):
    """Locate the main imshow axes' pixel bounding box, distinguishing it
    from the narrower colorbar strip by width."""
    h, w, _ = rgb.shape
    is_white = np.all(rgb > 245, axis=2)
    is_grayish = (np.abs(rgb[:, :, 0].astype(int) - rgb[:, :, 1].astype(int)) < 10) & \
                 (np.abs(rgb[:, :, 1].astype(int) - rgb[:, :, 2].astype(int)) < 10)
    colored = ~is_white & ~is_grayish  # colormap pixels have R!=B (red or blue tinted)

    row_spans = []
    for y in range(h):
        cols = np.where(colored[y])[0]
        if len(cols) < 20:
            continue
        # widest contiguous run in this row
        splits = np.where(np.diff(cols) > 1)[0]
        runs = np.split(cols, splits + 1)
        best = max(runs, key=len)
        if len(best) > 50:
            row_spans.append((y, best[0], best[-1]))

    widths = [e - s for _, s, e in row_spans]
    max_w = max(widths)
    main_rows = [(y, s, e) for y, s, e in row_spans if (e - s) > 0.8 * max_w]

    y0 = min(r[0] for r in main_rows)
    y1 = max(r[0] for r in main_rows)
    x0 = int(np.median([r[1] for r in main_rows]))
    x1 = int(np.median([r[2] for r in main_rows]))
    return x0, x1, y0, y1


def decode_phi_profile(png_path, phi_star_g=0.0, phi_star_l=1.0, n_x=300, n_y=300):
    img = Image.open(png_path).convert('RGB')
    rgb = np.array(img)
    x0, x1, y0, y1 = find_heatmap_bbox(rgb)

    cmap = matplotlib.colormaps['RdBu']
    lut_n = 512
    lut_vals = np.linspace(0, 1, lut_n)
    lut_colors = (np.array([cmap(v)[:3] for v in lut_vals]) * 255).astype(np.float32)

    # sample one pixel column per x-index, one pixel row per y-index
    xs_px = np.linspace(x0, x1, n_x).astype(int)
    ys_px = np.linspace(y1, y0, n_y).astype(int)  # y1 (bottom, high row#) -> y_idx=0

    interface_y = np.full(n_x, np.nan)
    for xi, xp in enumerate(xs_px):
        col_pixels = rgb[ys_px, xp, :].astype(np.float32)  # (n_y, 3), y_idx ascending
        dists = np.sum((col_pixels[:, None, :] - lut_colors[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(dists, axis=1)
        norm_vals = lut_vals[nearest]
        phi_col = phi_star_g + norm_vals * (phi_star_l - phi_star_g)

        sign = np.sign(phi_col - 0.5)
        changes = np.where(np.diff(sign) != 0)[0]
        if len(changes) == 0:
            continue
        idx = int(changes[0])
        y0v, y1v = idx, idx + 1
        p0, p1 = phi_col[idx], phi_col[idx + 1]
        interface_y[xi] = y0v if p1 == p0 else y0v + (0.5 - p0) * (y1v - y0v) / (p1 - p0)

    return np.arange(n_x), interface_y


if __name__ == '__main__':
    import sys
    x, y = decode_phi_profile(sys.argv[1])
    print('x range:', x.min(), x.max())
    print('interface y: center=', y[len(y)//2], ' left=', y[1], ' right=', y[-2])
    print('decoded rise (avg of left/right - center):',
          ((y[1] - y[len(y)//2]) + (y[-2] - y[len(y)//2])) / 2.0)
