import os, time
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy import ndimage

start = time.perf_counter()

file_path = "C:\\Users\\megan\\flies\\comp vision\\_1_MMStack_Default.ome.tif"
output_file = f'output{os.path.basename(file_path)}'
output_graph = f'output{os.path.basename(file_path)[:-8]}.svg'

# ---------- settings ----------
THRESH = 128
OPEN_RADIUS = 2
CLOSE_RADIUS = 5
MIN_AREA = 10000
MAX_AREA = 25000
DOT_RADIUS = 4
FPS = None
# ------------------------------


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x**2 + y**2 <= r**2

def blob_moments(mask):
    """Center, second central moments, orientation and ellipse axes of a binary blob."""
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    dx, dy = xs - cx, ys - cy
    mu20 = np.mean(dx * dx)          # variance in x
    mu02 = np.mean(dy * dy)          # variance in y
    mu11 = np.mean(dx * dy)          # covariance
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)   # major axis angle, image coords (y down)
    common = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
    l1 = (mu20 + mu02) / 2 + common  # variance along major axis
    l2 = (mu20 + mu02) / 2 - common  # variance along minor axis
    a = 2 * np.sqrt(l1)              # semi-axes of the ellipse with the same second moments
    b = 2 * np.sqrt(max(l2, 0))
    return cx, cy, mu20, mu02, mu11, theta, a, b

def draw_ellipse(rgb, cx, cy, a, b, theta, color=(0, 255, 0)):
    t = np.linspace(0, 2 * np.pi, 400)
    ex = cx + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    ey = cy + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
    ring = np.zeros((H, W), bool)
    ix, iy = np.round(ex).astype(int), np.round(ey).astype(int)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ring[iy[ok], ix[ok]] = True
    ring = ndimage.binary_dilation(ring, iterations=1)     # 3 px thick line
    rgb[ring] = color
    # major axis line
    for s in np.linspace(-a, a, 200):
        px, py = int(round(cx + s * np.cos(theta))), int(round(cy + s * np.sin(theta)))
        if 0 <= px < W and 0 <= py < H:
            rgb[py, px] = color

open_se, close_se = disk(OPEN_RADIUS), disk(CLOSE_RADIUS)
names = ["x", "y", "mu20", "mu02", "mu11", "theta", "a", "b"]
last = None

with tifffile.TiffFile(file_path, is_mmstack=False, is_ome=False) as tif, \
     tifffile.TiffWriter(output_file, bigtiff=True) as writer:

    T = len(tif.pages)
    H, W = tif.pages[0].shape[:2]
    print(f"{T} frames, {H}x{W}")
    meas = {k: np.full(T, np.nan) for k in names}
    yy, xx = np.ogrid[:H, :W]

    for i in range(T):
        frame = tif.pages[i].asarray().astype(np.float32)
        binary = ndimage.gaussian_filter(frame, sigma=5) > THRESH

        m = ndimage.binary_opening(binary, structure=open_se)
        m = ndimage.binary_closing(m, structure=close_se)
        lab, n = ndimage.label(m)

        mask = np.zeros((H, W), dtype=np.uint8)
        if n:
            idx = np.arange(1, n + 1)
            areas = ndimage.sum(m, lab, idx)
            ok = (areas >= MIN_AREA) & (areas <= MAX_AREA)
            if ok.any():
                smallest = idx[ok][np.argmin(areas[ok])]
                blob = lab == smallest
                mask = blob.astype(np.uint8) * 255
                vals = blob_moments(blob)
                for k, v in zip(names, vals):
                    meas[k][i] = v
                cx, cy, _, _, _, theta, a, b = vals
                last = (cx, cy, a, b, theta)

        rgb = np.repeat(mask[..., None], 3, axis=2)
        if last is not None:
            cx, cy, a, b, theta = last
            draw_ellipse(rgb, cx, cy, a, b, theta)
            rgb[(yy - cy)**2 + (xx - cx)**2 <= DOT_RADIUS**2] = (255, 0, 0)

        writer.write(rgb, photometric="rgb", compression="zlib")   # one frame at a time

        if i % 100 == 0:
            print(f"frame {i}/{T}  ({time.perf_counter() - start:.0f}s)")

theta = meas["theta"]
valid = ~np.isnan(theta)
angle_deg = np.full(T, np.nan)
angle_deg[valid] = -np.degrees(np.unwrap(2 * theta[valid]) / 2)

print(f"fly found in {valid.sum()}/{T} frames")
print(f'time: {time.perf_counter() - start:.2f} seconds')

t = np.arange(T) / FPS if FPS else np.arange(T)
xlabel = "time (s)" if FPS else "frame"

fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ax[0].plot(t, meas["x"], '.-', color='tab:blue');   ax[0].set_ylabel("x (px)")
ax[1].plot(t, meas["y"], '.-', color='tab:orange'); ax[1].set_ylabel("y (px)")
ax[1].invert_yaxis()
ax[2].plot(t, angle_deg, '.-', color='tab:green');  ax[2].set_ylabel("orientation (deg)")
ax[2].set_xlabel(xlabel)
plt.tight_layout()
plt.savefig(output_graph, format="svg")
plt.close(fig)