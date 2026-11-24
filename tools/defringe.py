"""Remove the old red backdrop from the cut-out's edges, for display on black.
1. Semi-transparent pixels get the RGB of their nearest opaque pixel.
2. Opaque pixels near the edge that carry red backdrop spill (rim light) have
   their excess red clamped, fading out over BAND px from the edge.
Alpha is untouched, so the cut-out shape stays identical."""
import sys, numpy as np
from PIL import Image
from scipy import ndimage
src, dst = sys.argv[1:3]
im = np.asarray(Image.open(src).convert('RGBA')).copy()
solid = im[..., 3] >= 250
_, (iy, ix) = ndimage.distance_transform_edt(~solid, return_indices=True)
im[..., :3] = im[iy, ix, :3]
BAND = 14.0
dist = ndimage.distance_transform_edt(im[..., 3] > 8)          # px to transparency
w = np.clip(1 - dist / BAND, 0, 1)
rgb = im[..., :3].astype(np.float32)
r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
r_cap = np.minimum(r, g * 1.15 + 12)                            # blonde/skin keep r ≈ g
rgb[..., 0] = r * (1 - w) + r_cap * w
im[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
Image.fromarray(im).save(dst)
