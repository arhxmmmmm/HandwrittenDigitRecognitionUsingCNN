"""Image preprocessing that converts an arbitrary photo/drawing of a digit into
the 28 x 28 white-on-black format used by MNIST."""
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage


def to_mnist(img: Image.Image, invert: str = "auto") -> np.ndarray:
    """Return a (28, 28) float32 array in [0, 255], white digit on black background.

    invert: "auto" (decide from the border brightness), "yes" or "no".
    """
    g = ImageOps.grayscale(img.convert("RGBA").convert("RGB") if img.mode != "L" else img)
    a = np.asarray(g, dtype=np.float32)

    if invert == "auto":
        border = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
        invert = "yes" if border.mean() > 127 else "no"
    if invert == "yes":
        a = 255.0 - a

    # stretch contrast and suppress faint background noise
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max() * 255.0
    a[a < 40] = 0

    ys, xs = np.where(a > 0)
    if len(ys) == 0:
        return np.zeros((28, 28), np.float32)
    a = a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    # scale longest side to 20 px, keep aspect ratio
    h, w = a.shape
    s = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    a = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((nw, nh), Image.LANCZOS), np.float32)

    # paste into 28 x 28 canvas, then centre by centre of mass (as MNIST does)
    canvas = np.zeros((28, 28), np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top : top + nh, left : left + nw] = a
    cy, cx = ndimage.center_of_mass(canvas)
    canvas = ndimage.shift(canvas, (14 - cy, 14 - cx), order=1, mode="constant")
    return np.clip(canvas, 0, 255).astype(np.float32)
