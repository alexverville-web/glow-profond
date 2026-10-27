"""Reference implementation of the Glow Profond effect (NumPy only).

This is the "source of truth" for the look: the After Effects plugin must
match its output. Images are float32 arrays of shape (H, W, 3) or (H, W, 4),
encoded in the comp's working space (sRGB-like gamma unless linear=True).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GlowParams:
    radius: float = 400.0            # largest blur sigma, in pixels
    exposure: float = 0.0            # stops applied to the glow
    threshold: float = 0.0           # in linear light
    threshold_smooth: float = 0.5    # soft knee, as a fraction of threshold (0..1)
    spread: float = 2.0              # falloff exponent p: glow ~ 1/r^p (2 = inverse square)
    aspect: float = 1.0              # >1 stretches horizontally, <1 vertically
    rgb_radius: tuple = (1.0, 1.0, 1.0)  # per-channel radius multipliers (chromatic aberration)
    tint: tuple = ((1.0, 1.0, 1.0),)     # colours from tight core to wide halo (multicolour tint)
    unmult: bool = False             # derive alpha from luminance (text on black)
    gamma_correct: bool = True       # linearize before blurring, re-encode after
    tonemap: bool = False            # ACES-fitted curve on the composite
    base_sigma: float = 1.5          # sigma of the tightest level, in pixels
    downsample: int = 1              # extra global downsample (speed / stylized grain)
    view: str = "final"              # "final" | "glow" | "source"


def srgb_to_linear(x):
    x = np.maximum(x, 0.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.maximum(x, 0.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def soft_threshold(rgb, t, smooth):
    """Unity-style soft-knee threshold on max(R,G,B). Smooth = less flicker.

    The knee is relative to the threshold, so t = 0 passes the image through
    unchanged instead of lifting dark pixels to ~knee/4.
    """
    br = rgb.max(axis=-1, keepdims=True)
    knee = max(t * smooth, 1e-5)
    soft = np.clip(br - t + knee, 0.0, 2 * knee) ** 2 / (4 * knee)
    contrib = np.maximum(soft, br - t) / np.maximum(br, 1e-6)
    return rgb * contrib


def _gauss_kernel(sigma):
    r = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / max(sigma, 1e-3)) ** 2)
    return k / k.sum()


def _blur_axis(img, sigma, axis):
    if sigma < 0.3:
        return img
    k = _gauss_kernel(sigma)
    r = len(k) // 2
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="constant")
    out = np.zeros_like(img)
    n = img.shape[axis]
    for i, w in enumerate(k):
        out += w * np.take(p, range(i, i + n), axis=axis)
    return out


def gaussian(img, sx, sy):
    return _blur_axis(_blur_axis(img, sx, 1), sy, 0)


def downsample2(img):
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    i = img[:h, :w]
    return 0.25 * (i[0::2, 0::2] + i[1::2, 0::2] + i[0::2, 1::2] + i[1::2, 1::2])


def upsample_to(img, shape):
    """Bilinear resize to (H, W)."""
    H, W = shape
    h, w = img.shape[:2]
    ys = np.clip((np.arange(H) + 0.5) * h / H - 0.5, 0, h - 1)
    xs = np.clip((np.arange(W) + 0.5) * w / W - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = img[y0][:, x0] * (1 - fx) + img[y0][:, x1] * fx
    bot = img[y1][:, x0] * (1 - fx) + img[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def level_weights(n, spread):
    """Weights for Gaussians at sigma_k = s0 * 2^k.

    A sum of area-normalized 2D Gaussians with log-spaced sigmas and weights
    w_k ~ sigma_k^(2-p) has a point-spread function ~ r^-p between the
    smallest and largest sigma. p = 2 gives equal weights per octave, i.e.
    inverse-square falloff.
    """
    k = np.arange(n, dtype=np.float32)
    w = (2.0 ** k) ** (2.0 - spread)
    return w / w.sum()


def _tint_for_level(tint, k, n):
    cols = np.asarray(tint, dtype=np.float32)
    if len(cols) == 1:
        return cols[0]
    t = k / max(n - 1, 1) * (len(cols) - 1)
    i = min(int(t), len(cols) - 2)
    return cols[i] * (1 - (t - i)) + cols[i + 1] * (t - i)


def upsample_tent2(img, shape):
    """3x3 tent upsample (Jimenez, CoD: AW bloom) to an exact (H, W).

    Bilinear taps at -1/0/+1 source texels around each target pixel's
    position in source coordinates, weights 1-2-1 / 2-4-2 / 1-2-1 over 16,
    edges clamped like upsample_to.
    """
    H, W = shape
    h, w = img.shape[:2]
    ys = (np.arange(H) + 0.5) * h / H - 0.5
    xs = (np.arange(W) + 0.5) * w / W - 0.5

    def taps(coords, n):
        out = []
        for d in (-1.0, 0.0, 1.0):
            c = np.clip(coords + d, 0, n - 1)
            i0 = np.floor(c).astype(int)
            out.append((i0, np.minimum(i0 + 1, n - 1), c - i0))
        return out

    ty, tx = taps(ys, h), taps(xs, w)
    tw = (0.25, 0.5, 0.25)
    # Separable: tent then bilinear along x, then along y.
    rows = sum(wx * (img[:, x0] * (1 - fx)[None, :, None] + img[:, x1] * fx[None, :, None])
               for wx, (x0, x1, fx) in zip(tw, tx))
    return sum(wy * (rows[y0] * (1 - fy)[:, None, None] + rows[y1] * fy[:, None, None])
               for wy, (y0, y1, fy) in zip(tw, ty))


# Variance (in source texels, per axis) that one 2x tent upsample adds:
# tent taps at +-1 (0.5) plus the bilinear split at +-0.25 texel (0.1875).
TENT_STEP_VARIANCE = 0.6875


def glow_layer(src_lin, p: GlowParams):
    """Glow only, in linear light, same size as src_lin (H, W, 3).

    Levels are blurred at 1/2^k resolution, then folded back from coarsest to
    finest with a 2x tent upsample per step (~1.33x the image in total work).
    Each level's blur is shrunk by the variance its upsample chain adds back,
    so the effective sigma stays base_sigma * 2^k.
    """
    H, W = src_lin.shape[:2]
    img = soft_threshold(src_lin, p.threshold, p.threshold_smooth)
    for _ in range(int(np.log2(max(p.downsample, 1)))):
        img = downsample2(img)
    scale = H / img.shape[0]
    n = max(1, int(np.floor(np.log2(max(p.radius / p.base_sigma, 1.0)))) + 1)
    weights = level_weights(n, p.spread)
    ax, ay = np.sqrt(p.aspect), 1.0 / np.sqrt(p.aspect)
    sigma_level = p.base_sigma / scale  # same in every level's own pixels
    levels = [img]
    for _ in range(1, n):
        levels.append(downsample2(levels[-1]))

    def blur_level(k):
        # Variance added by the k upsample steps back to level 0, in level-k texels.
        extra = TENT_STEP_VARIANCE * (4.0 / 3.0) * (1.0 - 4.0 ** -k)
        out = np.empty_like(levels[k])
        for c in range(3):
            s = sigma_level * p.rgb_radius[c]
            sx = np.sqrt(max((s * ax) ** 2 - extra, 0.0))
            sy = np.sqrt(max((s * ay) ** 2 - extra, 0.0))
            out[..., c] = gaussian(levels[k][..., c:c + 1], sx, sy)[..., 0]
        return weights[k] * _tint_for_level(p.tint, k, n) * out

    acc = blur_level(n - 1)
    for k in range(n - 2, -1, -1):
        acc = upsample_tent2(acc, levels[k].shape[:2]) + blur_level(k)
    if acc.shape[:2] != (H, W):
        acc = upsample_to(acc, (H, W))
    return acc * 2.0 ** p.exposure


def aces_fitted(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip(x * (a * x + b) / (x * (c * x + d) + e), 0, 1)


def render(src, p: GlowParams):
    rgb = src[..., :3].astype(np.float32)
    alpha = src[..., 3:4].astype(np.float32) if src.shape[-1] == 4 else np.ones_like(rgb[..., :1])
    if p.unmult:
        alpha = np.clip(rgb.max(axis=-1, keepdims=True), 0, 1)
        rgb = rgb / np.maximum(alpha, 1e-6)
    lin = srgb_to_linear(rgb) if p.gamma_correct else rgb
    lin_premult = lin * alpha
    if p.view == "source":
        out = lin_premult
    else:
        g = glow_layer(lin_premult, p)
        out = g if p.view == "glow" else lin_premult + g
    if p.tonemap:
        out = aces_fitted(out)
    out = linear_to_srgb(out) if p.gamma_correct else out
    return out.astype(np.float32)
