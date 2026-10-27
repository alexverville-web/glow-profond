"""Reference implementation of the Glow Profond effect (NumPy only).

This is the "source of truth" for the look: the After Effects plugin must
match its output. Images are float32 arrays of shape (H, W, 3) or (H, W, 4),
encoded in the comp's working space (sRGB-like gamma unless linear=True).
"""
from dataclasses import dataclass

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
    view: str = "final"              # "final" | "glow" | "source" | "behind" (glow only where alpha < 1)
    blend: str = "add"               # "add" | "screen" (screen never pushes src <= 1 past 1)
    dirt_amount: float = 0.0         # lens dirt strength (needs a dirt texture)
    depth_spread: float = 0.0        # 0..1: far sources (depth 1) push energy to wide levels
    occlusion: bool = False          # nearer pixels mask the glow of farther sources
    occlusion_softness: float = 0.05  # depth range of the occlusion smoothstep
    saturation: float = 1.0          # glow saturation around its Rec. 709 luma (0 grey, 2 double)


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
    """2x2 box average. Odd sizes round up, with zeros past the edge: texel i
    always covers source pixels 2i and 2i + 1, so every level stays aligned
    with the image (truncating made coarse levels drift by up to 2^k pixels,
    and the halo moved whenever the buffer size changed)."""
    h, w = img.shape[:2]
    i = np.pad(img, ((0, h % 2), (0, w % 2)) + ((0, 0),) * (img.ndim - 2))
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


def level_plan(radius, base_sigma, spread):
    """Sigmas and weights of the pyramid levels, continuous in the radius.

    Levels sit at sigma_k = s0 * 2^k, except the widest one, whose sigma is
    exactly the radius and whose weight fades in with the fractional octave:
    sliding the radius grows the halo smoothly, without the jumps a whole
    level appearing at once would cause.

    A sum of area-normalized 2D Gaussians with log-spaced sigmas and weights
    w_k ~ (sigma_k / s0)^(2-p) has a point-spread function ~ r^-p between the
    smallest and largest sigma. p = 2 gives equal weights per octave, i.e.
    inverse-square falloff. Weights are not normalized: a larger radius adds
    halo further out and leaves the glow near the source as it was.
    """
    nf = float(np.log2(max(radius / base_sigma, 1.0))) + 1.0
    n = max(1, int(np.ceil(nf - 1e-6)))
    sigmas = base_sigma * 2.0 ** np.arange(n, dtype=np.float64)
    sigmas[-1] = base_sigma * 2.0 ** (nf - 1.0)
    weights = (sigmas / base_sigma) ** (2.0 - spread)
    weights[-1] *= min(nf - (n - 1), 1.0)
    return sigmas, weights


def _tint_for_level(tint, k, n):
    cols = np.asarray(tint, dtype=np.float32)
    if len(cols) == 1:
        return cols[0]
    t = k / max(n - 1, 1) * (len(cols) - 1)
    i = min(int(t), len(cols) - 2)
    return cols[i] * (1 - (t - i)) + cols[i + 1] * (t - i)


def upsample_tent2(img, shape):
    """3x3 tent upsample (Jimenez, CoD: AW bloom) by exactly 2x, to (H, W).

    Bilinear taps at -1/0/+1 source texels around each target pixel's
    position in source coordinates, weights 1-2-1 / 2-4-2 / 1-2-1 over 16,
    edges clamped like upsample_to. The scale is exactly 2 (the inverse of
    downsample2), whatever the rounding of the sizes.
    """
    H, W = shape
    h, w = img.shape[:2]
    ys = (np.arange(H) + 0.5) / 2.0 - 0.5
    xs = (np.arange(W) + 0.5) / 2.0 - 0.5

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


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def upsample_by(img, shape, f):
    """Bilinear upsample by exactly f (the inverse of f/2 downsample2 steps)."""
    H, W = shape
    h, w = img.shape[:2]
    ys = np.clip((np.arange(H) + 0.5) / f - 0.5, 0, h - 1)
    xs = np.clip((np.arange(W) + 0.5) / f - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = img[y0][:, x0] * (1 - fx) + img[y0][:, x1] * fx
    bot = img[y1][:, x0] * (1 - fx) + img[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def _resize_to(img, shape):
    return img if img.shape[:2] == tuple(shape) else upsample_to(img, shape)


def glow_layer(src_lin, p: GlowParams, depth=None, dirt=None):
    """Glow only, in linear light, same size as src_lin (H, W, 3).

    Levels are blurred at 1/2^k resolution, then folded back from coarsest to
    finest with a 2x tent upsample per step (~1.33x the image in total work).
    Each level's blur is shrunk by the variance its upsample chain adds back,
    so the effective sigma stays base_sigma * 2^k.

    depth: optional (H, W) map, 0 = near, 1 = far. dirt: optional (H, W, 3)
    lens dirt texture, already in linear light.
    """
    H, W = src_lin.shape[:2]
    thr = soft_threshold(src_lin, p.threshold, p.threshold_smooth)
    d = None if depth is None else _resize_to(np.asarray(depth, np.float32)[..., None], (H, W))
    for _ in range(int(np.log2(max(p.downsample, 1)))):
        thr = downsample2(thr)
        d = None if d is None else downsample2(d)
    scale = float(2 ** int(np.log2(max(p.downsample, 1))))
    sigmas, weights = level_plan(p.radius, p.base_sigma, p.spread)
    n = len(sigmas)
    ax, ay = np.sqrt(p.aspect), 1.0 / np.sqrt(p.aspect)
    # Sigma of level k in its own pixels: base_sigma for all but the widest.
    sigma_level = sigmas / (2.0 ** np.arange(n) * scale)
    occlude = p.occlusion and d is not None
    spread = 0.0 if d is None else float(np.clip(p.depth_spread, 0.0, 1.0))
    large_from = n // 2  # levels feeding the lens dirt

    # Depth spread: a source at depth d sends w_k * lerp(1, 2^(k - n/2), spread * d)
    # to level k, renormalized so its total is 1. That factor is affine in d
    # once divided by Z(d), so two pyramids (a, b) replace one per level:
    # level input k = down^k(a) + c_k * down^k(b), c_k = 2^(k - n/2) - 1.
    c = 2.0 ** (np.arange(n) - n / 2.0) - 1.0
    if spread > 0:
        z = 1.0 + spread * d * float(np.dot(weights / weights.sum(), c))
        a_img, b_img = thr / z, thr * spread * d / z
    else:
        a_img, b_img = thr, None

    def extend(img):
        # Channels 3-4 carry intensity and intensity * depth for occlusion.
        if not occlude:
            return img
        i = img.mean(axis=-1, keepdims=True)
        return np.concatenate([img, i, i * d], axis=-1)

    pyr_a = [extend(a_img)]
    pyr_b = None if b_img is None else [extend(b_img)]
    for _ in range(1, n):
        pyr_a.append(downsample2(pyr_a[-1]))
        if pyr_b is not None:
            pyr_b.append(downsample2(pyr_b[-1]))

    def contribution(k):
        # Variance added by the k upsample steps back to level 0, in level-k texels.
        extra = TENT_STEP_VARIANCE * (4.0 / 3.0) * (1.0 - 4.0 ** -k)
        lvl = pyr_a[k] if pyr_b is None else pyr_a[k] + c[k] * pyr_b[k]
        blurred = np.empty_like(lvl)
        for ch in range(lvl.shape[-1]):
            s = sigma_level[k] * (p.rgb_radius[ch] if ch < 3 else 1.0)
            sx = np.sqrt(max((s * ax) ** 2 - extra, 0.0))
            sy = np.sqrt(max((s * ay) ** 2 - extra, 0.0))
            blurred[..., ch] = gaussian(lvl[..., ch:ch + 1], sx, sy)[..., 0]
        rgb = blurred[..., :3]
        # Folded channels: tinted glow (3), untinted wide glow (3), occlusion (2).
        parts = [weights[k] * _tint_for_level(p.tint, k, n) * rgb,
                 weights[k] * rgb * (k >= large_from)]
        if occlude:
            parts.append(weights[k] * blurred[..., 3:5])
        return np.concatenate(parts, axis=-1)

    acc = contribution(n - 1)
    for k in range(n - 2, -1, -1):
        acc = upsample_tent2(acc, pyr_a[k].shape[:2]) + contribution(k)
    if acc.shape[:2] != (H, W):
        acc = upsample_by(acc, (H, W), 2 ** int(np.log2(max(p.downsample, 1))))
    glow, large = acc[..., :3], acc[..., 3:6]
    if occlude:
        d_full = _resize_to(np.asarray(depth, np.float32)[..., None], (H, W))
        d_mean = acc[..., 7:8] / np.maximum(acc[..., 6:7], 1e-8)
        glow = glow * (1.0 - smoothstep(0.0, p.occlusion_softness, d_mean - d_full))
    if dirt is not None and p.dirt_amount > 0:
        glow = glow + p.dirt_amount * _resize_to(np.asarray(dirt, np.float32)[..., :3], (H, W)) * large
    glow = glow * 2.0 ** p.exposure
    if p.saturation != 1.0:
        # Scale the chroma around the luma: a neon keeps (or loses) its colour.
        luma = glow @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        glow = np.maximum(luma[..., None] + p.saturation * (glow - luma[..., None]), 0.0)
    return glow


def composite(src, glow, blend):
    if blend == "screen":
        return src + glow * (1.0 - np.clip(src, 0.0, 1.0))
    return src + glow


def aces_fitted(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip(x * (a * x + b) / (x * (c * x + d) + e), 0, 1)


def render(src, p: GlowParams, depth=None, dirt=None):
    """depth: optional (H, W), 0 = near, 1 = far. dirt: optional (H, W, 3)
    lens dirt texture in the same encoding as src."""
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
        if dirt is not None and p.gamma_correct:
            dirt = srgb_to_linear(np.asarray(dirt, np.float32)[..., :3])
        g = glow_layer(lin_premult, p, depth=depth, dirt=dirt)
        if p.view == "glow":
            out = g
        elif p.view == "behind":
            out = composite(lin_premult, g * (1 - alpha), p.blend)  # the halo passes under the layer
        else:
            out = composite(lin_premult, g, p.blend)
    if p.tonemap:
        out = aces_fitted(out)
    out = linear_to_srgb(out) if p.gamma_correct else out
    return out.astype(np.float32)
