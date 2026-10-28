"""Reference implementation of the Glow Profond effect (NumPy only).

This is the "source of truth" for the look: the After Effects plugin must
match its output. Images are float32 arrays of shape (H, W, 3) or (H, W, 4),
encoded in the comp's working space (sRGB-like gamma unless linear=True).
"""
from dataclasses import dataclass, replace

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
    # Core size: with base_sigma_ref > 0, level weights are relative to that
    # sigma and the light of the octaves between it and a larger base_sigma
    # folds into the first level, so a bigger core keeps the glow's energy.
    base_sigma_ref: float = 0.0
    angle: float = 0.0               # degrees: direction of the aspect stretch (0 = horizontal)
    # Lateral chromatic aberration on the glow: channel pair ("rb", "rg",
    # "gb") scaled apart about ca_center (pixel-centre coordinates); the first
    # moves out by ca_offset px at distance ca_rmax, the second in.
    ca_offset: float = 0.0
    ca_center: tuple = (0.0, 0.0)
    ca_rmax: float = 1.0
    ca_pair: str = "rb"
    tonemap_curve: str = "aces"      # "aces" | "hable" | "reinhard" (when tonemap is on)
    tonemap_mix: float = 1.0         # 0 = untouched, 1 = full curve
    source_opacity: float = 1.0      # the source in the composite (the glow is unaffected)
    dirt_inherit_color: bool = False  # lens dirt lit by the tinted glow instead of the untinted one
    dirt_threshold: float = 0.0      # cuts the dirt texture's dark values (0..1)


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
    """2x downsample with a [1 3 3 1] / 8 filter per axis. Odd sizes round up
    and pixels past the edge are zero: texel i is centred on source pixels 2i
    and 2i + 1, so every level stays aligned with the image whatever the
    buffer size (truncating made coarse levels drift by up to 2^k pixels).
    The filter is wider than a 2x2 box so the coarse levels don't alias: with
    a box, the glow depended on where the shape sat on the coarse grid (one
    side up to 4 % brighter, flicker on a slowly moving layer)."""

    def along(a, axis):
        n = a.shape[axis]
        n2 = (n + 1) // 2
        pad = [(0, 0)] * a.ndim
        pad[axis] = (1, 2 * n2 - n + 1)  # taps 2i-1 .. 2i+2, zeros outside
        p = np.pad(a, pad)
        t = [np.take(p, range(k, k + 2 * n2, 2), axis=axis) for k in range(4)]
        return (t[0] + 3.0 * t[1] + 3.0 * t[2] + t[3]) * 0.125

    return along(along(np.asarray(img, np.float32), 1), 0).astype(np.float32)


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


def level_plan(radius, base_sigma, spread, base_sigma_ref=0.0):
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

    base_sigma_ref (core size): when > 0, weights are relative to it and the
    octaves between it and a larger base_sigma add their weight (integrated,
    so it's continuous) to the first level. Raising the core size then
    softens the core instead of dropping light (the glow used to shrink).
    """
    ref = base_sigma_ref if base_sigma_ref > 0 else base_sigma
    nf = float(np.log2(max(radius / base_sigma, 1.0))) + 1.0
    n = max(1, int(np.ceil(nf - 1e-6)))
    sigmas = base_sigma * 2.0 ** np.arange(n, dtype=np.float64)
    sigmas[-1] = base_sigma * 2.0 ** (nf - 1.0)
    q = 2.0 - spread
    weights = (sigmas / ref) ** q
    weights[-1] *= min(nf - (n - 1), 1.0)
    if base_sigma > ref:
        span = float(np.log2(base_sigma / ref))
        weights[0] += span if abs(q) < 1e-9 else (2.0 ** (q * span) - 1.0) / (q * np.log(2.0))
    return sigmas, weights


def _sample(img, xs, ys):
    """Bilinear read of img (H, W, C) at pixel-centre coordinates (xs, ys),
    zero outside the image."""
    h, w = img.shape[:2]
    x, y = xs - 0.5, ys - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0).astype(np.float32)[..., None], (y - y0).astype(np.float32)[..., None]

    def at(yy, xx):
        ok = ((xx >= 0) & (xx < w) & (yy >= 0) & (yy < h))[..., None]
        return img[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)] * ok

    top = at(y0, x0) * (1 - fx) + at(y0, x0 + 1) * fx
    bot = at(y0 + 1, x0) * (1 - fx) + at(y0 + 1, x0 + 1) * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def _rotation(angle, w, h):
    """Size of the buffer holding a w x h image turned so that `angle` (deg)
    lies along x, and the unit vectors (d along the angle, n across it)."""
    t = np.radians(angle)
    c, s = np.cos(t), np.sin(t)
    wr = int(np.ceil(abs(w * c) + abs(h * s))) + 2
    hr = int(np.ceil(abs(w * s) + abs(h * c))) + 2
    return wr, hr, (c, -s), (s, c)


def _rotate_in(img, angle, wr, hr, d, n):
    h, w = img.shape[:2]
    v, u = np.mgrid[0:hr, 0:wr].astype(np.float64)
    qx, qy = u + 0.5 - wr / 2, v + 0.5 - hr / 2
    return _sample(img, w / 2 + qx * d[0] + qy * n[0], h / 2 + qx * d[1] + qy * n[1])


def _rotate_out(img, w, h, d, n):
    hr, wr = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    rx, ry = x + 0.5 - w / 2, y + 0.5 - h / 2
    return _sample(img, wr / 2 + rx * d[0] + ry * d[1], hr / 2 + rx * n[0] + ry * n[1])


def _lateral_ca(g, p):
    """Chromatic aberration on the glow: the channel pair scaled apart about
    ca_center, by (1 + k) and (1 - k), k = ca_offset / ca_rmax."""
    if p.ca_offset == 0:
        return g
    a, b = {"rb": (0, 2), "rg": (0, 1), "gb": (1, 2)}[p.ca_pair]
    k = float(np.clip(p.ca_offset / max(p.ca_rmax, 1e-6), -0.9, 0.9))
    h, w = g.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    x, y = x + 0.5, y + 0.5
    cx, cy = p.ca_center
    out = g.copy()
    for ch, scale in ((a, 1.0 + k), (b, 1.0 - k)):
        out[..., ch] = _sample(g[..., ch:ch + 1], cx + (x - cx) / scale, cy + (y - cy) / scale)[..., 0]
    return out


def _saturate(glow, s):
    if s == 1.0:
        return glow
    # Scale the chroma around the luma: a neon keeps (or loses) its colour.
    luma = glow @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return np.maximum(luma[..., None] + s * (glow - luma[..., None]), 0.0)


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


def _glow_core(src_lin, p: GlowParams, depth=None, dirt=None):
    """Glow only (before chromatic aberration and saturation), in linear light.

    Level k is blurred at 1/max(2^k, D) resolution (D = global downsample,
    the Quality setting), then levels are folded back from coarsest to finest
    with a 2x tent upsample wherever the resolution changes (~1.33x the image
    in total work). Each level's blur is shrunk by the variance its upsample
    chain adds back, so its effective sigma stays sigma_k. A lower Quality
    keeps the same look and only loses the finest detail: putting every
    level 2^e times coarser (as before) made each one ~2^e times too wide,
    and the glow went dim and washed out.

    depth: optional (H, W) map, 0 = near, 1 = far. dirt: optional (H, W, 3)
    lens dirt texture, already in linear light.
    """
    H, W = src_lin.shape[:2]
    thr = soft_threshold(src_lin, p.threshold, p.threshold_smooth)
    d = None if depth is None else _resize_to(np.asarray(depth, np.float32)[..., None], (H, W))
    for _ in range(int(np.log2(max(p.downsample, 1)))):
        thr = downsample2(thr)
        d = None if d is None else downsample2(d)
    e = int(np.log2(max(p.downsample, 1)))
    sigmas, weights = level_plan(p.radius, p.base_sigma, p.spread, p.base_sigma_ref)
    n = len(sigmas)
    ax, ay = np.sqrt(p.aspect), 1.0 / np.sqrt(p.aspect)
    # Level k lives at 1 / 2^res[k]; its sigma in its own pixels.
    res = [max(k, e) for k in range(n)]
    sigma_level = sigmas / 2.0 ** np.array(res, np.float64)
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
    for k in range(1, n):
        same = res[k] == res[k - 1]  # below the Quality resolution: shared image
        pyr_a.append(pyr_a[-1] if same else downsample2(pyr_a[-1]))
        if pyr_b is not None:
            pyr_b.append(pyr_b[-1] if same else downsample2(pyr_b[-1]))

    def contribution(k):
        # Variance added by the upsample steps back to the base level, in level-k texels.
        steps = res[k] - e
        extra = TENT_STEP_VARIANCE * (4.0 / 3.0) * (1.0 - 4.0 ** -steps)
        tint_k = _tint_for_level(p.tint, k, n)
        lvl = pyr_a[k] if pyr_b is None else pyr_a[k] + c[k] * pyr_b[k]
        blurred = np.empty_like(lvl)
        for ch in range(lvl.shape[-1]):
            s = sigma_level[k] * (p.rgb_radius[ch] if ch < 3 else 1.0)
            sx = np.sqrt(max((s * ax) ** 2 - extra, 0.0))
            sy = np.sqrt(max((s * ay) ** 2 - extra, 0.0))
            blurred[..., ch] = gaussian(lvl[..., ch:ch + 1], sx, sy)[..., 0]
        rgb = blurred[..., :3]
        # Folded channels: tinted glow (3), untinted wide glow (3), occlusion (2).
        parts = [weights[k] * tint_k * rgb,
                 weights[k] * rgb * (k >= large_from) * (tint_k if p.dirt_inherit_color else 1.0)]
        if occlude:
            parts.append(weights[k] * blurred[..., 3:5])
        return np.concatenate(parts, axis=-1)

    acc = contribution(n - 1)
    for k in range(n - 2, -1, -1):
        if res[k + 1] != res[k]:
            acc = upsample_tent2(acc, pyr_a[k].shape[:2])
        acc = acc + contribution(k)
    if acc.shape[:2] != (H, W):
        acc = upsample_by(acc, (H, W), 2 ** int(np.log2(max(p.downsample, 1))))
    glow, large = acc[..., :3], acc[..., 3:6]
    if occlude:
        d_full = _resize_to(np.asarray(depth, np.float32)[..., None], (H, W))
        d_mean = acc[..., 7:8] / np.maximum(acc[..., 6:7], 1e-8)
        glow = glow * (1.0 - smoothstep(0.0, p.occlusion_softness, d_mean - d_full))
    if dirt is not None and p.dirt_amount > 0:
        dt = _resize_to(np.asarray(dirt, np.float32)[..., :3], (H, W))
        if p.dirt_threshold > 0:  # matte: only the brighter smudges light up
            dt = np.maximum(dt - p.dirt_threshold, 0.0) / max(1.0 - p.dirt_threshold, 1e-6)
        glow = glow + p.dirt_amount * dt * large
    return glow * 2.0 ** p.exposure


def glow_layer(src_lin, p: GlowParams, depth=None, dirt=None):
    """Glow only, in linear light, same size as src_lin (H, W, 3).

    With an aspect ratio and an angle, the glow is computed in a buffer
    turned so the angle lies along x, then turned back. Lateral chromatic
    aberration, then saturation, apply to the finished glow.

    depth: optional (H, W) map, 0 = near, 1 = far. dirt: optional (H, W, 3)
    lens dirt texture, already in linear light.
    """
    q = replace(p, saturation=1.0, ca_offset=0.0, angle=0.0)
    if p.aspect != 1.0 and p.angle % 180.0 != 0.0:
        h, w = src_lin.shape[:2]
        wr, hr, d, n = _rotation(p.angle, w, h)
        turn = lambda img: _rotate_in(img, p.angle, wr, hr, d, n)  # noqa: E731
        rdepth = None if depth is None else turn(_resize_to(np.asarray(depth, np.float32)[..., None], (h, w)))[..., 0]
        rdirt = None if dirt is None else turn(_resize_to(np.asarray(dirt, np.float32)[..., :3], (h, w)))
        g = _rotate_out(_glow_core(turn(src_lin), q, rdepth, rdirt), w, h, d, n)
    else:
        g = _glow_core(src_lin, q, depth, dirt)
    return _saturate(_lateral_ca(g, p), p.saturation)


def _hable(x):
    a, b, c, d, e, f = 0.15, 0.5, 0.1, 0.2, 0.02, 0.3
    return (x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f) - e / f


def tonemap(x, p: GlowParams):
    """The tonemap curve chosen in p (ACES fitted, Hable filmic or Reinhard),
    mixed with the untouched value by p.tonemap_mix. Identity when off."""
    if not p.tonemap:
        return x
    if p.tonemap_curve == "hable":
        t = _hable(2.0 * np.maximum(x, 0.0)) / _hable(11.2)
    elif p.tonemap_curve == "reinhard":
        x0 = np.maximum(x, 0.0)
        t = x0 / (1.0 + x0)
    else:
        t = aces_fitted(x)
    return x + (t - x) * p.tonemap_mix


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
    # Source opacity: the source's share of the composite (the glow is still
    # made from the full source).
    shown, shown_alpha = lin_premult * p.source_opacity, alpha * p.source_opacity
    if p.view == "source":
        out = shown
    else:
        if dirt is not None and p.gamma_correct:
            dirt = srgb_to_linear(np.asarray(dirt, np.float32)[..., :3])
        g = glow_layer(lin_premult, p, depth=depth, dirt=dirt)
        if p.view == "glow":
            out = g
        elif p.view == "behind":
            out = composite(shown, g * (1 - shown_alpha), p.blend)  # the halo passes under the layer
        else:
            out = composite(shown, g, p.blend)
    out = tonemap(out, p)
    out = linear_to_srgb(out) if p.gamma_correct else out
    return out.astype(np.float32)
