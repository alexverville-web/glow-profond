"""Run: python3 reference/test_glow.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from glow import (GlowParams, composite, glow_layer, level_plan, render,  # noqa: E402
                  soft_threshold, upsample_tent2)


def test_zero_threshold_is_identity():
    rgb = np.random.default_rng(0).uniform(0, 2, (16, 16, 3)).astype(np.float32)
    rgb[0, 0] = (1e-3, 0, 0)
    out = soft_threshold(rgb, 0.0, 0.5)
    assert np.allclose(out, rgb, atol=1e-4), np.abs(out - rgb).max()


def test_threshold_never_amplifies():
    rgb = np.random.default_rng(1).uniform(0, 4, (32, 32, 3)).astype(np.float32)
    for t in (0.5, 1.0, 2.0):
        out = soft_threshold(rgb, t, 0.5)
        assert (out <= rgb + 1e-6).all()
        assert (out[rgb.max(-1) < t * 0.5] == 0).all()


def test_inverse_square_weights_are_equal():
    sigmas, w = level_plan(1.5 * 32, 1.5, 2.0)  # exactly 6 octaves
    assert np.allclose(w, 1.0) and np.allclose(sigmas, 1.5 * 2.0 ** np.arange(6))


def test_radius_is_continuous_and_monotonic():
    # Sliding the radius never shrinks the halo, and never jumps.
    img = _point()
    prev = None
    for r in np.linspace(20, 70, 26):
        g = glow_layer(img, GlowParams(radius=float(r)))
        far = g[128, 128 + 60, 1]
        if prev is not None:
            assert far >= prev[0] - 1e-9
            assert np.abs(g - prev[1]).max() < 0.05 * g.max()
        prev = (far, g)


def test_tent_upsample_keeps_constant_and_odd_shapes():
    out = upsample_tent2(np.full((5, 7, 3), 0.5, np.float32), (11, 15))
    assert out.shape == (11, 15, 3) and np.allclose(out, 0.5)


def test_point_glow_is_inverse_square_and_conserves_energy():
    img = np.zeros((512, 512, 3), np.float32)
    img[256, 256] = 1000
    g = glow_layer(img, GlowParams(radius=100))[..., 0]
    prof = g[256, 256:]
    slope = np.log(prof[64] / prof[16]) / np.log(4)
    assert abs(slope + 2) < 0.1, slope
    # Energy is 1000 per unit of level weight (weights are not normalized).
    total = level_plan(100, 1.5, 2.0)[1].sum()
    assert abs(g.sum() - 1000 * total) < 20 * total, g.sum()


def _point(n=256, value=1000.0):
    img = np.zeros((n, n, 3), np.float32)
    img[n // 2, n // 2] = value
    return img


def _second_moment(g):
    h, w = g.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (yy - h // 2) ** 2 + (xx - w // 2) ** 2
    lum = g[..., 0]
    return float((lum * r2).sum() / lum.sum())


def test_screen_never_exceeds_one():
    rng = np.random.default_rng(2)
    src, g = rng.uniform(0, 1, (2, 32, 32, 3)).astype(np.float32)
    out = composite(src, g, "screen")
    assert out.max() <= 1.0 + 1e-6 and (out >= src - 1e-6).all()
    assert np.allclose(composite(src, g, "add"), src + g)


def test_dirt_adds_untinted_wide_glow_only():
    img, p = _point(), GlowParams(radius=50, tint=((1, 0, 0),))
    base = glow_layer(img, p)
    assert np.array_equal(base, glow_layer(img, p, dirt=np.ones((256, 256, 3))))  # amount 0
    p.dirt_amount = 1.0
    extra = glow_layer(img, p, dirt=np.ones((256, 256, 3))) - base
    _, w = level_plan(50, 1.5, 2.0)
    expected = w[len(w) // 2:].sum() * 1000
    assert np.allclose(extra.sum(axis=(0, 1)), expected, rtol=0.03)  # untinted: G and B too


def test_depth_spread_widens_far_sources_and_keeps_energy():
    img, p = _point(), GlowParams(radius=50, depth_spread=1.0)
    near = glow_layer(img, p, depth=np.zeros((256, 256)))
    far = glow_layer(img, p, depth=np.ones((256, 256)))
    assert np.allclose(near, glow_layer(img, GlowParams(radius=50)), atol=1e-6)
    assert abs(far[..., 0].sum() - near[..., 0].sum()) < 0.01 * near[..., 0].sum()
    assert _second_moment(far) > 1.5 * _second_moment(near)


def test_occlusion_masks_glow_on_nearer_pixels_only():
    img = _point()
    depth = np.ones((256, 256), np.float32)
    depth[:, :100] = 0.0  # near wall left of a far source at x = 128
    p = GlowParams(radius=50, occlusion=True)
    g = glow_layer(img, p, depth=depth)
    free = glow_layer(img, GlowParams(radius=50))
    assert g[128, 90, 0] < 1e-3 * free[128, 90, 0]
    assert np.allclose(g[:, 100:], free[:, 100:], rtol=1e-4, atol=1e-7)


def test_new_features_off_match_defaults():
    src = np.random.default_rng(3).uniform(0, 1, (64, 64, 3)).astype(np.float32)
    a = render(src, GlowParams(radius=30))
    b = render(src, GlowParams(radius=30), depth=np.ones((64, 64)), dirt=np.ones((64, 64, 3)))
    assert np.array_equal(a, b)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


def test_buffer_size_does_not_move_the_glow():
    # Hosts pad the buffer by an amount that follows the radius: the glow
    # must stay centred on its source whatever the (odd) buffer size.
    def glow_in(size):
        img = np.zeros((size, size, 3), np.float32)
        img[40:90, 40:90] = 1.0
        return glow_layer(img, GlowParams(radius=100))[..., 0]
    a, b = glow_in(257), glow_in(300)
    assert np.abs(a[:128, :128] - b[:128, :128]).max() < 0.01 * a[65, 30]
    assert abs(a[65, 30] - a[65, 99]) < 0.03 * a[65, 30]
