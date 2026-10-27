"""Run: python3 reference/test_glow.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from glow import GlowParams, glow_layer, level_weights, soft_threshold, upsample_tent2  # noqa: E402


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
    assert np.allclose(level_weights(6, 2.0), 1 / 6)


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
    assert abs(g.sum() - 1000) < 20, g.sum()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
