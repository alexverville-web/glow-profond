"""Run: python3 reference/test_glow.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from glow import level_weights, soft_threshold  # noqa: E402


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
