"""Parity test: the C++ core must match reference/glow.py.

Usage: python3 tests/parity.py PATH_TO_GLOWPROFOND_CLI
Renders the reference demo scene with each preset in both implementations
and compares the float outputs.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
from demo import scene  # noqa: E402
from glow import GlowParams, render  # noqa: E402

TOLERANCE = 2e-4  # float32 summation order differs between NumPy and C++

# (name, params, equivalent CLI flags)
CASES = [
    ("default", GlowParams(radius=200), ["--radius", "200"]),
    ("spread_1.5_wide", GlowParams(radius=200, spread=1.5), ["--radius", "200", "--spread", "1.5"]),
    ("spread_3_tight", GlowParams(radius=200, spread=3.0, exposure=1),
     ["--radius", "200", "--spread", "3", "--exposure", "1"]),
    ("anamorphic", GlowParams(radius=200, aspect=6.0), ["--radius", "200", "--aspect", "6"]),
    ("chromatic", GlowParams(radius=200, rgb_radius=(1.4, 1.0, 0.7)),
     ["--radius", "200", "--rgb-radius", "1.4,1.0,0.7"]),
    ("multitint", GlowParams(radius=200, tint=((1, 1, 1), (1, 0.4, 0.8), (0.3, 0.3, 1.0)), exposure=1),
     ["--radius", "200", "--exposure", "1", "--tint", "1,1,1", "--tint", "1,0.4,0.8", "--tint", "0.3,0.3,1"]),
    ("glow_only", GlowParams(radius=200, view="glow"), ["--radius", "200", "--view", "glow"]),
    ("threshold_tonemap", GlowParams(radius=60, threshold=0.5, threshold_smooth=0.1, tonemap=True),
     ["--radius", "60", "--threshold", "0.5", "--threshold-smooth", "0.1", "--tonemap"]),
    ("linear_downsample", GlowParams(radius=120, gamma_correct=False, downsample=2),
     ["--radius", "120", "--linear", "--downsample", "2"]),
    ("unmult", GlowParams(radius=80, unmult=True), ["--radius", "80", "--unmult"]),
    ("screen", GlowParams(radius=200, blend="screen", exposure=1),
     ["--radius", "200", "--blend", "screen", "--exposure", "1"]),
    ("dirt", GlowParams(radius=200, dirt_amount=2.0, tint=((1, 0.5, 0.2),), exposure=1),
     ["--radius", "200", "--dirt-amount", "2", "--tint", "1,0.5,0.2", "--exposure", "1", "--dirt", "@dirt"]),
    ("depth_spread", GlowParams(radius=200, depth_spread=0.8),
     ["--radius", "200", "--depth-spread", "0.8", "--depth", "@depth"]),
    ("occlusion", GlowParams(radius=200, occlusion=True, exposure=1),
     ["--radius", "200", "--occlusion", "--exposure", "1", "--depth", "@depth"]),
    ("depth_all_downsample", GlowParams(radius=150, depth_spread=0.5, occlusion=True, occlusion_softness=0.2,
                                        downsample=2, dirt_amount=1.0),
     ["--radius", "150", "--depth-spread", "0.5", "--occlusion", "--occlusion-softness", "0.2",
      "--downsample", "2", "--dirt-amount", "1", "--depth", "@depth", "--dirt", "@dirt"]),
    ("saturation", GlowParams(radius=120, saturation=1.8), ["--radius", "120", "--saturation", "1.8"]),
    ("desaturated", GlowParams(radius=120, saturation=0.0), ["--radius", "120", "--saturation", "0"]),
    ("behind_unmult", GlowParams(radius=80, unmult=True, view="behind"),
     ["--radius", "80", "--unmult", "--view", "behind"]),
    ("quality_4", GlowParams(radius=160, spread=1.85, downsample=4),
     ["--radius", "160", "--spread", "1.85", "--downsample", "4"]),
    ("quality_8", GlowParams(radius=160, spread=1.85, downsample=8),
     ["--radius", "160", "--spread", "1.85", "--downsample", "8"]),
    ("core_size", GlowParams(radius=160, base_sigma=5.0, base_sigma_ref=1.5),
     ["--radius", "160", "--base-sigma", "5", "--base-sigma-ref", "1.5"]),
    ("anamorphic_angle", GlowParams(radius=120, aspect=4.0, angle=30.0),
     ["--radius", "120", "--aspect", "4", "--angle", "30"]),
    ("lateral_ca", GlowParams(radius=120, ca_offset=12.0, ca_center=(320.0, 180.0), ca_rmax=360.0, ca_pair="rb"),
     ["--radius", "120", "--ca-offset", "12", "--ca-center", "320,180", "--ca-rmax", "360", "--ca-pair", "rb"]),
    ("lateral_ca_gb", GlowParams(radius=120, ca_offset=-8.0, ca_center=(100.0, 50.0), ca_rmax=200.0, ca_pair="gb"),
     ["--radius", "120", "--ca-offset", "-8", "--ca-center", "100,50", "--ca-rmax", "200", "--ca-pair", "gb"]),
    ("tonemap_hable_mix", GlowParams(radius=120, tonemap=True, tonemap_curve="hable", tonemap_mix=0.5),
     ["--radius", "120", "--tonemap", "--tonemap-curve", "hable", "--tonemap-mix", "0.5"]),
    ("tonemap_reinhard", GlowParams(radius=120, tonemap=True, tonemap_curve="reinhard", exposure=2),
     ["--radius", "120", "--tonemap", "--tonemap-curve", "reinhard", "--exposure", "2"]),
    ("source_opacity", GlowParams(radius=120, source_opacity=0.4),
     ["--radius", "120", "--source-opacity", "0.4"]),
]


def aux_maps(shape):
    """Depth (left near, right far, with a near disc) and a blotchy dirt texture."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = x / (w - 1)
    depth = np.where((x - 0.6 * w) ** 2 + (y - 0.5 * h) ** 2 < (0.12 * h) ** 2, 0.1, depth)
    dirt = 0.5 + 0.5 * np.sin(x / 7.0)[..., None] * np.cos(y / 11.0)[..., None] * np.array([1, 0.8, 0.6])
    return depth.astype(np.float32), dirt.astype(np.float32)


def write_pfm(path, img):
    h, w = img.shape[:2]
    with open(path, "wb") as f:
        f.write(f"PF\n{w} {h}\n-1.0\n".encode())
        f.write(np.ascontiguousarray(img[::-1, :, :3], dtype="<f4").tobytes())


def read_pfm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"PF"
        w, h = map(int, f.readline().split())
        assert float(f.readline()) < 0
        data = np.frombuffer(f.read(), dtype="<f4")
    return data.reshape(h, w, 3)[::-1]


def main():
    cli = sys.argv[1]
    src = scene()
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "scene.pfm"
        write_pfm(src_path, src)
        depth, dirt = aux_maps(src.shape[:2])
        aux = {"@depth": Path(tmp) / "aux_depth.pfm", "@dirt": Path(tmp) / "aux_dirt.pfm"}
        write_pfm(aux["@depth"], np.repeat(depth[..., None], 3, axis=-1))
        write_pfm(aux["@dirt"], dirt)
        for name, params, flags in CASES:
            out_path = Path(tmp) / f"out_{name}.pfm"
            args = [str(aux.get(f, f)) for f in flags]
            subprocess.run([cli, str(src_path), str(out_path), *args], check=True)
            ours = read_pfm(out_path)
            ref = render(src, params, depth=depth if "@depth" in flags else None,
                         dirt=dirt if "@dirt" in flags else None)
            err = float(np.abs(ours - ref).max())
            ok = err <= TOLERANCE
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {name:20s} max|diff| = {err:.2e}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
