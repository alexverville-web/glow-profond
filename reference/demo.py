"""Render a synthetic test scene with a few parameter presets.

Usage: python3 reference/demo.py OUTPUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from glow import GlowParams, render  # noqa: E402


def scene(h=360, w=640):
    img = np.zeros((h, w, 3), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    img[(yy - 180) ** 2 + (xx - 160) ** 2 < 12 ** 2] = (1.0, 0.6, 0.2)   # warm dot
    img[170:190, 300:560] = (0.2, 0.7, 1.0)                              # cyan bar
    img[60:64, 40:600] = (1.0, 1.0, 1.0)                                 # thin line
    return img


PRESETS = {
    "default": GlowParams(radius=200),
    "spread_1.5_wide": GlowParams(radius=200, spread=1.5),
    "spread_3_tight": GlowParams(radius=200, spread=3.0, exposure=1),
    "anamorphic": GlowParams(radius=200, aspect=6.0),
    "chromatic": GlowParams(radius=200, rgb_radius=(1.4, 1.0, 0.7)),
    "multitint": GlowParams(radius=200, tint=((1, 1, 1), (1, 0.4, 0.8), (0.3, 0.3, 1.0)), exposure=1),
    "glow_only": GlowParams(radius=200, view="glow"),
}

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "renders")
    out.mkdir(parents=True, exist_ok=True)
    src = scene()
    tiles = []
    for name, p in PRESETS.items():
        img = (np.clip(render(src, p), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(out / f"{name}.png")
        tiles.append(img)
        print("wrote", out / f"{name}.png")
    tiles.insert(0, (np.clip(src, 0, 1) * 255).astype(np.uint8))
    grid = np.concatenate([np.concatenate(tiles[i:i + 2], 1) for i in range(0, 8, 2)], 0)
    Image.fromarray(grid).save(out / "contact_sheet.png")
