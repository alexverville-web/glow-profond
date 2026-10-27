# Glow Profond

Effet de glow « physiquement plausible » (falloff en 1/r², seuil HDR doux,
aberration chromatique, teinte multicolore) pour After Effects, développé
par Gimmick Studio.

- Spec : [docs/SPEC.md](docs/SPEC.md)
- Référence Python : `python3 reference/demo.py renders/` (numpy, pillow)
- Plug-in After Effects : [docs/AE_BUILD.md](docs/AE_BUILD.md)

## Organisation

| Dossier | Contenu |
| --- | --- |
| `reference/` | Implémentation NumPy, vérité terrain du look |
| `core/` | Port C++17 de la référence, sans dépendance (utilisé par le plug-in) |
| `cli/` | `glowprofond in.png out.png --radius 200 …` pour tester le cœur |
| `tests/` | Tests du cœur et test de parité C++ ↔ Python |
| `ae/` | Wrapper After Effects (SmartFX, 8/16/32 bpc), PiPL, CMake |

## Cœur, CLI et tests (Linux, macOS, Windows)

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure   # parité : numpy requis
./build/glowprofond scene.png out.png --radius 200 --tint 1,1,1 --tint 0.3,0.3,1
```

Le test de parité rend la scène de `reference/demo.py` avec chaque preset dans
les deux implémentations et exige un écart maximal de 2e-4 (actuellement
~2e-7).
