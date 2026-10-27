# Compiler le plug-in After Effects

## Prérequis

- macOS 11+ avec Xcode (pour `clang` et `Rez`) et CMake 3.20+.
- Le SDK After Effects 2022 ou plus récent, téléchargé depuis
  [developer.adobe.com](https://developer.adobe.com/after-effects/) (compte
  Adobe requis, le SDK ne peut pas être versionné dans ce dépôt).

## Compilation

```sh
cmake -S . -B build-ae -G Xcode -DAE_SDK_ROOT="$HOME/SDK/AfterEffectsSDK"
cmake --build build-ae --config Release --target GlowProfond
```

`AE_SDK_ROOT` est le dossier qui contient `Examples/Headers`. Le résultat est
`build-ae/ae/Release/GlowProfond.plugin` (universel arm64 + x86_64).

## Installation

Copier `GlowProfond.plugin` dans
`/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/`, puis
relancer After Effects. L'effet apparaît dans **Effet › Gimmick › Glow
Profond**.

## Paramètres

| Paramètre AE | Champ `GlowParams` |
| --- | --- |
| Rayon | `radius` (σ du niveau le plus large, en px) |
| Exposition | `exposure` (stops) |
| Diffusion (falloff) | `spread` (2 = 1/r²) |
| Entrée › Seuil, Douceur du seuil | `threshold`, `threshold_smooth` |
| Entrée › Unmult | `unmult` |
| Forme › Ratio d'aspect | `aspect` |
| Forme › Rayon rouge / vert / bleu | `rgb_radius` |
| Teinte › Mode, couleurs coeur / milieu / halo | `tint` (1, 2 ou 3 couleurs) |
| Rendu › Correction gamma | `gamma_correct` (à couper dans un projet linéaire 32 bpc) |
| Rendu › Tonemap ACES | `tonemap` |
| Rendu › Sigma de base, Sous-échantillonnage | `base_sigma`, `downsample` |
| Affichage | `view` |

Les tailles en pixels (rayon, sigma de base) suivent la résolution de
l'aperçu : un aperçu en demi-résolution garde le même look.

## Comportement propre au plug-in

- Les buffers AE sont prémultipliés. Le wrapper appelle
  `render_premultiplied`, qui linéarise la couleur droite, calcule le glow
  comme la référence, puis étend l'alpha (`a + (1 − a) × max(glow)`) pour que
  le glow déborde du calque. Sur une image opaque, le RGB est identique à
  `render()` (testé).
- Le buffer de sortie est agrandi de 3 × rayon autour du calque
  (`PF_OutFlag_I_EXPAND_BUFFER`), limité à la zone demandée par AE.
- Le rendu est sur CPU, multithreadé. Le portage GPU (Metal) est la prochaine
  étape de performance.

## Limites connues

- Windows : le wrapper compile en principe, mais l'étape PiPL (PiPLtool →
  `.rc`) n'est pas encore branchée dans CMake.
