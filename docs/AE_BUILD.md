# Compiler le plug-in After Effects

## Prérequis

- macOS 11+ avec les Command Line Tools d'Apple (`clang`, `Rez`). CMake 3.20+ et Xcode sont facultatifs.
- Le SDK After Effects 2022 ou plus récent, téléchargé depuis
  [developer.adobe.com](https://developer.adobe.com/after-effects/) (compte
  Adobe requis, le SDK ne peut pas être versionné dans ce dépôt).

## Compilation

Sans Xcode ni CMake (Command Line Tools seulement) :

```sh
AE_SDK_ROOT=~/Documents/AE_Development/AfterEffectsSDK ae/build_mac.sh
```

Le plug-in est produit dans `build-mac/GlowProfond.plugin`.

Avec CMake (et Xcode, ou `-G "Unix Makefiles"` sans Xcode) :

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

Interface en anglais (seul le nom « Glow Profond » reste en français). Les
réglages essentiels sont en haut ; chaque option avancée est dans son groupe,
fermé par défaut, derrière une case « Enable » (ses réglages sont grisés et
ignorés tant que la case est décochée).

| Paramètre AE | Champ `GlowParams` |
| --- | --- |
| Preset | aucun : remplit les réglages (Neon, Soft Sun, Cinematic, Anamorphic, Glowing Text, Subtle Bloom). Laisse intacts View, Quality, Lens Dirt, Depth, Anti-Flicker, Unmultiply et Linear Light |
| Radius (défaut 160) | `radius` (σ du niveau le plus large, en px, continu) |
| Exposure (défaut −1,25) | `exposure` (stops) |
| Threshold | `threshold` |
| Spread (%, défaut 57,5) | `spread = 3 − 2 × Spread/100` : 100 % = 1/r (le plus large), 50 % = 1/r², 0 % = 1/r³ |
| Saturation | `saturation` (%, 100 = inchangé, 0 = gris) |
| Blend Mode | `blend` : Add ou Screen |
| View | `view` : Final, Glow Only, Source, Glow Behind |
| Source Opacity | `source_opacity` (%) : part de la source dans le composite (le glow reste calculé sur la source entière) |
| Color Tint › Tint, Core / Middle / Halo Color | `tint` (Off, 2 ou 3 couleurs) |
| Chromatic Aberration › Enable, Amount, Channels, Lateral Offset, Red / Green / Blue Radius | `rgb_radius` = rayons fins × Amount (la paire choisie s'écarte : 1re × (1 + a), 2e × (1 − a), a = Amount / 200) ; `ca_offset` (px, aux coins du calque), `ca_pair`, centre = milieu du calque. Rayons 1, 1, 1 et pas de décalage si décoché |
| Anamorphic › Enable, Aspect Ratio, Angle | `aspect`, `angle` (le glow est calculé dans un buffer tourné) ; 1 et 0 si décoché |
| Input Mask › Enable, Mask Layer, Mask Mode, Invert | `GlowInputs::mask` : alpha ou luminance du calque (étiré sur le calque), inversé au besoin ; ne masque que la source du glow |
| Lens Dirt › Enable, Dirt Texture, Dirt Amount, Dirt Threshold, Inherit Glow Color | `GlowInputs::dirt` (calque étiré sur le calque de l'effet), `dirt_amount` (%), `dirt_threshold` (%), `dirt_inherit_color` |
| Depth › Enable, Depth Layer, Depth Spread, Occlusion, Invert Depth | `GlowInputs::depth` (rouge, 0 = proche ; zones transparentes = loin), `depth_spread`, `occlusion` ; Invert = `1 − rouge`. Ignoré sans calque |
| Anti-Flicker › Enable, Smoothing | `temporal_smooth` (%) : mélange les images t ± 1 dans la source du glow |
| Advanced › Threshold Smoothness, Unmultiply, Linear Light | `threshold_smooth`, `unmult`, `gamma_correct` |
| Advanced › Tonemap, Tonemap Curve, Tonemap Mix | `tonemap`, `tonemap_curve` (ACES, Filmic = Hable, Reinhard), `tonemap_mix` (%). L'ancienne case « ACES Tonemap » garde son identifiant : cochée = ACES |
| Advanced › Core Size | `base_sigma`, avec `base_sigma_ref` = 1,5 px : agrandir le cœur ramène la lumière des octaves plus fines dans le cœur (le glow ne rapetisse plus) |
| Advanced › Quality | `downsample` : Full, 1/2, 1/4, 1/8 ; même look, seuls les détails fins se perdent |
| Advanced › Expand Bounds | Auto (le glow déborde du calque, 4 × rayon) ou Off (limité aux bords du calque) |

Les valeurs par défaut reproduisent le glow par défaut de Deep Glow autour
d'une forme blanche (comparaison de rendus, voir
`build-mac/test/defaults_vs_deepglow.png` sur la machine de build).

Les identifiants disque ne sont jamais renumérotés : les nouveaux paramètres
sont ajoutés en fin de liste. L'ancien « Diffusion (falloff) » (exposant brut)
est remplacé par Spread (%) : les projets des versions précédentes s'ouvrent,
mais reprennent le Spread par défaut.

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
- Sur GPU : tout sauf profondeur, saleté, masque, anti-scintillement,
  anamorphique tourné et décalage latéral de l'aberration (AE repasse alors
  sur le CPU, automatiquement).
- Rendu GPU Metal (Mac) quand l'accélération GPU d'After Effects est active
  (Paramètres du projet › Rendu vidéo et effets › Accélération GPU Mercury
  (Metal)) : les images restent sur le GPU, le pipeline complet
  (`gpu/metal/`, noyaux compilés à l'exécution) donne le même résultat que le
  CPU à 2e-6 près (`GLOW_METAL=1 ae/build_mac.sh` le vérifie). Avec un calque
  de profondeur, la saleté de lentille ou l'anti-scintillement, AE repasse
  automatiquement sur le CPU (multithreadé). En « Logiciel Mercury
  uniquement », tout est sur CPU.
- Sorties 8 et 16 bpc : dither ordonné 8×8 (ancré dans l'espace du calque)
  contre le banding des grands halos ; la couleur reste ≤ alpha.
- Anti-scintillement : l'effet demande les images t − 1 et t + 1
  (`PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT`) seulement quand le réglage est > 0.

## Limites connues

- Windows : le wrapper compile en principe, mais l'étape PiPL (PiPLtool →
  `.rc`) n'est pas encore branchée dans CMake.
