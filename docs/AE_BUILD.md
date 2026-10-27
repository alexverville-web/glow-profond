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

| Paramètre AE | Champ `GlowParams` |
| --- | --- |
| Preset | aucun : remplit les réglages ci-dessous (Néon, Soleil doux, Ciné, Anamorphique, Texte lumineux, Bloom subtil). Laisse intacts l'affichage, la qualité, la saleté, l'anti-scintillement, l'unmult et le gamma |
| Rayon | `radius` (σ du niveau le plus large, en px) |
| Exposition | `exposure` (stops) |
| Diffusion (falloff) | `spread` (2 = 1/r²) |
| Saturation | `saturation` (%, 100 = inchangé, 0 = gris) |
| Mode de fusion | `blend` : Ajout ou Écran (le glow ne fait pas sauter les blancs) |
| Entrée › Seuil, Douceur du seuil | `threshold`, `threshold_smooth` |
| Entrée › Unmult | `unmult` |
| Entrée › Anti-scintillement | `temporal_smooth` (%) : mélange les images voisines (t ± 1 image) dans la source du glow |
| Forme › Ratio d'aspect | `aspect` |
| Forme › Rayon rouge / vert / bleu | `rgb_radius` |
| Teinte › Mode, couleurs cœur / milieu / halo | `tint` (1, 2 ou 3 couleurs) |
| Saleté de lentille › Texture, Intensité saleté | `GlowInputs::dirt` (calque étiré sur le calque de l'effet), `dirt_amount` (%) |
| Profondeur › Calque de profondeur | `GlowInputs::depth` (canal rouge, 0 = proche, 1 = loin ; étiré sur le calque ; zones transparentes = loin). Sans calque, tout le groupe est ignoré |
| Profondeur › Étalement selon la profondeur | `depth_spread` (0 à 1 : les sources lointaines diffusent plus large) |
| Profondeur › Occlusion | `occlusion` (les pixels plus proches masquent le glow des sources plus lointaines) |
| Profondeur › Inverser la profondeur | option du wrapper : `1 − rouge`, pour les passes Z où le blanc est proche |
| Rendu › Correction gamma | `gamma_correct` (à couper dans un projet linéaire 32 bpc) |
| Rendu › Tonemap ACES | `tonemap` |
| Rendu › Sigma de base, Qualité | `base_sigma`, `downsample` (Maximale 1x … Brouillon 8x) |
| Affichage | `view` : Résultat final, Glow seul, Source, Glow derrière (le glow passe sous le calque) |

Les nouveaux paramètres ont des identifiants disque ajoutés en fin de liste :
les projets enregistrés avec les versions précédentes s'ouvrent toujours.

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
- Le rendu est sur CPU, multithreadé. Le portage GPU (Metal) est en cours
  dans `gpu/metal/`.
- Sorties 8 et 16 bpc : dither ordonné 8×8 (ancré dans l'espace du calque)
  contre le banding des grands halos ; la couleur reste ≤ alpha.
- Anti-scintillement : l'effet demande les images t − 1 et t + 1
  (`PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT`) seulement quand le réglage est > 0.

## Limites connues

- Windows : le wrapper compile en principe, mais l'étape PiPL (PiPLtool →
  `.rc`) n'est pas encore branchée dans CMake.
