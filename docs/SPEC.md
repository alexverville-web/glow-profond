# Glow Profond : spécification fonctionnelle (salle blanche)

Objectif : un effet de glow pour After Effects, du même niveau que Deep Glow
(Plugin Everything), écrit de zéro pour Gimmick Studio.

## Méthode et limites

Cette spec est construite **sans désassembler Deep Glow**. Les fonctions
listées viennent de la documentation publique (pages Toolfarm / aescripts) ;
l'algorithme est une conception propre qui reproduit le comportement décrit.
Le binaire Deep Glow n'a pas été analysé : son EULA interdit très probablement
la rétro-ingénierie, et une réimplémentation inspirée de son code exposerait
le studio. Si une licence Deep Glow est disponible, on peut par contre
**comparer les rendus** (boîte noire) pour caler le look.

Ce que la documentation publique affirme (observé) :

| Fonction publique (Deep Glow / DG2) | Équivalent Glow Profond |
| --- | --- |
| Falloff « physiquement exact » en inverse du carré | Somme de gaussiennes par octave, poids égaux (voir plus bas) |
| Correction gamma, résultat linéaire même en espace non linéaire | `gamma_correct` : décodage sRGB → glow → réencodage |
| Seuil HDR avec lissage (moins de scintillement) | `threshold` + `threshold_smooth` (genou relatif au seuil, 0 à 1) |
| Spread, rayon, aspect ratio variable | `spread` (exposant p), `radius`, `aspect` |
| Aberration chromatique, multiplicateurs de rayon RGB | `rgb_radius` (σ par canal) |
| Teinte multicolore | `tint` : dégradé de couleurs du cœur vers le halo |
| Unmult (texte sur noir) | `unmult` : alpha = max(R,G,B) |
| Vue source / glow / rendu final | `view` |
| Contrôles de downsample et de qualité | `downsample`, `base_sigma` |
| Tonemapping cinématique | `tonemap` (courbe ACES ajustée) |
| Texture de saleté de lentille | `dirt_amount` + texture |
| Glow basé sur image, mode iris | Pas encore (phase 3) |
| 8/16/32 bpc, GPU requis | Cible du plugin (phase 2) |

Rien de public ne décrit l'implémentation interne de Deep Glow ; tout ce qui
suit est notre choix de conception.

## Pipeline

1. Unmult optionnel, puis linéarisation (sRGB → linéaire) si `gamma_correct`.
2. Seuil doux sur max(R,G,B), genou `k = t × threshold_smooth` (t = 0 laisse passer l'image intacte) : `soft = clamp(br - t + k, 0, 2k)² / 4k`,
   facteur = `max(soft, br - t) / br`.
3. Pyramide : niveau k à résolution 1/2^k, flou gaussien de σ_k = σ₀·2^k
   (donc σ constant en pixels du niveau : coût quasi indépendant du rayon).
   Tailles impaires arrondies au-dessus avec des zéros au-delà du bord, et
   remontée tente à l'échelle exacte 2 : le texel i du niveau k couvre
   toujours les pixels [2^k·i, 2^k·(i+1)), donc le halo reste centré quelle
   que soit la taille du buffer (l'hôte doit aligner l'origine du buffer sur
   un multiple de 2^(n−1) pour qu'il ne bouge pas du tout).
   Octaves continues : n_f = log2(radius / σ₀) + 1, n = ceil(n_f) niveaux ;
   le plus large a σ = radius exactement et son poids est multiplié par
   n_f − (n − 1), pour que le halo grandisse sans saut quand le rayon monte.
4. Poids w_k = (σ_k / σ₀)^(2−p), **non normalisés** : monter le rayon ajoute
   du halo plus loin sans assombrir celui qui est près de la source. Une somme de gaussiennes 2D normalisées
   à σ espacés en log, pondérées ainsi, donne une PSF ∝ r^−p entre σ₀ et le
   rayon max. **p = 2 → poids égaux par octave → falloff en 1/r².**
   p < 2 élargit le halo, p > 2 le resserre.
5. Par canal, σ multiplié par `rgb_radius[c]` ; en x/y par √aspect et 1/√aspect.
6. Chaque niveau est teinté (interpolation dans `tint`), puis la cascade tente
   les additionne du plus grossier au plus fin ; le tout × 2^exposure.
7. Composite en linéaire sur la source prémultipliée : `add` (src + g) ou
   `screen` (src + g·(1 − clamp(src, 0, 1)), ne dépasse jamais 1 si src, g ≤ 1).
   Tonemap optionnel, réencodage.

### Saleté de lentille, profondeur, occlusion

- **Saleté** (`dirt_amount`, texture H×W×3 linéarisée si `gamma_correct`) :
  glow += dirt_amount · dirt · G_large, où G_large est la somme non teintée
  des niveaux k ≥ n // 2, mêmes poids. Non soumise à l'occlusion (elle est
  sur la lentille).
- **Étalement par profondeur** (`depth_spread` ∈ [0, 1], carte 0 = proche,
  1 = loin) : une source de profondeur d envoie au niveau k
  w_k · lerp(1, 2^(k − n/2), s·d), renormalisé à 1 par source. Le facteur
  étant affine en d après division par Z(d) = 1 + s·d·Σ w_k c_k, on calcule
  deux pyramides a = src/Z et b = src·s·d/Z ; entrée du niveau k =
  a_k + c_k·b_k avec c_k = 2^(k − n/2) − 1.
- **Occlusion** (`occlusion`, `occlusion_softness` = ε) : deux canaux en plus
  dans la même cascade, I = moyenne RGB de l'entrée et I·d (σ sans
  multiplicateur RGB). d_moy = (I·d)_glow / I_glow, puis
  glow ×= 1 − smoothstep(0, ε, d_moy − d(p)).

## Implémentation de référence

`reference/glow.py` (NumPy seul) est la vérité terrain du look ;
`reference/demo.py` produit une planche de presets. Le plugin AE devra
reproduire ces images à une tolérance près (tests de régression image).

Remontée des niveaux : cascade du plus grossier au plus fin, un upsample
tente 3×3 ×2 par étape (Jimenez), environ 1,33× l'image en coût. Le σ de
chaque niveau est réduit de la variance ajoutée par sa chaîne d'upsamples, pour garder σ_k = σ₀·2^k et le falloff en 1/r² (pente mesurée −2,0 entre 32 et 64 px).

## Plan

1. **Phase 1 (fait)** : spec + référence Python, presets visuels.
2. **Phase 2** : plugin After Effects C++ (SDK AE, SmartFX, 32 bpc float),
   rendu GPU Metal sur Mac Apple Silicon et CUDA/DirectX sur Windows, chemin
   CPU de secours. Paramètres identiques à `GlowParams`.
3. **Phase 3** : saleté de lentille (texture × niveaux larges), glow basé sur
   image (convolution par FFT avec un noyau image), mode iris (noyau en
   polygone), stabilité temporelle du seuil.
4. Optionnel : calage du look par comparaison de rendus avec Deep Glow si une
   licence est disponible.
