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
| Seuil HDR avec lissage (moins de scintillement) | `threshold` + `threshold_smooth` (soft knee) |
| Spread, rayon, aspect ratio variable | `spread` (exposant p), `radius`, `aspect` |
| Aberration chromatique, multiplicateurs de rayon RGB | `rgb_radius` (σ par canal) |
| Teinte multicolore | `tint` : dégradé de couleurs du cœur vers le halo |
| Unmult (texte sur noir) | `unmult` : alpha = max(R,G,B) |
| Vue source / glow / rendu final | `view` |
| Contrôles de downsample et de qualité | `downsample`, `base_sigma` |
| Tonemapping cinématique | `tonemap` (courbe ACES ajustée) |
| Texture de saleté de lentille, glow basé sur image, mode iris | Pas encore (phase 3) |
| 8/16/32 bpc, GPU requis | Cible du plugin (phase 2) |

Rien de public ne décrit l'implémentation interne de Deep Glow ; tout ce qui
suit est notre choix de conception.

## Pipeline

1. Unmult optionnel, puis linéarisation (sRGB → linéaire) si `gamma_correct`.
2. Seuil doux sur max(R,G,B), genou `k = t × threshold_smooth` (t = 0 laisse passer l'image intacte) : `soft = clamp(br - t + k, 0, 2k)² / 4k`,
   facteur = `max(soft, br - t) / br`.
3. Pyramide : niveau k à résolution 1/2^k, flou gaussien de σ_k = σ₀·2^k
   (donc σ constant en pixels du niveau : coût quasi indépendant du rayon).
   Nombre de niveaux : `floor(log2(radius / σ₀)) + 1`.
4. Poids w_k ∝ σ_k^(2−p), normalisés. Une somme de gaussiennes 2D normalisées
   à σ espacés en log, pondérées ainsi, donne une PSF ∝ r^−p entre σ₀ et le
   rayon max. **p = 2 → poids égaux par octave → falloff en 1/r².**
   p < 2 élargit le halo, p > 2 le resserre.
5. Par canal, σ multiplié par `rgb_radius[c]` ; en x/y par √aspect et 1/√aspect.
6. Chaque niveau est teinté (interpolation dans `tint`) puis suréchantillonné
   et accumulé ; le tout × 2^exposure.
7. Composite additif en linéaire sur la source prémultipliée, tonemap
   optionnel, réencodage.

## Implémentation de référence

`reference/glow.py` (NumPy seul) est la vérité terrain du look ;
`reference/demo.py` produit une planche de presets. Le plugin AE devra
reproduire ces images à une tolérance près (tests de régression image).

Défaut connu : le suréchantillonnage bilinéaire des niveaux les plus petits
laisse un léger escalier dans les halos très larges. À remplacer par un
upsample tente 3×3 (style dual filter / bloom CoD) ou bicubique.

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
