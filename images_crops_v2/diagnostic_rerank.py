# -*- coding: utf-8 -*-
"""
diagnose_rerank.py
------------------
Imprime, pour une image requête donnée, le classement des 3 meilleures tenues :
1) AVANT préférences (baseline = score d'harmonie/cohérence)
2) APRÈS préférences avec λ faible (ex: 0.20)
3) APRÈS préférences avec λ élevé (ex: 0.60)

Optionnel : montre aussi l'effet d'une désactivation du poids "couleur" (w_color=0)
sur l'ordre baseline (diagnostic avant/après couleur).

Usage (exemple) :
    python diagnose_rerank.py \
        --ann /chemin/fashionpedia/annotations.json \
        --imgdir /chemin/images \
        --cropdir /chemin/crops \
        --query /chemin/ma_requete.jpg \
        --prefs "bas noir, chaussures blanches, chic" \
        --slots top bottom shoes \
        --lambda_lo 0.20 \
        --lambda_hi 0.60

Notes :
- Aucune interface, uniquement des prints de scores + chemins d'images (crop_path ou image_path).
- Ne modifie pas votre code; se contente d'appeler vos modules.
"""

from __future__ import annotations
import argparse
import copy
from typing import List, Optional, Dict, Any

# Vos modules
from fashion_recommendation_system import FashionRecommendationSystem
from user_preferences import UserPreferences, PreferenceScorer, apply_preference_rerank

def pretty_name(item: Any) -> str:
    return f"{getattr(item, 'group', '?')}/{getattr(item, 'category_name', '?')}#{getattr(item, 'id', '?')}"

def item_img_path(item: Any, crop_dir: Optional[str] = None) -> Optional[str]:
    # Reprend votre logique interne : crop_path > image_path > fallback crop_dir/id.jpg|jpeg
    import os
    from pathlib import Path
    p = getattr(item, 'crop_path', None) or getattr(item, 'image_path', None)
    if p and os.path.exists(p):
        return p
    if crop_dir and getattr(item, 'id', None):
        base = Path(crop_dir) / str(getattr(item, 'id'))
        for ext in ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG'):
            cand = base.with_suffix(ext)
            if cand.exists():
                return str(cand)
    return None

def print_outfits(title: str, outfits: List[Any], crop_dir: Optional[str] = None, topk: int = 3, show_components: bool = True):
    print(f"\n=== {title} ===")
    if not outfits:
        print("(aucune tenue)")
        return
    for rank, o in enumerate(outfits[:topk], start=1):
        adv = getattr(o, 'advanced_score', None)
        base = float(getattr(o, 'coherence_score', getattr(adv, 'overall_score', 0.0)) or 0.0)
        rnk = float(getattr(o, 'rank_score', base))
        pref = getattr(o, 'preference_score', None)
        delta = rnk - base
        # Affiche Δ et le vrai score de préférence s’il existe
        print(f"\n#{rank}  overall={base:.3f}  rank_score={rnk:.3f}  Δ={delta:+.3f}" + (f"  (pref={pref:.3f})" if pref is not None else ""))
        if show_components and adv is not None:
            # Ne PAS afficher un supposé "pref" dans les sous-composants (c’était trompeur)
            print(f"   • visual={getattr(adv, 'visual_similarity', 0.0):.3f}"
                  f"  • season={getattr(adv, 'seasonal_coherence', 0.0):.3f}"
                  f"  • color={getattr(adv, 'color_harmony', 0.0):.3f}"
                  f"  • style={getattr(adv, 'style_consistency', 0.0):.3f}")
        for it in getattr(o, 'items', []):
            path = item_img_path(it, crop_dir=crop_dir)
            print(f"   - {pretty_name(it)}  →  {path}")


def recompute_adv_scores_without_color(sys: FashionRecommendationSystem, outfits: List[Any]) -> List[Dict[str, float]]:
    """
    Recalcule, à titre diagnostic, le score avancé de chaque tenue avec w_color=0,
    sans toucher à l'objet d'origine. Retourne une liste de dicts {'overall_no_color': ...}.
    """
    from advanced_outfit_analysis import AdvancedOutfitScorer
    results = []
    scorer = sys.recommender.advanced_scorer  # instance existante
    old_weights = copy.deepcopy(getattr(scorer, 'weights', {}))
    try:
        # w_color = 0 ; renormalisation simple (optionnelle)
        w = scorer.weights
        tot = (w.get('visual_similarity',0)+w.get('seasonal_coherence',0)+w.get('color_harmony',0)+w.get('style_consistency',0))
        if tot <= 0: tot = 1.0
        # mettre couleur à 0 et renormaliser pour garder somme=1
        w_nc = {
            'visual_similarity': float(w.get('visual_similarity',0)),
            'seasonal_coherence': float(w.get('seasonal_coherence',0)),
            'color_harmony': 0.0,
            'style_consistency': float(w.get('style_consistency',0)),
        }
        s = w_nc['visual_similarity'] + w_nc['seasonal_coherence'] + w_nc['style_consistency']
        if s <= 0: s = 1.0
        w_nc = {k: (v/s) for k,v in w_nc.items()}
        scorer.weights = w_nc

        # Recalcule overall (les sous-scores restent ceux induits par vos analyzers)
        out = []
        for o in outfits:
            sims = [float(x) for x in getattr(o, 'similarities', [])]
            adv_nc = scorer.calculate_advanced_score(getattr(o, 'items', []), sims)
            out.append({
                'overall_no_color': float(getattr(adv_nc, 'overall_score', 0.0)),
                'color_harmony': float(getattr(adv_nc, 'color_harmony', 0.0)),
            })
        return out
    finally:
        # restauration
        scorer.weights = old_weights

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True, help='annotations Fashionpedia (json)')
    ap.add_argument('--imgdir', required=True, help='répertoire images source')
    ap.add_argument('--cropdir', required=True, help='répertoire de sortie/lecture des crops')
    ap.add_argument('--query', required=True, help='image requête (jpg/png)')
    ap.add_argument('--slots', nargs='+', default=['top', 'bottom', 'shoes'])
    ap.add_argument('--topk_per_slot', type=int, default=8)
    ap.add_argument('--beam_size', type=int, default=10)
    ap.add_argument('--max_outfits', type=int, default=3)
    ap.add_argument('--prefs', type=str, default='', help='texte libre en français')
    ap.add_argument('--lambda_lo', type=float, default=0.20)
    ap.add_argument('--lambda_hi', type=float, default=0.60)
    ap.add_argument('--show_color_diag', action='store_true')
    args = ap.parse_args()

    # 1) Charger/initialiser votre système
    sys = FashionRecommendationSystem(args.ann, args.imgdir, args.cropdir)
    sys.initialize(max_articles=2000) 

    # 2) Construire les tenues baseline (sans préférence)
    outfits_baseline = sys.recommender.recommend_outfits_builder(
        query_image_path=args.query,
        slots=args.slots,
        topk_per_slot=args.topk_per_slot,
        beam_size=args.beam_size,
        max_outfits=args.max_outfits,
        rules=None,
        locks=None,
        preferences=None,
        pref_weight=0.0,
    )

    print_outfits("Baseline (AVANT préférences)", outfits_baseline, crop_dir=args.cropdir, topk=args.max_outfits)

    # 3) Parser les préférences si fournies
    preferences = UserPreferences.from_text(args.prefs) if args.prefs else None

    if preferences:
        # 3a) Rerank λ faible
        outfits_lo = apply_preference_rerank(outfits_baseline.copy(), preferences, w_pref=args.lambda_lo)
        print_outfits(f"APRÈS préférences (λ={args.lambda_lo:.2f})", outfits_lo, crop_dir=args.cropdir, topk=args.max_outfits)

        # 3b) Rerank λ élevé
        outfits_hi = apply_preference_rerank(outfits_baseline.copy(), preferences, w_pref=args.lambda_hi)
        print_outfits(f"APRÈS préférences (λ={args.lambda_hi:.2f})", outfits_hi, crop_dir=args.cropdir, topk=args.max_outfits)

    # 4) (Optionnel) Diagnostic AVANT/APRÈS couleur sur la baseline
    if args.show_color_diag and outfits_baseline:
        diag = recompute_adv_scores_without_color(sys, outfits_baseline)
        print("\n--- Diagnostic couleur (baseline only) ---")
        for i, (o, d) in enumerate(zip(outfits_baseline, diag), start=1):
            base = float(getattr(o, 'coherence_score', getattr(getattr(o, 'advanced_score', None), 'overall_score', 0.0)) or 0.0)
            print(f"#{i}  overall_avec_couleur={base:.3f}   overall_sans_couleur={d['overall_no_color']:.3f}   (color_harmony={d['color_harmony']:.3f})")

if __name__ == '__main__':
    main()
