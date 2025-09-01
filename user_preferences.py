
# -*- coding: utf-8 -*-
"""
user_preferences.py
-------------------
Petit module autonome pour parser des préférences utilisateur en français
(texte libre + champs structurés) et scorer/reranker les tenues/articles
en fonction de ces préférences.

Intégration rapide :
    from user_preferences import UserPreferences, PreferenceScorer, apply_preference_rerank
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import re
import math

# Dépendances internes du projet
try:
    # pour l'analyse de couleurs et la saison prédite
    from advanced_outfit_analysis import ColorAnalyzer, SeasonalClassifier
except Exception as _e:  # pragma: no cover
    ColorAnalyzer = None  # type: ignore
    SeasonalClassifier = None  # type: ignore


# ------------------------- Utilitaires -------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

# Couleurs FR -> canonique
_COLOR_KEYWORDS: Dict[str, str] = {
    # basiques
    "noir": "noir", "noire": "noir",
    "blanc": "blanc", "blanche": "blanc",
    "gris": "gris", "grise": "gris", "anthracite": "gris",
    "rouge": "rouge", "bordeaux": "bordeaux",
    "orange": "orange", "corail": "orange",
    "jaune": "jaune", "moutarde": "jaune",
    "vert": "vert", "kaki": "kaki", "olive": "kaki", "émeraude": "vert", "emerald": "vert",
    "bleu": "bleu", "marine": "marine", "navy": "marine", "turquoise": "bleu",
    "violet": "violet", "lilas": "violet", "prune": "violet",
    "rose": "rose", "fuchsia": "rose",
    "marron": "marron", "brun": "marron", "chocolat": "marron",
    "beige": "beige", "sable": "beige", "écru": "beige",
    "doré": "doré", "or": "doré", "argent": "argent", "argenté": "argent",
    "multicolore": "multi", "imprimé": "imprime", "imprimée": "imprime",
}

# Saisons FR -> canonique
_SEASON_KEYWORDS: Dict[str, str] = {
    "hiver": "hiver", "hivernal": "hiver",
    "ete": "été", "été": "été", "estival": "été",
    "printemps": "printemps", "printanier": "printemps",
    "automne": "automne", "automnal": "automne",
    "mi-saison": "mi-saison", "midsaison": "mi-saison", "inter-saison": "mi-saison", "inter saison": "mi-saison",
}

# Styles (liste ouverte, purement heuristique)
_STYLE_KEYWORDS = {
    "sport": "sport", "sportif": "sport", "sportswear": "sport",
    "casual": "casual", "décontracté": "casual", "decontracte": "casual",
    "élégant": "chic", "elegant": "chic", "chic": "chic", "smart": "chic",
    "street": "street", "streetwear": "street",
    "bohème": "boheme", "boheme": "boheme",
    "minimaliste": "minimal", "minimal": "minimal",
    "formel": "formel", "business": "formel", "bureau": "formel",
    "soirée": "soiree", "soiree": "soiree",
}

# Vêtements/slots (FR) -> catégories simples (génériques) utilisées dans le code existant
_ITEM_KEYWORDS: Dict[str, str] = {
    # groupes
    "haut": "top", "top": "top", "chemise": "top", "chemisier": "top", "t-shirt": "top", "tee": "top",
    "pull": "top", "sweat": "top", "hoodie": "top", "gilet": "top", "cardigan": "top",
    "blazer": "top", "veste": "top", "manteau": "top", "doudoune": "top",
    "robe": "dress",
    "jupe": "bottom", "pantalon": "bottom", "jean": "bottom", "short": "bottom",
    "tailleur": "set",
    "chaussure": "shoes", "baskets": "shoes", "basket": "shoes", "sneakers": "shoes",
    "bottes": "shoes", "sandales": "shoes", "escarpins": "shoes",
    "sac": "accessories", "ceinture": "accessories", "chapeau": "accessories", "écharpe": "accessories", "foulard": "accessories",
}

COLOR_VOCAB = sorted(set(_COLOR_KEYWORDS.values()))
ITEM_VOCAB  = sorted(set([v for v in _ITEM_KEYWORDS.values()]))
SEASON_VOCAB = ["hiver","printemps","été","automne","mi-saison"]


def _unique(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


# ------------------------- Modèle de préférences -------------------------

@dataclass
class UserPreferences:
    """Modèle simple de préférences."""
    # Texte libre saisi par l'utilisateur
    text: Optional[str] = None

    # Champs structurés (tous facultatifs)
    desired_items: List[str] = field(default_factory=list)     # ex: ["top","bottom","dress"]
    colors: List[str] = field(default_factory=list)            # ex: ["noir","beige","marine"]
    season: Optional[str] = None                               # "hiver", "été", "printemps", "automne", "mi-saison"
    style: List[str] = field(default_factory=list)             # ex: ["chic","casual"]

    # Exclusions éventuelles
    exclude_items: List[str] = field(default_factory=list)
    exclude_colors: List[str] = field(default_factory=list)

    @staticmethod
    def from_text(text: str) -> "UserPreferences":
        """Parser minimaliste en FR pour extraire items, couleurs, saison, style."""
        t = _norm(text)
        items, colors, styles = [], [], []
        season: Optional[str] = None
        excl_items, excl_colors = [], []

        # Extractions simples par mots-clés
        for word, key in _ITEM_KEYWORDS.items():
            if re.search(rf"\b{re.escape(word)}s?\b", t):
                items.append(key)

        for word, key in _COLOR_KEYWORDS.items():
            if re.search(rf"\b{re.escape(word)}s?\b", t):
                colors.append(key)

        for word, key in _SEASON_KEYWORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", t):
                season = key

        for word, key in _STYLE_KEYWORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", t):
                styles.append(key)

        # Exclusions simples : "pas de X", "sans X"
        for word, key in _ITEM_KEYWORDS.items():
            if re.search(rf"(pas de|sans)\s+{re.escape(word)}", t):
                excl_items.append(key)
        for word, key in _COLOR_KEYWORDS.items():
            if re.search(rf"(pas de|sans)\s+{re.escape(word)}", t):
                excl_colors.append(key)

        return UserPreferences(
            text=text.strip(),
            desired_items=_unique(items),
            colors=_unique(colors),
            season=season,
            style=_unique(styles),
            exclude_items=_unique(excl_items),
            exclude_colors=_unique(excl_colors),
        )


# ------------------------- Scoring préférences -------------------------

class PreferenceScorer:
    """
    Calcule un score d'alignement aux préférences pour un Article ou un Outfit.
    - Catégorie / item voulu
    - Couleurs
    - Saison
    (Style laissé en TODO car dépend des attributs/labels dispo)
    """

    def __init__(self, color_analyzer=None, seasonal_classifier=None) -> None:
        self.color_analyzer = color_analyzer or (ColorAnalyzer() if ColorAnalyzer else None)
        self.seasonal_classifier = seasonal_classifier or (SeasonalClassifier() if SeasonalClassifier else None)
        self._ca_cache = {}  # cache ColorAnalysis par article

        self.w_item = 0.40
        self.w_color = 0.40
        self.w_season = 0.20

    # --------------------- Couleur: RGB -> nom FR simple ---------------------
    @staticmethod
    def _rgb_to_basic_name(rgb: Tuple[int, int, int]) -> str:
        """Map très simple via teinte/valeur/saturation -> nom FR canonique."""
        r, g, b = rgb
        # Conversion HSV "maison"
        mx, mn = max(r, g, b), min(r, g, b)
        v = mx / 255.0
        s = 0.0 if mx == 0 else (mx - mn) / mx
        # teinte approx
        if mx == mn:
            h = 0.0
        elif mx == r:
            h = (60 * ((g - b) / (mx - mn)) + 360) % 360
        elif mx == g:
            h = 60 * ((b - r) / (mx - mn)) + 120
        else:
            h = 60 * ((r - g) / (mx - mn)) + 240

        # gris/noir/blanc
        if s < 0.08:
            if v > 0.9:
                return "blanc"
            if v < 0.15:
                return "noir"
            return "gris"

        # teinte -> couleur
        if 0 <= h < 20 or 340 <= h <= 360:
            return "rouge"
        if 20 <= h < 45:
            return "orange"
        if 45 <= h < 70:
            return "jaune"
        if 70 <= h < 170:
            # vert à cyan
            if 150 <= h < 170:
                return "kaki"
            return "vert"
        if 170 <= h < 255:
            # bleu & marine
            if v < 0.35:
                return "marine"
            return "bleu"
        if 255 <= h < 290:
            return "violet"
        if 290 <= h < 340:
            return "rose"
        return "multi"

    # --------------------- Scores élémentaires ---------------------

    def _score_item(self, article, prefs: UserPreferences) -> float:
        if not prefs.desired_items and not prefs.exclude_items:
            return 0.0
        group = str(getattr(article, "group", "")).lower()
        cat = str(getattr(article, "category_name", "")).lower()
        wanted = (group in prefs.desired_items) or (cat in prefs.desired_items)
        unwanted = (group in prefs.exclude_items) or (cat in prefs.exclude_items)
        if unwanted and not wanted:
            return 0.0
        return 1.0 if wanted else 0.0

    def _score_color(self, article, prefs: UserPreferences) -> float:
        if not self.color_analyzer or not prefs.colors:
            return 0.0
        # On essaie d'utiliser un crop si possible
        path = getattr(article, "crop_path", None) or getattr(article, "image_path", None)
        if not path:
            return 0.0
        try:
            analysis = self.color_analyzer.analyze_colors(path, n_colors=4)
            # analysis peut être ColorAnalysis ou liste de couleurs suivant l'implémentation
            if hasattr(analysis, "dominant_colors"):
                cols = list(getattr(analysis, "dominant_colors"))
            else:
                cols = list(analysis)  # type: ignore
            names = {_norm(self._rgb_to_basic_name(tuple(map(int, c))))
                     for c in cols if c is not None}
            if not names:
                return 0.0
            desired = {c for c in prefs.colors}
            inter = names & desired
            # Jaccard simple
            score = len(inter) / float(len(names | desired))
            return float(max(0.0, min(1.0, score)))
        except Exception:
            return 0.0

    def _score_season(self, article, prefs: UserPreferences) -> float:
        if not self.seasonal_classifier or not prefs.season:
            return 0.0
        # Utilise nom de catégorie + attributs texte si présents
        cat = str(getattr(article, "category_name", ""))
        attrs = getattr(article, "attributes", [])
        attrs_txt = [str(a) for a in attrs]
        # Couleur pas nécessaire ici ; SeasonalClassifier la prend en compte si fournie
        try:
            # SeasonalClassifier.classify_season(category_name, attributes, color_analysis)
            pred = self.seasonal_classifier.classify_season(cat, attrs_txt, None)
            label = str(getattr(pred, "predicted_season", ""))
            if not label:
                return 0.0
            return 1.0 if _norm(label) == _norm(prefs.season) else 0.0
        except Exception:
            return 0.0

    # --------------------- Scores globaux ---------------------

    def article_preference_score(self, article, prefs: Optional[UserPreferences]) -> float:
        if not prefs:
            return 0.0
        s_item = self._score_item(article, prefs)
        s_col = self._score_color(article, prefs)
        s_sea = self._score_season(article, prefs)
        score = self.w_item * s_item + self.w_color * s_col + self.w_season * s_sea
        return float(max(0.0, min(1.0, score)))

    def outfit_preference_score(self, outfit, prefs: Optional[UserPreferences]) -> float:
        if not prefs:
            return 0.0
        items = list(getattr(outfit, "items", []))
        if not items:
            return 0.0
        # moyenne des scores par item
        scores = [self.article_preference_score(it, prefs) for it in items]
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))


# ------------------------- Rerank helper -------------------------
# Dans user_preferences.py, vérifiez que cette fonction existe et fonctionne :
def apply_preference_rerank(outfits, preferences, w_pref=0.25, color_analyzer=None, seasonal_classifier=None):
    scorer = PreferenceScorer(color_analyzer=color_analyzer, seasonal_classifier=seasonal_classifier)
    if not preferences or not outfits:
        return outfits
    
    scorer = PreferenceScorer()
    for outfit in outfits:
        try:
            pref_score = scorer.outfit_preference_score(outfit, preferences)
            outfit.preference_score = float(pref_score)
            base_score = float(getattr(outfit, "coherence_score", 0.0))
            outfit.rank_score = (1.0 - w_pref) * base_score + w_pref * pref_score
        except Exception as e:
            print(f"Erreur rerank préférences: {e}")
            outfit.preference_score = 0.0
            outfit.rank_score = float(getattr(outfit, "coherence_score", 0.0))
    
    # Trier par rank_score décroissant
    return sorted(outfits, key=lambda x: getattr(x, "rank_score", 0.0), reverse=True)