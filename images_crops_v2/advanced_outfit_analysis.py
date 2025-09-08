import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Optional,Sequence
from dataclasses import dataclass
from collections import defaultdict
import colorsys
import os


@dataclass
class ColorAnalysis:
    dominant_colors: List[Tuple[int, int, int]]
    color_harmony_score: float
    color_temperature: str  # "warm", "cool", "neutral"
    saturation_level: str  # "muted", "vibrant", "balanced"

@dataclass
class SeasonalProfile:
    season_scores: Dict[str, float]
    predicted_season: str
    confidence: float
    material_indicators: List[str]

@dataclass
class AdvancedOutfitScore:
    overall_score: float
    visual_similarity: float
    seasonal_coherence: float
    color_harmony: float
    style_consistency: float
    breakdown: Dict[str, float]

class ColorAnalyzer:
    def __init__(self):
        self.color_temperatures = {
            'warm': [(255, 0, 0), (255, 165, 0), (255, 255, 0), (139, 69, 19)],
            'cool': [(0, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 0)],
            'neutral': [(128, 128, 128), (255, 255, 255), (0, 0, 0), (245, 245, 220)]
        }
    
    def extract_dominant_colors(self, image_path: str, n_colors: int = 4) -> List[Tuple[int, int, int]]:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (150, 150))
        
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        return [tuple(color) for color in colors]
    
    def calculate_color_temperature(self, colors: List[Tuple[int, int, int]]) -> str:
        warm_score = cool_score = neutral_score = 0
        
        for color in colors:
            r, g, b = color
            if r > g and r > b:
                warm_score += 1
            elif b > r and b > g:
                cool_score += 1
            else:
                neutral_score += 1
        
        scores = {'warm': warm_score, 'cool': cool_score, 'neutral': neutral_score}

        return max(scores, key=scores.get)
    
    def calculate_saturation_level(self, colors: List[Tuple[int, int, int]]) -> str:
        saturations = []
        for r, g, b in colors:
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            saturations.append(s)
        
        avg_saturation = np.mean(saturations)

        if avg_saturation < 0.3:
            return "muted"
        elif avg_saturation > 0.7:
            return "vibrant"
        else:
            return "balanced"
    
    def calculate_harmony_score(self, colors: List[Tuple[int, int, int]]) -> float:
        if len(colors) < 2:
            return 1.0
        
        hues = []
        for r, g, b in colors:
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            hues.append(h * 360)
        
        harmony_scores = []
        for i, hue1 in enumerate(hues):
            for hue2 in hues[i+1:]:
                diff = abs(hue1 - hue2)
                diff = min(diff, 360 - diff)

                if 15 <= diff <= 45:  # Analogues
                    harmony_scores.append(0.9)
                elif 150 <= diff <= 210:  # Complémentaires
                    harmony_scores.append(0.8)
                elif diff <= 15:  # Monochromatique
                    harmony_scores.append(0.7)
                else:
                    harmony_scores.append(0.4)

        return np.mean(harmony_scores) if harmony_scores else 0.5
    
    def analyze_colors(self, image_path: str) -> ColorAnalysis:
        dominant_colors = self.extract_dominant_colors(image_path)
        color_temperature = self.calculate_color_temperature(dominant_colors)
        saturation_level = self.calculate_saturation_level(dominant_colors)
        harmony_score = self.calculate_harmony_score(dominant_colors)
        return ColorAnalysis(
            dominant_colors=dominant_colors,
            color_harmony_score=harmony_score,
            color_temperature=color_temperature,
            saturation_level=saturation_level
        )

class SeasonalClassifier:
    def __init__(self):
        self.seasonal_keywords = {
            "summer": {
                "categories": ["tank", "shorts", "sandal", "flip", "swimwear", "sleeveless"],
                "materials": ["cotton", "linen", "silk", "light", "breathable"],
                "colors": ["bright", "neon", "white", "pastels"],
                "weight": 1.0
            },
            "winter": {
                "categories": ["coat", "sweater", "boot", "scarf", "glove", "parka", "jacket"],
                "materials": ["wool", "cashmere", "fleece", "thick", "heavy", "fur"],
                "colors": ["dark", "deep", "burgundy", "navy"],
                "weight": 1.0
            },
            "spring": {
                "categories": ["cardigan", "light_jacket", "sneaker", "blazer", "trench"],
                "materials": ["cotton", "denim", "light_wool"],
                "colors": ["pastel", "fresh", "green", "pink"],
                "weight": 0.8
            },
            "fall": {
                "categories": ["cardigan", "blazer", "ankle_boot", "sweater", "trench"],
                "materials": ["wool", "corduroy", "suede", "knit"],
                "colors": ["brown", "orange", "rust", "burgundy"],
                "weight": 0.8
            }
        }
    
    def extract_material_indicators(self, article_attributes: List[str], category_name: str) -> List[str]:
        text_to_analyze = " ".join(article_attributes + [category_name]).lower()
        materials = []
        
        for season_data in self.seasonal_keywords.values():
            for material in season_data["materials"]:
                if material in text_to_analyze:
                    materials.append(material)

        return list(set(materials))
    
    def calculate_seasonal_scores(self, category_name: str, attributes: List[str], 
                                color_analysis: ColorAnalysis) -> Dict[str, float]:
        scores = {}
        text_to_analyze = " ".join(attributes + [category_name]).lower()
        for season, season_data in self.seasonal_keywords.items():
            score = 0.0
            
            for keyword in season_data["categories"]:
                if keyword in text_to_analyze:
                    score += 0.4
            
            for material in season_data["materials"]:
                if material in text_to_analyze:
                    score += 0.3
            
            color_temp_bonus = 0.0
            if season in ["summer", "spring"] and color_analysis.color_temperature in ["warm", "neutral"]:
                color_temp_bonus = 0.15
            elif season in ["winter", "fall"] and color_analysis.color_temperature in ["cool", "neutral"]:
                color_temp_bonus = 0.15
            
            saturation_bonus = 0.0
            if season == "summer" and color_analysis.saturation_level == "vibrant":
                saturation_bonus = 0.15
            elif season == "winter" and color_analysis.saturation_level == "muted":
                saturation_bonus = 0.15
            
            final_score = min(1.0, score + color_temp_bonus + saturation_bonus)
            scores[season] = final_score * season_data["weight"]
        return scores
    
    def classify_season(self, category_name: str, attributes: List[str], 
                       color_analysis: ColorAnalysis) -> SeasonalProfile:
        season_scores = self.calculate_seasonal_scores(category_name, attributes, color_analysis)


        predicted_season = max(season_scores, key=season_scores.get)
        confidence = season_scores[predicted_season]

        material_indicators = self.extract_material_indicators(attributes, category_name)
        
        return SeasonalProfile(
            season_scores=season_scores,
            predicted_season=predicted_season,
            confidence=confidence,
            material_indicators=material_indicators
        )
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import itertools
import colorsys
import os

# On suppose que ces classes existent déjà dans ton projet :
# - ColorAnalyzer (avec .analyze_colors -> ColorAnalysis)
# - SeasonalClassifier (avec .classify_season -> SeasonalProfile)
# - ColorAnalysis(dominant_colors, color_harmony_score, color_temperature, saturation_level)
# - SeasonalProfile(season_scores, predicted_season, confidence, material_indicators)

# ---------------------- Score enrichi retourné par le scorer ----------------------

@dataclass
class AdvancedOutfitScore:
    # Scores "classiques"
    overall_score: float
    visual_similarity: float
    seasonal_coherence: float
    color_harmony: float
    style_consistency: float
    breakdown: Dict[str, float]

    # Champs "riches" utiles à l’UI / mémoire
    predicted_season: Optional[str] = None
    season_confidence: float = 0.0
    redundancy: float = 0.0
    harmonious_pairs: List[Dict[str, Any]] = None  # [{"a": id, "b": id, "type": "analogous|complementary"}]
    item_color_summaries: List[Dict[str, Any]] = None  # par item : id, category, dominant_colors, temperature, saturation, color_score
    item_seasonal_profiles: Dict[str, Dict[str, Any]] = None  # par item_id : saison, confiance, scores, indicateurs matériaux


# ---------------------- Helpers internes ----------------------

def _norm_vec(v: np.ndarray) -> np.ndarray:
    """Return a 1D, L2-normalized vector. Accepts (D,), (1,D) or (D,1)."""
    v = np.asarray(v, dtype=float)
    if v.ndim > 1:
        v = v.reshape(-1)  # enlève la dimension batch/colonne
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0.0:
        return v
    return v / (n + 1e-12)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity that tolerates (1, D) or (D, 1) vectors."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _compute_redundancy_from_items(articles) -> float:
    """
    Redondance = max cosinus entre paires d'items (0 si <2 items ou pas d'embeddings).
    """
    embs = []
    for it in articles:
        emb = getattr(it, "embedding", None)
        if emb is None:
            continue
        embs.append(_norm_vec(np.asarray(emb)))
    if len(embs) < 2:
        return 0.0
    max_cos = 0.0
    for a, b in itertools.combinations(embs, 2):
        c = float(np.dot(a, b))
        if c > max_cos:
            max_cos = c
    return max(0.0, min(1.0, max_cos))

def _rgb_to_hue_deg(rgb: Tuple[int, int, int]) -> float:
    r, g, b = [max(0.0, min(255.0, float(x))) / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (h * 360.0) % 360.0

def _angle_dist(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d

def _dominant_hue_from_analysis(analysis) -> Optional[float]:
    """
    Extrait une teinte dominante (hue en degrés) depuis ColorAnalysis.
    Prend la première couleur dominante si dispo.
    """
    if analysis is None:
        return None
    cols = getattr(analysis, "dominant_colors", None)
    if not cols:
        return None
    c0 = cols[0]
    try:
        c0 = tuple(int(x) for x in c0)
        return _rgb_to_hue_deg(c0)
    except Exception:
        return None

def _derive_color_pairs(color_analyses: List, items: List) -> List[Dict[str, Any]]:
    """
    Déduit des paires harmonieuses (analogues / complémentaires) par heuristique de teinte.
    """
    infos = []
    for it, ca in zip(items, color_analyses):
        hue = _dominant_hue_from_analysis(ca)
        infos.append((getattr(it, "id", None), hue))

    pairs = []
    for (ida, ha), (idb, hb) in itertools.combinations(infos, 2):
        if ha is None or hb is None:
            continue
        d = _angle_dist(ha, hb)
        if d <= 20.0:        # teintes proches -> analogue
            pairs.append({"a": ida, "b": idb, "type": "analogous", "delta_h": d})
        elif 160.0 <= d <= 200.0:  # ~opposées -> complémentaires
            pairs.append({"a": ida, "b": idb, "type": "complementary", "delta_h": d})
    return pairs

def _vote_season_from_profiles(seasonal_profiles) -> Tuple[Optional[str], float]:
    """
    Agrège les profils saisonniers item->tenue.
    - label = saison avec la somme de scores pondérés la plus haute
    - confidence = max des scores de cette saison (lecture intuitive)
    """
    if not seasonal_profiles:
        return None, 0.0

    season_votes = defaultdict(float)
    for sp in seasonal_profiles:
        conf = float(getattr(sp, "confidence", 0.0))
        for season, score in getattr(sp, "season_scores", {}).items():
            season_votes[season] += float(score) * conf

    if not season_votes:
        return None, 0.0

    label = max(season_votes.items(), key=lambda kv: kv[1])[0]
    # confiance agrégée = somme pondérée pour la saison gagnante, bornée [0,1]
    conf = float(max(0.0, min(1.0, season_votes[label])))
    return label, conf


# ---------------------- Scorer enrichi ----------------------

class AdvancedOutfitScorer:
    def __init__(self,
                 color_analyzer: Optional['ColorAnalyzer'] = None,
                 seasonal_classifier: Optional['SeasonalClassifier'] = None):
        self.weights = {
            "visual_similarity": 0.25,
            "seasonal_coherence": 0.25,
            "color_harmony": 0.25,
            "style_consistency": 0.25
        }
        # Single source of truth
        self.color_analyzer = color_analyzer or ColorAnalyzer()
        self.seasonal_classifier = seasonal_classifier or SeasonalClassifier()

    def calculate_seasonal_coherence(self, seasonal_profiles: List['SeasonalProfile']) -> float:
        if len(seasonal_profiles) <= 1:
            return 1.0

        season_votes = defaultdict(float)
        for profile in seasonal_profiles:
            conf = float(getattr(profile, "confidence", 0.0))
            for season, score in getattr(profile, "season_scores", {}).items():
                season_votes[season] += float(score) * conf

        if not season_votes:
            return 0.0

        dominant_season = max(season_votes, key=season_votes.get)
        coherence_scores = []
        for profile in seasonal_profiles:
            season_score = float(getattr(profile, "season_scores", {}).get(dominant_season, 0.0))
            coherence_scores.append(season_score)
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0

    def calculate_color_harmony(self, color_analyses: List['ColorAnalysis']) -> float:
        if len(color_analyses) <= 1:
            return 1.0

        temps = [getattr(a, "color_temperature", None) for a in color_analyses]
        temp_consistency = len(set(t for t in temps if t is not None)) == 1

        individual = [float(getattr(a, "color_harmony_score", 0.5)) for a in color_analyses]
        avg_individual_harmony = float(np.mean(individual)) if individual else 0.5

        sats = [getattr(a, "saturation_level", None) for a in color_analyses]
        sat_consistency = len(set(s for s in sats if s is not None)) <= 2

        temp_score = 1.0 if temp_consistency else 0.6
        saturation_score = 1.0 if sat_consistency else 0.7
        return float(temp_score * 0.4 + avg_individual_harmony * 0.4 + saturation_score * 0.2)

    def calculate_style_consistency(self, articles, color_analyses: List['ColorAnalysis']) -> float:
        if len(articles) <= 1:
            return 1.0

        formality_indicators = {
            "formal": ["blazer", "dress_shirt", "suit", "heel", "leather"],
            "casual": ["t-shirt", "jeans", "sneaker", "hoodie", "shorts"],
            "smart_casual": ["cardigan", "chino", "loafer", "blouse"]
        }

        style_scores = defaultdict(float)
        total_articles = len(articles)

        for article in articles:
            text = " ".join([getattr(article, "category_name", "")] +
                            [str(attr) for attr in getattr(article, "attributes", [])]).lower()
            for style, keywords in formality_indicators.items():
                for kw in keywords:
                    if kw in text:
                        style_scores[style] += 1.0

        if not style_scores:
            return 0.5

        for style in style_scores:
            style_scores[style] /= float(total_articles)
        return float(max(style_scores.values()))

    def _analyze_items(self, articles) -> Tuple[List['ColorAnalysis'], List['SeasonalProfile'], List[Dict[str, Any]]]:
        """
        Retourne (color_analyses, seasonal_profiles, item_color_summaries).
        """
        color_analyses: List['ColorAnalysis'] = []
        seasonal_profiles: List['SeasonalProfile'] = []
        item_color_summaries: List[Dict[str, Any]] = []

        for art in articles:
            # Choix de chemin : crop si dispo sinon full
            crop_path = (getattr(art, 'feature_path', None)
                        or getattr(art, 'crop_path', None)
                        or getattr(art, 'image_path', None))

            if not crop_path or not os.path.exists(crop_path):
                # Fallback robustes dans crops_v2: d’abord PNG puis JPG
                alt_png = f"images/fashionpedia/crops_v2/{getattr(art, 'id', '')}.png"
                alt_jpg = f"images/fashionpedia/crops_v2/{getattr(art, 'id', '')}.jpg"
                crop_path = alt_png if os.path.exists(alt_png) else (alt_jpg if os.path.exists(alt_jpg) else None)


            # Analyse couleur
            try:
                ca = self.color_analyzer.analyze_colors(crop_path) if crop_path else None
            except Exception:
                ca = None
            color_analyses.append(ca)

            # Profil saisonnier
            try:
                sp = self.seasonal_classifier.classify_season(
                    getattr(art, "category_name", "") or "",
                    [str(a) for a in getattr(art, "attributes", [])],
                    ca
                )
            except Exception:
                sp = None
            seasonal_profiles.append(sp)

            # Résumé par item pour affichage
            item_color_summaries.append({
                "item_id": getattr(art, "id", None),
                "category": getattr(art, "category_name", None),
                "dominant_colors": getattr(ca, "dominant_colors", None) if ca else None,
                "color_temperature": getattr(ca, "color_temperature", None) if ca else None,
                "saturation_level": getattr(ca, "saturation_level", None) if ca else None,
                "color_score": float(getattr(ca, "color_harmony_score", 0.0)) if ca else None,
            })

        return color_analyses, seasonal_profiles, item_color_summaries

    def calculate_advanced_score(self, articles, similarities: List[float]) -> AdvancedOutfitScore:
        # Analyses item-level
        color_analyses, seasonal_profiles, item_color_summaries = self._analyze_items(articles)

        # Scores numériques
        visual_similarity = float(np.mean(similarities)) if similarities else 0.0
        seasonal_coherence = float(self.calculate_seasonal_coherence([sp for sp in seasonal_profiles if sp]))
        color_harmony = float(self.calculate_color_harmony([ca for ca in color_analyses if ca]))
        style_consistency = float(self.calculate_style_consistency(articles, [ca for ca in color_analyses if ca]))

        scores = {
            "visual_similarity": visual_similarity,
            "seasonal_coherence": seasonal_coherence,
            "color_harmony": color_harmony,
            "style_consistency": style_consistency
        }
        overall_score = float(sum(score * self.weights[metric] for metric, score in scores.items()))

        # Signaux riches
        predicted_season, season_confidence = _vote_season_from_profiles([sp for sp in seasonal_profiles if sp])
        redundancy = _compute_redundancy_from_items(articles)
        harmonious_pairs = _derive_color_pairs(color_analyses, articles)

        # Profils saisonniers par item (pour affichage détaillé)
        per_item_seasons: Dict[str, Dict[str, Any]] = {}
        for art, sp in zip(articles, seasonal_profiles):
            iid = getattr(art, "id", None)
            if iid is None or sp is None:
                continue
            per_item_seasons[str(iid)] = {
                "predicted_season": getattr(sp, "predicted_season", None),
                "confidence": float(getattr(sp, "confidence", 0.0)),
                "season_scores": dict(getattr(sp, "season_scores", {})),
                "material_indicators": list(getattr(sp, "material_indicators", []))
            }

        return AdvancedOutfitScore(
            overall_score=overall_score,
            visual_similarity=visual_similarity,
            seasonal_coherence=seasonal_coherence,
            color_harmony=color_harmony,
            style_consistency=style_consistency,
            breakdown=scores,

            predicted_season=predicted_season,
            season_confidence=season_confidence,
            redundancy=redundancy,
            harmonious_pairs=harmonious_pairs,
            item_color_summaries=item_color_summaries,
            item_seasonal_profiles=per_item_seasons
        )


# ---------------------- Filtre enrichi + propagation dans outfit ----------------------

class OutfitFilter:
    def __init__(self, min_seasonal_coherence: float = 0.6,
                 min_color_harmony: float = 0.5,
                 min_overall_score: float = 0.6,
                 scorer: Optional[AdvancedOutfitScorer] = None):
        self.min_seasonal_coherence = float(min_seasonal_coherence)
        self.min_color_harmony = float(min_color_harmony)
        self.min_overall_score = float(min_overall_score)
        # Utilise le même scorer que le reste du système si fourni
        self.scorer = scorer or AdvancedOutfitScorer()

    def _propagate_to_outfit(self, outfit, adv: AdvancedOutfitScore) -> None:
        """
        Copie sur l’outfit les champs utiles pour l’UI (et garde la rétro-compatibilité).
        """
        outfit.coherence_score = float(getattr(adv, "overall_score", 0.0))
        outfit.advanced_score = adv

        # Champs "outfit-level" lisibles par _explain_outfit_safe
        outfit.season_label = getattr(adv, "predicted_season", None)
        outfit.season_confidence = float(getattr(adv, "season_confidence", 0.0))
        outfit.redundancy = float(getattr(adv, "redundancy", 0.0))
        outfit.harmonious_pairs = getattr(adv, "harmonious_pairs", []) or []

        # EN BONUS (optionnel) : exposer les résumés item-level sur outfit pour affichage
        outfit.item_color_summaries = getattr(adv, "item_color_summaries", None)
        outfit.item_seasonal_profiles = getattr(adv, "item_seasonal_profiles", None)

    def filter_compatible_outfits(self, potential_outfits: List) -> List:
        filtered_outfits = []

        for outfit in potential_outfits:
            adv = self.scorer.calculate_advanced_score(outfit.items, outfit.similarities)

            if (adv.seasonal_coherence >= self.min_seasonal_coherence and 
                adv.color_harmony >= self.min_color_harmony and
                adv.overall_score >= self.min_overall_score):

                self._propagate_to_outfit(outfit, adv)
                filtered_outfits.append(outfit)

        return sorted(filtered_outfits, key=lambda x: getattr(x, "coherence_score", 0.0), reverse=True)



def topk_similar(query_vec: np.ndarray, gallery: Sequence[np.ndarray], k: int = 5):
    sims = []
    for i, v in enumerate(gallery):
        if v is None:
            sims.append((-1.0, i))
        else:
            sims.append((cosine_sim(query_vec, v), i))
    sims.sort(reverse=True, key=lambda x: x[0])
    return sims[:k]