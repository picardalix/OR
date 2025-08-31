

import os
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageFilter, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

# Optional FAISS
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

# FashionCLIP
from fashion_clip.fashion_clip import FashionCLIP

from advanced_outfit_analysis import (
    OutfitFilter,
    AdvancedOutfitScorer,
    ColorAnalyzer,
    SeasonalClassifier
)

from user_preferences import UserPreferences, apply_preference_rerank

# -------------------- Data structures --------------------


@dataclass
class Article:
    """Représentation d'un article de mode"""
    id: str
    category_id: int
    category_name: str
    group: str
    attributes: List[int]
    image_path: str
    embedding: Optional[np.ndarray] = None
    bbox: Optional[List[float]] = None
    crop_path: Optional[str] = None


@dataclass
class Outfit:
    """Représentation d'une tenue complète"""
    items: List[Article]
    similarities: List[float]
    coherence_score: float = 0.0


# -------------------- Data Loader --------------------

class DataLoader:
    """Gestionnaire des données Fashionpedia"""

    def __init__(self, annotation_file: str, image_dir: str):
        self.annotation_file = annotation_file
        self.image_dir = Path(image_dir)
        self.annotations = None
        self.category_mapping = self._build_category_mapping()

    def _build_category_mapping(self) -> Dict[str, str]:
        """Mapping des catégories vers des groupes principaux"""
        return {
            # Tops
            "shirt": "top", "blouse": "top", "t-shirt": "top", "tank": "top",
            "sweater": "top", "cardigan": "top", "jacket": "top", "coat": "top",
            "blazer": "top", "hoodie": "top", "vest": "top", "top": "top",

            # Bottoms
            "pants": "bottom", "jeans": "bottom", "trousers": "bottom",
            "shorts": "bottom", "skirt": "bottom", "leggings": "bottom",
            "bottom": "bottom", "short": "bottom",

            # Shoes
            "shoe": "shoes", "boot": "shoes", "sandal": "shoes", "sneaker": "shoes",
            "heel": "shoes", "flat": "shoes", "pump": "shoes",

            # Accessories
            "hat": "accessory", "cap": "accessory", "scarf": "accessory",
            "glove": "accessory", "belt": "accessory", "tie": "accessory",

            # Bags
            "bag": "bag", "purse": "bag", "handbag": "bag", "backpack": "bag"
        }

    def load_annotations(self) -> Dict:
        """Charge les annotations depuis le fichier JSON"""
        with open(self.annotation_file, 'r') as f:
            self.annotations = json.load(f)
        return self.annotations

    def _map_category_to_group(self, category_name: str) -> Optional[str]:
        """Mappe une catégorie vers son groupe principal"""
        category_clean = category_name.lower().split(',')[0].strip()
        for keyword, group in self.category_mapping.items():
            if keyword in category_clean:
                return group
        return None

    def extract_articles(self, max_articles: int = 2000,
                         target_groups: List[str] = None) -> List[Article]:
        if not self.annotations:
            self.load_annotations()

        if target_groups is None:
            target_groups = ["top", "bottom", "shoes", "bag", "accessory"]

        category_map = {cat["id"]: cat["name"] for cat in self.annotations["categories"]}
        images_by_id = {img["id"]: img for img in self.annotations["images"]}

        articles: List[Article] = []
        
        for annotation in self.annotations["annotations"]:
            
            if len(articles) >= max_articles:
                break

            category_name = category_map.get(annotation["category_id"], "unknown")
            group = self._map_category_to_group(category_name)
            
            if group not in target_groups:
                continue

            image_info = images_by_id.get(annotation["image_id"])
            if not image_info:
                continue
                
            image_path = self.image_dir / image_info["file_name"]
            if not image_path.exists():
                continue

            article = Article(
                id=f"{annotation['image_id']}_{annotation['id']}",
                category_id=annotation["category_id"],
                category_name=category_name,
                group=group,
                attributes=annotation.get("attributes", []),
                image_path=str(image_path),
                bbox=annotation["bbox"]
            )
            articles.append(article)
        return articles


# -------------------- Image Processor --------------------

class ImageProcessor:
    """Processeur d'images avec cropping amélioré (cache sur disque)"""

    def __init__(self, crop_dir: str, target_size: int = 224):
        self.crop_dir = Path(crop_dir)
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        self.target_size = target_size

    def _validate_bbox(self, bbox: List[float], img_width: int, img_height: int,
                       min_size: int = 80) -> bool:
        x, y, w, h = bbox
        return (w >= min_size and h >= min_size and
                x >= 0 and y >= 0 and
                x + w <= img_width and y + h <= img_height)

    def _expand_bbox(self, bbox: List[float], img_width: int, img_height: int,
                     expansion_factor: float = 0.1) -> List[float]:
        x, y, w, h = bbox
        pad_x = max(10, int(w * expansion_factor))
        pad_y = max(10, int(h * expansion_factor))
        new_x = max(0, x - pad_x)
        new_y = max(0, y - pad_y)
        new_w = min(img_width - new_x, w + 2 * pad_x)
        new_h = min(img_height - new_y, h + 2 * pad_y)
        return [new_x, new_y, new_w, new_h]

    def _create_smart_crop(self, image: Image.Image, bbox: List[float]) -> Image.Image:
        x, y, w, h = [int(c) for c in bbox]
        crop_size = max(w, h)
        cx, cy = x + w // 2, y + h // 2
        half = crop_size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(image.width, x1 + crop_size)
        y2 = min(image.height, y1 + crop_size)
        if x2 - x1 < crop_size:
            x1 = max(0, x2 - crop_size)
        if y2 - y1 < crop_size:
            y1 = max(0, y2 - crop_size)
        crop = image.crop((x1, y1, x2, y2))
        if crop.size != (self.target_size, self.target_size):
            crop = crop.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        return crop

    def process_article(self, article: Article) -> Optional[str]:
        """Retourne le chemin du crop. Ne recroppe pas si déjà présent."""
        crop_path = self.crop_dir / f"{article.id}.jpg"
        if crop_path.exists():
            article.crop_path = str(crop_path)
            return str(crop_path)

        try:
            with Image.open(article.image_path) as img:
                img = img.convert('RGB')
                if not self._validate_bbox(article.bbox, img.width, img.height):
                    return None
                expanded_bbox = self._expand_bbox(article.bbox, img.width, img.height)
                crop = self._create_smart_crop(img, expanded_bbox)
                crop = crop.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
                crop.save(crop_path, 'JPEG', quality=85, optimize=True)
                article.crop_path = str(crop_path)
                return str(crop_path)
        except Exception as e:
            print(f"Erreur crop {article.id}: {e}")
            return None


# -------------------- Embedding Generator --------------------
class EmbeddingGenerator:
    """Générateur d'embeddings avec FashionCLIP"""

    def __init__(self, model_name: str = "patrickjohncyh/fashion-clip"):
        self.model = FashionCLIP(model_name)

    def _sanitize(self, emb: np.ndarray, eps: float = 1e-8) -> Optional[np.ndarray]:
        # Remplace NaN/Inf, clip, puis L2-normalise
        if emb is None:
            return None
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        # Par sécurité, on clip des valeurs aberrantes
        if np.max(np.abs(emb)) > 1e6:
            emb = np.clip(emb, -1e6, 1e6)
        norm = np.linalg.norm(emb)
        if not np.isfinite(norm) or norm < eps:
            return None
        return emb / norm

    def _encode_safe(self, paths: List[str], batch_size: int) -> List[Optional[np.ndarray]]:
        """Encode en lots; si un lot retourne des NaN, re-essaie en unitaire."""
        out: List[Optional[np.ndarray]] = [None] * len(paths)
        if not paths:
            return out

        # Premier essai: batch
        try:
            with torch.inference_mode():
                batch_embs = self.model.encode_images(paths, batch_size=batch_size)
        except Exception as e:
            print(f"[encode_safe] Erreur batch, fallback unitaire: {e}")
            batch_embs = None

        if isinstance(batch_embs, np.ndarray) and batch_embs.ndim == 2:
            # Format (N, D)
            for i in range(len(paths)):
                out[i] = self._sanitize(batch_embs[i])
        elif isinstance(batch_embs, (list, tuple)):
            for i, emb in enumerate(batch_embs):
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                out[i] = self._sanitize(emb)
        else:
            # rien de valide, on fera unitaire
            pass

        # Retry unitaire là où c'est None
        for i, p in enumerate(paths):
            if out[i] is not None:
                continue
            try:
                with torch.inference_mode():
                    one = self.model.encode_images([p], batch_size=1)
                if isinstance(one, list):
                    emb = one[0]
                elif isinstance(one, np.ndarray) and one.ndim == 2:
                    emb = one[0]
                else:
                    emb = None
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                out[i] = self._sanitize(emb)
            except Exception as e:
                print(f"[encode_safe] Echec unitaire {p}: {e}")
                out[i] = None

        return out

    def generate_embeddings(self, image_paths: List[str], batch_size: int = 16) -> List[Optional[np.ndarray]]:
        embeddings: List[Optional[np.ndarray]] = []

        missing_files = [p for p in image_paths if not os.path.exists(p)]

        # IMPORTANT : encoder seulement les fichiers existants
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            valid_mask = [os.path.exists(p) for p in batch_paths]
            valid_paths = [p for p, ok in zip(batch_paths, valid_mask) if ok]

            if not valid_paths:
                # Tous manquants → on renvoie None pour chaque
                embeddings.extend([None] * len(batch_paths))
                continue

            # Encode les *valid_paths* (et NON batch_paths)
            batch_out = self._encode_safe(valid_paths, batch_size=min(len(valid_paths), batch_size))

            # Ré-injecte dans la position d’origine (manquants -> None)
            it = iter(batch_out)
            for ok in valid_mask:
                if ok:
                    embeddings.append(next(it))
                else:
                    embeddings.append(None)

        return embeddings



# -------------------- ANN Index --------------------

class ANNIndex:
    """Index ANN (FAISS si dispo, sinon sklearn NearestNeighbors brute/cosine)"""

    def __init__(self, dim: int, backend: str = "faiss"):
        self.dim = dim
        self.backend = backend if backend in ("faiss", "sklearn") else "sklearn"
        self._faiss_index = None
        self._sk_index = None

    def fit(self, embeddings: np.ndarray):
        # embeddings doivent être L2-normalisés
        if self.backend == "faiss" and _HAS_FAISS:
            # Inner product == cosine si L2-norm
            self._faiss_index = faiss.IndexFlatIP(self.dim)
            self._faiss_index.add(embeddings.astype(np.float32))
        else:
            # fallback exact search
            self._sk_index = NearestNeighbors(
                n_neighbors=min(embeddings.shape[0], 200),
                algorithm="brute", metric="cosine"
            ).fit(embeddings)

    def search(self, query: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retourne (distances, indices) avec distances = (1 - cosine_similarity).
        """
        topk = max(1, topk)
        if self._faiss_index is not None:
            q = query.astype(np.float32).reshape(1, -1)
            sims, idxs = self._faiss_index.search(q, topk)  # sims in [-1, 1]
            sims = sims[0]
            idxs = idxs[0]
            dists = 1.0 - sims  # convertir en "distance" style sklearn
            return dists, idxs
        elif self._sk_index is not None:
            q = query.reshape(1, -1)
            dists, idxs = self._sk_index.kneighbors(q, n_neighbors=topk)
            return dists[0], idxs[0]
        else:
            # sécurité : aucun index construit
            return np.array([]), np.array([])


# -------------------- Similarity Engine --------------------

class SimilarityEngine:
    """Moteur de recherche de similarité avec ANN + pool + re-ranking"""

    def __init__(self, articles: List[Article], ann_backend: str = "faiss"):
        self.articles = articles
        self.ann_backend = ann_backend
        self.articles_by_group: Dict[str, List[int]] = self._group_articles()
        self.emb_by_group: Dict[str, np.ndarray] = {}
        self.ann_by_group: Dict[str, ANNIndex] = {}
        self._build_indices()

    def _group_articles(self) -> Dict[str, List[int]]:
        groups = defaultdict(list)
        for i, art in enumerate(self.articles):
            if art.embedding is not None:
                groups[art.group].append(i)
        return dict(groups)

    def _build_indices(self):
        self.emb_by_group.clear()
        self.ann_by_group.clear()
        for group, idxs in self.articles_by_group.items():
            if len(idxs) < 2:
                continue
            emb = np.vstack([self.articles[i].embedding for i in idxs])
            emb = normalize(emb, norm='l2')
            self.emb_by_group[group] = emb
            index = ANNIndex(emb.shape[1], backend=self.ann_backend)
            index.fit(emb)
            self.ann_by_group[group] = index

    def rebuild(self, ann_backend: Optional[str] = None):
        if ann_backend:
            self.ann_backend = ann_backend
        self._build_indices()

    # Compat pour l'ancien chemin (kNN simple)
    def find_similar_items(self, query_embedding: np.ndarray, target_group: str, k: int = 10) -> List[Tuple[float, int]]:
        pool = self.ann_search(query_embedding, target_group, topk=k)
        return [(sim, idx) for sim, idx, _ in pool]

    # Nouveau : ANN pool top-K
    def ann_search(self, query_embedding: np.ndarray, target_group: str, topk: int = 100) -> List[Tuple[float, int, np.ndarray]]:
        if target_group not in self.ann_by_group:
            return []
        
        q = normalize(query_embedding.reshape(1, -1), norm='l2')[0]
        dists, neigh = self.ann_by_group[target_group].search(q, topk)
        idxs = self.articles_by_group[target_group]
        out = []
        for d, j in zip(dists, neigh):
            if j < 0:
                continue
            art_idx = idxs[j]
            sim = 1.0 - float(d)
            emb = self.emb_by_group[target_group][j]
            out.append((sim, art_idx, emb))
        # tri décroissant par similarité brute
        out.sort(key=lambda x: x[0], reverse=True)
        return out


# -------------------- Re-ranking utils --------------------

def mmr_rerank(candidates: List[Tuple[float, int, np.ndarray]],
               selected_embs: List[np.ndarray],
               lambda_div: float = 0.3,
               topk: int = 10) -> List[int]:
    """
    candidates: [(sim_to_query, article_idx, emb), ...]
    selected_embs: liste d'embeddings déjà sélectionnés (peut être vide)
    Retourne une liste d'indices d'articles (article_idx) rerankés via MMR.
    """
    cand = candidates.copy()
    chosen: List[np.ndarray] = selected_embs.copy()
    order: List[int] = []

    while cand and len(order) < topk:
        best_i = -1
        best_score = -1e9
        for i, (sim_q, art_idx, emb) in enumerate(cand):
            if chosen:
                redundancy = max(float(np.dot(emb, s)) for s in chosen)
            else:
                redundancy = 0.0
            score = (1.0 - lambda_div) * sim_q - lambda_div * redundancy
            if score > best_score:
                best_score = score
                best_i = i
        _, sel_idx, sel_emb = cand.pop(best_i)
        order.append(sel_idx)
        chosen.append(sel_emb)
    return order


# -------------------- Outfit Builder --------------------

class OutfitBuilder:
    """Compose des tenues par slots via beam search + règles"""

    def __init__(self, scorer: AdvancedOutfitScorer, rules: Dict):
        self.scorer = scorer
        self.rules = rules

    def _penalty_rules(self, items: List[Article]) -> float:
        penalty = 0.0
        # Limiter outerwear à 1
        if self.rules.get("limit_outer", True):
            if sum(1 for it in items if it.group == "outer") > 1:
                penalty += 0.2
        # Eviter clash d'imprimés (exemple simple : attributs contiennent un flag  'pattern')
        if self.rules.get("avoid_print_clash", True):
            has_pattern = [("pattern" in str(it.attributes)) for it in items]
            if sum(has_pattern) >= 2:
                penalty += 0.1
        # Exemple: pénaliser running + gala si détectable (placeholder)
        if self.rules.get("penalize_running_gala", False):
            # Ici on aurait besoin de tags/attributes plus riches
            pass
        return penalty

    def generate(self,
                 slot_to_candidates: Dict[str, List[Article]],
                 beam_size: int = 10,
                 max_outfits: int = 3) -> List[Outfit]:
        # beam: liste de (items, score)
        beams: List[Tuple[List[Article], float]] = [([], 0.0)]
        slot_names = list(slot_to_candidates.keys())

        for slot in slot_names:
            next_beams: List[Tuple[List[Article], float]] = []
            cands = slot_to_candidates[slot]
            for items, base_score in beams:
                for cand in cands:
                    new_items = items + [cand]
                    adv = self.scorer.calculate_advanced_score(new_items, [1.0] * len(new_items))
                    score = float(adv.overall_score) - self._penalty_rules(new_items)
                    next_beams.append((new_items, score))
            next_beams.sort(key=lambda x: x[1], reverse=True)
            beams = next_beams[:max(1, beam_size)]

        beams.sort(key=lambda x: x[1], reverse=True)
        outfits: List[Outfit] = []
        for items, score in beams[:max_outfits]:
            outfits.append(Outfit(items=items, similarities=[1.0] * len(items), coherence_score=float(score)))
        return outfits


# -------------------- Recommender --------------------

class OutfitRecommender:
    """Système de recommandation d'outfits avec ANN + re-ranking + MMR + Builder"""

    def __init__(self, similarity_engine: SimilarityEngine,
                 pool_size: int = 100,
                 w_cosine: float = 0.6,
                 w_coherence: float = 0.3,
                 w_redundancy: float = 0.1,
                 mmr_lambda: float = 0.3,
                 topk_per_slot: int = 3,
                 min_seasonal: float = 0.30,
                 min_color: float = 0.30,
                 min_overall: float = 0.30):
        self.engine = similarity_engine
        self.advanced_scorer = AdvancedOutfitScorer()
        self.outfit_filter = OutfitFilter(
            min_seasonal_coherence=min_seasonal,
            min_color_harmony=min_color,
            min_overall_score=min_overall
        )
        self.pool_size = pool_size
        self.w_cos = w_cosine
        self.w_coh = w_coherence
        self.w_red = w_redundancy
        self.mmr_lambda = mmr_lambda
        self.topk_per_slot = topk_per_slot

    def set_quality_thresholds(self, min_seasonal: float, min_color: float, min_overall: float):
        self.outfit_filter.min_seasonal_coherence = min_seasonal
        self.outfit_filter.min_color_harmony = min_color
        self.outfit_filter.min_overall_score = min_overall

        
    def _detect_query_group(self, query_embedding: np.ndarray) -> Optional[str]:
        best_group, best_sim = None, 0.0
        for group in self.engine.articles_by_group.keys():
            pool = self.engine.ann_search(query_embedding, group, topk=1)
            if pool and pool[0][0] > best_sim:
                best_sim = pool[0][0]
                best_group = group
        return best_group if best_sim > 0.1 else None

    def detect_slot_from_image(self, image_path: str) -> Optional[str]:
        """Détecte le groupe (slot) dominant de l'image requête via ANN,
        en réutilisant _detect_query_group côté recommender."""
        # 1) Encodage de l'image -> embedding (uniquement EmbeddingGenerator)
        emb_gen = EmbeddingGenerator()
        embs = emb_gen.generate_embeddings([image_path])
        if not embs or embs[0] is None:
            return None

        emb = np.asarray(embs[0])

        # 2) Appel de la logique existante côté recommender
        if hasattr(self, "recommender") and self.recommender:
            return self.recommender._detect_query_group(emb)

        return None

    def _select_from_group(self, query_embedding: np.ndarray,
                       group: str,
                       selected_items: List[Article],
                       selected_embs: List[np.ndarray]) -> Optional[Article]:
        pool = self.engine.ann_search(query_embedding, group, topk=self.pool_size)
        if not pool:
            return None

        # Re-ranking par score combiné
        scored = []
        for sim_q, idx, emb in pool:
            cand = self.engine.articles[idx]
            tmp_items = selected_items + [cand]
            adv = self.advanced_scorer.calculate_advanced_score(tmp_items, [1.0] * len(tmp_items))
            coh = float(adv.overall_score)
            red = max((float(np.dot(emb, s)) for s in selected_embs), default=0.0)
            score = self.w_cos * sim_q + self.w_coh * coh - self.w_red * red
            scored.append((score, sim_q, idx, emb, coh))  # Ajout de coh

        scored.sort(key=lambda x: x[0], reverse=True)
        mmr_order = mmr_rerank([(sim_q, idx, emb) for _, sim_q, idx, emb, _ in scored],
                            selected_embs,
                            lambda_div=self.mmr_lambda,
                            topk=max(1, self.topk_per_slot))
        if not mmr_order:
            return None

        selected_art = self.engine.articles[mmr_order[0]]

        for score, sim_q, idx, emb, coh in scored:
            if idx == mmr_order[0]:
                # Calcul des scores individuels pour cet article
                crop_path = getattr(selected_art, "crop_path", selected_art.image_path)
                if crop_path and os.path.exists(crop_path):
                    try:
                        color_analysis = self.advanced_scorer.color_analyzer.analyze_colors(crop_path)
                        seasonal_profile = self.advanced_scorer.seasonal_classifier.classify_season(
                            selected_art.category_name, 
                            [str(attr) for attr in selected_art.attributes],
                            color_analysis
                        )
                        selected_art.scores = {
                            "cosine": sim_q,
                            "color": color_analysis.color_harmony_score,
                            "season": seasonal_profile.confidence
                        }
                    except Exception as e:
                        print(f"Erreur calcul scores individuels: {e}")
                        selected_art.scores = {"cosine": sim_q, "color": 0.0, "season": 0.0}
                else:
                    selected_art.scores = {"cosine": sim_q, "color": 0.0, "season": 0.0}
                break

        return selected_art

    def recommend_outfit(self,
    query_image_path: str,
    target_groups: Optional[List[str]] = None,
    similarity_threshold: float = 0.3,
    use_advanced_scoring: bool = True,
    preferences: Optional[UserPreferences] = None,
    pref_weight: float = 0.35
) -> Optional[Outfit]:
        """Version single-outfit """
        if target_groups is None:
            target_groups = ["top", "bottom", "shoes"]

        # Obtenir embedding du query
        emb_gen = EmbeddingGenerator()
        q_embs = emb_gen.generate_embeddings([query_image_path])
        if not q_embs or q_embs[0] is None:
            return None
        q = q_embs[0]

        # Détecter et exclure le slot du query
        detected_group = self._detect_query_group(q)
        if detected_group and len(target_groups) > 1:
            groups = [g for g in target_groups if g != detected_group]
        else:
            groups = target_groups

        selected: List[Article] = []
        selected_embs: List[np.ndarray] = []
        sims: List[float] = []

        for g in groups:
            art = self._select_from_group(q, g, selected, selected_embs)
            if art is None:
                continue
            selected.append(art)
            selected_embs.append(art.embedding)
            sims.append(float(np.dot(q, art.embedding)))

        if not selected:
            return None

        if use_advanced_scoring:
            adv = self.advanced_scorer.calculate_advanced_score(selected, sims)
            ok = (
                adv.seasonal_coherence >= self.outfit_filter.min_seasonal_coherence and
                adv.color_harmony     >= self.outfit_filter.min_color_harmony and
                adv.overall_score     >= self.outfit_filter.min_overall_score
            )
            if not ok:
                # --- Auto-relaxation progressive ---
                relax_steps = [
                    (0.25, 0.25, 0.28),
                    (0.20, 0.20, 0.25),
                    (0.15, 0.15, 0.22),
                    (0.10, 0.10, 0.20),
                ]
                for ms, mc, mo in relax_steps:
                    if (adv.seasonal_coherence >= ms and
                        adv.color_harmony     >= mc and
                        adv.overall_score     >= mo):
                        outfit = Outfit(items=selected, similarities=sims, coherence_score=float(adv.overall_score))
                        outfit.advanced_score = adv  # type: ignore[attr-defined]
                        outfit.relaxed_thresholds = (ms, mc, mo)  # aide UI
                        return outfit
                return None

            outfit = Outfit(items=selected, similarities=sims, coherence_score=float(adv.overall_score))
            outfit.advanced_score = adv  # type: ignore[attr-defined]

            # Si même la relaxation échoue, retourner le meilleur disponible
            if not outfit:
                for g in groups:
                    art = self._select_from_group(q, g, selected, selected_embs)
                    if art:
                        selected.append(art)
                        selected_embs.append(art.embedding)
                        sims.append(float(np.dot(q, art.embedding)))
                
                if selected:
                    outfit = Outfit(items=selected, similarities=sims, coherence_score=0.1)
                    outfit.fallback_mode = True  # type: ignore[attr-defined]
            # Appliquer le re-ranking par préférences si fourni
            pref_weight = float(max(0.0, min(1.0, pref_weight)))
            if preferences:
                try:
                    from user_preferences import PreferenceScorer
                    scorer = PreferenceScorer()
                    pref = float(scorer.outfit_preference_score(outfit, preferences))
                    outfit.preference_score = pref
                    base = float(getattr(outfit, "coherence_score", 0.0))
                    outfit.rank_score = (1.0 - pref_weight) * base + pref_weight * pref
                except Exception as e:
                    print(f"Erreur calcul préférences: {e}")  # Pour débugger
                    outfit.preference_score = 0.0
                    outfit.rank_score = float(getattr(outfit, "coherence_score", 0.0))
            else:
                outfit.preference_score = None
                outfit.rank_score = float(getattr(outfit, "coherence_score", 0.0))

            return outfit


    # --------- Mode Builder (plusieurs tenues possibles) ---------
    def recommend_outfits_builder(self,
    query_image_path: str,
    slots: List[str],
    topk_per_slot: int = 10,
    beam_size: int = 10,
    max_outfits: int = 3,
    rules: Optional[Dict] = None,
    locks: Optional[Dict[str, Any]] = None,
    preferences: Optional[UserPreferences] = None,
    pref_weight: float = 0.35
) -> List[Outfit]:
        
        """Construit des tenues par slots, applique filtres qualité et rerank préférences."""
     
        pref_weight = float(max(0.0, min(1.0, pref_weight)))

        emb_gen = EmbeddingGenerator()
        q_embs = emb_gen.generate_embeddings([query_image_path])
        if not q_embs or q_embs[0] is None:
            return []
        q = np.asarray(q_embs[0])
        # — 2) candidats par slot (ANN + MMR)
        slot_to_cands: Dict[str, List[Article]] = {}
        for slot in slots:
            pool = self.engine.ann_search(q, slot, topk=max(self.pool_size, topk_per_slot))
            if not pool:
                slot_to_cands[slot] = []
                continue
            order = mmr_rerank(pool, selected_embs=[], lambda_div=self.mmr_lambda, topk=topk_per_slot)
            cand_articles = [self.engine.articles[idx] for idx in order]

            # — 2.b) locks natifs: si un article est verrouillé pour ce slot, on le met en tête et on déduplique
            if locks and locks.get(slot):
                locked = locks[slot]
                # normaliser en Article si besoin (id -> Article)
                if not hasattr(locked, "id") and hasattr(self, "get_article"):
                    try:
                        locked = self.get_article(locked)
                    except Exception:
                        locked = None
                if locked:
                    lid = getattr(locked, "id", None)
                    cand_articles = [locked] + [a for a in cand_articles if getattr(a, "id", None) != lid]

            slot_to_cands[slot] = cand_articles[:topk_per_slot]

        # — 3) build
        builder = OutfitBuilder(self.advanced_scorer, rules or {
            "limit_outer": True,
            "avoid_print_clash": True,
            "penalize_running_gala": False,
        })
        outfits = builder.generate(slot_to_cands, beam_size=beam_size, max_outfits=max_outfits)

        # — 4) filtres qualité + auto-relax
        def pass_with(outf, ms, mc, mo):
            adv = self.advanced_scorer.calculate_advanced_score(outf.items, outf.similarities)
            return (adv.seasonal_coherence >= ms and adv.color_harmony >= mc and adv.overall_score >= mo), adv

        ms0 = self.outfit_filter.min_seasonal_coherence
        mc0 = self.outfit_filter.min_color_harmony
        mo0 = self.outfit_filter.min_overall_score
        relax_steps = [(ms0, mc0, mo0), (0.25,0.25,0.28), (0.20,0.20,0.25), (0.15,0.15,0.22), (0.10,0.10,0.20)]

        filtered: List[Outfit] = []
        for o in outfits:
            for ms, mc, mo in relax_steps:
                ok, adv = pass_with(o, ms, mc, mo)
                if ok:
                    o.coherence_score = float(adv.overall_score)
                    o.advanced_score = adv  # hint UI
                    if (ms,mc,mo) != (ms0,mc0,mo0):
                        o.relaxed_thresholds = (ms,mc,mo)
                    filtered.append(o)
                    break

        # — 5) rerank par préférences (avec le bon poids)
        if preferences and filtered:
            filtered = apply_preference_rerank(filtered, preferences, w_pref=pref_weight)
        else:
            filtered = sorted(filtered, key=lambda x: getattr(x, "coherence_score", 0.0), reverse=True)

        return filtered


    
    def top_candidates_by_slot(self, slot: str, query: str, k: int) -> List[Article]:
        emb_gen = EmbeddingGenerator()
        q_embs = emb_gen.generate_embeddings([query])
        if not q_embs or q_embs[0] is None:
            return []
        
        pool = self.engine.ann_search(q_embs[0], slot, topk=k)
        return [self.engine.articles[idx] for _, idx, _ in pool]



# -------------------- Top-level system --------------------

class FashionRecommendationSystem:
    """Système principal de recommandation de mode"""

    def __init__(self, annotation_file: str, image_dir: str, crop_dir: str,
                 ann_backend: str = "faiss"):
        self.data_loader = DataLoader(annotation_file, image_dir)
        self.image_processor = ImageProcessor(crop_dir)
        self.embedding_generator = EmbeddingGenerator()
        self.articles: List[Article] = []
        self.similarity_engine: Optional[SimilarityEngine] = None
        self.recommender: Optional[OutfitRecommender] = None
        self.ann_backend = ann_backend

    def initialize(self, max_articles: int = 2000):
        print("Chargement des données...")
        self.articles = self.data_loader.extract_articles(max_articles)
        self.advanced_scorer = AdvancedOutfitScorer()
        self.color_analyzer = ColorAnalyzer()
        self.season_classifier = SeasonalClassifier()
        print(f"Articles extraits: {len(self.articles)}")

        print("Traitement des images (crops en cache si existants)...")
        valid_articles: List[Article] = []
        crop_paths: List[str] = []
        for art in self.articles:
            crop_path = self.image_processor.process_article(art)
            if crop_path:
                valid_articles.append(art)
                crop_paths.append(crop_path)
        self.articles = valid_articles
        print(f"Images prêtes: {len(self.articles)}")

        print("Génération des embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(
            [getattr(a, "crop_path", a.image_path) for a in self.articles]
        )

        final_articles: List[Article] = []
        for article, emb in zip(self.articles, embeddings):
            if emb is not None:
                article.embedding = emb
                final_articles.append(article)
        self.articles = final_articles
        print(f"Articles avec embeddings: {len(self.articles)}")

        print("Construction des index ANN...")
        self.similarity_engine = SimilarityEngine(self.articles, ann_backend=self.ann_backend)
        self.recommender = OutfitRecommender(self.similarity_engine)
        print("Système initialisé ✔")

    def rebuild_ann_indices(self, ann_backend: str):
        """Reconstruit uniquement les index ANN (rapide)"""
        if not self.similarity_engine:
            raise RuntimeError("Engine non initialisé")
        self.similarity_engine.rebuild(ann_backend=ann_backend)
        self.ann_backend = ann_backend

    def get_recommendation(self,
    query_image_path: str,
    target_groups: Optional[List[str]] = None,
    similarity_threshold: float = 0.3,
    use_advanced_scoring: bool = True,
    preferences: Optional[UserPreferences] = None,
    pref_weight: float = 0.35
) -> Optional[Outfit]:
        if not self.recommender:
            raise RuntimeError("Système non initialisé. Appelez initialize() d'abord.")
        return self.recommender.recommend_outfit(
            query_image_path,
            target_groups,
            similarity_threshold,
            use_advanced_scoring,
            preferences,
            pref_weight=pref_weight
        )



    def get_recommendations_builder(self,
    query_image_path: str,
    slots: List[str],
    topk_per_slot: int = 10,
    beam_size: int = 10,
    max_outfits: int = 3,
    rules: Dict | None = None,
    locks: Dict | None = None,
    preferences: Optional[UserPreferences] = None,
    pref_weight: float = 0.35
):
        if not self.recommender:
            raise RuntimeError("Système non initialisé. Appelez initialize() d'abord.")
        try:
            return self.recommender.recommend_outfits_builder(
                query_image_path, slots, topk_per_slot, beam_size, max_outfits,
                rules, locks=locks, preferences=preferences, pref_weight=pref_weight
            )
        except TypeError:
            pass



    def get_outfit_analysis(self, outfit) -> dict:
        """
        Analyse une tenue en s'appuyant sur advanced_outfit_analysis:
        - ColorAnalyzer / SeasonalClassifier pour signaux par item
        - AdvancedOutfitScorer pour agrégats (cohérence saison, harmonie couleurs, etc.)
        - Similarité intra-tenue pour la redondance
        """
        assert hasattr(self, "advanced_scorer") and self.advanced_scorer is not None, "AdvancedOutfitScorer non initialisé"
        assert hasattr(self, "color_analyzer") and self.color_analyzer is not None, "ColorAnalyzer non initialisé"
        assert hasattr(self, "season_classifier") and self.season_classifier is not None, "SeasonalClassifier non initialisé"

        items = list(getattr(outfit, "items", []))

        # ---- helpers chemin image (prend crop_path sinon image_path, sinon CROP_DIR/id.jpg)
        def _image_path_for_item(it) -> str | None:
            p = getattr(it, "crop_path", None) or getattr(it, "image_path", None)
            if p and os.path.exists(p):
                return p
            # fallback sur CROP_DIR/<id>.jpg si dispo dans ton projet
            try:
                crop_dir = getattr(self, "crop_dir", None)
                if crop_dir and getattr(it, "id", None):
                    cand = str(Path(crop_dir) / f"{it.id}.jpg")
                    if os.path.exists(cand):
                        return cand
            except Exception:
                pass
            return None

        # ---- Similarités vs requête (si outfit.similarities pas fourni, on met 0.0 par item)
        similarities = list(getattr(outfit, "similarities", []))
        if not similarities or len(similarities) != len(items):
            similarities = [float(getattr(getattr(it, "scores", {}), "get", lambda *_: 0.0)("cosine")) if isinstance(getattr(it, "scores", {}), dict) else float(getattr(it, "scores", {}).get("cosine", 0.0)) for it in items]
            # si scores pas présents → 0.0
            similarities = [s if isinstance(s, (int, float)) else 0.0 for s in similarities]
            if len(similarities) != len(items):
                similarities = [0.0] * len(items)

        # ---- Analyses couleurs + profils saisonniers par item (avec tes classes)
        color_analyses = []
        seasonal_profiles = []
        for it in items:
            img_path = _image_path_for_item(it)
            # ColorAnalyzer a besoin d'une image lisible; si manquante on met des valeurs neutres
            if img_path and os.path.exists(img_path):
                try:
                    ca = self.color_analyzer.analyze_colors(img_path)
                except Exception:
                    # neutre si erreur lecture
                    ca = None
            else:
                ca = None

            if ca is None:
                # neutre : harmonie moyenne, température neutre, saturation balanced
                ca = type("CAProxy", (), {})()
                ca.dominant_colors = []
                ca.color_harmony_score = 0.5
                ca.color_temperature = "neutral"
                ca.saturation_level = "balanced"

            color_analyses.append(ca)

            # Seasonal profile à partir de la catégorie + attributs + analyse couleurs
            cat = getattr(it, "category_name", "") or ""
            attrs = [str(a) for a in getattr(it, "attributes", [])] if getattr(it, "attributes", None) is not None else []
            sp = self.season_classifier.classify_season(cat, attrs, ca)
            seasonal_profiles.append(sp)

        # ---- Score agrégé avancé (ta classe)
        adv = self.advanced_scorer.calculate_advanced_score(items, similarities)

        # ---- Saison majoritaire (label + confiance) à partir des profils
        #     (ton AdvancedOutfitScorer retourne seasonal_coherence mais pas le label dominant)
        votes = {}
        for sp in seasonal_profiles:
            for season, sc in sp.season_scores.items():
                votes[season] = votes.get(season, 0.0) + sc * sp.confidence
        if votes:
            season_label = max(votes, key=votes.get)
            # normalise une "confiance" simple
            total = sum(votes.values())
            season_conf = float(votes[season_label] / total) if total > 0 else float(adv.seasonal_coherence)
        else:
            season_label = "unknown"
            season_conf = float(adv.seasonal_coherence)

        # ---- Harmonie couleurs "OK ?" à partir de ton agrégat
        color_ok = bool(adv.color_harmony >= 0.5)
        color_pairs = []  # ton ColorAnalyzer ne retourne pas les paires; on laisse vide (UI sait gérer)

        # ---- Redondance intra-tenue = moyenne des cosines entre embeddings d’items
        #     (si tu as item.embedding, on l’utilise; sinon redondance=0)
        embs = []
        for it in items:
            v = getattr(it, "embedding", None)
            if v is not None:
                v = np.asarray(v)
                if np.isfinite(v).all() and v.size > 0:
                    n = np.linalg.norm(v)
                    if n > 1e-8:
                        embs.append(v / n)
        redundancy = 0.0
        if len(embs) >= 2:
            sims = []
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    sims.append(float(np.dot(embs[i], embs[j])))
            redundancy = float(np.mean(sims)) if sims else 0.0

        # ---- Scores par item (lis directement it.scores si présent)
        item_scores = []
        for it in items:
            s = getattr(it, "scores", {}) or {}
            item_scores.append({
                "item_id": getattr(it, "id", None),
                "slot": getattr(it, "group", None),
                "name": getattr(it, "category_name", None),
                "cosine": float(s.get("cosine", 0.0)) if isinstance(s, dict) else 0.0,
                "color_score": float(s.get("color", 0.0)) if isinstance(s, dict) else 0.0,
                "season_score": float(s.get("season", 0.0)) if isinstance(s, dict) else 0.0,
            })

        # ---- Texte brut (si tu veux conserver l’existant)
        raw_text = None
        if hasattr(self, "get_outfit_explanation"):
            try:
                raw_text = self.get_outfit_explanation(outfit)
            except Exception:
                raw_text = None

        return {
            "color_harmony": {"ok": color_ok, "pairs": color_pairs},
            "season": {"label": season_label, "confidence": season_conf},
            "redundancy": redundancy,
            "item_scores": item_scores,
            "advanced": {
                "overall": float(adv.overall_score),
                "visual_similarity": float(adv.visual_similarity),
                "seasonal_coherence": float(adv.seasonal_coherence),
                "color_harmony": float(adv.color_harmony),
                "style_consistency": float(adv.style_consistency),
            },
            "raw_text": raw_text,
        }


    # Explication textuelle
    def get_outfit_explanation(self, outfit: Outfit) -> str:
        if not hasattr(outfit, 'advanced_score'):
            return f"Tenue avec score de cohérence: {outfit.coherence_score:.2f}"
        score = outfit.advanced_score
        explanation = f"""
Score global: {score.overall_score:.2f}/1.0

Détail des scores:
• Similarité visuelle: {score.visual_similarity:.2f}
• Cohérence saisonnière: {score.seasonal_coherence:.2f}
• Harmonie des couleurs: {score.color_harmony:.2f}
• Consistance du style: {score.style_consistency:.2f}

Composition:
"""
        for item in outfit.items:
            explanation += f"• {item.group}: {item.category_name}\n"
        return explanation.strip()
