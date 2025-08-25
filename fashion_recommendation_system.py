import numpy as np
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from PIL import Image, ImageFilter
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from fashion_clip.fashion_clip import FashionCLIP
from advanced_outfit_analysis import AdvancedOutfitScorer, OutfitFilter


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


@dataclass
class Outfit:
    """Représentation d'une tenue complète"""
    items: List[Article]
    similarities: List[float]
    coherence_score: float = 0.0


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
        """Extrait les articles des annotations"""
        if not self.annotations:
            self.load_annotations()
        
        if target_groups is None:
            target_groups = ["top", "bottom", "shoes"]
        
        # Créer le mapping des catégories
        category_map = {cat["id"]: cat["name"] for cat in self.annotations["categories"]}
        
        articles = []
        for annotation in self.annotations["annotations"]:
            if len(articles) >= max_articles:
                break
                
            category_name = category_map.get(annotation["category_id"], "unknown")
            group = self._map_category_to_group(category_name)
            
            if group not in target_groups:
                continue
            
            # Construire le chemin de l'image
            image_info = next(
                (img for img in self.annotations["images"] 
                 if img["id"] == annotation["image_id"]), None
            )
            
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


class ImageProcessor:
    """Processeur d'images avec cropping amélioré"""
    
    def __init__(self, crop_dir: str, target_size: int = 224):
        self.crop_dir = Path(crop_dir)
        self.crop_dir.mkdir(exist_ok=True)
        self.target_size = target_size
    
    def _validate_bbox(self, bbox: List[float], img_width: int, img_height: int,
                      min_size: int = 80) -> bool:
        """Valide une bounding box"""
        x, y, w, h = bbox
        return (w >= min_size and h >= min_size and 
                x >= 0 and y >= 0 and 
                x + w <= img_width and y + h <= img_height)
    
    def _expand_bbox(self, bbox: List[float], img_width: int, img_height: int,
                    expansion_factor: float = 0.1) -> List[float]:
        """Expand une bounding box avec padding intelligent"""
        x, y, w, h = bbox
        
        # Calculer l'expansion basée sur la taille de l'objet
        pad_x = max(10, int(w * expansion_factor))
        pad_y = max(10, int(h * expansion_factor))
        
        # Nouvelles coordonnées avec contraintes
        new_x = max(0, x - pad_x)
        new_y = max(0, y - pad_y)
        new_w = min(img_width - new_x, w + 2 * pad_x)
        new_h = min(img_height - new_y, h + 2 * pad_y)
        
        return [new_x, new_y, new_w, new_h]
    
    def _create_smart_crop(self, image: Image.Image, bbox: List[float]) -> Image.Image:
        """Crée un crop intelligent en gardant l'aspect ratio"""
        x, y, w, h = [int(coord) for coord in bbox]
        
        # Déterminer la taille du crop (carré basé sur la plus grande dimension)
        crop_size = max(w, h)
        
        # Centrer le crop
        center_x = x + w // 2
        center_y = y + h // 2
        
        half_size = crop_size // 2
        crop_x1 = max(0, center_x - half_size)
        crop_y1 = max(0, center_y - half_size)
        crop_x2 = min(image.width, crop_x1 + crop_size)
        crop_y2 = min(image.height, crop_y1 + crop_size)
        
        # Ajuster si on dépasse les limites
        if crop_x2 - crop_x1 < crop_size:
            crop_x1 = max(0, crop_x2 - crop_size)
        if crop_y2 - crop_y1 < crop_size:
            crop_y1 = max(0, crop_y2 - crop_size)
        
        # Extraire et redimensionner
        crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        
        if crop.size != (self.target_size, self.target_size):
            crop = crop.resize((self.target_size, self.target_size), 
                             Image.Resampling.LANCZOS)
        
        return crop
    
    def process_article(self, article: Article) -> Optional[str]:
        """Traite un article et retourne le chemin du crop"""
        crop_path = self.crop_dir / f"{article.id}.jpg"
        
        if crop_path.exists():
            return str(crop_path)
        
        try:
            with Image.open(article.image_path) as img:
                img = img.convert('RGB')
                
                if not self._validate_bbox(article.bbox, img.width, img.height):
                    return None
                
                # Expand la bbox
                expanded_bbox = self._expand_bbox(article.bbox, img.width, img.height)
                
                # Créer le crop
                crop = self._create_smart_crop(img, expanded_bbox)
                
                # Appliquer un léger sharpening pour améliorer la qualité
                crop = crop.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
                
                # Sauvegarder avec qualité optimisée
                crop.save(crop_path, 'JPEG', quality=85, optimize=True)
                
                return str(crop_path)
                
        except Exception as e:
            print(f"Erreur lors du traitement de {article.id}: {e}")
            return None


class EmbeddingGenerator:
    """Générateur d'embeddings avec FashionCLIP"""
    
    def __init__(self, model_name: str = "patrickjohncyh/fashion-clip"):
        self.model = FashionCLIP(model_name)
    
    def generate_embeddings(self, image_paths: List[str], 
                          batch_size: int = 16) -> List[Optional[np.ndarray]]:
        """Génère les embeddings pour une liste d'images"""
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            try:
                batch_embeddings = self.model.encode_images(batch_paths, batch_size=len(batch_paths))
                
                for emb in batch_embeddings:
                    if isinstance(emb, torch.Tensor):
                        emb = emb.cpu().numpy()
                    
                    # Normaliser l'embedding
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        embeddings.append(emb / norm)
                    else:
                        embeddings.append(None)
                        
            except Exception as e:
                print(f"Erreur lors de la génération des embeddings: {e}")
                embeddings.extend([None] * len(batch_paths))
        
        return embeddings


class SimilarityEngine:
    """Moteur de recherche de similarité"""
    
    def __init__(self, articles: List[Article]):
        self.articles = articles
        self.articles_by_group = self._group_articles()
        self.nn_models = {}
        self._build_indices()
    
    def _group_articles(self) -> Dict[str, List[int]]:
        """Groupe les articles par catégorie"""
        groups = defaultdict(list)
        for i, article in enumerate(self.articles):
            if article.embedding is not None:
                groups[article.group].append(i)
        return dict(groups)
    
    def _build_indices(self):
        """Construit les index de similarité pour chaque groupe"""
        for group, indices in self.articles_by_group.items():
            if len(indices) < 2:
                continue
            
            embeddings = np.vstack([self.articles[i].embedding for i in indices])
            embeddings = normalize(embeddings, norm='l2')
            
            n_neighbors = min(20, len(indices))
            self.nn_models[group] = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric='cosine',
                algorithm='brute'
            ).fit(embeddings)
    
    def find_similar_items(self, query_embedding: np.ndarray, 
                          target_group: str, k: int = 10) -> List[Tuple[float, int]]:
        """Trouve les items similaires dans un groupe donné"""
        if target_group not in self.nn_models:
            return []
        
        indices = self.articles_by_group[target_group]
        k_actual = min(k, len(indices))
        
        query_norm = normalize(query_embedding.reshape(1, -1), norm='l2')
        distances, neighbors = self.nn_models[target_group].kneighbors(
            query_norm, n_neighbors=k_actual
        )
        
        results = []
        for dist, neighbor_idx in zip(distances[0], neighbors[0]):
            similarity = 1 - dist  # Convertir distance cosine en similarité
            article_idx = indices[neighbor_idx]
            results.append((similarity, article_idx))
        
        return results

class OutfitRecommender:
    """Système de recommandation d'outfits"""
    
    def __init__(self, similarity_engine: SimilarityEngine):
        self.similarity_engine = similarity_engine
        self.advanced_scorer = AdvancedOutfitScorer()
        self.outfit_filter = OutfitFilter(
            min_seasonal_coherence=0.3,
            min_color_harmony=0.3,
            min_overall_score=0.3
        )
    
    def _calculate_outfit_coherence(self, articles: List[Article]) -> float:
        """ANCIEN : Calcule la cohérence globale d'une tenue (méthode simple)"""
        if len(articles) <= 1:
            return 1.0
        
        embeddings = [art.embedding for art in articles]
        similarities = []
        
        weights = {"top": 1.0, "bottom": 1.0, "shoes": 0.7, "accessory": 0.5, "bag": 0.5}
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                group1, group2 = articles[i].group, articles[j].group
                weight = min(weights.get(group1, 0.5), weights.get(group2, 0.5))
                weighted_sim = max(0, sim) * weight
                similarities.append(weighted_sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def recommend_outfit(self, query_image_path: str, 
                        target_groups: List[str] = None,
                        similarity_threshold: float = 0.3,
                        use_advanced_scoring: bool = True) -> Optional[Outfit]:
        """Recommande une tenue complète à partir d'une image"""
        if target_groups is None:
            target_groups = ["top", "bottom", "shoes"]
        
        embedding_gen = EmbeddingGenerator()
        query_embeddings = embedding_gen.generate_embeddings([query_image_path])
        
        if not query_embeddings or query_embeddings[0] is None:
            return None
        
        query_embedding = query_embeddings[0]
        
        # Détecter le groupe de l'image de référence
        detected_group = self._detect_query_group(query_embedding)
        
        if detected_group:
            target_groups = [g for g in target_groups if g != detected_group]
            print(f"Groupe détecté: {detected_group}, groupes restants: {target_groups}")
        
        # Trouver les meilleurs items pour chaque groupe
        outfit_items = []
        similarities = []
        
        for group in target_groups:
            similar_items = self.similarity_engine.find_similar_items(
                query_embedding, group, k=10
            )
            
            if not similar_items:
                continue
            
            for similarity, article_idx in similar_items:
                if similarity >= similarity_threshold:
                    article = self.similarity_engine.articles[article_idx]
                    outfit_items.append(article)
                    similarities.append(similarity)
                    break
        
        if not outfit_items:
            return None
        
        # NOUVEAU : Utiliser le scoring avancé si activé
        if use_advanced_scoring:
            advanced_score = self.advanced_scorer.calculate_advanced_score(outfit_items, similarities)
            
            # Vérifier si la tenue respecte les critères de qualité
            if (advanced_score.seasonal_coherence < self.outfit_filter.min_seasonal_coherence or
                advanced_score.color_harmony < self.outfit_filter.min_color_harmony or
                advanced_score.overall_score < self.outfit_filter.min_overall_score):
                
                print(f"Tenue rejetée - Score: {advanced_score.overall_score:.2f}, "
                      f"Cohérence saisonnière: {advanced_score.seasonal_coherence:.2f}, "
                      f"Harmonie couleurs: {advanced_score.color_harmony:.2f}")
                return None
            
            coherence_score = advanced_score.overall_score
            
            # Créer l'outfit avec les métadonnées avancées
            outfit = Outfit(
                items=outfit_items,
                similarities=similarities,
                coherence_score=coherence_score
            )
            
            # Ajouter les informations détaillées
            outfit.advanced_score = advanced_score
            
            return outfit
        
        else:
            # ANCIEN : Utiliser l'ancien système de scoring
            coherence_score = self._calculate_outfit_coherence(outfit_items)
            
            return Outfit(
                items=outfit_items,
                similarities=similarities,
                coherence_score=coherence_score
            )
    
    def recommend_multiple_outfits(self, query_image_path: str,
                                 target_groups: List[str] = None,
                                 n_outfits: int = 5,
                                 similarity_threshold: float = 0.3) -> List[Outfit]:
        """Génère plusieurs tenues alternatives avec scoring avancé"""
        if target_groups is None:
            target_groups = ["top", "bottom", "shoes"]
        
        embedding_gen = EmbeddingGenerator()
        query_embeddings = embedding_gen.generate_embeddings([query_image_path])
        
        if not query_embeddings or query_embeddings[0] is None:
            return []
        
        query_embedding = query_embeddings[0]
        detected_group = self._detect_query_group(query_embedding)
        
        if detected_group:
            target_groups = [g for g in target_groups if g != detected_group]
        
        potential_outfits = []
        
        # Générer plusieurs combinaisons
        for group in target_groups:
            similar_items = self.similarity_engine.find_similar_items(
                query_embedding, group, k=20  # Plus d'options
            )
            
            if not similar_items:
                continue
            
            # Prendre les 3 meilleurs items de ce groupe
            for similarity, article_idx in similar_items[:3]:
                if similarity >= similarity_threshold:
                    # Créer une tenue avec cet item comme base
                    outfit_items = [self.similarity_engine.articles[article_idx]]
                    similarities = [similarity]
                    
                    # Compléter avec les autres groupes
                    remaining_groups = [g for g in target_groups if g != group]
                    for other_group in remaining_groups:
                        other_items = self.similarity_engine.find_similar_items(
                            query_embedding, other_group, k=3
                        )
                        if other_items and other_items[0][0] >= similarity_threshold:
                            outfit_items.append(self.similarity_engine.articles[other_items[0][1]])
                            similarities.append(other_items[0][0])
                    
                    if len(outfit_items) >= 2:  # Au moins 2 pièces
                        potential_outfits.append(Outfit(
                            items=outfit_items,
                            similarities=similarities,
                            coherence_score=0.0
                        ))
        
        # Filtrer et scorer avec le système avancé
        filtered_outfits = self.outfit_filter.filter_compatible_outfits(potential_outfits)
        
        return filtered_outfits[:n_outfits]
    
    def _detect_query_group(self, query_embedding: np.ndarray) -> Optional[str]:
        """Détecte le groupe le plus probable de l'image de requête"""
        best_group = None
        best_similarity = 0.0
        
        for group in self.similarity_engine.articles_by_group.keys():
            similar_items = self.similarity_engine.find_similar_items(
                query_embedding, group, k=1
            )
            
            if similar_items and similar_items[0][0] > best_similarity:
                best_similarity = similar_items[0][0]
                best_group = group
        
        return best_group if best_similarity > 0.4 else None
    
    def get_outfit_explanation(self, outfit: Outfit) -> str:
        """Génère une explication textuelle de la tenue recommandée"""
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
        
        Composition de la tenue:
        """
        
        for item in outfit.items:
            explanation += f"• {item.group}: {item.category_name}\n"
        
        return explanation.strip()

class FashionRecommendationSystem:
    """Système principal de recommandation de mode"""
    
    def __init__(self, annotation_file: str, image_dir: str, crop_dir: str):
        self.data_loader = DataLoader(annotation_file, image_dir)
        self.image_processor = ImageProcessor(crop_dir)
        self.embedding_generator = EmbeddingGenerator()
        self.articles = []
        self.similarity_engine = None
        self.recommender = None
        
    def initialize(self, max_articles: int = 2000):
        """Initialise le système complet"""
        print("Chargement des données...")
        self.articles = self.data_loader.extract_articles(max_articles)
        print(f"Articles extraits: {len(self.articles)}")
        
        print("Traitement des images...")
        valid_articles = []
        crop_paths = []
        
        for article in self.articles:
            crop_path = self.image_processor.process_article(article)
            if crop_path:
                valid_articles.append(article)
                crop_paths.append(crop_path)
        
        self.articles = valid_articles
        print(f"Images traitées avec succès: {len(self.articles)}")
        
        print("Génération des embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(crop_paths)
        
        # Associer les embeddings aux articles
        final_articles = []
        for article, embedding in zip(self.articles, embeddings):
            if embedding is not None:
                article.embedding = embedding
                final_articles.append(article)
        
        self.articles = final_articles
        print(f"Articles avec embeddings: {len(self.articles)}")
        
        # Construire les index
        print("Construction des index de similarité...")
        self.similarity_engine = SimilarityEngine(self.articles)
        self.recommender = OutfitRecommender(self.similarity_engine)
        
        print("Système initialisé avec succès!")
    
    def get_recommendation(self, query_image_path: str, 
                          target_groups: List[str] = None,
                          similarity_threshold: float = 0.3, use_advanced_scoring: bool = True) -> Optional[Outfit]:
        """Interface principale pour obtenir des recommandations"""
        if not self.recommender:
            raise RuntimeError("Système non initialisé. Appelez initialize() d'abord.")
        

        return self.recommender.recommend_outfit(
            query_image_path, target_groups, similarity_threshold, use_advanced_scoring
        )

    
    def get_stats(self) -> Dict[str, int]:
        """Retourne des statistiques sur le système"""
        if not self.articles:
            return {}
        
        stats = defaultdict(int)
        for article in self.articles:
            stats[article.group] += 1
        
        return dict(stats)


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    ANN_FILE = "images/fashionpedia/instances_attributes_train2020.json"
    IMG_DIR = "images/fashionpedia/train"
    CROP_DIR = "images/fashionpedia/crops_v2"
    
    # Initialiser le système
    system = FashionRecommendationSystem(ANN_FILE, IMG_DIR, CROP_DIR)
    system.initialize(max_articles=1000)
    
    # Afficher les statistiques
    stats = system.get_stats()
    print("Statistiques du système:", stats)
    
    # Exemple de recommandation
    # query_image = "path/to/your/query/image.jpg"
    # outfit = system.get_recommendation(query_image)
    # 
    # if outfit:
    #     print(f"Tenue recommandée (cohérence: {outfit.coherence_score:.3f}):")
    #     for item, sim in zip(outfit.items, outfit.similarities):
    #         print(f"  {item.group}: {item.category_name} (similarité: {sim:.3f})")