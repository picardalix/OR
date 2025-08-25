import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Optional
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
    
    def extract_dominant_colors(self, image_path: str, n_colors: int = 5) -> List[Tuple[int, int, int]]:
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

class AdvancedOutfitScorer:
    def __init__(self):
        self.weights = {
            "visual_similarity": 0.25,
            "seasonal_coherence": 0.25,
            "color_harmony": 0.25,
            "style_consistency": 0.25
        }
        self.color_analyzer = ColorAnalyzer()
        self.seasonal_classifier = SeasonalClassifier()
    
    def calculate_seasonal_coherence(self, seasonal_profiles: List[SeasonalProfile]) -> float:
        if len(seasonal_profiles) <= 1:
            return 1.0
        
        season_votes = defaultdict(float)
        for profile in seasonal_profiles:
            for season, score in profile.season_scores.items():
                season_votes[season] += score * profile.confidence
        
        dominant_season = max(season_votes, key=season_votes.get)
        
        coherence_scores = []
        for profile in seasonal_profiles:
            season_score = profile.season_scores.get(dominant_season, 0.0)
            coherence_scores.append(season_score)
        
        return np.mean(coherence_scores)
    
    def calculate_color_harmony(self, color_analyses: List[ColorAnalysis]) -> float:
        if len(color_analyses) <= 1:
            return 1.0
        
        temps = [analysis.color_temperature for analysis in color_analyses]
        temp_consistency = len(set(temps)) == 1
        
        individual_harmonies = [analysis.color_harmony_score for analysis in color_analyses]
        avg_individual_harmony = np.mean(individual_harmonies)
        
        saturations = [analysis.saturation_level for analysis in color_analyses]
        saturation_consistency = len(set(saturations)) <= 2
        
        temp_score = 1.0 if temp_consistency else 0.6
        saturation_score = 1.0 if saturation_consistency else 0.7
        
        return (temp_score * 0.4 + avg_individual_harmony * 0.4 + saturation_score * 0.2)
    
    def calculate_style_consistency(self, articles, color_analyses: List[ColorAnalysis]) -> float:
        if len(articles) <= 1:
            return 1.0
        
        formality_indicators = {
            "formal": ["blazer", "dress_shirt", "suit", "heel", "leather"],
            "casual": ["t-shirt", "jeans", "sneaker", "hoodie", "shorts"],
            "smart_casual": ["cardigan", "chino", "loafer", "blouse"]
        }
        
        style_scores = defaultdict(float)
        for article in articles:
            text = " ".join([article.category_name] + [str(attr) for attr in article.attributes]).lower()
            
            for style, keywords in formality_indicators.items():
                for keyword in keywords:
                    if keyword in text:
                        style_scores[style] += 1
        
        total_articles = len(articles)
        for style in style_scores:
            style_scores[style] /= total_articles
        
        if style_scores:
            max_style_score = max(style_scores.values())
            return max_style_score
        
        return 0.5
    
    def calculate_advanced_score(self, articles, similarities: List[float]) -> AdvancedOutfitScore:
        color_analyses = []
        seasonal_profiles = []
        
        for article in articles:
            crop_path = f"images/fashionpedia/crops_v2/{article.id}.jpg"
            
            color_analysis = self.color_analyzer.analyze_colors(crop_path)
            color_analyses.append(color_analysis)
            
            seasonal_profile = self.seasonal_classifier.classify_season(
                article.category_name, 
                [str(attr) for attr in article.attributes],
                color_analysis
            )
            seasonal_profiles.append(seasonal_profile)
        
        visual_similarity = np.mean(similarities) if similarities else 0.0
        seasonal_coherence = self.calculate_seasonal_coherence(seasonal_profiles)
        color_harmony = self.calculate_color_harmony(color_analyses)
        style_consistency = self.calculate_style_consistency(articles, color_analyses)
        
        scores = {
            "visual_similarity": visual_similarity,
            "seasonal_coherence": seasonal_coherence,
            "color_harmony": color_harmony,
            "style_consistency": style_consistency
        }
        
        overall_score = sum(score * self.weights[metric] for metric, score in scores.items())
        
        return AdvancedOutfitScore(
            overall_score=overall_score,
            visual_similarity=visual_similarity,
            seasonal_coherence=seasonal_coherence,
            color_harmony=color_harmony,
            style_consistency=style_consistency,
            breakdown=scores
        )

class OutfitFilter:
    def __init__(self, min_seasonal_coherence: float = 0.6, 
                 min_color_harmony: float = 0.5,
                 min_overall_score: float = 0.6):
        self.min_seasonal_coherence = min_seasonal_coherence
        self.min_color_harmony = min_color_harmony
        self.min_overall_score = min_overall_score
        self.scorer = AdvancedOutfitScorer()
    
    def filter_compatible_outfits(self, potential_outfits: List) -> List:
        filtered_outfits = []
        
        for outfit in potential_outfits:
            score = self.scorer.calculate_advanced_score(outfit.items, outfit.similarities)
            
            if (score.seasonal_coherence >= self.min_seasonal_coherence and 
                score.color_harmony >= self.min_color_harmony and
                score.overall_score >= self.min_overall_score):
                
                outfit.coherence_score = score.overall_score
                outfit.advanced_score = score
                filtered_outfits.append(outfit)
        
        return sorted(filtered_outfits, key=lambda x: x.coherence_score, reverse=True)