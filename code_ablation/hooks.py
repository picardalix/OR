# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json, math

import numpy as np

# Embeddings multi-backends (tu l'as dans fashion_recommendation_system.py)
from fashion_recommendation_system import EmbeddingGenerator

# Analyse optionnelle si tu as ces classes
try:
    from advanced_outfit_analysis import ColorAnalyzer, SeasonalClassifier, ColorAnalysis
    HAVE_ANALYSIS = True
except Exception:
    ColorAnalyzer = None  # type: ignore
    SeasonalClassifier = None  # type: ignore
    HAVE_ANALYSIS = False


class RecommenderWithAblations:
    """
    Reco simple avec:
      - Embeddings (F-CLIP / OpenCLIP / timm)
      - Règles: couleur, saison, luminance, texture
      - Calibration: z-score des similarités visuelles
      - Re-ranking MMR (diversité)
      - Poids appris (logistique) optionnels

    Paramètres __init__ supportés via --recommender_kwargs:
      - model_name: str (ex. "patrickjohncyh/fashion-clip", "openclip:ViT-B-32/laion2b_s34b_b79k", "timm:vit_base_patch16_224", "timm:convnext_base")
      - learned_weights_path: str|None -> JSON de poids appris (voir train_rule_weights.py)
    """
    # --------------- init / caches ---------------
    def __init__(self, **kwargs):
        model_name = kwargs.get("model_name", "patrickjohncyh/fashion-clip")
        self.emb = EmbeddingGenerator(model_name=model_name)

        self.color = ColorAnalyzer() if HAVE_ANALYSIS and ColorAnalyzer else None  # type: ignore
        self.season = SeasonalClassifier() if HAVE_ANALYSIS and SeasonalClassifier else None  # type: ignore

        self.learned_weights_path: Optional[str] = kwargs.get("learned_weights_path")
        self.learned: Optional[Dict] = None
        if self.learned_weights_path and Path(self.learned_weights_path).exists():
            try:
                self.learned = json.loads(Path(self.learned_weights_path).read_text(encoding="utf-8"))
            except Exception:
                self.learned = None

        self._emb_cache: Dict[str, np.ndarray] = {}
        self._color_cache: Dict[str, object] = {}

    # --------------- low-level utils ---------------
    def _embed(self, path: str) -> np.ndarray:
        if path in self._emb_cache:
            return self._emb_cache[path]
        vec = self.emb.encode_images([path])[0]
        self._emb_cache[path] = vec.astype(np.float32)
        return self._emb_cache[path]

    def _analyze_color(self, path: str):
        if not self.color:
            return None
        if path in self._color_cache:
            return self._color_cache[path]
        try:
            res = self.color.analyze_colors(path)  # -> ColorAnalysis
        except Exception:
            res = None
        self._color_cache[path] = res
        return res

    def _color_harmony_pair(self, qa: ColorAnalysis, ca: ColorAnalysis) -> float:
        # 1) harmonie de la palette combinée
        comb = (qa.dominant_colors[:3] + ca.dominant_colors[:3])
        base = float(self.color.calculate_harmony_score(comb))  # ~[0..1]

        # 2) bonus température
        temp_bonus = 0.15 if qa.color_temperature == ca.color_temperature else 0.0

        # 3) bonus saturation "compatible"
        compat = {
            ("vibrant","vibrant"), ("balanced","balanced"), ("muted","muted"),
            ("balanced","vibrant"), ("balanced","muted")
        }
        sat_pair = (qa.saturation_level, ca.saturation_level)
        sat_bonus = 0.10 if sat_pair in compat or sat_pair[::-1] in compat else 0.0

        return max(0.0, min(1.0, base + temp_bonus + sat_bonus))

  
    def _color_harmony_score(self, q_path: str, c_path: str) -> float:
        if not self.color:
            return 0.0
        qa = self._analyze_color(q_path)
        ca = self._analyze_color(c_path)
        if qa and ca:
            return self._color_harmony_pair(qa, ca)
        # Fallback si analyse indisponible
        return 0.5

                    
    def _season_pair(self, q_path: str, c_path: str) -> float:
        qa = self._analyze_color(q_path)
        ca = self._analyze_color(c_path)
        if qa and ca:
            return 1.0 if qa.color_temperature == ca.color_temperature else 0.6
        # fallback HSV si analyse KO
        (qS,_), (cS,_) = self._hsv_stats(q_path), self._hsv_stats(c_path)
        if qS < 0.15 or cS < 0.15:
            return 1.0
        return 0.8


    def _rgb_to_token(self, rgb: tuple[int,int,int]) -> str:
        """
        Map RGB (0..255) -> nom de couleur textuel robuste (anglais, pour CLIP).
        - Gère black/white/gray par faible saturation ou valeurs extrêmes
        - Utilise la teinte (H) pour discriminer 12+ couleurs
        - Détecte brown/beige/olive/navy via (S, V) + H
        """
        r, g, b = rgb
        # Normalisation 0..1
        rf, gf, bf = r/255.0, g/255.0, b/255.0
        mx, mn = max(rf, gf, bf), min(rf, gf, bf)
        c = mx - mn                     # chroma
        v = mx                          # value
        s = 0.0 if mx == 0 else c / mx  # saturation "HSV"

        # Cas neutres (ordre important)
        if v < 0.08:
            return "black"
        if s < 0.12 and v > 0.92:
            return "white"
        if s < 0.12:
            return "gray"

        # Hue en degrés [0, 360)
        if c == 0:
            h_deg = 0.0
        else:
            if mx == rf:
                h = ((gf - bf) / c) % 6
            elif mx == gf:
                h = ((bf - rf) / c) + 2
            else:
                h = ((rf - gf) / c) + 4
            h_deg = float(h * 60.0)

        # Heuristiques pour brown / beige / olive / navy (avant le mappage standard)
        # brown: teinte orange-jaune mais sombre/modérément saturée
        if 15 <= h_deg < 50 and v < 0.60 and s > 0.25:
            return "brown"
        # beige: proche des jaunes/oranges mais faible saturation et assez clair
        if 20 <= h_deg < 60 and v > 0.80 and s < 0.35:
            return "beige"
        # olive: jaune-vert peu saturé et pas très lumineux
        if 55 <= h_deg < 95 and s < 0.40 and v < 0.75:
            return "olive"
        # navy: bleu très sombre
        if 200 <= h_deg < 250 and v < 0.35:
            return "navy"

        # Mappage principal par plages de teinte
        # (Plages légèrement élargies pour être plus tolérantes)
        if (h_deg >= 345) or (h_deg < 15):
            base = "red"
        elif h_deg < 35:
            base = "orange"
        elif h_deg < 65:
            base = "yellow"
        elif h_deg < 90:
            base = "lime"      # entre yellow et green
        elif h_deg < 150:
            base = "green"
        elif h_deg < 180:
            base = "teal"
        elif h_deg < 195:
            base = "cyan"
        elif h_deg < 210:
            base = "sky"
        elif h_deg < 240:
            base = "blue"
        elif h_deg < 270:
            base = "indigo"
        elif h_deg < 300:
            base = "purple"
        elif h_deg < 330:
            base = "magenta"
        elif h_deg < 345:
            base = "pink"
        else:
            base = "red"  # fallback

        # Optionnel: raffiner avec luminosité pour "dark/ light"
        # (évite de renvoyer "dark white" etc.)
        if base not in {"black","white","gray","navy","brown","beige","olive"}:
            if v < 0.25:
                return f"dark {base}"
            if v > 0.88 and s < 0.55:
                # un peu plus doux / pastel si très clair & peu saturé
                return f"light {base}"

        return base

    # --- Texte / CLIP prompt ---
    
    def _color_token(self, path: str) -> str:
        qa = self._analyze_color(path)
        if qa and qa.dominant_colors:
            return self._rgb_to_token(qa.dominant_colors[0])
        # fallback HSV
        S, V = self._hsv_stats(path)
        if S < 0.15 and V > 0.8: return "white"
        if S < 0.15 and V < 0.25: return "black"
        if S < 0.15:               return "gray"
        return "brown"

    def _text_prompt_score(self, query_path: str, cand_vec: np.ndarray, cand_path: str) -> float:
        """
        Score texte→image avec un prompt simple dépendant des couleurs (q,c).
        Retourne 0.0 si le backend texte n'est pas dispo.
        """
        qcol = self._color_token(query_path)
        ccol = self._color_token(cand_path)
        prompt = f"a {ccol} clothing item that matches a {qcol} piece in an outfit"
        try:
            if hasattr(self.emb, "encode_texts"):
                t = self.emb.encode_texts([prompt])[0]
                t = t.astype(np.float32)
                t /= (np.linalg.norm(t) + 1e-8)
                return float(np.dot(t, cand_vec))
        except Exception:
            pass
        return 0.0

    
        # --- Pattern / Busyness ---
    def _pattern_busyness(self, path: str) -> float:
        """
        Renvoie ~[0..1] : 0 = très calme (peu de motifs), 1 = très chargé.
        Implémentation simple: normalise l'énergie de gradient (déjà calculée via _texture_energy).
        """
        try:
            t = self._texture_energy(path)  # ~[0..2]
            return float(max(0.0, min(1.0, t / 0.6)))  # 0.6 ≈ échelle empirique
        except Exception:
            return 0.0

    def _pattern_compat(self, q_path: str, c_path: str) -> float:
        """
        Compatibilité de motifs: 1 - (busy_q * busy_c).
        Si les deux sont très chargés -> score ↓ ; sinon score ≈ 1.
        """
        bq = self._pattern_busyness(q_path)
        bc = self._pattern_busyness(c_path)
        return float(1.0 - (bq * bc))


    def _visual_sims_zscore(self, q_vec: np.ndarray, pool_vecs: np.ndarray) -> np.ndarray:
        sims = pool_vecs @ q_vec
        mu, sd = float(sims.mean()), float(sims.std() + 1e-6)
        return (sims - mu) / sd

    # ----------- HSV / texture helpers -----------
    def _hsv_stats(self, path: str):
        """Retourne (meanS, meanV) dans [0..1]."""
        try:
            from PIL import Image
            im = Image.open(path).convert("RGB").resize((128,128))
            hsv = np.array(im.convert("HSV"), dtype=np.float32) / 255.0
            S = float(hsv[...,1].mean())
            V = float(hsv[...,2].mean())
            return S, V
        except Exception:
            return 0.0, 0.0

    def _avg_saturation(self, path: str) -> float:
        s, _ = self._hsv_stats(path)
        return s

    def _texture_energy(self, path: str) -> float:
        """Énergie de gradient approximative (~[0..2])."""
        try:
            from PIL import Image
            g = Image.open(path).convert("L").resize((128,128))
            x = np.asarray(g, dtype=np.float32) / 255.0
            dx = np.abs(np.diff(x, axis=1)).mean()
            dy = np.abs(np.diff(x, axis=0)).mean()
            return float(dx + dy)
        except Exception:
            return 0.0

    # ----------- MMR (diversité) -----------
    def _mmr_rerank_indices(self,
                            cand_idx: List[int],
                            base_score: np.ndarray,
                            vis_mat: np.ndarray,
                            lamb: float = 0.75,
                            out_k: int = 10) -> List[int]:
        """
        Maximal Marginal Relevance:
          val = λ * pertinence - (1-λ) * redondance_max
        """
        selected: List[int] = []
        remaining = cand_idx[:]

        # normalise pertinence parmi les candidats
        sub = np.array([base_score[i] for i in remaining], dtype=np.float32)
        mu, sd = float(sub.mean()), float(sub.std() + 1e-6)
        norm = {i: (base_score[i]-mu)/sd for i in remaining}

        idx2pos = {i:p for p,i in enumerate(remaining)}
        Vsub = vis_mat[[idx2pos[i] for i in remaining]][:,[idx2pos[i] for i in remaining]]

        while remaining and len(selected) < out_k:
            best_i, best_val = None, -1e9
            for i in remaining:
                if not selected:
                    redund = 0.0
                else:
                    pos_i = idx2pos[i]
                    pos_sel = [idx2pos[j] for j in selected]
                    redund = float(np.max(Vsub[pos_i, pos_sel]))
                val = lamb * norm[i] - (1.0 - lamb) * redund
                if val > best_val:
                    best_val, best_i = val, i
            selected.append(best_i)
            remaining.remove(best_i)
        return selected

    # ----------- Logistique -----------
    def _linear_from_learned(self, feats: Dict[str, float]) -> float:
        """
        Calcule w·x + b en respectant l'ordre des features appris.
        feats: dict avec au moins les clés présentes dans learned["feature_order"].
        """
        if not self.learned:
            raise RuntimeError("learned weights non chargés")
        order = self.learned.get("feature_order", ["clip","color","season","luminance","texture"])
        use_clip_z = bool(self.learned.get("use_clip_z", False))
        wmap = self.learned.get("weights", {})
        b = float(self.learned.get("bias", 0.0))
        # assemble le vecteur x dans le même ordre
        x = []
        for name in order:
            if name == "clip" and use_clip_z:
                x.append(float(feats.get("clip_z", feats.get("clip", 0.0))))
            else:
                x.append(float(feats.get(name, 0.0)))
        lin = float(sum(float(wmap.get(order[i], 0.0)) * x[i] for i in range(len(order))) + b)
        return lin

    
    # --------------- API principale ---------------
    def recommend(self,
                  query_path: str,
                  pool_paths: List[str],
                  k: int = 10,
                  weights: Optional[Dict[str, float]] = None,
                  use_mmr: bool = True,
                  mmr_top: int = 30,
                  mmr_lambda: float = 0.75) -> List[Tuple[str, float]]:

        if not pool_paths:
            return []

        # Embeddings batch
        qv = self._embed(query_path)
        pool_vecs = np.vstack([self._embed(p) for p in pool_paths])

        # Similarités visuelles
        clip_raw = (pool_vecs @ qv).astype(np.float32)            # cosine brut
        clip_z   = self._visual_sims_zscore(qv, pool_vecs)        # z-score (pool)

        # Règles: couleur + saison + luminance + texture
        q_sat = self._avg_saturation(query_path)  # 0..1
        # boost couleur: 0.5→1.0 selon saturation requête (si pas de poids appris)
        color_boost = 0.5 + 0.5 * min(1.0, q_sat / 0.35)

        col = np.zeros(len(pool_paths), dtype=np.float32)
        sea = np.zeros(len(pool_paths), dtype=np.float32)
        lum = np.zeros(len(pool_paths), dtype=np.float32)
        tex = np.zeros(len(pool_paths), dtype=np.float32)

        pat = np.zeros(len(pool_paths), dtype=np.float32)
        txt = np.zeros(len(pool_paths), dtype=np.float32)
        for i, cand in enumerate(pool_paths):
            pat[i] = float(self._pattern_compat(query_path, cand))
            # texte: utilise l'embedding déjà calculé pour cand
            txt[i] = float(self._text_prompt_score(query_path, pool_vecs[i], cand))


        # stats requête
        _, qV = self._hsv_stats(query_path)
        qTex = self._texture_energy(query_path)

        for i, cand in enumerate(pool_paths):
            ccol = self._color_harmony_score(query_path, cand)
            col[i] = float(ccol)
            sea[i] = float(self._season_pair(query_path, cand))

            # luminance
            _, cV = self._hsv_stats(cand)
            lum[i] = float(1.0 - abs(qV - cV))

            # texture
            cTex = self._texture_energy(cand)
            tex[i] = float(1.0 - min(1.0, abs(qTex - cTex) / 0.3))

        # ----- Score final -----
        if self.learned:
            # Score logistique appris (sans boost couleur pour rester fidèle à l'entraînement)
            base = np.zeros(len(pool_paths), dtype=np.float32)
            for i in range(len(pool_paths)):
                feats = {
                    "clip": float(clip_raw[i]),
                    "clip_z": float(clip_z[i]),
                    "color": float(col[i]),
                    "season": float(sea[i]),
                    "luminance": float(lum[i]),
                    "texture": float(tex[i]),
                }
                z = self._linear_from_learned(feats)
                base[i] = float(1.0 / (1.0 + math.exp(-z)))
        else:
            # Somme pondérée avec boost couleur adaptatif

            w = {"clip":1.0, "color":0.0, "season":0.0, "luminance":0.0, "texture":0.0,
                "pattern":0.0, "text":0.0}
            if weights:
                w.update({k2: float(v) for k2, v in weights.items() if k2 in w})

            base = (w["clip"]*clip_z
                    + (w["color"]*color_boost)*col
                    + w["season"]*sea
                    + w["luminance"]*lum
                    + w["texture"]*tex
                    + w["pattern"]*pat
                    + w["text"]*txt).astype(np.float32)


        # ----- MMR re-ranking -----
        if use_mmr and len(pool_paths) > 2:
            topN = min(mmr_top, len(pool_paths))
            idx_by_vis = np.argsort(-clip_z)[:topN].tolist()  # pré-sélection visuelle
            V = pool_vecs[idx_by_vis]
            vis_mat = (V @ V.T).astype(np.float32)            # cosines
            mmr_idx = self._mmr_rerank_indices(idx_by_vis, base, vis_mat,
                                               lamb=mmr_lambda, out_k=min(k, topN))
            chosen = set(mmr_idx)
            rest = [i for i in np.argsort(-base).tolist() if i not in chosen]
            order = mmr_idx + rest
        else:
            order = np.argsort(-base).tolist()

        order = order[:k]
        return [(pool_paths[i], float(base[i])) for i in order]
