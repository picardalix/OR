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
    from advanced_outfit_analysis import ColorAnalyzer, SeasonalClassifier
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
            res = self.color.analyze(path)  # si ton ColorAnalyzer expose analyze(path)
        except Exception:
            res = None
        self._color_cache[path] = res
        return res

    def _color_harmony_score(self, q_path: str, c_path: str) -> float:
        """
        Essaie une méthode dédiée si dispo, sinon fallback: similarité entre histogrammes HSV.
        Retourne ~[0..1].
        """
        # méthode native si dispo
        if self.color:
            for name in ("harmony_score", "compatibility_score", "score"):
                fn = getattr(self.color, name, None)
                if callable(fn):
                    try:
                        val = float(fn(q_path, c_path))
                        # clamp
                        return max(0.0, min(1.0, val))
                    except Exception:
                        pass

        # fallback HSV
        try:
            from PIL import Image
            def _hist(path: str) -> np.ndarray:
                im = Image.open(path).convert("RGB").resize((128,128))
                hsv = np.array(im.convert("HSV"), dtype=np.float32) / 255.0
                H = hsv[...,0].reshape(-1)
                S = hsv[...,1].reshape(-1)
                V = hsv[...,2].reshape(-1)
                hH, _ = np.histogram(H, bins=24, range=(0,1), density=True)
                hS, _ = np.histogram(S, bins=8,  range=(0,1), density=True)
                hV, _ = np.histogram(V, bins=8,  range=(0,1), density=True)
                h = np.concatenate([hH, hS, hV]).astype(np.float32)
                h /= (np.linalg.norm(h) + 1e-8)
                return h
            hq = _hist(q_path); hc = _hist(c_path)
            sim = float(np.dot(hq, hc))
            return max(0.0, min(1.0, sim))
        except Exception:
            return 0.0

    def _season_pair(self, q_path: str, c_path: str) -> float:
        """
        1.0 si même 'color_temperature' si dispo, sinon 0.6 si différent.
        Fallback: proche moyenne H/S (grossier).
        """
        qc = self._analyze_color(q_path)
        cc = self._analyze_color(c_path)
        if qc and cc:
            t1 = getattr(qc, "color_temperature", "neutral")
            t2 = getattr(cc, "color_temperature", "neutral")
            return 1.0 if t1 == t2 else 0.6

        # fallback très simple: compare mean hue bucket
        (qS, _), (cS, _) = self._hsv_stats(q_path), self._hsv_stats(c_path)
        # si saturation très faible, on considère neutre (évite faux désaccords)
        if qS < 0.15 or cS < 0.15:
            return 1.0
        return 0.8  # neutre mais pas d'info -> quasi match
    
    # --- Texte / CLIP prompt ---
    def _color_token(self, path: str) -> str:
        """Retourne un nom de couleur grossier à partir de HSV (anglais pour CLIP)."""
        S, V = self._hsv_stats(path)
        try:
            from PIL import Image
            import numpy as np
            im = Image.open(path).convert("RGB").resize((64,64))
            hsv = np.array(im.convert("HSV"), dtype=np.float32) / 255.0
            H = float(hsv[...,0].mean())
        except Exception:
            H = 0.0
        if S < 0.15 and V > 0.8: return "white"
        if S < 0.15 and V < 0.25: return "black"
        if S < 0.15: return "gray"
        # hue → label
        h = H
        if h < 0.04 or h >= 0.96: return "red"
        if h < 0.08: return "orange"
        if h < 0.16: return "yellow"
        if h < 0.30: return "green"
        if h < 0.40: return "cyan"
        if h < 0.60: return "blue"
        if h < 0.75: return "purple"
        if h < 0.90: return "pink"
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
