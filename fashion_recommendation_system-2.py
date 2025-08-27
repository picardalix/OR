
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import os
import numpy as np
from PIL import Image

# ---------- Optional deps ----------
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    # pip install fashion-clip
    from fashion_clip.fashion_clip import FashionCLIP  # type: ignore
except Exception:
    FashionCLIP = None

# ----------------------------------

@dataclass
class Article:
    id: int
    image_path: str
    embedding: np.ndarray  # L2-normalized
    category: str
    color: str
    season: str
    meta: Dict[str, Any] = field(default_factory=dict)

DEFAULT_SLOTS = ["top", "bottom", "shoes"]

# -------- FAISS Index --------

class FAISSIndex:
    def __init__(self, dim: int):
        if not _HAS_FAISS:
            raise ImportError("faiss n'est pas installé. Installez faiss-cpu pour utiliser FAISSIndex.")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine/IP with normalized vectors
        self.ids: List[int] = []

    def add(self, vectors: np.ndarray, ids: List[int]):
        assert vectors.shape[1] == self.dim
        faiss.normalize_L2(vectors)  # safety
        self.index.add(vectors.astype(np.float32))
        self.ids.extend(ids)

    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        q = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        sims, idx = self.index.search(q, top_k)
        out: List[Tuple[int, float]] = []
        for i, s in zip(idx[0], sims[0]):
            if i == -1:
                continue
            out.append((self.ids[i], float(s)))
        return out

# -------- MMR --------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))

def mmr_rerank(candidates: List[Tuple[int, float]],
               item_vectors: Dict[int, np.ndarray],
               lambda_div: float = 0.3,
               top_k: int = 10) -> List[Tuple[int, float]]:
    selected: List[int] = []
    result: List[Tuple[int, float]] = []
    cand_ids = [cid for cid, _ in candidates]
    rel = {cid: score for cid, score in candidates}
    while cand_ids and len(result) < top_k:
        best_id, best_score = None, -1e9
        for cid in cand_ids:
            if not selected:
                div_penalty = 0.0
            else:
                sim_to_selected = max(cosine_sim(item_vectors[cid], item_vectors[sid]) for sid in selected)
                div_penalty = sim_to_selected
            score = (1 - lambda_div) * rel[cid] - lambda_div * div_penalty
            if score > best_score:
                best_id, best_score = cid, score
        selected.append(best_id)  # type: ignore[arg-type]
        cand_ids.remove(best_id)  # type: ignore[arg-type]
        result.append((best_id, rel[best_id]))  # type: ignore[index]
    return result

# -------- Category Matrix --------

def default_category_matrix(categories: List[str]) -> Dict[Tuple[str,str], float]:
    cats = sorted(set(categories + DEFAULT_SLOTS))
    comp: Dict[Tuple[str,str], float] = {}
    for a in cats:
        for b in cats:
            if a == b:
                comp[(a,b)] = 0.6
            else:
                s = set([a,b])
                if {"top","bottom"} <= s:
                    comp[(a,b)] = 0.95
                elif {"top","shoes"} <= s:
                    comp[(a,b)] = 0.8
                elif {"bottom","shoes"} <= s:
                    comp[(a,b)] = 0.85
                else:
                    comp[(a,b)] = 0.7
    return comp

# -------- FashionCLIP Encoder with cache --------

class FashionCLIPEncoder:
    """
    Wraps FashionCLIP with disk cache. Falls back to a deterministic mock if FashionCLIP/torch aren't available.
    """
    def __init__(self, model_name: str = "patrickjohncyh/fashion-clip", cache_file: Optional[str] = None, device: Optional[str] = None):
        self.cache_file = cache_file
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.dim: Optional[int] = None

        self.model = None
        self.device = device
        self._mock = False

        if FashionCLIP is None or torch is None:
            # graceful fallback
            self._mock = True
        else:
            try:
                self.model = FashionCLIP(model_name)
                if self.device is None:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.model = None
                self._mock = True

        if self.cache_file and os.path.exists(self.cache_file):
            self._load_cache()

        # initialize dim
        _d = self.encode_text("init-dim")
        self.dim = int(_d.shape[0])

    # ---- cache helpers ----
    def _load_cache(self):
        try:
            cache_data = np.load(self.cache_file, allow_pickle=True).item()
            self.embedding_cache = cache_data
            print(f"[FCLIP] Cache chargé: {len(self.embedding_cache)} entrées")
        except Exception as e:
            print(f"[FCLIP] Erreur chargement cache: {e}")
            self.embedding_cache = {}

    def _save_cache(self):
        if self.cache_file:
            try:
                np.save(self.cache_file, self.embedding_cache)
                print(f"[FCLIP] Cache sauvegardé: {len(self.embedding_cache)} entrées")
            except Exception as e:
                print(f"[FCLIP] Erreur sauvegarde cache: {e}")

    # ---- encoding ----
    def encode_text(self, text: str) -> np.ndarray:
        if self._mock or self.model is None:
            rng = np.random.default_rng(abs(hash(("text", text))) % (2**32))
            dim = self.dim or 512
            v = rng.standard_normal(dim).astype(np.float32)
        else:
            try:
                emb = self.model.encode_text([text])[0]
                if isinstance(emb, np.ndarray):
                    v = emb.astype(np.float32)
                else:
                    v = emb.detach().cpu().numpy().astype(np.float32)
            except Exception:
                # fallback mock if anything fails
                rng = np.random.default_rng(abs(hash(("text", text))) % (2**32))
                dim = self.dim or 512
                v = rng.standard_normal(dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-8)
        return v

    def generate_embeddings(self, image_paths: List[str], batch_size: int = 16, article_ids: Optional[List[str]] = None) -> List[Optional[np.ndarray]]:
        if article_ids and len(article_ids) != len(image_paths):
            article_ids = None
        embeddings: List[Optional[np.ndarray]] = []
        to_process: List[str] = []
        to_indices: List[int] = []

        # hit cache
        for i, p in enumerate(image_paths):
            key = article_ids[i] if article_ids else os.path.basename(p)
            if key in self.embedding_cache:
                embeddings.append(self.embedding_cache[key])
            else:
                embeddings.append(None)
                to_process.append(p)
                to_indices.append(i)

        if to_process:
            if self._mock or self.model is None:
                # mock: histogram fallback per image (deterministic)
                for idx, p in zip(to_indices, to_process):
                    img = Image.open(p).convert("RGB").resize((128,128))
                    arr = np.asarray(img, dtype=np.float32)
                    hist = []
                    for c in range(3):
                        h, _ = np.histogram(arr[..., c], bins=16, range=(0,255), density=True)
                        hist.append(h.astype(np.float32))
                    v = np.concatenate(hist, axis=0).astype(np.float32)
                    v /= (np.linalg.norm(v) + 1e-8)
                    embeddings[idx] = v
                    key = article_ids[idx] if article_ids else os.path.basename(p)
                    self.embedding_cache[key] = v
                # set dim based on produced vector
                if self.dim is None and embeddings[to_indices[0]] is not None:
                    self.dim = int(embeddings[to_indices[0]].shape[0])
            else:
                # real model batch encoding
                for i in range(0, len(to_process), batch_size):
                    batch_paths = to_process[i:i+batch_size]
                    batch_indices = to_indices[i:i+batch_size]
                    try:
                        batch_embeddings = self.model.encode_images(batch_paths, batch_size=len(batch_paths))
                        for j, (emb, idx) in enumerate(zip(batch_embeddings, batch_indices)):
                            if torch is not None and isinstance(emb, torch.Tensor):
                                emb = emb.detach().cpu().numpy()
                            v = emb.astype(np.float32)
                            v /= (np.linalg.norm(v) + 1e-8)
                            embeddings[idx] = v
                            key = article_ids[idx] if article_ids else os.path.basename(batch_paths[j])
                            self.embedding_cache[key] = v
                    except Exception as e:
                        print(f"[FCLIP] Erreur génération batch: {e}")
                        for idx in batch_indices:
                            embeddings[idx] = None
                # set dim
                for v in embeddings:
                    if v is not None:
                        self.dim = int(v.shape[0])
                        break
            self._save_cache()

        return embeddings

# -------- System --------

class FashionRecommendationSystem:
    def __init__(self, img_dir: str, encoder: Optional[FashionCLIPEncoder] = None, cache_file: Optional[str] = None):
        self.img_dir = Path(img_dir)
        self.encoder = encoder or FashionCLIPEncoder(cache_file=cache_file or str(self.img_dir / "emb_cache.npy"))
        self.dim = int(self.encoder.dim or 512)
        self.articles: List[Article] = []
        self.faiss_index: Optional[FAISSIndex] = None
        self.cat_matrix: Dict[Tuple[str,str], float] = {}

    def _infer_color(self, arr: np.ndarray) -> str:
        mean = arr.mean(axis=(0,1))
        idx = int(np.argmax(mean))  # 0=R,1=G,2=B
        return ["red","green","blue"][idx]

    def _assign_category(self, fname: str) -> str:
        f = fname.lower()
        if "top" in f: return "top"
        if "bottom" in f: return "bottom"
        if "shoe" in f or "sneaker" in f: return "shoes"
        # heuristic fallback
        return np.random.choice(["top","bottom","shoes"]).item()

    def _assign_season(self) -> str:
        return np.random.choice(["spring/summer","fall/winter"]).item()

    def initialize(self, max_articles: Optional[int] = None) -> None:
        self.img_dir.mkdir(parents=True, exist_ok=True)
        image_paths: List[Path] = []
        for ext in ("*.jpg","*.jpeg","*.png","*.webp","*.bmp"):
            image_paths.extend(self.img_dir.rglob(ext))
        image_paths = sorted(image_paths)
        if not image_paths:
            ensure_demo_images(self.img_dir, num=12)
            image_paths = sorted(self.img_dir.glob("*.png"))

        if max_articles is not None:
            image_paths = image_paths[:max_articles]

        # generate embeddings with cache
        paths_str = [str(p) for p in image_paths]
        embs = self.encoder.generate_embeddings(paths_str, batch_size=16, article_ids=[p.name for p in image_paths])

        arts: List[Article] = []
        cats: List[str] = []
        for i, (p, e) in enumerate(zip(image_paths, embs)):
            if e is None:
                # skip images that failed
                continue
            img = Image.open(p).convert("RGB").resize((128,128))
            arr = np.asarray(img, dtype=np.float32)
            color = self._infer_color(arr)
            cat = self._assign_category(p.name)
            season = self._assign_season()
            arts.append(Article(id=i, image_path=str(p), embedding=e, category=cat, color=color, season=season))
            cats.append(cat)
        self.articles = arts

        # build FAISS
        if _HAS_FAISS:
            self.dim = int(self.encoder.dim or arts[0].embedding.shape[0])
            self.faiss_index = FAISSIndex(self.dim)
            vectors = np.vstack([a.embedding for a in self.articles]).astype(np.float32)
            ids = [a.id for a in self.articles]
            self.faiss_index.add(vectors, ids)
        else:
            self.faiss_index = None

        # category matrix
        self.cat_matrix = default_category_matrix(cats)

    # --- ANN stage ---
    def _ann_candidates(self, query_vec: np.ndarray, top_k: int = 50) -> List[Tuple[int, float]]:
        if not self.faiss_index:
            raise RuntimeError("FAISS indisponible. Installez faiss-cpu pour l'ANN.")
        return self.faiss_index.search(query_vec, top_k)

    # --- Coherence ---
    def _coherence_score(self, ids: List[int]) -> float:
        if len(ids) <= 1:
            return 0.0
        items = [self.articles[i] for i in ids]
        embs = [x.embedding for x in items]
        s_emb = 0.0; cnt=0
        for i in range(len(embs)):
            for j in range(i+1,len(embs)):
                s_emb += cosine_sim(embs[i], embs[j]); cnt += 1
        s_emb = s_emb / max(1, cnt)
        s_cat = 0.0; cnt=0
        for i in range(len(items)):
            for j in range(i+1,len(items)):
                s_cat += self.cat_matrix.get((items[i].category, items[j].category), 0.7); cnt += 1
        s_cat = s_cat / max(1, cnt)
        return 0.6 * s_emb + 0.4 * s_cat

    # --- Public API ---
    def text_to_embedding(self, text: str) -> np.ndarray:
        v = self.encoder.encode_text(text)
        return v / (np.linalg.norm(v) + 1e-8)

    def recommend_outfit(self,
                         query: str,
                         constraints: Optional[Dict[str, str]] = None,
                         slots: Optional[List[str]] = None,
                         ann_topk: int = 60,
                         per_slot: int = 20,
                         lambda_div: float = 0.3) -> Tuple[List[Article], Dict[str, float], List[Tuple[str,float]]]:
        from fashion_recommendation_system import explain_reasons  # local

        if slots is None:
            slots = DEFAULT_SLOTS
        constraints = constraints or {}

        q_vec = self.text_to_embedding(query)
        ann = self._ann_candidates(q_vec, top_k=ann_topk)

        def match(a: Article) -> bool:
            ok = True
            if 'category' in constraints:
                ok &= (a.category == constraints['category'])
            if 'color' in constraints:
                ok &= (a.color == constraints['color'])
            if 'season' in constraints:
                ok &= (a.season == constraints['season'])
            return bool(ok)

        filtered = [(i, s) for i, s in ann if match(self.articles[i])]
        if not filtered:
            filtered = ann

        slot_items: Dict[str, List[Tuple[int, float]]] = {sl: [] for sl in slots}
        for i, s in filtered:
            cat = self.articles[i].category
            if cat in slot_items and len(slot_items[cat]) < per_slot:
                slot_items[cat].append((i, s))

        flat: List[Tuple[int,float]] = []
        for sl in slots:
            flat.extend(slot_items.get(sl, []))
        flat.sort(key=lambda x: x[1], reverse=True)

        item_vectors = {i: self.articles[i].embedding for i, _ in flat}
        mmr = mmr_rerank(flat, item_vectors, lambda_div=lambda_div, top_k=len(slots))
        selected_ids = [i for i, _ in mmr]
        selected = [self.articles[i] for i in selected_ids]

        relevance = float(np.mean([s for _, s in mmr])) if mmr else 0.0
        coherence = self._coherence_score(selected_ids)
        reasons = explain_reasons(selected, q_vec, self.cat_matrix)

        global_scores = {"relevance": relevance, "coherence": coherence, "mmr_lambda": lambda_div}
        return selected, global_scores, reasons

# -------- Explanations (shared) --------

def explain_reasons(items: List[Article], q_vec: np.ndarray, cat_matrix: Dict[Tuple[str,str], float]) -> List[Tuple[str,float]]:
    if not items:
        return []
    align = np.mean([cosine_sim(q_vec, it.embedding) for it in items])
    colors = [it.color for it in items]
    color_harmony = 1.0 if len(set(colors)) == 1 else 0.6
    if len(items) == 1:
        texture = 0.0
        catc = 0.0
    else:
        embs = [it.embedding for it in items]
        sim_sum = 0.0; cnt=0
        for i in range(len(embs)):
            for j in range(i+1,len(embs)):
                sim_sum += cosine_sim(embs[i], embs[j]); cnt += 1
        texture = sim_sum / max(1, cnt)
        cats = [it.category for it in items]
        csum=0.0; cnt=0
        for i in range(len(cats)):
            for j in range(i+1,len(cats)):
                csum += cat_matrix.get((cats[i], cats[j]), 0.7); cnt += 1
        catc = csum / max(1, cnt)
    raw = {
        "Alignement texte↔image": float(align),
        "Harmonie des couleurs": float(color_harmony),
        "Similarité de texture": float(texture),
        "Compatibilité catégories": float(catc),
    }
    vals = np.array(list(raw.values()), dtype=np.float32)
    if vals.ptp() > 1e-6:
        vals = (vals - vals.min()) / (vals.ptp() + 1e-8)
    else:
        vals = np.ones_like(vals)
    ranked = sorted(zip(raw.keys(), vals.tolist()), key=lambda x: x[1], reverse=True)[:3]
    return ranked

# -------- Demo images --------

def ensure_demo_images(dest_dir: Path, num: int = 12) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = list(dest_dir.glob("*.png"))
    if len(existing) >= num:
        return
    rng = np.random.default_rng(42)
    for i in range(num):
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        img = Image.fromarray(np.tile(color, (128, 128, 1)), mode="RGB")
        cat = np.random.choice(["top","bottom","shoes"])
        img.save(dest_dir / f"{cat}_demo_{i:02d}.png")
