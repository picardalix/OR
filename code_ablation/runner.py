# -*- coding: utf-8 -*-
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Any, Optional
from dataclasses import dataclass

from configs import VARIANTS, SHAPLEY_REFERENCE, COMPONENTS
from hooks import RecommenderWithAblations

# ======================== Métriques (fallback sûrs) ========================
_FALLBACK = False
try:
    from metrics import precision_at_k, recall_at_k, map_at_k, ndcg_at_k, fitb_accuracy  # type: ignore
    _FALLBACK = any(f is None for f in [precision_at_k, recall_at_k, map_at_k, ndcg_at_k, fitb_accuracy])
except Exception:
    _FALLBACK = True

if _FALLBACK:
    import math
    def precision_at_k(positives: Sequence[str], ranked: Sequence[str], k: int) -> float:
        if k <= 0: return 0.0
        topk = ranked[:k]
        hits = sum(1 for r in topk if r in set(positives))
        return hits / float(k)

    def recall_at_k(positives: Sequence[str], ranked: Sequence[str], k: int) -> float:
        P = len(positives)
        if P == 0 or k <= 0: return 0.0
        topk = ranked[:k]
        hits = sum(1 for r in topk if r in set(positives))
        return hits / float(P)

    def map_at_k(positives: Sequence[str], ranked: Sequence[str], k: int) -> float:
        P = len(positives)
        if P == 0 or k <= 0: return 0.0
        pos = set(positives)
        ap_sum, hits = 0.0, 0
        for i, r in enumerate(ranked[:k], start=1):
            if r in pos:
                hits += 1
                ap_sum += hits / float(i)
        return ap_sum / float(min(P, k))

    def ndcg_at_k(positives: Sequence[str], ranked: Sequence[str], k: int) -> float:
        if k <= 0: return 0.0
        import math
        pos = set(positives)
        dcg = 0.0
        for i, r in enumerate(ranked[:k], start=1):
            if r in pos:
                dcg += 1.0 / math.log2(i + 1)
        p = min(len(pos), k)
        if p == 0: return 0.0
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, p + 1))
        return dcg / idcg if idcg > 0 else 0.0

    def fitb_accuracy(batches: List[List[float]], correct_indices: List[int]) -> float:
        if not batches: return 0.0
        ok = 0
        for scores, ci in zip(batches, correct_indices):
            if not scores: continue
            pred = max(range(len(scores)), key=lambda i: scores[i])
            ok += int(pred == int(ci))
        return ok / float(len(batches))

# ======================== Utils canon ========================
def _canon(s: str) -> str:
    if s is None: return ""
    s = s.strip()
    # enlève quotes parasites
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    # normalise séparateurs
    s = s.replace("\\", "/")
    # supprime doubles espaces autour des ';' éventuels déjà splittés
    return s

def _canon_list(xs: Sequence[str]) -> List[str]:
    # garde ordre et déduplique proprement
    seen, out = set(), []
    for x in xs:
        cx = _canon(x)
        if not cx: continue
        if cx in seen: continue
        seen.add(cx); out.append(cx)
    return out

def _ensure_id_list(ranked: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for x in ranked:
        if isinstance(x, (tuple, list)) and x:
            out.append(_canon(str(x[0])))
        elif hasattr(x, "id"):
            out.append(_canon(str(getattr(x, "id"))))
            
        else:
            out.append(_canon(str(x)))
    return out

# ======================== Données ========================
@dataclass
class EvalItem:
    query_id: str
    query_path: str
    pool_paths: List[str]
    positives: List[str]
    fitb_options: Optional[List[str]] = None
    fitb_answer_idx: Optional[int] = None

def load_eval_csv(path: Path) -> List[EvalItem]:
    items: List[EvalItem] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_path = _canon(row.get("query_path", ""))
            pool_paths  = _canon_list(row.get("pool_paths", "").split(";"))
            positives   = _canon_list(row.get("positives", "").split(";"))
            # ne garde que les positifs présents dans le pool
            pool_set = set(pool_paths)
            positives_in = [p for p in positives if p in pool_set]
            if positives and not positives_in:
                print(f"[WARN] Positifs absents du pool pour {row.get('query_id','?')} -> ils seront ignorés.")
            fitb_opts = _canon_list(row.get("fitb_options", "").split(";")) if row.get("fitb_options") else None
            fitb_idx  = int(row["fitb_answer_idx"]) if row.get("fitb_answer_idx") not in (None, "",) else None
            items.append(EvalItem(
                query_id=_canon(row.get("query_id","")),
                query_path=query_path,
                pool_paths=pool_paths,
                positives=positives_in,  # filtrés
                fitb_options=fitb_opts,
                fitb_answer_idx=fitb_idx,
            ))
    total = len(items)
    with_pos = sum(1 for it in items if len(it.positives) > 0)
    print(f"[INFO] Chargé {total} items ; {with_pos} avec au moins 1 positif présent dans le pool.")
    return items

# ======================== Évaluation ========================
def compute_metrics_at_k(ranked: List[str], positives: List[str], k: int) -> Dict[str, float]:
    return {
        "precision@k": precision_at_k(positives, ranked, k),
        "recall@k":    recall_at_k(positives, ranked, k),
        "map@k":       map_at_k(positives, ranked, k),
        "ndcg@k":      ndcg_at_k(positives, ranked, k),
    }

def run_variant(rec: RecommenderWithAblations, items: List[EvalItem], k: int, weights: Dict[str, float]) -> Dict[str, float]:
    metrics_accum = {"precision@k": 0.0, "recall@k": 0.0, "map@k": 0.0, "ndcg@k": 0.0}
    eval_count = 0

    fitb_batches: List[List[float]] = []
    fitb_correct: List[int] = []

    for it in items:
        # skip les items sans positif (sinon biais 0)
        if len(it.positives) == 0:
            continue

        ranked = rec.recommend(it.query_path, it.pool_paths, k=k, weights=weights)
        ranked_ids = _ensure_id_list(ranked)

        # métriques retrieval
        m = compute_metrics_at_k(ranked_ids, it.positives, k)
        for key in metrics_accum:
            metrics_accum[key] += float(m.get(key, 0.0))
        eval_count += 1

        # FITB (optionnel)
        if it.fitb_options and it.fitb_answer_idx is not None:
            rank_pos = {rid: idx for idx, rid in enumerate(ranked_ids)}
            scores = [(len(ranked_ids) - rank_pos[o]) if o in rank_pos else 0.0 for o in it.fitb_options]
            fitb_batches.append(scores)
            fitb_correct.append(int(it.fitb_answer_idx))

    if eval_count == 0:
        print("[ERROR] Aucun item évaluable (0 positif dans le pool après nettoyage). Vérifie ton CSV.")
        return {"precision@k":0.0,"recall@k":0.0,"map@k":0.0,"ndcg@k":0.0,"fitb_acc":0.0}

    results = {k2: v / eval_count for k2, v in metrics_accum.items()}
    results["fitb_acc"] = fitb_accuracy(fitb_batches, fitb_correct) if fitb_batches else 0.0
    return results

# ======================== Shapley-like ========================
def shapley_like(full_scores: Dict[str, float], leave_one_out: Dict[str, Dict[str, float]], on_metric: str="ndcg@k") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for comp in [c.key for c in COMPONENTS if c.key in leave_one_out]:
        diff = full_scores.get(on_metric, 0.0) - leave_one_out[comp].get(on_metric, 0.0)
        out[comp] = diff
    return out

# ======================== Main ========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to eval CSV")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "results.csv"))
    ap.add_argument("--recommender_kwargs", type=str, default="{}", help="JSON dict passed to RecommenderWithAblations")
    args = ap.parse_args()

    items = load_eval_csv(Path(args.data))
    rec = RecommenderWithAblations(**json.loads(args.recommender_kwargs))

    results_rows: List[Dict[str, Any]] = []
    cache_variant_scores: Dict[str, Dict[str, float]] = {}

    for name, weights in VARIANTS.items():
        scores = run_variant(rec, items, k=args.k, weights=weights)
        cache_variant_scores[name] = scores
        row = {"variant": name, **scores}
        results_rows.append(row)

    # Leave-one-out autour de SHAPLEY_REFERENCE (si présent)
    if SHAPLEY_REFERENCE in cache_variant_scores:
        full_weights = VARIANTS[SHAPLEY_REFERENCE]
        loo_scores: Dict[str, Dict[str, float]] = {}
        for comp_key, w in full_weights.items():
            if w == 0.0:
                continue
            w_loo = dict(full_weights); w_loo[comp_key] = 0.0
            scores = run_variant(rec, items, k=args.k, weights=w_loo)
            loo_scores[comp_key] = scores
        shapley = shapley_like(cache_variant_scores[SHAPLEY_REFERENCE], loo_scores, on_metric="ndcg@k")
    else:
        shapley = {}

    # Écriture résultats
    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["variant", "precision@k", "recall@k", "map@k", "ndcg@k", "fitb_acc"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)

    # JSON Shapley-like
    shapley_path = out_path.with_suffix(".shapley.json")
    with shapley_path.open("w", encoding="utf-8") as f:
        json.dump({"reference": SHAPLEY_REFERENCE, "delta_vs_leave_one_out": shapley}, f, indent=2)

    print(f"Saved: {out_path}")
    print(f"Saved: {shapley_path}")

if __name__ == "__main__":
    main()
