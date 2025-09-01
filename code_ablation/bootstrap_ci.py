# -*- coding: utf-8 -*-
"""
Bootstrap CI (95%) pour la différence de performance entre deux variantes
(ex. 'clip_only' vs '+color') sur MAP@k et NDCG@k, en paire par requête.

Usage (exemples) :
    python bootstrap_ci.py \
      --data /mnt/data/eval_manual_from_alix.csv \
      --variant_a "clip_only" \
      --variant_b "+color" \
      --k 10 \
      --n_boot 5000 \
      --recommender_kwargs '{"model_name":"patrickjohncyh/fashion-clip"}' \
      --out /mnt/data/ci_clip_vs_color.json \
      --save_per_query /mnt/data/per_query_clip_vs_color.csv

Interprétation :
- Si l'IC 95% du Δ (B - A) **n'inclut pas 0**, le gain est significatif.
- p_boot (approx) = fraction des moyennes bootstrap ≤ 0 (si Δ_obs > 0),
  ou ≥ 0 (si Δ_obs < 0).
"""
import argparse, csv, json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence, Any, Optional, Tuple

import numpy as np

from configs import VARIANTS  # on lit les poids par nom
from hooks import RecommenderWithAblations

# ======================== Métriques (fallback sûrs) ========================
_FALLBACK = False
try:
    from metrics import precision_at_k, recall_at_k, map_at_k, ndcg_at_k  # type: ignore
    _FALLBACK = any(f is None for f in [precision_at_k, recall_at_k, map_at_k, ndcg_at_k])
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

# ======================== Utils canon ========================
def _canon(s: str) -> str:
    if s is None: return ""
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s.replace("\\", "/")

def _canon_list(xs: Sequence[str]) -> List[str]:
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

def load_eval_csv(path: Path) -> List[EvalItem]:
    items: List[EvalItem] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = _canon(row.get("query_id",""))
            query_path = _canon(row.get("query_path",""))
            pool_paths = _canon_list(row.get("pool_paths","").split(";"))
            positives  = _canon_list(row.get("positives","").split(";"))
            # ne garde que les positifs présents dans le pool
            pool_set = set(pool_paths)
            positives_in = [p for p in positives if p in pool_set]
            if not positives_in:
                continue  # on écarte les lignes non évaluables
            items.append(EvalItem(query_id, query_path, pool_paths, positives_in))
    return items

# ======================== Évaluation par requête ========================
def eval_variant_per_query(rec: RecommenderWithAblations,
                           items: List[EvalItem],
                           k: int,
                           weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    Retourne un dict: query_id -> { 'map@k': x, 'ndcg@k': y, 'precision@k': ..., 'recall@k': ... }
    """
    out: Dict[str, Dict[str, float]] = {}
    for it in items:
        ranked = rec.recommend(it.query_path, it.pool_paths, k=k, weights=weights)
        ranked_ids = _ensure_id_list(ranked)
        out[it.query_id] = {
            "precision@k": precision_at_k(it.positives, ranked_ids, k),
            "recall@k":    recall_at_k(it.positives, ranked_ids, k),
            "map@k":       map_at_k(it.positives, ranked_ids, k),
            "ndcg@k":      ndcg_at_k(it.positives, ranked_ids, k),
        }
    return out

# ======================== Bootstrap ========================
def bootstrap_ci(deltas: np.ndarray, n_boot: int = 5000, seed: int = 123) -> Tuple[float, float, float]:
    """
    Retourne (mean_obs, ci_low, ci_high). p_boot approx = frac(means_boot <= 0) si mean_obs>0, symétrique sinon.
    """
    rng = np.random.default_rng(seed)
    n = deltas.shape[0]
    mean_obs = float(deltas.mean())
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = float(deltas[idx].mean())
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    if mean_obs >= 0:
        p_boot = float((means <= 0).mean())
    else:
        p_boot = float((means >= 0).mean())
    return mean_obs, float(ci_low), float(ci_high), p_boot

# ======================== Main ========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--variant_a", type=str, default="clip_only")
    ap.add_argument("--variant_b", type=str, default="+color")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--recommender_kwargs", type=str, default="{}")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--save_per_query", type=str, default="")
    args = ap.parse_args()

    data_path = Path(args.data)
    items = load_eval_csv(data_path)
    if not items:
        raise SystemExit("[ERROR] Aucun item évaluable (aucun positif dans le pool). Vérifie ton CSV.")

    # initialisation reco
    rk = json.loads(args.recommender_kwargs or "{}")
    rec = RecommenderWithAblations(**rk)

    # poids par variante
    if args.variant_a not in VARIANTS or args.variant_b not in VARIANTS:
        raise SystemExit(f"[ERROR] Variantes inconnues. Dispo: {list(VARIANTS.keys())}")
    wA = VARIANTS[args.variant_a]
    wB = VARIANTS[args.variant_b]

    # évaluation par requête
    perA = eval_variant_per_query(rec, items, k=args.k, weights=wA)
    perB = eval_variant_per_query(rec, items, k=args.k, weights=wB)

    # intersection des requêtes (sécurité)
    qs = sorted(set(perA.keys()) & set(perB.keys()))
    if not qs:
        raise SystemExit("[ERROR] Aucune requête en intersection entre A et B.")

    def arr(metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.array([perA[q][metric] for q in qs], dtype=np.float64)
        b = np.array([perB[q][metric] for q in qs], dtype=np.float64)
        d = b - a
        return a, b, d

    res = {}

    for metric in ("map@k", "ndcg@k", "precision@k", "recall@k"):
        A, B, D = arr(metric)
        meanA, meanB = float(A.mean()), float(B.mean())
        meanD, lo, hi, p = bootstrap_ci(D, n_boot=args.n_boot, seed=args.seed)
        res[metric] = {
            "mean_A": meanA,
            "mean_B": meanB,
            "delta_B_minus_A": meanD,
            "ci95_low": lo,
            "ci95_high": hi,
            "p_boot": p,
            "n_queries": int(len(qs)),
        }

    # Affichage console
    print(f"\n[INFO] {args.variant_a}  vs  {args.variant_b}   (k={args.k}, n={len(qs)})")
    for m in ("map@k", "ndcg@k", "precision@k", "recall@k"):
        r = res[m]
        print(f"- {m}:  A={r['mean_A']:.4f}  B={r['mean_B']:.4f}  Δ={r['delta_B_minus_A']:.4f} "
              f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]  p_boot={r['p_boot']:.4f}")

    # Sauvegardes
    if args.out:
        Path(args.out).write_text(json.dumps({
            "data": str(data_path),
            "variant_a": args.variant_a,
            "variant_b": args.variant_b,
            "k": args.k,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "results": res,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] JSON écrit: {args.out}")

    if args.save_per_query:
        # on écrit un CSV avec les métriques par requête pour A et B
        outp = Path(args.save_per_query)
        with outp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query_id",
                        "A_map@k","B_map@k","D_map@k",
                        "A_ndcg@k","B_ndcg@k","D_ndcg@k",
                        "A_precision@k","B_precision@k","D_precision@k",
                        "A_recall@k","B_recall@k","D_recall@k"])
            for q in qs:
                Amap, Bmap = perA[q]["map@k"], perB[q]["map@k"]
                Andc, Bndc = perA[q]["ndcg@k"], perB[q]["ndcg@k"]
                Aprc, Bprc = perA[q]["precision@k"], perB[q]["precision@k"]
                Arec, Brec = perA[q]["recall@k"], perB[q]["recall@k"]
                w.writerow([q,
                            f"{Amap:.6f}", f"{Bmap:.6f}", f"{(Bmap-Amap):.6f}",
                            f"{Andc:.6f}", f"{Bndc:.6f}", f"{(Bndc-Andc):.6f}",
                            f"{Aprc:.6f}", f"{Bprc:.6f}", f"{(Bprc-Aprc):.6f}",
                            f"{Arec:.6f}", f"{Brec:.6f}", f"{(Brec-Arec):.6f}"])
        print(f"[OK] CSV per-query écrit: {outp}")

if __name__ == "__main__":
    main()
