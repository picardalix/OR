
from __future__ import annotations
import numpy as np
from typing import Sequence, List, Tuple

def auc_pairwise(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """
    AUC based on pairwise comparisons between positive and negative samples.
    y_true: list of 0/1
    y_score: list of scores (higher is better)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    # proportion of correctly ordered pairs
    wins = 0
    ties = 0
    total = len(pos) * len(neg)
    for p in pos:
        wins += np.sum(p > neg)
        ties += np.sum(p == neg)
    auc = (wins + 0.5 * ties) / total
    return float(auc)

def recall_at_k(y_true: Sequence[int], y_pred: Sequence[int], k: int) -> float:
    y_true_set = set(y_true)
    if not y_true_set:
        return 0.0
    topk = list(y_pred)[:k]
    hits = sum(1 for x in topk if x in y_true_set)
    return hits / len(y_true_set)

def ndcg_at_k(y_true: Sequence[int], y_pred: Sequence[int], k: int) -> float:
    y_true_set = set(y_true)
    topk = list(y_pred)[:k]
    if not topk:
        return 0.0
    dcg = 0.0
    for i, x in enumerate(topk, start=1):
        rel = 1.0 if x in y_true_set else 0.0
        if rel > 0:
            dcg += rel / np.log2(i + 1)
    ideal_hits = min(len(y_true_set), len(topk))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0

def intra_list_diversity(item_vectors: List[np.ndarray]) -> float:
    """
    Average pairwise dissimilarity (1 - cosine similarity).
    """
    if len(item_vectors) <= 1:
        return 0.0
    sims = []
    for i in range(len(item_vectors)):
        for j in range(i+1, len(item_vectors)):
            a = item_vectors[i]; b = item_vectors[j]
            na = np.linalg.norm(a) + 1e-8
            nb = np.linalg.norm(b) + 1e-8
            sims.append(np.dot(a, b) / (na * nb))
    dis = [1 - s for s in sims]
    return float(np.mean(dis)) if dis else 0.0
