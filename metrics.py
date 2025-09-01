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
    wins = 0
    ties = 0
    total = len(pos) * len(neg)
    for p in pos:
        wins += np.sum(p > neg)
        ties += np.sum(p == neg)
    auc = (wins + 0.5 * ties) / total
    return float(auc)

def recall_at_k(y_true: Sequence[int], y_pred: Sequence[int], k: int) -> float:
    """
    Recall@K: proportion d'items pertinents récupérés dans le top-K.
    y_true: items pertinents (identifiants uniques de préférence)
    y_pred: liste ordonnée de prédictions (du meilleur au pire)
    """
    y_true_set = set(y_true)
    if not y_true_set or k <= 0:
        return 0.0
    topk = list(y_pred)[:k]
    hits = sum(1 for x in topk if x in y_true_set)
    return hits / len(y_true_set)

def precision_at_k(y_true: Sequence[int], y_pred: Sequence[int], k: int) -> float:
    """
    Precision@K: proportion de prédictions du top-K qui sont pertinentes.
    """
    if k <= 0:
        return 0.0
    y_true_set = set(y_true)
    topk = list(y_pred)[:k]
    if not topk:
        return 0.0
    hits = sum(1 for x in topk if x in y_true_set)
    return hits / len(topk)

def map_at_k(y_true: Sequence[int], y_pred: Sequence[int], k: int) -> float:
    """
    MAP@K (Mean Average Precision @ K) pour une requête.
    Déf: AP@K = (1/min(|y_true|, K)) * somme_{i=1..K} [Precision@i * rel_i]
    où rel_i = 1 si la prédiction à la position i est pertinente, 0 sinon.
    - Les doublons dans y_pred ne sont pas multi-comptés (on ignore les secondes occurrences).
    """
    y_true_set = set(y_true)
    if not y_true_set or k <= 0:
        return 0.0
    topk = list(y_pred)[:k]

    num_rel = min(len(y_true_set), len(topk))
    if num_rel == 0:
        return 0.0

    ap = 0.0
    hits = 0
    seen = set()
    for i, x in enumerate(topk, start=1):
        if x in seen:
            continue
        seen.add(x)
        if x in y_true_set:
            hits += 1
            ap += hits / i  # precision@i (locale) accumulée aux positions pertinentes
    return ap / num_rel

def ndcg_at_k(y_true: Sequence[int], y_pred: Sequence[int], k: int) -> float:
    """
    NDCG@K avec gains binaires (1 si pertinent, 0 sinon).
    """
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

# ----------------------- FITB (Polyvore) -----------------------

def fitb_accuracy_single(candidate_scores: Sequence[float], correct_index: int) -> float:
    """
    Accuracy pour une question FITB (Fill-In-The-Blank).
    - candidate_scores: scores pour chaque candidat (plus grand = mieux)
    - correct_index: index (0-based) du candidat correct
    Retourne 1.0 si l'argmax est le bon index, sinon 0.0.
    """
    if len(candidate_scores) == 0:
        return 0.0
    pred_idx = int(np.argmax(np.asarray(candidate_scores, dtype=float)))
    return 1.0 if pred_idx == int(correct_index) else 0.0

def fitb_accuracy(batched_candidate_scores: Sequence[Sequence[float]],
                  correct_indices: Sequence[int]) -> float:
    """
    Accuracy moyenne sur un lot de questions FITB.
    - batched_candidate_scores: liste de listes de scores par question
    - correct_indices: liste des indices corrects (0-based), même taille que le lot
    """
    if not batched_candidate_scores:
        return 0.0
    accs = []
    for scores, ci in zip(batched_candidate_scores, correct_indices):
        accs.append(fitb_accuracy_single(scores, ci))
    return float(np.mean(accs)) if accs else 0.0

def catalog_coverage(
    recommendations: list[list[str] | list[int]], 
    catalog_items: list[str] | list[int]
) -> float:
    """
    Calcule la proportion d'articles distincts du catalogue
    qui apparaissent au moins une fois dans les recommandations.

    Parameters
    ----------
    recommendations : list of lists
        Chaque sous-liste contient les IDs d'articles recommandés pour une requête.
    catalog_items : list
        Liste (ou ensemble) des IDs d'articles du catalogue complet.

    Returns
    -------
    float
        Coverage entre 0 et 1
    """
    # transformer en ensemble pour éviter les doublons
    all_recommended = set()
    for rec in recommendations:
        all_recommended.update(rec)

    catalog_set = set(catalog_items)
    if not catalog_set:
        return 0.0

    return len(all_recommended & catalog_set) / len(catalog_set)
