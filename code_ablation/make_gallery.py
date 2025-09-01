# -*- coding: utf-8 -*-
"""
Génère des planches d'images comparant deux variantes (A vs B) :
- 1 ligne = requête
- Colonne gauche: image requête
- Ligne A: Top-k de la variante A
- Ligne B: Top-k de la variante B
- Bordure VERTE = positif, GRIS = négatif
- Titre de la planche: métriques par requête et Δ (B - A)

Sorties:
- /chemin/out_dir/gallery_*.png
- /chemin/out_dir/index.html (galerie)
- /chemin/out_dir/summary.csv (résumé des deltas)

Exemple:
    python make_gallery.py \
      --data /mnt/data/eval_manual_from_alix.csv \
      --variant_a "clip_only" \
      --variant_b "+color" \
      --k 8 \
      --m 12 \
      --metric "map@k" \
      --recommender_kwargs '{"model_name":"patrickjohncyh/fashion-clip"}' \
      --out_dir /mnt/data/gallery
"""

import argparse, csv, json, os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence, Any, Optional, Tuple
import numpy as np

from hooks import RecommenderWithAblations
from configs import VARIANTS

# ========== métriques fallback (si metrics.py absent) ==========
_FALLBACK = False
try:
    from metrics import map_at_k, ndcg_at_k, precision_at_k, recall_at_k  # type: ignore
    _FALLBACK = any(f is None for f in [map_at_k, ndcg_at_k, precision_at_k, recall_at_k])
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

# ========== utils canon ==========
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

# ========== data ==========
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
            qid = _canon(row.get("query_id",""))
            qpath = _canon(row.get("query_path",""))
            pool = _canon_list(row.get("pool_paths","").split(";"))
            pos  = _canon_list(row.get("positives","").split(";"))
            pool_set = set(pool)
            pos_in = [p for p in pos if p in pool_set]
            if not pos_in:
                continue
            items.append(EvalItem(qid, qpath, pool, pos_in))
    return items

# ========== reco & per-query ==========
def eval_variant_per_query(rec: RecommenderWithAblations, items: List[EvalItem], k: int, weights: Dict[str, float]):
    """
    Retourne deux dicts:
      per_metrics[qid] = {map, ndcg, precision, recall}
      per_topk[qid] = liste top-k (IDs/chemins)
    """
    per_metrics: Dict[str, Dict[str, float]] = {}
    per_topk: Dict[str, List[str]] = {}
    for it in items:
        ranked = rec.recommend(it.query_path, it.pool_paths, k=k, weights=weights)
        ranked_ids = _ensure_id_list(ranked)
        per_topk[it.query_id] = ranked_ids[:k]
        per_metrics[it.query_id] = {
            "map@k": map_at_k(it.positives, ranked_ids, k),
            "ndcg@k": ndcg_at_k(it.positives, ranked_ids, k),
            "precision@k": precision_at_k(it.positives, ranked_ids, k),
            "recall@k": recall_at_k(it.positives, ranked_ids, k),
        }
    return per_metrics, per_topk

# ========== dessin des planches ==========
def _draw_tile(path: str, tile_size: int, border: int, is_positive: bool) -> "Image.Image":
    from PIL import Image, ImageDraw, ImageFont
    W = H = tile_size
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((W, H))
    except Exception:
        # placeholder rouge si image illisible
        im = Image.new("RGB", (W, H), (200, 50, 50))
        d = ImageDraw.Draw(im)
        d.line((0,0,W,H), fill=(255,255,255), width=3)
        d.line((W,0,0,H), fill=(255,255,255), width=3)

    # canvas avec bordure
    canvas = Image.new("RGB", (W + 2*border, H + 2*border), (240,240,240))
    canvas.paste(im, (border + (W - im.width)//2, border + (H - im.height)//2))

    # couleur bordure
    col = (40, 170, 90) if is_positive else (160, 160, 160)
    d = ImageDraw.Draw(canvas)
    for i in range(border):
        d.rectangle([i, i, canvas.width-1-i, canvas.height-1-i], outline=col)
    return canvas

def make_board(query_path: str,
               positives: set,
               topA: List[str],
               topB: List[str],
               metricA: Dict[str, float],
               metricB: Dict[str, float],
               out_path: Path,
               k: int,
               title: str = "",
               tile_size: int = 160,
               border: int = 6):
    from PIL import Image, ImageDraw, ImageFont

    # layout: 1 colonne query + k colonnes résultats, 2 rangées (A, B) + 1 rangée titre
    cols = 1 + k
    rows = 2
    pad = 12
    label_h = 56  # hauteur bloc titre

    cell_w = tile_size + 2*border
    cell_h = tile_size + 2*border

    W = cols * cell_w + (cols + 1) * pad
    H = label_h + rows * cell_h + (rows + 1) * pad

    board = Image.new("RGB", (W, H), (250, 250, 250))
    draw = ImageDraw.Draw(board)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # titre
    if not title:
        title = "Comparaison variantes"
    # texte métriques
    tA = f"A: P={metricA['precision@k']:.3f}  R={metricA['recall@k']:.3f}  MAP={metricA['map@k']:.3f}  NDCG={metricA['ndcg@k']:.3f}"
    tB = f"B: P={metricB['precision@k']:.3f}  R={metricB['recall@k']:.3f}  MAP={metricB['map@k']:.3f}  NDCG={metricB['ndcg@k']:.3f}"
    delta = f"Δ(B-A): MAP={metricB['map@k']-metricA['map@k']:.3f} | NDCG={metricB['ndcg@k']-metricA['ndcg@k']:.3f}"

    draw.text((pad, 8), title, fill=(10,10,10), font=font)
    draw.text((pad, 28), tA, fill=(60,60,60), font=font)
    draw.text((pad, 44), tB + "   " + delta, fill=(60,60,60), font=font)

    # case requête (affichée à gauche de chaque ligne)
    qtile = _draw_tile(query_path, tile_size, border, False)

    # Ligne A
    yA = label_h + pad
    board.paste(qtile, (pad, yA))
    x = pad + cell_w + pad
    for i in range(k):
        pid = topA[i] if i < len(topA) else ""
        is_pos = pid in positives
        tile = _draw_tile(pid, tile_size, border, is_pos)
        board.paste(tile, (x, yA))
        # rang
        draw.text((x+6, yA+6), f"{i+1}", fill=(20,20,20), font=font)
        x += cell_w + pad

    # Ligne B
    yB = label_h + pad + cell_h + pad
    board.paste(qtile, (pad, yB))
    x = pad + cell_w + pad
    for i in range(k):
        pid = topB[i] if i < len(topB) else ""
        is_pos = pid in positives
        tile = _draw_tile(pid, tile_size, border, is_pos)
        board.paste(tile, (x, yB))
        draw.text((x+6, yB+6), f"{i+1}", fill=(20,20,20), font=font)
        x += cell_w + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(out_path, format="PNG")

# ========== main ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--variant_a", type=str, default="clip_only")
    ap.add_argument("--variant_b", type=str, default="+color")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--m", type=int, default=12, help="nb total de planches: moitié meilleures améliorations, moitié pires")
    ap.add_argument("--metric", type=str, default="map@k", choices=["map@k","ndcg@k"])
    ap.add_argument("--recommender_kwargs", type=str, default="{}")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--only_ids", type=str, default="", help="liste CSV de query_id à visualiser (ignore --m)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # charge data
    items = load_eval_csv(Path(args.data))
    if not items:
        raise SystemExit("[ERROR] Aucune requête évaluable (positifs absents du pool).")

    # init reco
    rec = RecommenderWithAblations(**json.loads(args.recommender_kwargs or "{}"))

    # poids des variantes
    if args.variant_a not in VARIANTS or args.variant_b not in VARIANTS:
        raise SystemExit(f"[ERROR] Variantes inconnues. Dispo: {list(VARIANTS.keys())}")
    wA = VARIANTS[args.variant_a]
    wB = VARIANTS[args.variant_b]

    # évalue A et B
    perA, topA = eval_variant_per_query(rec, items, k=args.k, weights=wA)
    perB, topB = eval_variant_per_query(rec, items, k=args.k, weights=wB)

    # set des ids à produire
    if args.only_ids:
        keep = set([x.strip() for x in args.only_ids.split(",") if x.strip()])
    else:
        # tri par delta (B - A) sur la métrique choisie
        deltas = []
        for it in items:
            q = it.query_id
            if q in perA and q in perB:
                d = perB[q][args.metric] - perA[q][args.metric]
                deltas.append((q, d))
        # meilleurs et pires
        deltas.sort(key=lambda t: t[1], reverse=True)
        half = max(1, args.m // 2)
        keep = set([q for q,_ in deltas[:half]] + [q for q,_ in deltas[-half:]])

    # summary csv
    sum_csv = out_dir / "summary.csv"
    with sum_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id","metric","A","B","delta","png_path"])
        # génère planches
        for it in items:
            q = it.query_id
            if q not in keep: continue
            A = perA[q]; B = perB[q]
            delta = B[args.metric] - A[args.metric]
            title = f"{q}  |  {args.variant_a} vs {args.variant_b}"
            png_path = out_dir / f"gallery_{q.replace('/','_')}.png"
            make_board(
                query_path=it.query_path,
                positives=set(it.positives),
                topA=topA[q],
                topB=topB[q],
                metricA=A,
                metricB=B,
                out_path=png_path,
                k=args.k,
                title=title
            )
            w.writerow([q, args.metric, f"{A[args.metric]:.6f}", f"{B[args.metric]:.6f}", f"{delta:.6f}", str(png_path)])

    # index.html
    html_path = out_dir / "index.html"
    rows = list(csv.DictReader(sum_csv.open("r", encoding="utf-8")))
    rows.sort(key=lambda r: float(r["delta"]), reverse=True)
    with html_path.open("w", encoding="utf-8") as h:
        h.write("<html><head><meta charset='utf-8'><title>Gallery</title></head><body>")
        h.write(f"<h1>Comparaison {args.variant_a} vs {args.variant_b} (k={args.k})</h1>")
        h.write("<p>Bordure verte = positif; gris = négatif. Δ = B - A.</p>")
        for r in rows:
            rel = os.path.basename(r["png_path"])
            h.write(f"<div style='margin:16px 0;'><h3>{r['query_id']} — Δ {args.metric} = {float(r['delta']):+.4f}</h3>")
            h.write(f"<img src='{rel}' style='max-width:100%; border:1px solid #ddd;' />")
            h.write("</div>")
        h.write("</body></html>")

    print(f"[OK] Planches : {out_dir}")
    print(f"[OK] Index    : {html_path}")
    print(f"[OK] Résumé   : {sum_csv}")

if __name__ == "__main__":
    main()
