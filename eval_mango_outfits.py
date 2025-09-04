# -*- coding: utf-8 -*-
"""
Mango "styled-with" — P@3 & FITB avec ablation (couleur/matière/texte) + super-catégories.

Fonctions clés
--------------
- Lit un CSV: anchor_url,anchor_img,pos_url,pos_img,brand,name,color,price,composition,description
- Encode les images avec FashionCLIP si disponible, sinon OpenCLIP (fallback).
- Baseline = similarité image→image (cosine) pour P@3 et FITB.
- + Attributs (optionnels) = bonus léger:
    * Couleur (familles + analogues) — couleur ancre via CSV, couleur candidate via dominante HSV.
    * Matière (mots-clés compo/desc + hints URL).
    * Texte (OpenCLIP: prompt anchor texte -> image candidate), normalisé [0..1].
- FITB: 1 positive + 3 négatives; on essaie d'échantillonner dans la même super-catégorie.
- Super-catégories lisibles depuis pos_url: chaussures, sacs, vestes_manteaux, bas, hauts_maille, robes_combinaisons, accessoires.
- Exports:
    * ablation_summary.json (scores globaux + poids)
    * ablation_p3_details.csv (rangs avant/après, delta…)
    * ablation_supercats_p3.csv, ablation_supercats_fitb.csv (agrégés)
    * top5_gains.csv, top5_drops.csv
    * (si --plot) PNGs: p3_ab.png, fitb_ab.png, delta_hist.png, p3_supercats.png

Exemple
-------
python mango_eval_ablation.py \
  --input mango_pairs.csv \
  --out_dir mango_eval_ablation \
  --use_color --use_material --use_text \
  --w_img 1.0 --w_color 0.12 --w_matiere 0.08 --w_text 0.08 \
  --fitb_same_supercat --plot
"""
import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch

# ---------------------- Utils généraux ----------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def device_select():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def ensure_paths_exist(paths):
    ok = []
    miss = []
    for p in paths:
        if Path(p).exists():
            ok.append(p)
        else:
            miss.append(p)
    if miss:
        print(f"[WARN] {len(miss)} images manquantes; on les ignore (extrait):")
        for m in miss[:10]:
            print("  -", m)
    return ok

# ---------------------- Chargement modèles ----------------------
def load_image_model(device: str):
    """
    Retourne: (encode_images_fn, dim, model_name)
    encode_images_fn(imgs: List[PIL.Image]) -> torch.Tensor (N, D) normalisé
    """
    # 1) FashionCLIP si possible
    try:
        from fashion_clip.fashion_clip import FashionCLIP
        fclip = FashionCLIP('fashion-clip')
        def enc_imgs(imgs):
            z = fclip.encode_images(imgs, batch_size=32)
            z = torch.tensor(z, device="cpu", dtype=torch.float32)
            z = z / z.norm(dim=-1, keepdim=True)
            return z
        return enc_imgs, 512, "FashionCLIP"
    except Exception:
        pass

    # 2) OpenCLIP fallback
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    model = model.to(device)
    model.eval()

    @torch.no_grad()
    def enc_imgs(imgs):
        batch = []
        for im in imgs:
            if isinstance(im, (str, Path)):
                im = Image.open(im).convert("RGB")
            batch.append(preprocess(im))
        x = torch.stack(batch).to(device)
        z = model.encode_image(x)
        z = z / z.norm(dim=-1, keepdim=True)
        z = z.detach().to("cpu")
        return z

    return enc_imgs, model.visual.output_dim, "OpenCLIP ViT-B/32"

def load_openclip_text_and_image(device: str):
    """Toujours OpenCLIP pour le petit signal texte->image."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    tok = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device)
    model.eval()

    @torch.no_grad()
    def enc_text(texts):
        toks = tok(texts).to(device)
        zt = model.encode_text(toks)
        zt = zt / zt.norm(dim=-1, keepdim=True)
        return zt.detach().to("cpu")

    @torch.no_grad()
    def enc_imgs(imgs):
        batch = []
        for im in imgs:
            if isinstance(im, (str, Path)):
                im = Image.open(im).convert("RGB")
            batch.append(preprocess(im))
        x = torch.stack(batch).to(device)
        zi = model.encode_image(x)
        zi = zi / zi.norm(dim=-1, keepdim=True)
        return zi.detach().to("cpu")

    return enc_text, enc_imgs

# ---------------------- Lecture CSV robuste ----------------------
COLS = ["anchor_url","anchor_img","pos_url","pos_img","brand","name","color","price","composition","description"]

def read_pairs(path: str, limit: int = None) -> pd.DataFrame:
    p = Path(path)
    try:
        df = pd.read_csv(p, encoding="utf-8")
        if not all(c in df.columns for c in COLS):
            raise ValueError("Colonnes inattendues.")
    except Exception:
        # Fallback si "description" contient des virgules
        rows = []
        with p.open("r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\n").split(",")
                head = parts[:9]
                desc = ",".join(parts[9:])
                if len(head) == 9:
                    rows.append(head + [desc])
        df = pd.DataFrame(rows, columns=COLS)
    if limit:
        df = df.head(limit)
    return df.dropna(subset=["anchor_img","pos_img","pos_url"]).copy()

# ---------------------- Couleurs ----------------------
NEUTRALS = {"noir","blanc","gris","beige","ecru","écru","ivory","ivoire"}
COLOR_CANON = {
    "noir":"noir","black":"noir",
    "blanc":"blanc","white":"blanc","écru":"beige","ecru":"beige","ivory":"beige","ivoire":"beige",
    "gris":"gris","gray":"gris","argent":"gris","silver":"gris",
    "beige":"beige","sable":"beige","taupe":"beige","camel":"marron","marron":"marron","brun":"marron","chataigne":"marron","châtain":"marron","ocre":"marron",
    "kaki":"kaki","olive":"kaki","vert":"vert",
    "bleu":"bleu","marine":"marine","navy":"marine","denim":"bleu",
    "rouge":"rouge","bordeaux":"bordeaux","rose":"rose",
    "orange":"orange","jaune":"jaune",
    "violet":"violet","lilas":"violet","pourpre":"violet",
    "doré":"doré","or":"doré","gold":"doré",
}
ANALOG = {
    "marron": {"camel","beige","ocre"},
    "camel": {"marron","beige"},
    "beige": {"marron","camel","gris","blanc","kaki"},
    "kaki": {"marron","vert","olive","beige"},
    "vert": {"kaki","bleu"},
    "bleu": {"marine","denim","vert"},
    "marine": {"bleu","gris","noir"},
    "rouge": {"bordeaux","rose","orange"},
    "bordeaux": {"rouge","violet","marron"},
    "rose": {"rouge","violet","beige"},
    "orange": {"jaune","rouge","marron"},
    "jaune": {"orange","beige"},
    "violet": {"rose","bordeaux","bleu"},
}

def norm_color_str(s: str) -> str:
    if not isinstance(s, str): return "unknown"
    t = s.lower()
    for k, v in COLOR_CANON.items():
        if k in t: return v
    return t.split()[0] if t else "unknown"

def infer_color_from_image(img: Image.Image) -> str:
    # Dominante HSV simple (ignorer zones trop peu saturées/lumineuses)
    arr = np.array(img.convert("HSV"))
    H, S, V = arr[:,:,0].astype(np.int16), arr[:,:,1].astype(np.float32)/255.0, arr[:,:,2].astype(np.float32)/255.0
    mask = (S > 0.25) & (V > 0.25)
    if mask.sum() < 50: return "neutre"
    hdeg = (H[mask] * (360.0/255.0))
    m = float(np.mean(hdeg))
    if   m < 15 or m >= 345: fam = "rouge"
    elif m < 45:  fam = "orange"
    elif m < 70:  fam = "jaune"
    elif m < 95:  fam = "vert"
    elif m < 150: fam = "kaki"
    elif m < 200: fam = "bleu"
    elif m < 250: fam = "violet"
    elif m < 290: fam = "rose"
    elif m < 330: fam = "bordeaux"
    else:         fam = "rouge"
    return fam

def color_score(ca: str, cb: str) -> float:
    if ca == "unknown" or cb == "unknown": return 0.5
    if ca == cb: return 1.0
    if ca in NEUTRALS or cb in NEUTRALS or ca=="neutre" or cb=="neutre": return 0.7
    if cb in ANALOG.get(ca, set()) or ca in ANALOG.get(cb, set()): return 0.8
    return 0.55

# ---------------------- Matières ----------------------
MAT_MAP = {
    r"\blaine(s)?\b": "laine",
    r"\bcachemire\b": "laine",
    r"\bcoton\b": "coton",
    r"\blin\b": "lin",
    r"\bdenim\b|\bjean\b": "denim",
    r"\bsoie\b": "soie",
    r"\bviscose\b": "viscose",
    r"\bpolyester\b|\bpolyamide\b|\bélast(anne|hane)\b|\belastane\b|\belasthanne\b": "synthetique",
    r"\bcuir\b|\bsimilicuir\b|\bdaim\b|\bsuede\b": "cuir",
    r"\bmaille\b|\btricot\b|\bjersey\b": "maille",
    r"\btweed\b": "tweed",
    r"\bvelours\b": "velours",
}
URL_HINTS = [
    (r"/pull", "maille"),
    (r"/pantalon", "tisse"),
    (r"/jean|denim", "denim"),
    (r"/cuir|leather", "cuir"),
    (r"/veste|blazer|manteau|coat", "laine"),
    (r"/robe", "tisse"),
]

def extract_materials_from_text(comp: str, desc: str):
    txt = f"{comp or ''} {desc or ''}".lower()
    found = set()
    for pat, lab in MAT_MAP.items():
        if re.search(pat, txt):
            found.add(lab)
    return found

def material_from_url(url: str):
    for pat, lab in URL_HINTS:
        if re.search(pat, url):
            return lab
    return None

def mat_score(ma: set, mb: set, url_b: str) -> float:
    """Jaccard entre familles matière; si mb vide, on tente un hint via l'URL candidate."""
    if not mb or len(mb)==0:
        hint = material_from_url(url_b)
        if hint: mb = {hint}
    if not ma or not mb: return 0.5
    inter = len(ma & mb); union = len(ma | mb)
    return inter/union if union else 0.5

# ---------------------- Super-catégories ----------------------
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def supercat_from_url(url: str) -> str:
    if not isinstance(url, str) or "/femme/" not in url:
        return "autres"
    path = url.split("/femme/")[1]
    segs = [t for t in path.split("/") if t and "_" not in t]  # on ignore le slug final _ID
    segs = [_strip_accents(t.lower()) for t in segs]
    toks = set(segs + sum([t.split("-") for t in segs], []))   # éclate les mots composés

    def has_any(keys): return any(any(k in t for t in toks) for k in keys)

    FOOT = {"chaussure","bottes","bottine","escarpin","sandale","sneakers","basket","derby","mocassin","talon","plate","boots"}
    BAGS = {"sac","cabas","bandouliere","pochette","banane"}
    OUTW = {"veste","blazer","manteau","parka","trench","doudoune","gilet"}
    BOTT = {"pantalon","pantalons","jean","jupe","short","legging","wideleg","straight","paperbag","flare","cargo"}
    TOPS = {"pull","cardigan","tshirt","t","chemise","blouse","top","sweat","hoodie"}
    DRES = {"robe","combinaison","jumpsuit"}
    ACCS = {"ceinture","collier","bracelet","boucle","boucle-doreille","echarpe","foulard","chaussettes","collants","chapeau","gants"}

    if has_any(FOOT): return "chaussures"
    if has_any(BAGS): return "sacs"
    if has_any(OUTW): return "vestes_manteaux"
    if has_any(BOTT): return "bas"
    if has_any(TOPS): return "hauts_maille"
    if has_any(DRES): return "robes_combinaisons"
    if has_any(ACCS): return "accessoires"
    return "autres"

# ---------------------- Texte (prompts) ----------------------
def build_prompt(name, color, mats, desc):
    mtxt = "/".join(sorted(mats)) if mats else "inconnue"
    desc = (desc or "").strip().replace("\n", " ")
    if len(desc) > 180: desc = desc[:180] + "..."
    return f"{name}. Couleur: {color}. Matière: {mtxt}. {desc}"

# ---------------------- Corps principal ----------------------
def main(args):
    set_seed(args.seed)
    device = device_select()
    print(f"[INFO] Device = {device}")

    enc_imgs_main, dim, img_model_name = load_image_model(device)
    print(f"[INFO] Image model = {img_model_name} (dim={dim})")

    # Texte->image (OpenCLIP) pour petit signal facultatif
    enc_text = enc_imgs_openclip = None
    if args.use_text:
        enc_text, enc_imgs_openclip = load_openclip_text_and_image(device)
        print("[INFO] OpenCLIP chargé pour texte->image")

    # Lire et prétraiter les paires
    df = read_pairs(args.input, limit=args.limit)
    df["anchor_img"] = df["anchor_img"].str.strip()
    df["pos_img"]    = df["pos_img"].str.strip()
    df["color_norm"] = df["color"].map(norm_color_str)
    df["mats_anchor"] = df.apply(lambda r: extract_materials_from_text(r.get("composition",""), r.get("description","")), axis=1)

    # Encodage images (principal)
    uniq_imgs = sorted(set(df["anchor_img"].tolist() + df["pos_img"].tolist()))
    uniq_imgs = ensure_paths_exist(uniq_imgs)
    emb_img = {}
    B = args.batch_size
    for i in tqdm(range(0, len(uniq_imgs), B), desc="Encodage images (principal)"):
        batch_paths = uniq_imgs[i:i+B]
        pil_batch = []
        for p in batch_paths:
            try:
                pil_batch.append(Image.open(p).convert("RGB"))
            except Exception:
                if not args.skip_missing:
                    raise
        if not pil_batch: continue
        z = enc_imgs_main(pil_batch)  # CPU tensor (N,d)
        for pth, vec in zip(batch_paths, z):
            emb_img[pth] = vec.detach().to("cpu")

    # Dominante couleur des candidats (depuis l'image)
    pos_color_from_img = {}
    for p in tqdm(sorted(set(df["pos_img"])), desc="Couleur candidates (HSV)"):
        try:
            im = Image.open(p).convert("RGB").resize((160,160))
            pos_color_from_img[p] = infer_color_from_image(im)
        except Exception:
            pos_color_from_img[p] = "unknown"

    # OpenCLIP: prompts pour ancres + encodage des images candidates
    text_feat_by_anchor = {}
    pos_img_openclip = {}
    if args.use_text and enc_text is not None:
        prompts = []
        keys = []
        for _, r in df.iterrows():
            prompts.append(build_prompt(r["name"], r["color_norm"], r["mats_anchor"], r["description"]))
            keys.append(r["anchor_img"])
        zt = enc_text(prompts)  # (N,d) CPU
        for k, v in zip(keys, zt):
            text_feat_by_anchor[k] = v

        pos_paths = sorted(set(df["pos_img"]))
        for i in tqdm(range(0, len(pos_paths), B), desc="OpenCLIP encode pos images"):
            batch_paths = pos_paths[i:i+B]
            pil_batch = []
            for p in batch_paths:
                try:
                    pil_batch.append(Image.open(p).convert("RGB"))
                except Exception:
                    pil_batch.append(Image.new("RGB",(224,224),(200,200,200)))
            zi = enc_imgs_openclip(pil_batch)  # CPU
            for pth, vec in zip(batch_paths, zi):
                pos_img_openclip[pth] = vec

    # Candidats = positives uniques avec embeddings dispos
    pos_items = df[["pos_img","pos_url"]].drop_duplicates().reset_index(drop=True)
    ok_mask = [p in emb_img for p in pos_items["pos_img"]]
    pos_items = pos_items.loc[ok_mask].reset_index(drop=True)
    if pos_items.empty:
        raise SystemExit("Aucun candidat positif encodé. Vérifie les chemins d'images.")
    pos_vecs = torch.stack([emb_img[p] for p in pos_items["pos_img"]])  # (N,d) CPU
    pos_url_by_img = {row["pos_img"]: row["pos_url"] for _, row in pos_items.iterrows()}

    # --------- P@3 (baseline + attributs) ---------
    rows = []
    hits_img, hits_attr = [], []
    for _, r in df.iterrows():
        a_img = r["anchor_img"]; p_img = r["pos_img"]
        if a_img not in emb_img or p_img not in emb_img: continue

        a_vec = emb_img[a_img]  # (d,)
        sims_img = (pos_vecs @ a_vec)  # (N,)
        order_img = torch.argsort(sims_img, descending=True).cpu().numpy().tolist()
        ranked = pos_items.iloc[order_img]["pos_img"].tolist()
        rank_img = ranked.index(p_img)+1 if p_img in ranked else 10**9
        hit_img = int(rank_img <= 3); hits_img.append(hit_img)

        # Attributs par candidate
        c_anchor = r["color_norm"]; mats_anchor = r["mats_anchor"]
        color_s, mat_s, text_s = [], [], []
        for pi in pos_items["pos_img"]:
            cs = color_score(c_anchor, pos_color_from_img.get(pi,"unknown")) if args.use_color else 0.0
            ms = mat_score(mats_anchor, set(), pos_url_by_img[pi]) if args.use_material else 0.0
            ts = 0.0
            if args.use_text and text_feat_by_anchor and pos_img_openclip:
                ta = text_feat_by_anchor.get(a_img, None)
                ti = pos_img_openclip.get(pi, None)
                if ta is not None and ti is not None:
                    ts = float((ta @ ti.T).item())
            color_s.append(cs); mat_s.append(ms); text_s.append(ts)
        color_s = torch.tensor(color_s)
        mat_s   = torch.tensor(mat_s)
        text_s  = torch.tensor(text_s)
        # normalisation douce texte (-1..1)->(0..1) si jamais
        if len(text_s) > 0:
            text_s = (text_s + 1.0) * 0.5

        final = sims_img*args.w_img + color_s*args.w_color + mat_s*args.w_matiere + text_s*args.w_text
        order_attr = torch.argsort(final, descending=True).cpu().numpy().tolist()
        ranked_attr = pos_items.iloc[order_attr]["pos_img"].tolist()
        rank_attr = ranked_attr.index(p_img)+1 if p_img in ranked_attr else 10**9
        hit_attr = int(rank_attr <= 3); hits_attr.append(hit_attr)

        rows.append({
            "anchor_url": r["anchor_url"],
            "anchor_img": a_img,
            "pos_url": r["pos_url"],
            "pos_img": p_img,
            "supercat": supercat_from_url(r["pos_url"]),
            "anchor_color": c_anchor,
            "pos_color_est": pos_color_from_img.get(p_img,"unknown"),
            "rank_img_only": rank_img,
            "rank_with_attrs": rank_attr,
            "delta_rank": rank_img - rank_attr,
            "hit_top3_img": hit_img,
            "hit_top3_attrs": hit_attr,
        })

    p3_img = float(np.mean(hits_img)) if hits_img else 0.0
    p3_attr = float(np.mean(hits_attr)) if hits_attr else 0.0

    # --------- FITB (baseline + attributs) ---------
    fitb_img_hits, fitb_attr_hits = [], []
    for idx, r in df.iterrows():
        a_img = r["anchor_img"]; p_img = r["pos_img"]
        if a_img not in emb_img or p_img not in emb_img: continue

        # Négatifs: de la même supercat si demandé et disponible
        pos_super = supercat_from_url(r["pos_url"])
        pool_all = [o for o in pos_items["pos_img"].tolist() if o != p_img]
        if args.fitb_same_supercat:
            pool_same = [o for o in pool_all if supercat_from_url(pos_url_by_img[o]) == pos_super]
        else:
            pool_same = []
        if len(pool_same) >= args.n_fitb_negs:
            negs = random.sample(pool_same, args.n_fitb_negs)
        else:
            negs = random.sample(pool_all, min(args.n_fitb_negs, len(pool_all)))

        options = [p_img] + negs
        random.Random(args.seed + idx).shuffle(options)

        a_vec = emb_img[a_img]
        opt_vecs = torch.stack([emb_img[o] for o in options])  # (k,d)
        sims_img = (opt_vecs @ a_vec).numpy()  # (k,)

        # attributs par option
        c_anchor = r["color_norm"]; mats_anchor = r["mats_anchor"]
        color_s = np.array([color_score(c_anchor, pos_color_from_img.get(o,"unknown")) if args.use_color else 0.0 for o in options], dtype=np.float32)
        mat_s   = np.array([mat_score(mats_anchor, set(), pos_url_by_img[o]) if args.use_material else 0.0 for o in options], dtype=np.float32)
        text_s  = np.zeros(len(options), dtype=np.float32)
        if args.use_text and text_feat_by_anchor and pos_img_openclip:
            ta = text_feat_by_anchor.get(a_img, None)
            if ta is not None:
                tv = torch.stack([pos_img_openclip[o] for o in options])
                text_s = (ta @ tv.T).numpy().astype(np.float32)
                text_s = (text_s + 1.0) * 0.5

        final = sims_img*args.w_img + color_s*args.w_color + mat_s*args.w_matiere + text_s*args.w_text

        pred_img  = options[int(np.argmax(sims_img))]
        pred_attr = options[int(np.argmax(final))]
        fitb_img_hits.append(int(pred_img == p_img))
        fitb_attr_hits.append(int(pred_attr == p_img))

    fitb_img = float(np.mean(fitb_img_hits)) if fitb_img_hits else 0.0
    fitb_attr = float(np.mean(fitb_attr_hits)) if fitb_attr_hits else 0.0

    # --------- Sauvegardes ---------
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(out / "ablation_p3_details.csv", index=False)

    # Agrégés P@3 par supercat
    per_super_p3 = (df_rows.groupby("supercat")
                    .agg(n=("pos_img","count"),
                         p3_img=("hit_top3_img","mean"),
                         p3_attrs=("hit_top3_attrs","mean"),
                         median_delta_rank=("delta_rank","median"))
                    .reset_index())
    per_super_p3["delta_p3"] = per_super_p3["p3_attrs"] - per_super_p3["p3_img"]
    per_super_p3.to_csv(out / "ablation_supercats_p3.csv", index=False)

    # Agrégés FITB par supercat
    # (on ré-exécute rapidement sur la base des logs FITB → ici on reconstruit depuis fitb_img_hits/fitb_attr_hits par supercat)
    # Pour garder les supercats, on relance un petit passage dédié:
    fitb_logs = []
    for idx, r in df.iterrows():
        a_img = r["anchor_img"]; p_img = r["pos_img"]
        if a_img not in emb_img or p_img not in emb_img: continue
        pos_super = supercat_from_url(r["pos_url"])
        fitb_logs.append({"supercat": pos_super})
    # On mappe au bon nombre de lignes
    df_fitb = pd.DataFrame(fitb_logs)
    if not df_fitb.empty and len(df_fitb) == len(fitb_img_hits):
        df_fitb["correct_img"] = fitb_img_hits
        df_fitb["correct_attrs"] = fitb_attr_hits
        per_super_fitb = (df_fitb.groupby("supercat")
                          .agg(n=("supercat","count"),
                               fitb_img=("correct_img","mean"),
                               fitb_attrs=("correct_attrs","mean"))
                          .reset_index())
        per_super_fitb["delta_fitb"] = per_super_fitb["fitb_attrs"] - per_super_fitb["fitb_img"]
        per_super_fitb.to_csv(out / "ablation_supercats_fitb.csv", index=False)

    # Top gains/drops de rang
    chg = df_rows[["anchor_img","pos_img","pos_url","supercat","rank_img_only","rank_with_attrs","delta_rank"]].copy()
    chg.sort_values("delta_rank", ascending=False).head(5).to_csv(out/"top5_gains.csv", index=False)
    chg.sort_values("delta_rank", ascending=True).head(5).to_csv(out/"top5_drops.csv", index=False)

    summary = {
        "image_model": img_model_name,
        "device": device,
        "pairs": int(len(df)),
        "weights": {"w_img":args.w_img,"w_color":args.w_color,"w_matiere":args.w_matiere,"w_text":args.w_text},
        "options": {
            "use_color": bool(args.use_color),
            "use_material": bool(args.use_material),
            "use_text": bool(args.use_text),
            "fitb_same_supercat": bool(args.fitb_same_supercat),
            "n_fitb_negs": int(args.n_fitb_negs)
        },
        "P@3_baseline": round(p3_img,4),
        "P@3_with_attrs": round(p3_attr,4),
        "FITB_baseline": round(fitb_img,4),
        "FITB_with_attrs": round(fitb_attr,4)
    }
    with (out / "ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== RÉSUMÉ =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # --------- Plots (optionnels) ---------
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # P@3 A/B
        plt.figure(figsize=(5,4))
        plt.bar(["Visuel seul","Visuel+attributs"], [p3_img, p3_attr])
        plt.title("P@3 — A/B")
        plt.ylabel("P@3")
        plt.tight_layout(); plt.savefig(out / "p3_ab.png"); plt.close()

        # FITB A/B
        plt.figure(figsize=(5,4))
        plt.bar(["Visuel seul","Visuel+attributs"], [fitb_img, fitb_attr])
        plt.title("FITB — A/B")
        plt.ylabel("Exactitude")
        plt.tight_layout(); plt.savefig(out / "fitb_ab.png"); plt.close()

        # Δ rang histogram
        deltas = df_rows["delta_rank"].replace([np.inf,-np.inf], np.nan).dropna().astype(float).values
        if deltas.size:
            plt.figure(figsize=(6,4))
            plt.hist(deltas, bins=15)
            plt.title("Distribution des améliorations de rang (+ = mieux)")
            plt.xlabel("Δ rang = rank_img_only - rank_with_attrs")
            plt.ylabel("Nombre de cas")
            plt.tight_layout(); plt.savefig(out / "delta_hist.png"); plt.close()

        # P@3 par supercat (A/B)
        if not per_super_p3.empty:
            top = per_super_p3.sort_values("n", ascending=False).head(8)
            x = np.arange(len(top))
            w = 0.35
            plt.figure(figsize=(8,4))
            plt.bar(x - w/2, top["p3_img"], width=w, label="Visuel seul")
            plt.bar(x + w/2, top["p3_attrs"], width=w, label="Visuel+attributs")
            plt.title("P@3 par super-catégorie (A/B)")
            plt.xticks(x, top["supercat"], rotation=30, ha="right")
            plt.ylabel("P@3"); plt.legend()
            plt.tight_layout(); plt.savefig(out / "p3_supercats.png"); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV des paires Mango")
    ap.add_argument("--out_dir", default="mango_eval_ablation", help="Dossier de sortie")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="Limiter le nombre de lignes (debug)")

    # Heuristiques à activer
    ap.add_argument("--use_color", action="store_true")
    ap.add_argument("--use_material", action="store_true")
    ap.add_argument("--use_text", action="store_true")

    # Poids
    ap.add_argument("--w_img", type=float, default=1.0)
    ap.add_argument("--w_color", type=float, default=0.12)
    ap.add_argument("--w_matiere", type=float, default=0.08)
    ap.add_argument("--w_text", type=float, default=0.08)

    # FITB
    ap.add_argument("--fitb_same_supercat", action="store_true", help="Négatifs FITB dans la même supercat")
    ap.add_argument("--n_fitb_negs", type=int, default=3)

    # Robustesse
    ap.add_argument("--skip_missing", action="store_true", help="Ignorer les images illisibles")

    # Plots
    ap.add_argument("--plot", action="store_true")

    args = ap.parse_args()
    main(args)
