# -*- coding: utf-8 -*-
"""
Influence Mode — Figures & Exports pour mémoire
- Charge influence_mode.xlsx (multi-feuilles)
- Produit:
  1) Graphe d'influence multi-couches (Influenceur -> Marque)
  2) Heatmap co-occurrence marques
  3) Graphe marques↔marques + communautés (greedy modularity)
- Exporte tables LaTeX (Top-N) + JSON (D3) + PNG/SVG haute résolution

Prérequis: pandas, openpyxl, matplotlib, networkx, numpy
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

# ---------------- Config ----------------
EXCEL_PATH = r"influence_mode.xlsx"   # <-- mets ton chemin ici
OUT_DIR = Path("figures_memoires")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOPK_INFLUENCERS = 15
TOPK_BRANDS = 15
DPI = 300  # qualité publication

# ---------- Lecture Excel ----------
xls = pd.ExcelFile(EXCEL_PATH)
df_inf_scores = pd.read_excel(xls, "Influenceurs_scores")
df_br_scores  = pd.read_excel(xls, "Marques_scores")
df_edges_inf_brand = pd.read_excel(xls, "Edges_InfBrand")
df_edges_brand_brand = pd.read_excel(xls, "Edges_BrandBrand")
df_co = pd.read_excel(xls, "Cooccurrence", index_col=0)

# Nettoyages légers
for c in ["influence_composite", "collab_avg", "followers_num", "engagement_rate"]:
    if c in df_inf_scores.columns:
        df_inf_scores[c] = pd.to_numeric(df_inf_scores[c], errors="coerce")

for c in ["brand_influence_score", "mentions_per_month", "collab_freq_avg", "stories_pct"]:
    if c in df_br_scores.columns:
        df_br_scores[c] = pd.to_numeric(df_br_scores[c], errors="coerce")

# ---------- Tables Top-N -> LaTeX ----------
top_inf = (df_inf_scores
           .sort_values("influence_composite", ascending=False)
           .head(TOPK_INFLUENCERS)
           [["Handle","Catégorie","followers_num","engagement_rate","collab_avg","influence_composite"]])

top_br = (df_br_scores
          .sort_values("brand_influence_score", ascending=False)
          .head(TOPK_BRANDS)
          [["Marque","Catégorie principale","mentions_per_month","collab_freq_avg","stories_pct","brand_influence_score"]])

(top_inf
 .rename(columns={
     "Handle":"Influenceur", "Catégorie":"Cat.",
     "followers_num":"Abonnés", "engagement_rate":"Taux d’engagement",
     "collab_avg":"Collabs/mois (moy.)", "influence_composite":"Score composite"
 })
 .to_latex(OUT_DIR/"table_top_influenceurs.tex", index=False, float_format="%.3f"))

(top_br
 .rename(columns={
     "Marque":"Marque", "Catégorie principale":"Cat.",
     "mentions_per_month":"Mentions/mois", "collab_freq_avg":"Collabs/mois (moy.)",
     "stories_pct":"Stories (%)", "brand_influence_score":"Score marque"
 })
 .to_latex(OUT_DIR/"table_top_marques.tex", index=False, float_format="%.3f"))

print("Tables LaTeX écrites:",
      OUT_DIR/"table_top_influenceurs.tex",
      OUT_DIR/"table_top_marques.tex", sep="\n- ")

# ---------- 1) Graphe d'influence multi-couches ----------
# Nœuds: influenceurs (Macro/Micro), marques. Arêtes: source_influencer -> target_brand (weight)
G = nx.DiGraph()

# Ajouter marques d'abord (utiles pour taille/couleur)
brand_nodes = set(df_br_scores["Marque"].dropna().astype(str).tolist())
for b, s in zip(df_br_scores["Marque"], df_br_scores["brand_influence_score"]):
    G.add_node(("brand", b), label=b, ntype="brand",
               brand_score=float(s) if pd.notna(s) else 0.0)

# Influenceurs
for _, r in df_inf_scores.iterrows():
    handle = str(r["Handle"])
    cat = str(r["Catégorie"])
    score = float(r["influence_composite"]) if pd.notna(r["influence_composite"]) else 0.0
    G.add_node(("influencer", handle), label=handle, ntype="macro" if cat.lower().startswith("macro") else "micro",
               inf_score=score)

# Arêtes
for _, e in df_edges_inf_brand.iterrows():
    src = ("influencer", str(e["source_influencer"]))
    tgt = ("brand", str(e["target_brand"]))
    if src in G and tgt in G:
        w = float(e["weight"]) if pd.notna(e["weight"]) else 0.0
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += w
        else:
            G.add_edge(src, tgt, weight=w)

# Sous-graphe top: garder nœuds reliés aux top-K (pour lisibilité)
top_inf_keys = set([("influencer", h) for h in top_inf["Handle"].astype(str)])
top_br_keys  = set([("brand", m) for m in top_br["Marque"].astype(str)])
keep_nodes = set()
for u, v, d in G.edges(data=True):
    if u in top_inf_keys or v in top_br_keys:
        keep_nodes.add(u); keep_nodes.add(v)

H = G.subgraph(keep_nodes).copy()

# Layout (spring sur graph non orienté pour stabilité visuelle)
pos = nx.spring_layout(H.to_undirected(), seed=42, k=0.7)

# Styles
node_sizes = []
node_colors = []
for n, attrs in H.nodes(data=True):
    if attrs["ntype"] == "brand":
        # Taille ∝ score marque
        s = max(200, 2000 * attrs.get("brand_score", 0.0))
        node_sizes.append(s)
        node_colors.append("#E74C3C")   # rouge marques
    elif attrs["ntype"] == "macro":
        s = max(150, 1800 * attrs.get("inf_score", 0.0))
        node_sizes.append(s)
        node_colors.append("#3498DB")   # bleu macro
    else:
        s = max(80, 1200 * attrs.get("inf_score", 0.0))
        node_sizes.append(s)
        node_colors.append("#27AE60")   # vert micro

edge_widths = [max(0.5, 6.0 * d.get("weight", 0.0)) for _,_,d in H.edges(data=True)]

plt.figure(figsize=(12, 9))
nx.draw_networkx_edges(H, pos, width=edge_widths, alpha=0.25, arrows=False)
nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color=node_colors, linewidths=0.5, edgecolors="#333333")
# Labels: seulement pour nœuds les plus “grands”
labels = {}
for n, s in zip(H.nodes(), node_sizes):
    if s >= np.percentile(node_sizes, 65):
        labels[n] = H.nodes[n]["label"]
nx.draw_networkx_labels(H, pos, labels=labels, font_size=9)
plt.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR/"fig_graphe_influence_multicouches.png", dpi=DPI)
plt.savefig(OUT_DIR/"fig_graphe_influence_multicouches.svg")
plt.close()
print("- Figure 1 OK:", OUT_DIR/"fig_graphe_influence_multicouches.png")

# ---------- 2) Heatmap co-occurrence marques ----------
co_mat = df_co.values.astype(float)
brands = df_co.index.astype(str).tolist()

# Option: réordonner par degré pour lisibilité
deg = np.argsort(-np.sum(co_mat>0, axis=1))
co_ord = co_mat[deg][:,deg]
brands_ord = [brands[i] for i in deg]

plt.figure(figsize=(10, 9))
plt.imshow(co_ord, interpolation="nearest", aspect="auto")
plt.title("Co-occurrence des marques (pondérée par influence des profils)")
plt.xlabel("Marques")
plt.ylabel("Marques")
plt.xticks(ticks=np.arange(len(brands_ord)), labels=brands_ord, rotation=90, fontsize=7)
plt.yticks(ticks=np.arange(len(brands_ord)), labels=brands_ord, fontsize=7)
cbar = plt.colorbar()
cbar.set_label("Poids de co-mention")
plt.tight_layout()
plt.savefig(OUT_DIR/"fig_heatmap_cooccurrence_marques.png", dpi=DPI)
plt.savefig(OUT_DIR/"fig_heatmap_cooccurrence_marques.svg")
plt.close()
print("- Figure 2 OK:", OUT_DIR/"fig_heatmap_cooccurrence_marques.png")

# ---------- 3) Graphe marques↔marques + communautés ----------
GB = nx.Graph()
for _, row in df_edges_brand_brand.iterrows():
    a, b = str(row["brand_a"]), str(row["brand_b"])
    w = float(row["weight"]) if pd.notna(row["weight"]) else 0.0
    if w > 0:
        GB.add_edge(a, b, weight=w)

# Filtrage léger (optionnel) pour lisibilité: enlever très faibles poids
THRESH = np.percentile([d["weight"] for *_, d in GB.edges(data=True)], 30) if GB.number_of_edges() else 0.0
to_remove = [(u,v) for u,v,d in GB.edges(data=True) if d["weight"] < THRESH]
GB.remove_edges_from(to_remove)
GB.remove_nodes_from([n for n in list(GB.nodes()) if GB.degree(n) == 0])

# Communautés
if GB.number_of_edges() > 0:
    comms = list(greedy_modularity_communities(GB, weight="weight"))
else:
    comms = []

# Mapping communauté -> couleur
palette = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728",
    "#9467bd","#8c564b","#e377c2","#7f7f7f",
    "#bcbd22","#17becf"
]
node_color_map = {}
for i, com in enumerate(comms):
    for n in com:
        node_color_map[n] = palette[i % len(palette)]
# défaut si isolé
for n in GB.nodes():
    node_color_map.setdefault(n, "#7f7f7f")

pos_b = nx.spring_layout(GB, seed=42, k=0.6)
node_sizes_b = [600 + 60*GB.degree(n) for n in GB.nodes()]
node_colors_b = [node_color_map[n] for n in GB.nodes()]
edge_widths_b = [max(0.4, 4.0*d.get("weight", 0.0)) for *_, d in GB.edges(data=True)]

plt.figure(figsize=(11, 9))
nx.draw_networkx_edges(GB, pos_b, width=edge_widths_b, alpha=0.3)
nx.draw_networkx_nodes(GB, pos_b, node_size=node_sizes_b, node_color=node_colors_b, edgecolors="#333333", linewidths=0.6)
nx.draw_networkx_labels(GB, pos_b, font_size=9)
plt.title("Graphe de co-occurrence des marques (clusters de communautés)")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR/"fig_graphe_cooccurrence_communautes.png", dpi=DPI)
plt.savefig(OUT_DIR/"fig_graphe_cooccurrence_communautes.svg")
plt.close()
print("- Figure 3 OK:", OUT_DIR/"fig_graphe_cooccurrence_communautes.png")

# ---------- 4) Export JSON (D3-ready) ----------
# Pour un Sankey/force-directed dans le web
nodes_json = []
node_index = {}

# Ajoute influenceurs du sous-graphe H
for i, (n_type, name) in enumerate(H.nodes()):
    node_index[(n_type, name)] = len(nodes_json)
    if n_type == "brand":
        nodes_json.append({
            "id": name, "group": "brand",
            "score": float(H.nodes[(n_type, name)].get("brand_score", 0.0))
        })
    else:
        nodes_json.append({
            "id": name, "group": H.nodes[(n_type, name)]["ntype"],  # "macro"/"micro"
            "score": float(H.nodes[(n_type, name)].get("inf_score", 0.0))
        })

links_json = []
for u, v, d in H.edges(data=True):
    links_json.append({
        "source": H.nodes[u]["label"],
        "target": H.nodes[v]["label"],
        "value": float(d.get("weight", 0.0))
    })

with open(OUT_DIR/"graph_influence_d3.json", "w", encoding="utf-8") as f:
    json.dump({"nodes": nodes_json, "links": links_json}, f, ensure_ascii=False, indent=2)

print("- JSON D3 écrit:", OUT_DIR/"graph_influence_d3.json")

print("\nTout est prêt. Figures dans:", OUT_DIR.resolve())

# ================== 5) Graphe biparti (propre & lisible) ==================
# Influenceurs (gauche) → Marques (droite). On garde seulement les liens entre
# TOP influenceurs et TOP marques, et on filtre sous la médiane des poids.

# Reprend les TOP déjà calculés plus haut (top_inf, top_br)
top_inf_set = set(top_inf["Handle"].astype(str))
top_br_set  = set(top_br["Marque"].astype(str))

E = df_edges_inf_brand[
    df_edges_inf_brand["source_influencer"].astype(str).isin(top_inf_set)
    & df_edges_inf_brand["target_brand"].astype(str).isin(top_br_set)
].copy()

if not E.empty:
    thr = np.nanpercentile(pd.to_numeric(E["weight"], errors="coerce"), 50)  # médiane
    E = E[pd.to_numeric(E["weight"], errors="coerce") >= thr]

# Construire un graphe orienté biparti
GBI = nx.DiGraph()

# Nœuds influenceurs
sub_inf = df_inf_scores[df_inf_scores["Handle"].astype(str).isin(top_inf_set)]
for _, r in sub_inf.iterrows():
    GBI.add_node(("influencer", r["Handle"]),
                 layer=0,
                 label=str(r["Handle"]),
                 ntype=("macro" if str(r["Catégorie"]).lower().startswith("macro") else "micro"),
                 size=float(r["influence_composite"]) if pd.notna(r["influence_composite"]) else 0.0)

# Nœuds marques
sub_br = df_br_scores[df_br_scores["Marque"].astype(str).isin(top_br_set)]
for _, r in sub_br.iterrows():
    GBI.add_node(("brand", r["Marque"]),
                 layer=1,
                 label=str(r["Marque"]),
                 ntype="brand",
                 size=float(r["brand_influence_score"]) if pd.notna(r["brand_influence_score"]) else 0.0)

# Arêtes
for _, e in E.iterrows():
    u = ("influencer", str(e["source_influencer"]))
    v = ("brand", str(e["target_brand"]))
    if u in GBI and v in GBI:
        w = float(e["weight"]) if pd.notna(e["weight"]) else 0.0
        if GBI.has_edge(u, v):
            GBI[u][v]["weight"] += w
        else:
            GBI.add_edge(u, v, weight=w)

# Positionnement biparti : deux colonnes alignées
left_nodes  = [n for n, a in GBI.nodes(data=True) if a.get("layer") == 0]
right_nodes = [n for n, a in GBI.nodes(data=True) if a.get("layer") == 1]

pos = {}
y_left  = np.linspace(0, 1, len(left_nodes))  if left_nodes  else []
y_right = np.linspace(0, 1, len(right_nodes)) if right_nodes else []
for i, n in enumerate(left_nodes):
    pos[n] = (0.0, float(y_left[i]))
for i, n in enumerate(right_nodes):
    pos[n] = (1.0, float(y_right[i]))

# Tailles des nœuds (échelle simple, sans imposer de couleurs)
size_map = {}
for n, a in GBI.nodes(data=True):
    if a["ntype"] == "brand":
        s = max(150, 2200 * a.get("size", 0.0))
    elif a["ntype"] == "macro":
        s = max(120, 1800 * a.get("size", 0.0))
    else:  # micro
        s = max(90, 1300 * a.get("size", 0.0))
    size_map[n] = s

edge_widths = [max(0.5, 5.0 * d.get("weight", 0.0)) for *_, d in GBI.edges(data=True)]

plt.figure(figsize=(12, 8))
nx.draw_networkx_edges(GBI, pos, width=edge_widths, alpha=0.25, arrows=False)

macro_nodes = [n for n, a in GBI.nodes(data=True) if a["ntype"] == "macro"]
micro_nodes = [n for n, a in GBI.nodes(data=True) if a["ntype"] == "micro"]
brand_nodes = [n for n, a in GBI.nodes(data=True) if a["ntype"] == "brand"]

nx.draw_networkx_nodes(GBI, pos,
                       nodelist=macro_nodes,
                       node_size=[size_map[n] for n in macro_nodes],
                       node_shape="s", linewidths=0.6, edgecolors="black")
nx.draw_networkx_nodes(GBI, pos,
                       nodelist=micro_nodes,
                       node_size=[size_map[n] for n in micro_nodes],
                       node_shape="^", linewidths=0.6, edgecolors="black")
nx.draw_networkx_nodes(GBI, pos,
                       nodelist=brand_nodes,
                       node_size=[size_map[n] for n in brand_nodes],
                       node_shape="o", linewidths=0.6, edgecolors="black")

# Labels seulement pour les plus gros nœuds (évite le fouillis)
sizes_sorted = sorted(size_map.values())
if sizes_sorted:
    cutoff = sizes_sorted[int(0.6 * len(sizes_sorted))]  # top 40% par taille
    labels = {n: GBI.nodes[n]["label"] for n, s in size_map.items() if s >= cutoff}
    nx.draw_networkx_labels(GBI, pos, labels=labels, font_size=9)

plt.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_graphe_influence_multicouches_biparti.png", dpi=DPI)
plt.savefig(OUT_DIR / "fig_graphe_influence_multicouches_biparti.svg")
plt.close()
print("- Figure 1-bis OK:", OUT_DIR / "fig_graphe_influence_multicouches_biparti.png")

# ================== 6) Exports TOP en CSV/LaTeX 'pretty' + PNG ==================
def _fmt_int(x):
    if pd.isna(x): return ""
    try:
        return f"{int(round(float(x))):,}".replace(",", " ")
    except Exception:
        return str(x)

def _fmt_pct(x, digits=2, factor=100.0):
    if pd.isna(x): return ""
    try:
        return f"{float(x)*factor:.{digits}f}%"
    except Exception:
        return str(x)

def _fmt_float(x, digits=3):
    if pd.isna(x): return ""
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)

def dataframe_to_png(df: pd.DataFrame, out_path: Path, title: str | None = None, fontsize=9, rowheight=0.9):
    """Rendu simple d'un DataFrame en PNG via matplotlib (sans styles/couleurs forcés)."""
    n_rows, n_cols = df.shape
    fig_w = max(8, min(16, n_cols * 1.2))
    fig_h = max(2.5, n_rows * rowheight + (1.0 if title else 0.4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize+2, pad=12)
    table = ax.table(cellText=df.astype(str).values,
                     colLabels=list(df.columns),
                     loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.2)
    for i in range(df.shape[1]):
        table.auto_set_column_width(col=(i,))
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# Prépare TOP influenceurs "pretty"
top_inf_pretty = (top_inf
    .rename(columns={
        "Handle":"Influenceur","Catégorie":"Catégorie",
        "followers_num":"Abonnés","engagement_rate":"Taux d’engagement",
        "collab_avg":"Collabs/mois (moy.)","influence_composite":"Score composite"
    }).copy())

top_inf_pretty["Abonnés"] = top_inf_pretty["Abonnés"].map(_fmt_int)
top_inf_pretty["Taux d’engagement"] = top_inf_pretty["Taux d’engagement"].map(lambda x: _fmt_pct(x, 2, 100))
top_inf_pretty["Collabs/mois (moy.)"] = top_inf_pretty["Collabs/mois (moy.)"].map(lambda x: _fmt_float(x, 1))
top_inf_pretty["Score composite"] = top_inf_pretty["Score composite"].map(lambda x: _fmt_float(x, 3))

# Prépare TOP marques "pretty"
top_br_pretty = (top_br
    .rename(columns={
        "Marque":"Marque","Catégorie principale":"Catégorie",
        "mentions_per_month":"Mentions/mois","collab_freq_avg":"Collabs/mois (moy.)",
        "stories_pct":"Stories (%)","brand_influence_score":"Score marque"
    }).copy())

top_br_pretty["Mentions/mois"] = top_br_pretty["Mentions/mois"].map(_fmt_int)
top_br_pretty["Collabs/mois (moy.)"] = top_br_pretty["Collabs/mois (moy.)"].map(lambda x: _fmt_float(x, 1))
top_br_pretty["Stories (%)"] = top_br_pretty["Stories (%)"].map(lambda x: _fmt_pct(x, 0, 100))
top_br_pretty["Score marque"] = top_br_pretty["Score marque"].map(lambda x: _fmt_float(x, 3))

# Exports CSV
(top_inf_pretty).to_csv(OUT_DIR / "table_top_influenceurs.csv", index=False)
(top_br_pretty).to_csv(OUT_DIR / "table_top_marques.csv", index=False)

# Exports LaTeX "pretty" (valeurs déjà formatées en str)
(top_inf_pretty).to_latex(OUT_DIR / "table_top_influenceurs_pretty.tex", index=False, escape=True)
(top_br_pretty).to_latex(OUT_DIR / "table_top_marques_pretty.tex", index=False, escape=True)

# PNG des tableaux
dataframe_to_png(top_inf_pretty, OUT_DIR / "table_top_influenceurs.png", title="TOP Influenceurs (score composite)")
dataframe_to_png(top_br_pretty,  OUT_DIR / "table_top_marques.png",     title="TOP Marques (score influence)")

print("- Tables TOP exportées (CSV, LaTeX pretty, PNG) dans:", OUT_DIR.resolve())
