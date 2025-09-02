# -*- coding: utf-8 -*-
"""
Interactive draggable networks (PyVis) — HTML only (no PNG)
- Lit influence_mode.xlsx
- Construit 2 graphes interactifs (déplaçables à la souris) :
  1) Influenceurs → Marques (sous-graphe top, biparti gauche→droite)
  2) Marques ↔ Marques (communautés, layout physique libre)
- Exporte centralités en CSV pour analyse

Usage:
  python topology_interactive_draggable.py

Sorties (dans ./figures_interactive):
  - influenceurs_marques_draggable.html
  - marques_communautes_draggable.html
  - centralites_influenceurs_marques.csv
  - centralites_marques_graph.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from pyvis.network import Network

# ---------------- Config ----------------
DATA_PATH = Path("influence_mode.xlsx")
OUT_DIR = Path("figures_interactive")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOPK_INFLUENCERS = 15
TOPK_BRANDS = 15
SEED = 42

# ---------------- Chargement ----------------
print("Lecture Excel…")
xls = pd.ExcelFile(DATA_PATH)
df_inf = pd.read_excel(xls, "Influenceurs_scores")
df_br  = pd.read_excel(xls, "Marques_scores")
df_eib = pd.read_excel(xls, "Edges_InfBrand")
df_ebb = pd.read_excel(xls, "Edges_BrandBrand")
# df_co  = pd.read_excel(xls, "Cooccurrence", index_col=0)  # utile si besoin plus tard

# Casts utiles
for c in ["influence_composite", "collab_avg", "followers_num", "engagement_rate"]:
    if c in df_inf.columns:
        df_inf[c] = pd.to_numeric(df_inf[c], errors="coerce")
for c in ["brand_influence_score", "mentions_per_month", "collab_freq_avg", "stories_pct"]:
    if c in df_br.columns:
        df_br[c] = pd.to_numeric(df_br[c], errors="coerce")

# ---------------- Sous-ensembles top (lisibilité) ----------------
top_inf = (df_inf
           .sort_values("influence_composite", ascending=False)
           .head(TOPK_INFLUENCERS)
           [["Handle","Catégorie","followers_num","engagement_rate","collab_avg","influence_composite"]])

top_br = (df_br
          .sort_values("brand_influence_score", ascending=False)
          .head(TOPK_BRANDS)
          [["Marque","Catégorie principale","mentions_per_month","collab_freq_avg","stories_pct","brand_influence_score"]])

top_inf_keys = {("influencer", h) for h in top_inf["Handle"].astype(str)}
top_br_keys  = {("brand", m) for m in top_br["Marque"].astype(str)}

# ---------------- Graphe influenceurs → marques (DiGraph) ----------------
G = nx.DiGraph()

# Nœuds marques
for _, r in df_br.iterrows():
    m = str(r["Marque"])
    G.add_node(("brand", m),
               label=m,
               ntype="brand",
               brand_score=float(r.get("brand_influence_score", 0) or 0.0))

# Nœuds influenceurs
for _, r in df_inf.iterrows():
    h = str(r["Handle"])
    cat = str(r.get("Catégorie",""))
    ntype = "macro" if cat.lower().startswith("macro") else "micro"
    G.add_node(("influencer", h),
               label=h,
               ntype=ntype,
               inf_score=float(r.get("influence_composite", 0) or 0.0))

# Arêtes
for _, e in df_eib.iterrows():
    u = ("influencer", str(e["source_influencer"]))
    v = ("brand", str(e["target_brand"]))
    if u in G and v in G:
        w = float(e.get("weight", 0) or 0.0)
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

# Sous-graphe lisible : garder les edges liés aux top
keep_nodes = set()
for u, v in G.edges():
    if (u in top_inf_keys) or (v in top_br_keys):
        keep_nodes.add(u); keep_nodes.add(v)
H = G.subgraph(keep_nodes).copy()

# ---- Centralités sur H (non orienté) pour tooltips
H_und = H.to_undirected()
deg_H = nx.degree_centrality(H_und)
btw_H = nx.betweenness_centrality(H_und, weight="weight", normalized=True)
try:
    eig_H = nx.eigenvector_centrality_numpy(H_und, weight="weight")
except Exception:
    eig_H = nx.eigenvector_centrality(H_und, weight="weight", max_iter=1000)

# Export centralités
cent_rows = []
for n, a in H.nodes(data=True):
    base = a.get("inf_score", a.get("brand_score", 0.0))
    cent_rows.append({
        "noeud": a["label"],
        "type": a["ntype"],
        "degree_centrality": deg_H.get(n, 0.0),
        "betweenness_centrality": btw_H.get(n, 0.0),
        "eigenvector_centrality": eig_H.get(n, 0.0),
        "score_attribut": base
    })
df_cent_H = (pd.DataFrame(cent_rows)
             .sort_values(["type","eigenvector_centrality"], ascending=[True, False]))
df_cent_H.to_csv(OUT_DIR/"centralites_influenceurs_marques.csv", index=False)

# ---- PyVis (biparti, L→R), DRAG ENABLED
net_ib = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="#222222", directed=True, notebook=False)
net_ib.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=180, spring_strength=0.02, damping=0.4)

# JSON VALIDE (pas de var/options JS)
net_ib.set_options("""
{
  "interaction": {
    "hover": true,
    "dragNodes": true,
    "dragView": true,
    "zoomView": true
  },
  "physics": {
    "enabled": true,
    "stabilization": {"iterations": 150}
  },
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "LR",
      "sortMethod": "directed"
    }
  },
  "edges": {
    "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
    "smooth": {"type": "dynamic"}
  }
}
""")

def node_style(ntype):
    if ntype == "brand":
        return dict(color="#E74C3C", shape="dot")     # marque: cercle rouge
    if ntype == "macro":
        return dict(color="#3498DB", shape="square")  # macro: carré bleu
    return dict(color="#27AE60", shape="triangle")    # micro: triangle vert

# Ajout des nœuds
for n, a in H.nodes(data=True):
    base = a.get("inf_score", a.get("brand_score", 0.0))
    style = node_style(a["ntype"])
    size = 10 + 30*float(base)
    title = (f"<b>{a['label']}</b><br>"
             f"type={a['ntype']}<br>"
             f"score={base:.3f}<br>"
             f"deg={deg_H.get(n,0):.3f} | betw={btw_H.get(n,0):.3f} | eig={eig_H.get(n,0):.3f}")
    net_ib.add_node(str(n), label=a["label"], title=title,
                    color=style["color"], shape=style["shape"], size=size)

# Ajout des arêtes
for u, v, d in H.edges(data=True):
    w = float(d.get("weight", 0.0))
    net_ib.add_edge(str(u), str(v), value=w, width=max(1, 4*w), title=f"weight={w:.3f}")

html_ib = OUT_DIR / "influenceurs_marques_draggable.html"
net_ib.write_html(str(html_ib), notebook=False, open_browser=False)
print(f"- HTML interactif (draggable) écrit: {html_ib}")

# ---------------- Graphe marques ↔ marques ----------------
GB = nx.Graph()
for _, r in df_ebb.iterrows():
    a, b = str(r["brand_a"]), str(r["brand_b"])
    w = float(r.get("weight", 0) or 0.0)
    if w > 0:
        GB.add_edge(a, b, weight=w)

# Filtrage léger pour lisibilité
if GB.number_of_edges() > 0:
    weights = [d["weight"] for *_, d in GB.edges(data=True)]
    thr = np.percentile(weights, 30) if len(weights) > 0 else 0.0
    GB.remove_edges_from([(u, v) for u, v, d in GB.edges(data=True) if d["weight"] < thr])
    GB.remove_nodes_from([n for n in list(GB.nodes()) if GB.degree(n) == 0])

# Communautés
comms = list(greedy_modularity_communities(GB, weight="weight")) if GB.number_of_edges() else []
com_id = {}
for i, com in enumerate(comms):
    for n in com:
        com_id[n] = i
for n in GB.nodes():
    com_id.setdefault(n, -1)

# Centralités pour tooltips
deg_B = nx.degree_centrality(GB) if GB.number_of_nodes() else {}
btw_B = nx.betweenness_centrality(GB, weight="weight", normalized=True) if GB.number_of_nodes() else {}
try:
    eig_B = nx.eigenvector_centrality_numpy(GB, weight="weight") if GB.number_of_nodes() else {}
except Exception:
    eig_B = nx.eigenvector_centrality(GB, weight="weight", max_iter=1000) if GB.number_of_nodes() else {}

df_cent_B = pd.DataFrame([
    {"marque": n, "community": com_id.get(n, -1),
     "degree_centrality": deg_B.get(n, 0.0),
     "betweenness_centrality": btw_B.get(n, 0.0),
     "eigenvector_centrality": eig_B.get(n, 0.0),
     "degree": GB.degree(n)}
    for n in GB.nodes()
]).sort_values(["community","eigenvector_centrality"], ascending=[True, False])
df_cent_B.to_csv(OUT_DIR/"centralites_marques_graph.csv", index=False)

# PyVis pour GB (physique libre), DRAG ENABLED
net_bb = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="#222222", notebook=False)
net_bb.barnes_hut(gravity=-16000, central_gravity=0.15, spring_length=180, spring_strength=0.02, damping=0.4)

net_bb.set_options("""
{
  "interaction": {
    "hover": true,
    "dragNodes": true,
    "dragView": true,
    "zoomView": true
  },
  "physics": {
    "enabled": true,
    "stabilization": {"iterations": 200}
  },
  "edges": {
    "smooth": {"type": "dynamic"}
  }
}
""")

# Palette simple par communauté
palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
           "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

for n in GB.nodes():
    com = com_id.get(n, -1)
    color = palette[com % len(palette)] if com >= 0 else "#777777"
    size = 10 + 6*GB.degree(n)
    title = (f"<b>{n}</b><br>"
             f"com={com}<br>"
             f"deg={deg_B.get(n,0):.3f} | betw={btw_B.get(n,0):.3f} | eig={eig_B.get(n,0):.3f}")
    net_bb.add_node(n, label=n, title=title, color=color, shape="dot", size=size)

for u, v, d in GB.edges(data=True):
    w = float(d.get("weight", 0.0))
    net_bb.add_edge(u, v, value=w, width=max(1, 4*w), title=f"weight={w:.3f}")

html_bb = OUT_DIR / "marques_communautes_draggable.html"
net_bb.write_html(str(html_bb), notebook=False, open_browser=False)
print(f"- HTML interactif (draggable) écrit: {html_bb}")

print("\n✅ Ouvre ces fichiers dans ton navigateur et déplace les nœuds librement :")
print(f" - {html_ib}")
print(f" - {html_bb}")
