# -*- coding: utf-8 -*-
"""
Topology & Influence (Plotly) — prêt pour mémoire
- Lit influence_mode.xlsx
- Calcule centralités (degree, betweenness, eigenvector)
- Figures Plotly (HTML interactif) + PNG statique via kaleido
- Exports CSV (centralités + tops utiles pour guider l'analyse)

Usage:
  python topology_plotly.py

Sorties (dans ./figures_plotly):
  - graphe_influenceurs_marques.html/.png
  - heatmap_cooccurrence_marques.html/.png
  - graphe_marques_communautes.html/.png
  - centralites_influenceurs_marques.csv
  - centralites_marques_graph.csv
  - top_influenceurs_centr_eig.csv
  - top_marques_centr_eig.csv
  - marques_ponts_betweenness.csv
  - top_paires_marques_edges.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import plotly.graph_objects as go
import plotly.express as px

# ---------------- Config ----------------
DATA_PATH = Path("influence_mode.xlsx")  # adapte si besoin
OUT_DIR = Path("figures_plotly")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOPK_INFLUENCERS = 15
TOPK_BRANDS = 15
SEED = 42

# ---------------- Utils ----------------
def to_num(s):
    """Cast numérique safe."""
    try:
        return float(s)
    except Exception:
        return np.nan

def safe_eig_centrality(G, weight="weight"):
    """Eigenvector centrality robuste (numpy -> fallback power method)."""
    try:
        return nx.eigenvector_centrality_numpy(G, weight=weight)
    except Exception:
        try:
            return nx.eigenvector_centrality(G, weight=weight, max_iter=1000)
        except Exception:
            return {n: 0.0 for n in G.nodes()}

def write_both(fig: go.Figure, stem: Path):
    """Enregistre HTML interactif + PNG statique (kaleido)."""
    html_path = stem.with_suffix(".html")
    png_path = stem.with_suffix(".png")
    fig.write_html(html_path, include_plotlyjs="cdn")
    try:
        fig.write_image(png_path, scale=2, width=1300, height=900)
    except Exception as e:
        print(f"[!] PNG non généré (kaleido manquant ?): {e}")
    print(f"- {html_path.name}  (et PNG)")

# ---------------- Chargement ----------------
print("Lecture Excel…")
xls = pd.ExcelFile(DATA_PATH)
df_inf = pd.read_excel(xls, "Influenceurs_scores")
df_br  = pd.read_excel(xls, "Marques_scores")
df_eib = pd.read_excel(xls, "Edges_InfBrand")
df_ebb = pd.read_excel(xls, "Edges_BrandBrand")
df_co  = pd.read_excel(xls, "Cooccurrence", index_col=0)

# casts utiles
for c in ["influence_composite","collab_avg","followers_num","engagement_rate"]:
    if c in df_inf.columns:
        df_inf[c] = pd.to_numeric(df_inf[c], errors="coerce")
for c in ["brand_influence_score","mentions_per_month","collab_freq_avg","stories_pct"]:
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

# nœuds marques
for _, r in df_br.iterrows():
    m = str(r["Marque"])
    G.add_node(("brand", m),
               label=m,
               ntype="brand",
               brand_score=float(r.get("brand_influence_score", 0) or 0.0))

# nœuds influenceurs
for _, r in df_inf.iterrows():
    h = str(r["Handle"])
    cat = str(r.get("Catégorie",""))
    ntype = "macro" if cat.lower().startswith("macro") else "micro"
    G.add_node(("influencer", h),
               label=h,
               ntype=ntype,
               inf_score=float(r.get("influence_composite", 0) or 0.0))

# arêtes
for _, e in df_eib.iterrows():
    u = ("influencer", str(e["source_influencer"]))
    v = ("brand", str(e["target_brand"]))
    if u in G and v in G:
        w = float(e.get("weight", 0) or 0.0)
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

# sous-graphe lisible : garder les edges liés aux top
keep_nodes = set()
for u, v in G.edges():
    if (u in top_inf_keys) or (v in top_br_keys):
        keep_nodes.add(u); keep_nodes.add(v)
H = G.subgraph(keep_nodes).copy()
pos_H = nx.spring_layout(H.to_undirected(), seed=SEED, k=0.7)

# centralités sur H (non orienté)
H_und = H.to_undirected()
deg_H = nx.degree_centrality(H_und)
btw_H = nx.betweenness_centrality(H_und, weight="weight", normalized=True)
eig_H = safe_eig_centrality(H_und, weight="weight")

# table centralités (influenceurs + marques)
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

# --- Figure Plotly: H
edge_x, edge_y = [], []
for u, v, d in H.edges(data=True):
    x0, y0 = pos_H[u]; x1, y1 = pos_H[v]
    edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                        line=dict(width=1, color="#BBBBBB"), hoverinfo="none", opacity=0.5)

node_x, node_y, sizes, texts, colors, symbols = [], [], [], [], [], []
for n, a in H.nodes(data=True):
    x, y = pos_H[n]
    node_x.append(x); node_y.append(y)
    base = a.get("inf_score", a.get("brand_score", 0.0))
    sizes.append(10 + 50 * float(base))
    texts.append(
        f"{a['label']}<br>type={a['ntype']}<br>"
        f"score={base:.3f}<br>"
        f"deg={deg_H.get(n,0):.3f} | betw={btw_H.get(n,0):.3f} | eig={eig_H.get(n,0):.3f}"
    )
    if a["ntype"] == "brand":
        colors.append("#E74C3C"); symbols.append("circle")
    elif a["ntype"] == "macro":
        colors.append("#3498DB"); symbols.append("square")
    else:
        colors.append("#27AE60"); symbols.append("triangle-up")

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    text=[H.nodes[n]["label"] for n in H.nodes()],
    textposition="top center",
    hovertext=texts, hoverinfo="text",
    marker=dict(size=sizes, color=colors, symbol=symbols, line=dict(width=1, color="#333"))
)
fig_H = go.Figure([edge_trace, node_trace])
fig_H.update_layout(
    title="Graphe influenceurs → marques (sous-graphe top)",
    hovermode="closest", showlegend=False,
    margin=dict(l=0, r=0, t=40, b=0)
)
write_both(fig_H, OUT_DIR/"graphe_influenceurs_marques")

# ---------------- Heatmap co-occurrences marques ----------------
co_mat = df_co.values.astype(float)
brands = df_co.index.astype(str).tolist()
# réordonner par “degré binaire” (nb co-mentions > 0)
order = np.argsort(-np.sum(co_mat > 0, axis=1))
co_ord = co_mat[order][:, order]
brands_ord = [brands[i] for i in order]

fig_heat = go.Figure(go.Heatmap(z=co_ord, x=brands_ord, y=brands_ord, coloraxis="coloraxis"))
fig_heat.update_layout(
    title="Co-occurrence des marques (pondérée)",
    xaxis_nticks=40, yaxis_autorange="reversed",
    margin=dict(l=60, r=20, t=40, b=100),
    coloraxis=dict(colorbar=dict(title="Poids"))
)
write_both(fig_heat, OUT_DIR/"heatmap_cooccurrence_marques")

# ---------------- Graphe marques↔marques (+ communautés) ----------------
GB = nx.Graph()
for _, r in df_ebb.iterrows():
    a, b = str(r["brand_a"]), str(r["brand_b"])
    w = float(r.get("weight", 0) or 0.0)
    if w > 0:
        GB.add_edge(a, b, weight=w)

if GB.number_of_edges() > 0:
    weights = [d["weight"] for *_, d in GB.edges(data=True)]
    thr = np.percentile(weights, 30) if len(weights) > 0 else 0.0
    GB.remove_edges_from([(u, v) for u, v, d in GB.edges(data=True) if d["weight"] < thr])
    GB.remove_nodes_from([n for n in list(GB.nodes()) if GB.degree(n) == 0])

comms = list(greedy_modularity_communities(GB, weight="weight")) if GB.number_of_edges() else []
com_id = {}
for i, com in enumerate(comms):
    for n in com: com_id[n] = i
for n in GB.nodes():
    com_id.setdefault(n, -1)

deg_B = nx.degree_centrality(GB) if GB.number_of_nodes() else {}
btw_B = nx.betweenness_centrality(GB, weight="weight", normalized=True) if GB.number_of_nodes() else {}
eig_B = safe_eig_centrality(GB, weight="weight") if GB.number_of_nodes() else {}

df_cent_B = pd.DataFrame([
    {"marque": n, "community": com_id.get(n, -1),
     "degree_centrality": deg_B.get(n, 0.0),
     "betweenness_centrality": btw_B.get(n, 0.0),
     "eigenvector_centrality": eig_B.get(n, 0.0),
     "degree": GB.degree(n)}
    for n in GB.nodes()
]).sort_values(["community","eigenvector_centrality"], ascending=[True, False])
df_cent_B.to_csv(OUT_DIR/"centralites_marques_graph.csv", index=False)

pos_B = nx.spring_layout(GB, seed=SEED, k=0.6) if GB.number_of_nodes() else {}

edge_xB, edge_yB = [], []
for u, v in GB.edges():
    x0, y0 = pos_B[u]; x1, y1 = pos_B[v]
    edge_xB += [x0, x1, None]; edge_yB += [y0, y1, None]
edge_trace_B = go.Scatter(x=edge_xB, y=edge_yB, mode="lines",
                          line=dict(width=1, color="#BBBBBB"), hoverinfo="none", opacity=0.5)

palette = px.colors.qualitative.Set3 + px.colors.qualitative.Safe + px.colors.qualitative.T10
node_xB, node_yB, sizeB, colorB, textB, labelB = [], [], [], [], [], []
for n in GB.nodes():
    x, y = pos_B[n]
    node_xB.append(x); node_yB.append(y)
    sizeB.append(10 + 6 * GB.degree(n))
    c = com_id.get(n, -1)
    colorB.append(palette[c % len(palette)] if c >= 0 else "#777777")
    textB.append(f"{n}<br>deg={deg_B.get(n,0):.3f} | betw={btw_B.get(n,0):.3f} | eig={eig_B.get(n,0):.3f} | com={c}")
    labelB.append(n)

node_trace_B = go.Scatter(
    x=node_xB, y=node_yB,
    mode="markers+text",
    text=labelB, textposition="top center",
    hovertext=textB, hoverinfo="text",
    marker=dict(size=sizeB, color=colorB, line=dict(width=1, color="#333"))
)
fig_B = go.Figure([edge_trace_B, node_trace_B])
fig_B.update_layout(
    title="Graphe marques ↔ marques (communautés)",
    hovermode="closest", showlegend=False,
    margin=dict(l=0, r=0, t=40, b=0)
)
write_both(fig_B, OUT_DIR/"graphe_marques_communautes")

# ---------------- Exports “guides d’analyse” ----------------
topk = 10
top_infl_by_eig = (df_cent_H[df_cent_H["type"].isin(["macro","micro"])]
                   .nlargest(topk, "eigenvector_centrality")
                   [["noeud","type","eigenvector_centrality","betweenness_centrality","degree_centrality"]]
                   .reset_index(drop=True))
top_br_by_eig = (df_cent_H[df_cent_H["type"]=="brand"]
                 .nlargest(topk, "eigenvector_centrality")
                 [["noeud","eigenvector_centrality","betweenness_centrality","degree_centrality"]]
                 .reset_index(drop=True))
bridge_brands = (df_cent_B.nlargest(topk, "betweenness_centrality")
                 [["marque","community","betweenness_centrality","eigenvector_centrality","degree"]]
                 .reset_index(drop=True))
edges_B = (pd.DataFrame([(u, v, d.get("weight", 0.0)) for u, v, d in GB.edges(data=True)],
                        columns=["brand_a","brand_b","weight"])
           .sort_values("weight", ascending=False).head(20).reset_index(drop=True))

top_infl_by_eig.to_csv(OUT_DIR/"top_influenceurs_centr_eig.csv", index=False)
top_br_by_eig.to_csv(OUT_DIR/"top_marques_centr_eig.csv", index=False)
bridge_brands.to_csv(OUT_DIR/"marques_ponts_betweenness.csv", index=False)
edges_B.to_csv(OUT_DIR/"top_paires_marques_edges.csv", index=False)

print("\nFini ✅  Figures & CSV dans:", OUT_DIR.resolve())
