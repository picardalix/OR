
from __future__ import annotations
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from fashion_recommendation_system import (
    FashionRecommendationSystem,
    ensure_demo_images,
    DEFAULT_SLOTS,
)
from advanced_outfit_analysis import AdvancedOutfitScorer
from metrics import recall_at_k, ndcg_at_k, intra_list_diversity

st.set_page_config(page_title="Fashion Reco (FAISS + MMR)", page_icon="🎽", layout="wide")

# -------------- Cached system -----------------
@st.cache_resource(show_spinner=True)
def load_system(img_dir: str, max_items: Optional[int] = 200):
    sys = FashionRecommendationSystem(img_dir)
    sys.initialize(max_items)
    return sys

def _filters_ui():
    st.sidebar.header("Filtres")
    cat = st.sidebar.selectbox("Catégorie", options=["", "top", "bottom", "shoes"], index=0)
    color = st.sidebar.selectbox("Couleur", options=["", "red", "green", "blue"], index=0)
    season = st.sidebar.selectbox("Saison", options=["", "spring/summer", "fall/winter"], index=0)
    constraints = {}
    if cat: constraints["category"] = cat
    if color: constraints["color"] = color
    if season: constraints["season"] = season
    return constraints

def _export_figures(selected_paths: List[str], cat_matrix, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Export top images in a single grid
    imgs = [Image.open(p).convert("RGB").resize((256,256)) for p in selected_paths]
    cols = min(4, len(imgs))
    rows = int(np.ceil(len(imgs)/cols))
    canvas = Image.new("RGB", (256*cols, 256*rows), (255,255,255))
    for idx, im in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        canvas.paste(im, (256*c, 256*r))
    grid_path = out_dir / "tops_grid.png"
    canvas.save(grid_path)

    # Export category compatibility heatmap (heuristic)
    cats = ["top", "bottom", "shoes"]
    M = np.array([[cat_matrix.get((a,b),0.7) for b in cats] for a in cats], dtype=float)
    plt.figure()
    plt.imshow(M, cmap="viridis")
    plt.xticks(range(len(cats)), cats)
    plt.yticks(range(len(cats)), cats)
    plt.colorbar()
    plt.title("Compatibilité catégories (heuristique)")
    heatmap_path = out_dir / "category_compatibility.png"
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()

    return [grid_path, heatmap_path]

def main():
    st.title("🎽 Fashion Recommendation — FAISS + MMR + Explications")
    default_dir = Path("/mnt/data/images")
    ensure_demo_images(default_dir)
    img_dir = st.text_input("Dossier images", value=str(default_dir))

    constraints = _filters_ui()
    query_text = st.text_input("Requête (CLIP-like)", value="minimal chic summer outfit")

    lambda_div = st.slider("Diversité (λ pour MMR)", 0.0, 0.9, 0.3, 0.05)
    diversify = st.button("🔀 Diversifier (λ +0.2)")  # increases diversity once

    # Load system
    try:
        system = load_system(img_dir, max_items=400)
    except Exception as e:
        st.error(f"Erreur d'initialisation: {e}")
        st.stop()

    if diversify:
        lambda_div = min(0.9, lambda_div + 0.2)

    tab1, tab2 = st.tabs(["Recommandation article", "Construction d’outfit"])

    with tab1:
        st.subheader("Top articles similaires")
        # Use the pipeline but request 1 slot to get a single top item re-ranked
        items, scores, reasons = system.recommend_outfit(
            query_text,
            constraints=constraints,
            slots=["top"],  # treat as single-slot for items
            ann_topk=60,
            per_slot=30,
            lambda_div=lambda_div
        )
        if not items:
            st.info("Aucun résultat.")
        else:
            cols = st.columns(min(5, len(items)))
            for c, art in zip(cols, items):
                with c:
                    st.image(Image.open(art.image_path), caption=f"{art.category} | {art.color} | {art.season}", use_container_width=True)
            st.caption(f"Scores globaux — pertinence: {scores['relevance']:.3f} | cohérence: {scores['coherence']:.3f} | λ={scores['mmr_lambda']:.2f}")
            # Reasons
            st.markdown("**Pourquoi ? (top 3 contributions)**")
            for name, w in reasons:
                st.write(f"- {name}: {w:.3f}")

    with tab2:
        st.subheader("Outfit multi-slots")
        slots = DEFAULT_SLOTS
        items, scores, reasons = system.recommend_outfit(
            query_text,
            constraints=constraints,
            slots=slots,
            ann_topk=80,
            per_slot=25,
            lambda_div=lambda_div
        )
        if not items:
            st.info("Aucun résultat.")
        else:
            cols = st.columns(len(items))
            for c, art in zip(cols, items):
                with c:
                    st.image(Image.open(art.image_path), caption=f"{art.category} | {art.color} | {art.season}", use_container_width=True)

            st.caption(f"Scores globaux — pertinence: {scores['relevance']:.3f} | cohérence: {scores['coherence']:.3f} | λ={scores['mmr_lambda']:.2f}")

            # Advanced scorer explanations
            scorer = AdvancedOutfitScorer(system.cat_matrix)
            q_vec = system.text_to_embedding(query_text, system.dim)
            br = scorer.compatibility(items, q_vec)
            st.markdown("**Explications (façon SHAP)**")
            comps = sorted(br.components.items(), key=lambda x: x[1], reverse=True)
            for k, v in comps[:3]:
                st.write(f"- {k}: {v:.3f}")

            # Metrics (toy): compute diversity on selected
            from metrics import intra_list_diversity
            ild = intra_list_diversity([a.embedding for a in items])
            st.write({"intra_list_diversity": float(ild)})

            # Export figures
            if st.button("📤 Exporter figures pour mémoire"):
                out_dir = Path("/mnt/data/exports")
                paths = _export_figures([a.image_path for a in items], system.cat_matrix, out_dir)
                st.success("Figures exportées.")
                for p in paths:
                    st.markdown(f"- [{p.name}]({p.as_posix()})")

if __name__ == "__main__":
    main()
