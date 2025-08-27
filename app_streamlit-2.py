import streamlit as st
from pathlib import Path
from typing import Optional, List
from PIL import Image
import os

from fashion_recommendation_system import (
    FashionRecommendationSystem, Outfit, Article
)

# -------------------- Page config --------------------
st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon="🎽",
    layout="wide"
)

# -------------------- Init système (cache) --------------------
@st.cache_resource
def initialize_system():
    """Initialise le système de recommandation (une seule fois)"""
    try:
        ANN_FILE = "images/fashionpedia/instances_attributes_train2020.json"
        IMG_DIR = "images/fashionpedia/train"
        CROP_DIR = "images/fashionpedia/crops_v2"

        st.write("### Diagnostic des fichiers")
        st.write(f"Fichier annotations existe: {os.path.exists(ANN_FILE)}")
        st.write(f"Dossier images existe: {os.path.exists(IMG_DIR)}")
        st.write(f"Dossier crops existe: {os.path.exists(CROP_DIR)}")


        if os.path.exists(IMG_DIR):
            img_files = list(Path(IMG_DIR).glob("*.jpg"))
            st.write(f"Nombre d'images dans {IMG_DIR}: {len(img_files)}")
            
        if os.path.exists(ANN_FILE):
            import json
            with open(ANN_FILE, 'r') as f:
                data = json.load(f)
            st.write(f"Annotations chargées: {len(data.get('annotations', []))} annotations")
            st.write(f"Categories: {len(data.get('categories', []))} catégories")
            st.write(f"Images référencées: {len(data.get('images', []))} images")

        system = FashionRecommendationSystem(ANN_FILE, IMG_DIR, CROP_DIR, ann_backend="faiss")
        system.initialize(max_articles=1500)
        return system, True
    except Exception as e:
        st.error(f"Erreur d'initialisation: {e}")
        return None, False

system, ok = initialize_system()
if not ok or system is None:
    st.stop()

# Après la ligne "system, ok = initialize_system()"
if system:
    st.sidebar.write(f"Articles chargés: {len(system.articles)}")
    st.sidebar.write(f"Groupes disponibles: {list(system.similarity_engine.articles_by_group.keys())}")
    for group, count in system.similarity_engine.articles_by_group.items():
        st.sidebar.write(f"  - {group}: {len(count)} articles")

# -------------------- Sidebar controls --------------------
st.sidebar.header("Paramètres ANN & Ranking")

ann_backend = st.sidebar.selectbox("Backend ANN", options=["faiss", "sklearn"], index=0)
pool_size = st.sidebar.slider("Pool size ANN (top-K)", min_value=50, max_value=300, value=50, step=10)
mmr_lambda = st.sidebar.slider("Diversité MMR (λ)", min_value=0.0, max_value=0.8, value=0.1, step=0.05)

st.sidebar.subheader("Poids du score (re-ranking)")
st.sidebar.subheader("Seuils de qualité (filtres)")
min_seasonal = st.sidebar.slider("Min cohérence saisonnière", 0.0, 1.0, 0.30, 0.01)
min_color = st.sidebar.slider("Min harmonie couleurs", 0.0, 1.0, 0.30, 0.01)
min_overall = st.sidebar.slider("Min score global", 0.0, 1.0, 0.30, 0.01)

st.sidebar.subheader("Autres")
exclude_detected_slot = st.sidebar.checkbox("Exclure le slot détecté dans l'image requête", value=True)

w_cos = st.sidebar.slider("Poids cosine", 0.0, 1.0, 0.1, 0.05)
w_coh = st.sidebar.slider("Poids cohérence outfit", 0.0, 1.0, 0.1, 0.05)
w_red = st.sidebar.slider("Poids pénalité redondance", 0.0, 1.0, 0.1, 0.05)

st.sidebar.markdown("---")
use_builder = st.sidebar.checkbox("Activer Outfit Builder (slots)", value=False)
slots_selected: List[str] = []
topk_slot = 10
beam_size = 10
max_outfits = 3

if use_builder:
    st.sidebar.subheader("Slots")
    slots_selected = st.sidebar.multiselect(
        "Choisir les slots",
        options=["top", "bottom", "shoes", "outer", "bag"],
        default=["top", "bottom", "shoes"]
    )
    topk_slot = st.sidebar.slider("Top-K candidats par slot", 3, 30, 10, 1)
    beam_size = st.sidebar.slider("Beam size", 3, 30, 10, 1)
    max_outfits = st.sidebar.slider("Nombre de tenues à générer", 1, 8, 3, 1)

    st.sidebar.subheader("Règles")
    rule_limit_outer = st.sidebar.checkbox("Limiter outerwear à 1", value=True)
    rule_avoid_print = st.sidebar.checkbox("Éviter clash d’imprimés", value=True)
else:
    # defaults silencieux si builder off
    rule_limit_outer = True
    rule_avoid_print = True

st.sidebar.markdown("---")
if st.sidebar.button("Reconstruire index ANN"):
    try:
        system.rebuild_ann_indices(ann_backend=ann_backend)
        st.sidebar.success("Index ANN reconstruits ✔")
    except Exception as e:
        st.sidebar.error(f"Erreur rebuild ANN: {e}")

# Ajoutez ce test dans streamlit_app.py
if st.sidebar.button("Test direct FashionCLIP"):
    from fashion_clip.fashion_clip import FashionCLIP
    model = FashionCLIP("patrickjohncyh/fashion-clip")
    
    # Test sur une seule image
    test_img = next(Path("images/fashionpedia/crops_v2").glob("*.jpg"), None)
    if test_img:
        try:
            result = model.encode_images([str(test_img)], batch_size=1)
            st.sidebar.write(f"Résultat brut: {type(result)}")
            if hasattr(result, '__len__'):
                st.sidebar.write(f"Longueur: {len(result)}")
                if len(result) > 0:
                    emb = result[0]
                    st.sidebar.write(f"Premier élément: type={type(emb)}, shape={getattr(emb, 'shape', 'no_shape')}")
                    if hasattr(emb, 'detach'):
                        emb_np = emb.detach().cpu().numpy()
                        st.sidebar.write(f"Après conversion: shape={emb_np.shape}, norm={np.linalg.norm(emb_np)}")
        except Exception as e:
            st.sidebar.error(f"Erreur test: {e}")

# Pousse les hyperparamètres au recommender courant
if system.recommender:
    system.recommender.pool_size = pool_size
    system.recommender.w_cos = w_cos
    system.recommender.w_coh = w_coh
    system.recommender.w_red = w_red
    system.recommender.mmr_lambda = mmr_lambda
    system.recommender.set_quality_thresholds(min_seasonal, min_color, min_overall)




# -------------------- UI principale --------------------
st.title("🎽 Fashion Recommendation System — ANN + Builder")

st.markdown("Charge une image de référence (article porté) pour recommander des pièces compatibles ou composer une tenue complète.")

uploaded = st.file_uploader("Image de requête", type=["jpg", "jpeg", "png"])
colL, colR = st.columns([1, 2])

with colL:
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Image de requête", use_container_width=True)

run = st.button("Lancer la recommandation")

if run:
    if not uploaded:
        st.warning("Charge d'abord une image.")
        st.stop()

    # Sauvegarde temporaire
    tmp_dir = Path("tmp_queries")
    tmp_dir.mkdir(exist_ok=True)
    query_path = str(tmp_dir / "query.jpg")
    Image.open(uploaded).convert("RGB").save(query_path, "JPEG", quality=90)

    def debug_simple_search(system, query_path):
        """Test basique sans filtres"""
        emb_gen = EmbeddingGenerator() 
        q_embs = emb_gen.generate_embeddings([query_path])
        if not q_embs or q_embs[0] is None:
            return "Pas d'embedding généré"
        
        q = q_embs[0]
        results = {}
        
        for group in ["top", "bottom", "shoes"]:
            if group in system.similarity_engine.articles_by_group:
                pool = system.similarity_engine.ann_search(q, group, topk=5)
                results[group] = len(pool)
            else:
                results[group] = "Groupe absent"
        
        return results

    # Ajoutez ce bouton dans l'interface :
    if st.sidebar.button("Debug search"):
        if uploaded:
            tmp_path = "tmp_queries/debug.jpg"
            Image.open(uploaded).convert("RGB").save(tmp_path)
            debug_results = debug_simple_search(system, tmp_path)
            st.sidebar.json(debug_results)

    if not use_builder:
        with st.spinner("Recherche ANN + re-ranking + MMR…"):
            outfit = system.get_recommendation(
                query_image_path=query_path,
                target_groups=["top", "bottom", "shoes"] if exclude_detected_slot else ["top","bottom","shoes"],
                similarity_threshold=0.3,
                use_advanced_scoring=True
            )

        if outfit is None:
            st.warning("Aucune tenue n'a passé les seuils actuels.")
            with st.expander("👉 Astuces pour débloquer"):
                st.markdown(
                    "- Augmente **Pool size** à 200–300\n"
                    "- Baisse **MMR λ** vers 0.1–0.2 (moins de diversité)\n"
                    "- Réduis **Pénalité redondance** à 0.05\n"
                    "- Abaisse **Min score global** à 0.22–0.25\n"
                    "- (Builder) Monte **Top-K par slot** à 15–20 et **Beam** à 15\n"
                )
            # Proposer un essai auto plus permissif
            if st.button("Essayer automatiquement avec paramètres permissifs"):
                # temporairement assouplir
                system.recommender.pool_size = max(system.recommender.pool_size, 200)
                system.recommender.mmr_lambda = min(system.recommender.mmr_lambda, 0.2)
                system.recommender.w_red = min(system.recommender.w_red, 0.05)
                system.recommender.set_quality_thresholds(
                    min(min_seasonal, 0.20),
                    min(min_color, 0.20),
                    min(min_overall, 0.25),
                )
                with st.spinner("Relance en mode permissif…"):
                    outfit = system.get_recommendation(
                        query_image_path=query_path,
                        target_groups=["top", "bottom", "shoes"],
                        similarity_threshold=0.25,
                        use_advanced_scoring=True
                    )
            if outfit is None:
                st.error("Toujours rien. Essaie une autre image (produit plus centré) ou repasse en mode Builder.")
                st.stop()

        # affichage si dispo (inchangé)
        with colR:
            st.subheader(f"Tenue recommandée — score {outfit.coherence_score:.2f}")
            if hasattr(outfit, "relaxed_thresholds"):
                ms, mc, mo = outfit.relaxed_thresholds
                st.caption(f"(Seuils auto-relâchés pour afficher un résultat : saison {ms:.2f}, couleurs {mc:.2f}, global {mo:.2f})")
            gcols = st.columns(len(outfit.items))
            for c, art in zip(gcols, outfit.items):
                path = getattr(art, "crop_path", None) or art.image_path
                c.image(path, caption=f"{art.group} — {art.category_name}", use_container_width=True)

            st.markdown("#### Explication")
            st.code(system.get_outfit_explanation(outfit))


    else:
        rules = {
            "limit_outer": rule_limit_outer,
            "avoid_print_clash": rule_avoid_print,
            "penalize_running_gala": False,
        }
        with st.spinner("Génération de tenues (Builder + beam)…"):
            outfits = system.get_recommendations_builder(
                query_image_path=query_path,
                slots=slots_selected,
                topk_per_slot=topk_slot,
                beam_size=beam_size,
                max_outfits=max_outfits,
                rules=rules
            )

        if not outfits:
            st.warning("Aucune tenue n'a passé les seuils actuels (Builder).")
            with st.expander("👉 Astuces pour débloquer"):
                st.markdown(
                    "- Augmente **Top-K par slot** à 15–20\n"
                    "- Augmente **Beam size** à 15–20\n"
                    "- Diminue **MMR λ** vers 0.1–0.2\n"
                    "- Réduis **Pénalité redondance** à 0.05\n"
                    "- Assouplis les seuils: global 0.22–0.25, saison/couleurs 0.20\n"
                    "- Désactive temporairement la règle clash d’imprimés"
                )
            if st.button("Essayer automatiquement (Builder permissif)"):
                system.recommender.pool_size = max(system.recommender.pool_size, 220)
                system.recommender.mmr_lambda = min(system.recommender.mmr_lambda, 0.2)
                system.recommender.w_red = min(system.recommender.w_red, 0.05)
                system.recommender.set_quality_thresholds(
                    min(min_seasonal, 0.20),
                    min(min_color, 0.20),
                    min(min_overall, 0.24),
                )
                with st.spinner("Relance Builder en mode permissif…"):
                    outfits = system.get_recommendations_builder(
                        query_image_path=query_path,
                        slots=slots_selected or ["top","bottom","shoes"],
                        topk_per_slot=max(topk_slot, 15),
                        beam_size=max(beam_size, 15),
                        max_outfits=max_outfits,
                        rules={
                            "limit_outer": rule_limit_outer,
                            "avoid_print_clash": False,  # relax
                            "penalize_running_gala": False,
                        }
                    )

        if not outfits:
            st.error("Toujours aucune tenue. Essaie une autre image (plus centrée) ou active moins de slots.")
            st.stop()


        st.subheader("Tenues générées")
        for i, outfit in enumerate(outfits, 1):
            st.markdown(f"### Tenue {i} — score {outfit.coherence_score:.2f}")
            gcols = st.columns(len(outfit.items))
            for c, art in zip(gcols, outfit.items):
                path = getattr(art, "crop_path", None) or art.image_path
                c.image(path, caption=f"{art.group} — {art.category_name}", use_container_width=True)
            with st.expander("Explication"):
                st.code(system.get_outfit_explanation(outfit))
