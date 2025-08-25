import streamlit as st
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional

# Import du système de recommandation
from fashion_recommendation_system import FashionRecommendationSystem, Outfit

# Configuration de la page
st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon="🎽",
    layout="wide"
)

@st.cache_resource
def initialize_system():
    """Initialise le système de recommandation (une seule fois)"""
    try:
        ANN_FILE = "images/fashionpedia/instances_attributes_train2020.json"
        IMG_DIR = "images/fashionpedia/train"
        CROP_DIR = "images/fashionpedia/crops_v2"
        
        system = FashionRecommendationSystem(ANN_FILE, IMG_DIR, CROP_DIR)
        system.initialize(max_articles=1500)
        return system, True
    except Exception as e:
        st.error(f"Erreur d'initialisation: {e}")
        return None, False

def display_outfit_results(outfit: Outfit, query_image_path: str):
    """Affiche les résultats de recommandation"""
    st.success(f"Outfit généré - Cohérence: {outfit.coherence_score:.3f}")
    
    # Calculer le nombre de colonnes nécessaires
    num_items = len(outfit.items)
    cols = st.columns(num_items + 1)  # +1 pour l'image de référence
    
    # Afficher l'image de référence
    with cols[0]:
        st.subheader("Référence")
        try:
            ref_img = Image.open(query_image_path)
            st.image(ref_img, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur affichage référence: {e}")
    
    # Afficher les recommandations
    for i, (item, similarity) in enumerate(zip(outfit.items, outfit.similarities)):
        with cols[i + 1]:
            st.subheader(item.group.title())
            
            try:
                # Afficher l'image cropée
                crop_path = Path("images/fashionpedia/crops_v2") / f"{item.id}.jpg"
                if crop_path.exists():
                    item_img = Image.open(crop_path)
                    st.image(item_img, use_container_width=True)
                else:
                    st.warning("Image non disponible")
                
                # Informations sur l'item
                st.write(f"**Catégorie:** {item.category_name}")
                st.write(f"**Similarité:** {similarity:.3f}")
                
                # Barre de progression pour la similarité
                progress_value = min(similarity, 1.0)
                st.progress(float(progress_value))
                
                # Afficher les attributs si disponibles
                if item.attributes:
                    st.write(f"**Attributs:** {len(item.attributes)}")
                
            except Exception as e:
                st.error(f"Erreur affichage item: {e}")

def main():
    st.title("Système de Recommandation de Mode")
    st.markdown("Système basé sur FashionCLIP et le dataset Fashionpedia")
    st.markdown("---")
    
    # Initialisation du système
    system, is_loaded = initialize_system()
    
    if not is_loaded:
        st.error("Impossible de charger le système de recommandation")
        st.stop()
    
    # Sidebar pour la configuration
    st.sidebar.title("Configuration")
    
    # Sélection des groupes cibles
    available_groups = ["top", "bottom", "shoes", "accessory", "bag"]
    target_groups = st.sidebar.multiselect(
        "Types de vêtements à recommander:",
        available_groups,
        default=["top", "bottom", "shoes"]
    )
    
    # Seuil de similarité
    similarity_threshold = st.sidebar.slider(
        "Seuil de similarité minimum:",
        0.0, 1.0, 0.25, 0.05,
        help="Items avec une similarité inférieure seront ignorés"
    )
    
    # Affichage des statistiques du système
    if st.sidebar.checkbox("Afficher les statistiques"):
        stats = system.get_stats()
        st.sidebar.write("**Articles disponibles:**")
        for group, count in stats.items():
            st.sidebar.write(f"- {group}: {count}")
        st.sidebar.write(f"**Total:** {sum(stats.values())}")
    
    # Interface principale
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Image de référence")
        
        uploaded_file = st.file_uploader(
            "Sélectionnez une image de vêtement",
            type=["jpg", "jpeg", "png"],
            help="Uploadez une image claire d'un vêtement"
        )
        
        if uploaded_file is not None:
            # Afficher l'image uploadée
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Image de référence", use_container_width=True)
                
                # Validation de base de l'image
                if image.size[0] < 100 or image.size[1] < 100:
                    st.warning("Image trop petite. Recommandé: au moins 100x100 pixels")
                
                # Informations sur l'image
                st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} pixels")
                st.write(f"**Format:** {image.format}")
                
                # Bouton de génération
                if st.button("Générer les recommandations", type="primary"):
                    if len(target_groups) == 0:
                        st.warning("Sélectionnez au moins un type de vêtement")
                    else:
                        generate_recommendations(
                            system, uploaded_file, target_groups, similarity_threshold
                        )
                        
            except Exception as e:
                st.error(f"Erreur lors du chargement de l'image: {e}")
    
    with col2:
        st.subheader("Recommandations")
        
        # Zone d'affichage des résultats
        if 'recommendation_results' in st.session_state:
            outfit, query_path = st.session_state.recommendation_results
            display_outfit_results(outfit, query_path)
        else:
            st.info("Uploadez une image et cliquez sur 'Générer' pour voir les recommandations")
    
    # Section d'aide
    st.markdown("---")
    with st.expander("Guide d'utilisation"):
        st.markdown("""
        **Comment utiliser le système:**
        
        1. **Upload d'image:** Sélectionnez une image claire d'un vêtement
        2. **Configuration:** Ajustez les paramètres dans la sidebar
        3. **Génération:** Cliquez sur le bouton pour obtenir les recommandations
        
        **Conseils pour de meilleures recommandations:**
        - Utilisez des images avec un bon contraste
        - Préférez des images où le vêtement est bien visible
        - Évitez les images trop sombres ou floues
        - Ajustez le seuil de similarité si peu de résultats
        """)
    
    # Informations techniques
    with st.expander("Informations techniques"):
        st.markdown(f"""
        **Architecture du système:**
        - Modèle: FashionCLIP (patrickjohncyh/fashion-clip)
        - Dataset: Fashionpedia
        - Recherche: k-NN avec distance cosine
        - Articles chargés: {sum(system.get_stats().values()) if system else 0}
        """)


def generate_recommendations(system: FashionRecommendationSystem, 
                           uploaded_file, target_groups, similarity_threshold):
    """Génère les recommandations et met à jour l'interface"""
    
    with st.spinner("Génération des recommandations en cours..."):
        try:
            # Créer un fichier temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Générer les recommandations
            outfit = system.get_recommendation(
                tmp_file_path, 
                target_groups=target_groups,
                similarity_threshold=similarity_threshold,
                use_advanced_scoring = True
            )
            
            if outfit and len(outfit.items) > 0:
                # GARDE le fichier pour l'affichage - ne le supprime PAS ici
                st.session_state.recommendation_results = (outfit, tmp_file_path)
                st.success(f"Recommandations générées avec succès!")
                st.rerun()
            else:
                # Supprime seulement si pas de résultat
                os.unlink(tmp_file_path)
                st.warning(
                    "Aucune recommandation trouvée avec les critères actuels. "
                    "Essayez de réduire le seuil de similarité."
                )
                    
        except Exception as e:
            st.error(f"Erreur lors de la génération: {e}")

# Point d'entrée
if __name__ == "__main__":
    main()