
import streamlit as st
from pathlib import Path
from PIL import Image
import os
import io
from glob import glob
import pandas as pd
from user_preferences import UserPreferences, PreferenceScorer, apply_preference_rerank

from fashion_recommendation_system import FashionRecommendationSystem

# ==================== CONFIGURATION ET CACHE ====================
st.set_page_config(
    page_title="Fashion AI Studio",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    /* Variables CSS */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-accent: #f1f5f9;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }

    .block-container { padding: 1rem 2rem 2rem 2rem; }
    div[data-testid="stHorizontalBlock"],
    div[data-testid="column"],
    .element-container { background: transparent !important; border: none !important; box-shadow: none !important; }
    div[data-testid="column"] { padding: 0 !important; }
    div[data-testid="stExpander"] { background: transparent !important; border: 1px solid var(--border-color) !important; border-radius: 12px !important; }

    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 2rem 1.5rem; border-radius: 20px; margin-bottom: 2rem; color: white; text-align: center; box-shadow: var(--shadow-lg);
    }
    .main-header h1 {
        margin: 0; font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(45deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem; }

    .upload-zone { border: 3px dashed var(--primary-color); border-radius: 20px; padding: 2rem; text-align: center;
                   background: linear-gradient(135deg, #f8faff, #f1f5f9); transition: all 0.3s ease; margin: 1rem 0; }
    .upload-zone:hover { border-color: var(--secondary-color); background: linear-gradient(135deg, #f1f5f9, #e0e7ff); }

    .status-badge { display: inline-block; padding: 0.5rem 1rem; border-radius: 50px; font-size: 0.875rem; font-weight: 600; margin: 0.25rem; }
    .badge-locked { background: linear-gradient(135deg, var(--warning-color), #fbbf24); color: white; }
    .badge-active { background: linear-gradient(135deg, var(--success-color), #34d399); color: white; }
    .badge-info { background: linear-gradient(135deg, var(--accent-color), #22d3ee); color: white; }

    .stButton > button { border-radius: 12px; border: none; font-weight: 600; transition: all 0.3s ease; box-shadow: var(--shadow); width: 100%; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: var(--shadow-lg); }

    .actions-container { background: var(--bg-secondary); border-radius: 12px; padding: 1rem; margin: 1rem 0; border: 1px solid var(--border-color); }

    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px);} to { opacity: 1; transform: translateY(0);} }
    .animate-fade-in { animation: fadeInUp 0.6s ease-out; }

    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .main-header p { font-size: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des states
if "locks" not in st.session_state:
    st.session_state["locks"] = {"top": None, "bottom": None, "shoes": None, "outer": None, "bag": None}
if "query_path" not in st.session_state:
    st.session_state["query_path"] = None
if "target_groups" not in st.session_state:
    st.session_state["target_groups"] = ["top", "bottom", "shoes"]
if "show_gallery" not in st.session_state:
    st.session_state["show_gallery"] = False
if "use_advanced" not in st.session_state:
    st.session_state["use_advanced"] = False
if "ui_mode" not in st.session_state:
    st.session_state["ui_mode"] = "idle"
if "last_outfit_view" not in st.session_state:
    st.session_state["last_outfit_view"] = None
if "uploaded_preview" not in st.session_state:
    st.session_state["uploaded_preview"] = None
if "include_shoes" not in st.session_state:
    st.session_state["include_shoes"] = True
if "pref_weight" not in st.session_state:
    st.session_state["pref_weight"] = 0.35
if "new_similar_idx" not in st.session_state:
    st.session_state["new_similar_idx"] = 0 



# ==================== FONCTIONS UTILITAIRES ====================

def _resolve_display_path(art):
    # Priorité à display_path (JPG)
    p = getattr(art, "display_path", None)
    if p and os.path.exists(p):
        return p
    # Sinon, tenter le même nom en .jpg
    cand = (getattr(art, "crop_path", None)
            or getattr(art, "feature_path", None)
            or getattr(art, "image_path", None))
    if cand:
        jpg = str(Path(cand).with_suffix(".jpg"))
        if os.path.exists(jpg):
            return jpg
        if os.path.exists(cand):
            return cand
    return getattr(art, "image_path", None)

def safe_get_color_summaries(system, outfit):
    adv = getattr(outfit, "advanced_score", None)
    summaries = getattr(adv, "item_color_summaries", None) if adv else None
    return summaries


def _explain_outfit_safe(system, outfit) -> dict:
    analysis = {}
    
    adv = getattr(outfit, "advanced_score", None)
    season_label = (
        getattr(outfit, "season_label", None)
        or getattr(outfit, "season", None)
        or (getattr(adv, "predicted_season", None) if adv else None)
        or (getattr(adv, "season_label", None) if adv else None)
        or "inconnue"
    )
    season_conf = float(
        getattr(outfit, "season_confidence", None)
        or (getattr(adv, "season_confidence", None) if adv else 0.0)
        or (getattr(adv, "seasonal_coherence", 0.0) if adv else 0.0)
    )
    analysis["season"] = {"label": season_label, "confidence": season_conf}

    redundancy = float(
        getattr(outfit, "redundancy", None)
        or (getattr(adv, "redundancy", 0.0) if adv else 0.0)
    )
    analysis["redundancy"] = redundancy

    pairs = (
        getattr(outfit, "harmonious_pairs", None)
        or (getattr(adv, "harmonious_pairs", None) if adv else None)
        or (getattr(adv, "color_pairs", None) if adv else None)
    )
    analysis["color_harmony"] = {"ok": bool(pairs), "pairs": pairs or []}

    # --- Détails scores par item (comme avant)
    rows = []
    for it in getattr(outfit, "items", []):
        sdict = getattr(it, "scores", {}) or {}
        cosine = getattr(it, "cosine", None) or sdict.get("cosine")
        color_score = getattr(it, "color_score", None)
        if color_score is None:
            color_score = sdict.get("color") or sdict.get("color_harmony")
        season_score = getattr(it, "season_score", None) or sdict.get("season")

        rows.append({
            "slot": getattr(it, "group", getattr(it, "slot", None)),
            "item_id": getattr(it, "id", None),
            "name": getattr(it, "category_name", "") or getattr(it, "name", ""),
            "cosine": cosine,
            "color_score": color_score,
            "season_score": season_score,
        })
    analysis["item_scores"] = rows

    # --- NOUVEAU : résumés couleurs par item (palette/tonalité/temp/saturation/score)
    analysis["item_color_summaries"] = safe_get_color_summaries(system, outfit)

    print("SCORES ITEMS:", [(r["item_id"], r["cosine"], r["color_score"], r["season_score"]) for r in rows])
    return analysis



def format_explanation_md(analysis: dict) -> str:
    ch = analysis.get("color_harmony") or {}
    ok = bool(ch.get("ok"))
    pairs = ch.get("pairs") or []

    season_d = analysis.get("season") or {}
    season = season_d.get("label") or "inconnue"
    conf = float(season_d.get("confidence") or 0.0)
    redund = float(analysis.get("redundancy") or 0.0)

    lines = []
    lines.append(f"- **Harmonie couleurs** : {'OK' if ok else 'à améliorer'}")
    if ok and pairs:
        def pair_to_str(p):
            if isinstance(p, (list, tuple)) and len(p) >= 2: return f"{p[0]}-{p[1]}"
            if isinstance(p, dict): return f"{p.get('a','?')}-{p.get('b','?')}"
            return str(p)
        lines.append("  - Paires : " + ", ".join(pair_to_str(p) for p in pairs[:3]))
    lines.append(f"- **Saison** : {season} *(confiance {conf:.0%})*")
    lines.append(f"- {' **Diversité optimale**' if redund < 0.3 else '⚠️ **Redondance détectée**'}")
    return "\n".join(lines)



def show_item_scores_table(analysis: dict):
    """Affiche le tableau des scores + une vignette par item."""
    items = analysis.get("item_scores", []) or []
    if not items:
        return

    df = pd.DataFrame(items)

    # Colonnes utiles
    keep_cols = [c for c in ["slot", "item_id", "name", "cosine", "color_score", "season_score"]
                 if c in df.columns]
    if keep_cols:
        df = df[keep_cols]

    # Mise au propre des numériques
    for c in ["cosine", "color_score", "season_score", "style", "pairwise", "pref"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round(3)

    # Helper: retrouve un chemin d'image pour l'item
    def _resolve_img_path(row: pd.Series) -> str | None:
        row_item = str(row.get("item_id") or "").strip()
        if not row_item:
            return None

        _system = globals().get("system", None)
        art = None
        if _system:
            try:
                art = _system.get_article(row_item)
                if art is not None:
                    p = _resolve_display_path(art)
                    print("path display path : ",p)
                    if p and os.path.exists(p):
                        return p
            except Exception:
                art = None

        crop_dir = st.session_state.get("crop_dir")
        if not crop_dir:
            return None
        item_id = str(row.get("item_id") or "").strip()

        candidates = []
        if item_id:
            candidates += [
                os.path.join(crop_dir, f"{item_id}.png"),
                os.path.join(crop_dir, f"{item_id}.jpg"),
            ]
    
        for cand in candidates:
            if os.path.exists(cand):
                print("candidat :", cand)
                return cand

        return None

    df["__img_path"] = df.apply(_resolve_img_path, axis=1)
    if {"slot", "cosine"}.issubset(df.columns):
        df = df.sort_values(["slot", "cosine"], ascending=[True, False]).reset_index(drop=True)

    st.markdown("#### Détails des scores")
    
    # Affichage en colonnes avec images
    for idx, row in df.iterrows():
        with st.container():
            col_img, col_info = st.columns([1, 4])
            
            with col_img:
                img_path = row.get("__img_path")
                if img_path and os.path.exists(img_path):
                    st.image(img_path, width=80)
                else:
                    st.write("📦")  # Icône de remplacement
            
            with col_info:
                # Titre avec slot et nom
                slot = row.get("slot", "unknown")
                name = row.get("name", "")
                item_id = row.get("item_id", "")
                st.write(f"**{slot.upper()}** - {name} `({item_id})`")
                
                # Métriques en ligne
                score_cols = st.columns(3)
                with score_cols[0]:
                    cosine = row.get("cosine", 0)
                    if cosine:
                        st.metric("Similarité", f"{cosine:.3f}")
                with score_cols[1]:
                    color_score = row.get("color_score", 0)
                    if color_score:
                        st.metric("Couleur", f"{color_score:.3f}")
                with score_cols[2]:
                    season_score = row.get("season_score", 0)
                    if season_score:
                        st.metric("Saison", f"{season_score:.3f}")
            
            if idx < len(df) - 1:  # Séparateur sauf pour le dernier
                st.divider()

def _get_top_candidates_safe(system, query_image_path: str, slot: str, k: int, exclude_paths: set[str] | None = None):
    """Récupère les top-K candidats pour un slot donné, normalisés en List[Article]."""
    exclude_paths = exclude_paths or set()

    def to_articles(maybe_items):
        out = []
        if not maybe_items:
            return out
        for it in maybe_items:
            # cas 1: déjà un Article
            if hasattr(it, "image_path") or hasattr(it, "crop_path"):
                out.append(it)
                continue
            # cas 2: tuple (score, id) ou (id, score)
            if isinstance(it, (tuple, list)) and it:
                cand_id = it[1] if isinstance(it[0], (int, float)) else it[0]
                try:
                    art = system.get_article(cand_id)
                    if art:
                        out.append(art)
                except Exception:
                    pass
                continue
            # cas 3: id direct
            try:
                art = system.get_article(it)
                if art:
                    out.append(art)
            except Exception:
                pass
        # filtrer la source si nécessaire
        norm = []
        for a in out:
            p = getattr(a, "crop_path", None) or getattr(a, "image_path", None)
            if p and os.path.abspath(p) in exclude_paths:
                continue
            norm.append(a)
        return norm[:k]
    try:
        if hasattr(system, "top_candidates_by_slot"):
            return to_articles(system.top_candidates_by_slot(slot=slot, query=query_image_path, k=k))
    except Exception:
        pass
    try:
        if hasattr(system, "recommender") and hasattr(system.recommender, "top_candidates_by_slot"):
            return to_articles(system.recommender.top_candidates_by_slot(slot=slot, query=query_image_path, k=k))
    except Exception:
        pass
    try:
        if hasattr(system, "similarity_engine") and hasattr(system.similarity_engine, "retrieve_by_slot"):
            ids = system.similarity_engine.retrieve_by_slot(query_image_path=query_image_path, slot=slot, top_k=k)
            return to_articles(ids)
    except Exception:
        pass

    return []


def show_candidates_gallery(system, query_image_path: str, slots=("top", "bottom", "shoes"), k: int = 6):
    """Galerie moderne des candidats par catégorie"""
    st.markdown("#### 🛍️ Explorez les options par catégorie")

    tab_labels = [f"{slot.upper()} ({len(_get_top_candidates_safe(system, query_image_path, slot, k))})"
                  for slot in slots]
    tabs = st.tabs(tab_labels)

    for tab_idx, slot in enumerate(slots):
        with tabs[tab_idx]:
            locked_id = st.session_state["locks"].get(slot)

            if locked_id:
                st.info(f"🔒 Catégorie verrouillée: {locked_id}")
                if st.button(f"Déverrouiller {slot}", key=f"unlock-tab-{slot}", use_container_width=True):
                    st.session_state["locks"][slot] = None
                    st.rerun()
                continue

            candidates = _get_top_candidates_safe(system, query_image_path, slot, k)
            if not candidates:
                st.warning(f"Aucun candidat trouvé pour {slot}")
                continue

            cols = st.columns(3)
            for idx, item in enumerate(candidates):
                col_idx = idx % 3
                with cols[col_idx]:

                    #path = getattr(item, "crop_path", None) or getattr(item, "image_path", None)
                    path = _resolve_display_path(item)
                    if path and os.path.exists(path):
                        st.image(path, width=150)

                    category = getattr(item, "category_name", f"item-{idx}")
                    st.caption(f"**{category}**")

                    lock_key = f"lock-{slot}-{getattr(item, 'id', str(item))}-{idx}"
                    if st.button("🔒 Sélectionner", key=lock_key, type="secondary", use_container_width=True):
                        if show_locked_items():
                            st.markdown("---")
                        st.success(f"✅ {category} sélectionné!")
                        st.rerun()

                    st.markdown('</div>', unsafe_allow_html=True)

def display_outfit_result(outfit_view):
    """Affiche une tenue de manière élégante"""
    if not outfit_view or not outfit_view.get("items"):
        return

    st.markdown("#### ✨ Tenue recommandée")

    # Score global + préférence
    score = outfit_view.get("score")
    pref = outfit_view.get("pref")
    if score is not None:
        color = "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🔴"
        extra = f" · 🎯 Préférences: {pref:.2f}" if pref is not None else ""
        st.markdown(f'{color} Score: {score:.2f}/1.0{extra}', unsafe_allow_html=True)

    cols = st.columns(len(outfit_view["items"]))

    for i, (col, item) in enumerate(zip(cols, outfit_view["items"])):
        with col:
            if os.path.exists(item["path"]):
                st.image(item["path"], width=160)
            st.markdown(f"**{item['category'].upper()}**")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def analyze_query_image_colors(system, query_path: str) -> dict:
    if not (query_path and os.path.exists(query_path)):
        return {"dominant_colors": None, "color_temperature": None, "saturation_level": None, "color_score": None}
    try:
        from pathlib import Path
        p = Path(query_path)
        if p.suffix.lower() in (".jpg", ".jpeg"):
            cand = p.with_suffix(".png")
            if cand.exists():
                query_path = str(cand)
    except Exception:
        pass
    try:
        ca = system.recommender.advanced_scorer.color_analyzer.analyze_colors(query_path)
        return {
            "dominant_colors": getattr(ca, "dominant_colors", None),
            "color_temperature": getattr(ca, "color_temperature", None),
            "saturation_level": getattr(ca, "saturation_level", None),
            "color_score": float(getattr(ca, "color_harmony_score", 0.0))
        }
    except Exception as e:
        print("Query color analysis error:", e)
        return {"dominant_colors": None, "color_temperature": None, "saturation_level": None, "color_score": None}

# ---------- Helpers d'affichage ----------

def _rgb_to_hex(rgb):
    if not rgb: 
        return "#000000"
    r, g, b = [int(max(0, min(255, x))) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"

def render_color_swatches(colors, label="Palette"):
    if not colors:
        st.caption(f"{label} : —")
        return
    st.caption(label)
    cols = st.columns(len(colors))
    for i, c in enumerate(colors):
        hexc = _rgb_to_hex(c)
        with cols[i]:
            st.markdown(
                f"""
                <div style="border-radius:8px;width:100%;height:40px;border:1px solid #ddd;background:{hexc};"></div>
                <div style="font-size:12px;margin-top:4px;text-align:center;">{hexc}</div>
                """, unsafe_allow_html=True
            )

def describe_color_summary(summary: dict) -> str:
    if not summary:
        return "Pas d’analyse couleur disponible."
    temp = summary.get("color_temperature")
    sat = summary.get("saturation_level")
    score = summary.get("color_score")
    parts = []
    if temp: parts.append(f"température **{temp}**")
    if sat: parts.append(f"saturation **{sat}**")
    if score is not None: parts.append(f"harmonie **{score:.2f}**")
    if not parts:
        return "Analyse couleur non déterminée."
    return " ; ".join(parts)


#---- Verrouillage des items

def _resolve_article(system, val):
    """Retourne un Article à partir d'un id / Article / dict, sinon None."""
    if val is None:
        return None
    # déjà un Article ?
    if hasattr(val, "image_path") or hasattr(val, "id"):
        return val
    # id simple
    if isinstance(val, (int, str)):
        try:
            return system.get_article(val)
        except Exception:
            return None
    # dict avec id / path
    if isinstance(val, dict):
        vid = val.get("id") or val.get("item_id")
        if vid is not None:
            try:
                art = system.get_article(vid)
                if art:
                    return art
            except Exception:
                pass
        vpath = val.get("path")
        if vpath and hasattr(system, "find_article_by_path"):
            try:
                return system.find_article_by_path(vpath)
            except Exception:
                pass
    return None

def show_locked_items(system= None):
    """Affiche les éléments verrouillés de manière élégante. Retourne True si quelque chose est affiché."""
    # autoriser un system global si non passé
    if system is None:
        try:
            system = globals().get("system")
        except Exception:
            system = None

    locks = st.session_state.get("locks", {}) or {}
    # normaliser
    active_locks = {}
    for slot, raw in locks.items():
        art = _resolve_article(system, raw) if raw else None
        if art is not None:
            active_locks[slot] = art

    if not active_locks:
        return False

    st.markdown("#### 🔒 Sélections verrouillées")
    cols = st.columns(len(active_locks))

    for i, (slot, art) in enumerate(active_locks.items()):
        with cols[i]:
            #path = getattr(art, "crop_path", None) or getattr(art, "image_path", None)
            path = _resolve_display_path(art)
            if path and os.path.exists(path):
                st.image(path, width=120)
            label = getattr(art, "category_name", None) or f"#{getattr(art, 'id', '?')}"
            st.markdown(f"**{slot.upper()}**")
            st.caption(label)
            st.markdown('<span class="status-badge badge-locked">🔒 Verrouillé</span>', unsafe_allow_html=True)

            if st.button("Déverrouiller", key=f"unlock-{slot}-{i}", use_container_width=True):
                st.session_state["locks"][slot] = None
                st.rerun()
    return True

def _normalize_active_locks_for_builder(system, locks: dict):
    """Retourne {slot: Article} en filtrant les None."""
    out = {}
    for slot, raw in (locks or {}).items():
        if raw is None:  # Ajout de cette vérification explicite
            continue
        art = _resolve_article(system, raw)
        if art is not None:
            out[slot] = art
    return out


# ==================== INITIALISATION SYSTÈME ====================
@st.cache_resource
def initialize_system(feature_mode: str = "auto_png"):   # <-- ajout du paramètre
    """Initialise le système de recommandation (une seule fois)"""
    try:
        ANN_FILE = "images/fashionpedia/instances_attributes_train2020.json"
        IMG_DIR = "images/fashionpedia/train"
        CROP_DIR = "images/fashionpedia/crops_v2"

        if os.path.exists(ANN_FILE):
            import json
            with open(ANN_FILE, "r") as f:
                _ = json.load(f)

        system = FashionRecommendationSystem(
            ANN_FILE, IMG_DIR, CROP_DIR,
            ann_backend="faiss",
            feature_mode=feature_mode
        )
        system.initialize(max_articles=1000)
        st.session_state["crop_dir"] = CROP_DIR
        return system, True
    except Exception as e:
        st.error(f"Erreur d'initialisation: {e}")
        return None, False

# ==================== INTERFACE PRINCIPALE ====================
st.markdown("""
<div class="main-header animate-fade-in">
    <h1>👗 Fashion AI Studio</h1>
    <p>Intelligence artificielle pour la mode — Créez des tenues parfaites avec l'IA</p>
</div>
""", unsafe_allow_html=True)

# Initialisation du système
with st.spinner("Initialisation du système de recommandation..."):
    feature_mode = st.session_state.get("feature_mode", "auto_png")
    system, ok = initialize_system(feature_mode)
    

if not ok or system is None:
    st.error(" Impossible d'initialiser le système. Vérifiez les fichiers de données.")
    st.stop()

# ==================== SIDEBAR AMÉLIORÉE ====================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Statut du système
    st.markdown("**Statut du système**")
    if system:
        st.success(f"{len(system.articles)} articles chargés")
        with st.expander("Détails par catégorie"):
            try:
                for group, ids in system.similarity_engine.articles_by_group.items():
                    st.write(f"• **{group}**: {len(ids)} articles")
            except Exception:
                pass

    st.markdown("### Options d'ablation")
    _feature_choice = st.radio(
        "Source des features (couleurs / embeddings)",
        options=["PNG si dispo (fallback JPG)", "JPG uniquement"],
        index=0,
        help="Affichage utilisateur : toujours JPG de crops_v2. Ce réglage ne concerne que les calculs."
    )
    st.session_state["feature_mode"] = "auto_png" if _feature_choice.startswith("PNG") else "jpg_only"

    # Image source
    st.markdown("**Image source**")
    if st.session_state.get("uploaded_preview"):
        st.image(st.session_state["uploaded_preview"], width=220, caption="Image d'entrée")
    elif st.session_state.get("query_path") and os.path.exists(st.session_state["query_path"]):
        st.image(st.session_state["query_path"], width=150, caption="Image d'entrée")

    # Préférences utilisateur
    st.markdown("**✨ Préférences**")
    if st.session_state.get("ui_mode") == "result":
        prefs = st.session_state.get("user_prefs")
        if prefs:
            st.write(f"- Items: {', '.join(getattr(prefs,'desired_items',[]) or []) or '—'}")
            st.write(f"- Couleurs: {', '.join(getattr(prefs,'colors',[]) or []) or '—'}")
            st.write(f"- Saison: {getattr(prefs,'season','—') or '—'}")
            st.write(f"- Poids: {st.session_state.get('pref_weight',0.35):.2f}")
        else:
            st.caption("Aucune préférence définie.")
    else:
        st.caption("Renseignez vos préférences dans le panneau principal.")


# ==================== INTERFACE PRINCIPALE ====================

# Zone d'upload ou interface principale
if not st.session_state.get("query_path"):
    # Mode upload initial
    st.markdown("### Commencez par uploader votre image")

    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Glissez votre image ici ou cliquez pour parcourir",
        type=["jpg", "jpeg", "png"],
        help="Uploadez une image de vêtement ou une tenue pour obtenir des recommandations"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded: #quand une image est mise
        img = Image.open(uploaded).convert("RGB")
        st.session_state["uploaded_preview"] = uploaded.getvalue()

        st.markdown("#### Vos préférences")

        # Widgets "principaux" 
        pref_text = st.text_area(
            "Décrivez vos préférences",
            value=st.session_state.get("pref_text", ""),
            height=100, key="pref_text"
        )
        pref_items = st.multiselect(
            "Pièces recherchées", ["top","bottom","dress","shoes","accessories"], key="pref_items"
        )
        pref_colors = st.multiselect(
            "Couleurs", ["noir","blanc","gris","rouge","orange","jaune","vert","bleu","violet","rose","marron","beige","kaki","marine","bordeaux"],
            key="pref_colors"
        )
        pref_season = st.selectbox(
            "Saison", ["", "hiver","printemps","été","automne","mi-saison"], key="pref_season"
        )
        st.session_state["pref_weight"] = st.slider(
            "Poids des préférences (rerank)", 0.0, 1.0, st.session_state.get("pref_weight", 0.35), 0.05,
            help="0 = ignorer les préférences, 1 = trier principalement selon vos préférences",
            key="pref_weight_slider"
        )

        # Construire l'objet prefs et le garder en session
        prefs = None
        try:
            prefs = UserPreferences.from_text(pref_text or "")
            if pref_items:  prefs.desired_items = pref_items
            if pref_colors: prefs.colors = pref_colors
            if pref_season: prefs.season = pref_season or None
        except Exception as e:
            st.warning(f"Analyse des préférences impossible: {e}")
        st.session_state["user_prefs"] = prefs

        st.markdown('<div class="actions-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
    
            if st.button("Analyser et créer une tenue", type="primary", use_container_width=True):
                # Sauvegarde temporaire
                tmp_dir = Path("tmp_queries"); tmp_dir.mkdir(exist_ok=True)
                query_path = str(tmp_dir / "query.jpg")
                img.save(query_path, "JPEG", quality=90)

                try:
                    detected_slot = system.detect_slot_from_image(query_path)
                except Exception:
                    detected_slot = None

                detected_slot = (detected_slot or "").lower()
                BASE = ["top", "bottom", "shoes"]
                default_groups = [g for g in BASE if g != detected_slot]
                if not st.session_state.get("target_groups"):
                    st.session_state["target_groups"] = default_groups

                with st.spinner("Création de la tenue..."):

                    outfits = system.get_recommendations_builder(
                        query_image_path=query_path,
                        slots=st.session_state["target_groups"],
                        topk_per_slot=12,
                        beam_size=12,
                        max_outfits=5,
                        rules={"limit_outer": True, "avoid_print_clash": True},
                        locks={},  # pas de locks pour la première génération
                        preferences=st.session_state.get("user_prefs"),
                        pref_weight=st.session_state.get("pref_weight", 0.35)
                    )
                    outfit = outfits[0] if outfits else None 
                st.session_state["new_similar_idx"] = 0
                st.session_state["query_path"] = query_path

                if outfit:

                    analysis = _explain_outfit_safe(system, outfit)
                    try:
                        if st.session_state.get("user_prefs"):
                            scorer = PreferenceScorer()
                            pref_score = float(scorer.outfit_preference_score(outfit, st.session_state["user_prefs"]))
                            setattr(outfit, "preference_score", pref_score)
                    except Exception:
                        pass

                    # Préparer l'affichage
                    src = st.session_state.get("query_path")
                    view_items = []
                    for art in outfit.items:
                        #p = getattr(art, "crop_path", None) or getattr(art, "image_path", None)
                        p = _resolve_display_path(art)
                        if p and src and os.path.abspath(p) == os.path.abspath(src):
                            continue
                        view_items.append({
                            "path": p,
                            "category": getattr(art, "category_name", "item"),
                            "id": getattr(art, "id", None)
                        })
                    st.session_state["last_outfit_view"] = {
                        "items": view_items,
                        "score": getattr(outfit, "coherence_score", None),
                        "pref": getattr(outfit, "preference_score", None),
                        "analysis": analysis,
                    }
                    st.session_state["ui_mode"] = "result"
                    st.success(" Tenue créée avec succès!")
                    st.rerun()
                else:
                    st.warning("Aucune tenue satisfaisante trouvée. Essayez avec une autre image.")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Mode analyse -
    tab1, tab2, tab3 = st.tabs(["Résultats", "Autre choix", "Paramètres"])

    # === TAB 1: RÉSULTATS ===
    with tab1:

        if st.session_state.get("query_path") and st.session_state.get("ui_mode") != "result":
            hdr_left, hdr_right = st.columns([1, 1])
            with hdr_right:
                st.markdown("#### Préférences")
                prefs = st.session_state.get("user_prefs")
                if prefs:
                    st.write(f"**Items**: {', '.join(getattr(prefs,'desired_items',[]) or []) or '—'}")
                    st.write(f"**Couleurs**: {', '.join(getattr(prefs,'colors',[]) or []) or '—'}")
                    st.write(f"**Exclusions items**: {', '.join(getattr(prefs,'exclude_items',[]) or []) or '—'}")
                    st.write(f"**Exclusions couleurs**: {', '.join(getattr(prefs,'exclude_colors',[]) or []) or '—'}")
                    st.write(f"**Saison**: {getattr(prefs,'season', '—') or '—'}")
                    st.write(f"**Poids prefs**: {st.session_state.get('pref_weight', 0.35):.2f}")
                else:
                    st.caption("Aucune préférence définie.")

        col_action1, col_action2, col_action3 = st.columns(3)

        with col_action1:
            if st.button("Nouvelle tenue similaire", key="new_similar", use_container_width=True):
                with st.spinner("Génération d'une nouvelle tenue..."):
                    locks_raw = st.session_state.get("locks", {})
                    active_locks = _normalize_active_locks_for_builder(system, locks_raw)

                    # s'assurer que les slots incluent les slots lockés
                    slots = st.session_state.get("target_groups", ["top", "bottom", "shoes"])
                    for k in active_locks.keys():
                        if k not in slots:
                            slots.append(k)

                    try:
                        if active_locks and getattr(system.recommender, "exclude_detected_slot", False):
                            system.recommender.exclude_detected_slot = False
                    except Exception:
                        pass

                    outfits = system.get_recommendations_builder(
                        query_image_path=st.session_state["query_path"],
                        slots=slots,
                        topk_per_slot=12 if not active_locks else 10,
                        beam_size=12 if not active_locks else 10,
                        max_outfits=5, 
                        rules={"limit_outer": True, "avoid_print_clash": True},
                        locks=active_locks, 
                        preferences=st.session_state.get("user_prefs")
                    )

                    # rerank par préférences si présent
                    if outfits and st.session_state.get("user_prefs"):
                        outfits = apply_preference_rerank(
                            outfits,
                            st.session_state["user_prefs"],
                            w_pref=st.session_state.get("pref_weight", 0.35)
                        )
                        

                    # choisir la tenue suivante (2e, puis 3e, etc.)
                    outfit2 = None
                    if outfits:
                        st.session_state["new_similar_idx"] += 1
                        idx = min(st.session_state["new_similar_idx"], len(outfits) - 1)
                        outfit2 = outfits[idx]

                if outfit2:
                    analysis2 = _explain_outfit_safe(system, outfit2)
                    src = st.session_state.get("query_path")
                    # chemins des locks pour ne pas les filtrer
                    locked_paths = set()
                    for v in st.session_state.get("locks", {}).values():
                        if not v: 
                            continue
                        lp = getattr(v, "crop_path", None) or getattr(v, "image_path", None) or v.get("path") if isinstance(v, dict) else None
                        if lp:
                            locked_paths.add(os.path.abspath(lp))

                    view_items = []
                    for art in outfit2.items:
                        #p = getattr(art, "crop_path", None) or getattr(art, "image_path", None)
                        p = _resolve_display_path(art)
                        # on skippe la source SEULEMENT si ce n'est pas un lock
                        if p and src and os.path.abspath(p) == os.path.abspath(src) and (not locked_paths or os.path.abspath(p) not in locked_paths):
                            continue
                        view_items.append({
                            "path": p,
                            "category": getattr(art, "category_name", "item"),
                            "id": getattr(art, "id", None)
                        })
                    st.session_state["last_outfit_view"] = {
                        "items": view_items,
                        "score": getattr(outfit2, "coherence_score", None),
                        "pref": getattr(outfit2, "preference_score", None),
                        "analysis": analysis2,
                    }
                    st.success("✨ Nouvelle tenue générée!")
                    st.rerun()
                else:
                    st.warning("Aucune autre tenue trouvée.")

        with col_action2:
            if st.button("Recommencer", key="reset_all", use_container_width=True):
                # Reset complet
                for key in ["query_path", "last_outfit_view", "uploaded_preview", "locks", "new_similar_idx"]:
                    if key in st.session_state:
                        if key == "locks":
                            st.session_state[key] = {"top": None, "bottom": None, "shoes": None, "outer": None, "bag": None}
                        else:
                            del st.session_state[key]
                st.session_state["ui_mode"] = "idle"
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Affichage des sélections verrouillées
        if show_locked_items():
            st.markdown("---")

        # Résultat principal

        if st.session_state["last_outfit_view"]:
            display_outfit_result(st.session_state["last_outfit_view"])

            st.markdown("---")
            st.markdown("### Explications")
            analysis = st.session_state["last_outfit_view"].get("analysis")

            if analysis:
                # ---- (1) Affichage global existant (si tu as déjà format_explanation_md)
                st.markdown(format_explanation_md(analysis))

                # ---- (2) Image requête : palette & résumé
                qp = st.session_state.get("query_path")
                if qp:
                    qsum = analyze_query_image_colors(system, qp)  # NEW
                    st.markdown("#### Image requête — Couleurs")
                    render_color_swatches(qsum.get("dominant_colors"), label="Palette dominante")  # NEW
                    st.write("• " + describe_color_summary(qsum))  # NEW
                    st.markdown("---")

                # ---- (3) Détails des scores (comme avant)
                with st.expander(" Détails des scores par item"):
                    show_item_scores_table(analysis)

                    item_summaries = analysis.get("item_color_summaries") or []
                    if item_summaries:
                        st.markdown("#### Explications visuelles par article")
                        for s in item_summaries:
                            iid = s.get("item_id")
                            cat = s.get("category") or "item"
                            st.markdown(f"**{cat}** — *{iid}*")
                            render_color_swatches(s.get("dominant_colors"), label="Palette dominante")
                            st.write("• " + describe_color_summary(s))
                            st.markdown("")  # petit espace
            else:
                st.caption("Aucune analyse disponible pour cette tenue.")


        else:
            st.info("Aucune tenue à afficher. Utilisez les autres onglets pour explorer.")

    # === TAB 2: EXPLORER ===
    with tab2:
        show_candidates_gallery(
            system,
            st.session_state["query_path"],
            st.session_state.get("target_groups", ["top", "bottom", "shoes"]),
            k=8
        )

        # Bouton pour générer avec les sélections
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(" Créer une tenue avec mes sélections",
                       key="create_with_locks",
                       type="primary",
                       use_container_width=True):
                locks_raw = st.session_state.get("locks", {})
                active_locks = _normalize_active_locks_for_builder(system, locks_raw)

                if not active_locks:
                    st.warning("Veuillez d'abord sélectionner au moins un élément.")
                else:
                    with st.spinner(" Création de votre tenue personnalisée..."):
                        outfits = system.get_recommendations_builder(
                        query_image_path=st.session_state["query_path"],
                        slots=(list(set(st.session_state.get("target_groups", ["top", "bottom", "shoes"])) 
                            | set(active_locks.keys()))),
                        topk_per_slot=10,
                        beam_size=10,
                        max_outfits=3,
                        rules={"limit_outer": True, "avoid_print_clash": True},
                        locks=active_locks, 
                        preferences=st.session_state.get("user_prefs")
                    )

                    # Rerank préférences
                    if outfits and st.session_state.get("user_prefs"):
                        outfits = apply_preference_rerank(outfits, st.session_state["user_prefs"], w_pref=st.session_state.get("pref_weight", 0.35))

                    if outfits:
                        st.success(" Tenues créées avec vos sélections!")

                        for i, outfit in enumerate(outfits, 1):
                            st.markdown(f"### Option {i}")

                            score = getattr(outfit, "coherence_score", 0)
                            pref = getattr(outfit, "preference_score", None)
                            color_flag = "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🔴"
                            base = f"{color_flag} Score: {score:.2f}/1.0"
                            if pref is not None:
                                base += f" ·  Préférences: {pref:.2f}"
                            st.markdown(base, unsafe_allow_html=True)

                            cols = st.columns(len(outfit.items))
                            for col, art in zip(cols, outfit.items):
                                with col:

                                    path = _resolve_display_path(art)
                                    if path and os.path.exists(path):
                                        st.image(path, width=150)
                                    st.caption(f"**{getattr(art, 'category_name', 'Item')}**")
                                    st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            analysis_i = _explain_outfit_safe(system, outfit)

                            with st.expander("🔎 Diagnostic préférences (par item)"):
                                prefs = st.session_state.get("user_prefs")
                                if prefs:
                                    from user_preferences import PreferenceScorer
                                    import pandas as pd

                                    ps = PreferenceScorer(
                                        color_analyzer=system.recommender.color_analyzer,
                                        seasonal_classifier=system.recommender.season_classifier
                                    )

                                    rows = []
                                    # si tu as l'outfit courant sous la main :
                                    for art in outfit.items:   # ou outfit2.items / option.items selon le contexte
                                        try:
                                            s_item = ps._score_item(art, prefs)
                                            s_col  = ps._score_color(art, prefs)
                                            s_sea  = ps._score_season(art, prefs)
                                            rows.append({
                                                "id": getattr(art, "id", None),
                                                "slot": getattr(art, "group", ""),
                                                "cat": getattr(art, "category_name", ""),
                                                "pref_item": round(s_item, 3),
                                                "pref_color": round(s_col, 3),
                                                "pref_season": round(s_sea, 3),
                                                "pref_total(art)": round(ps.w_item*s_item + ps.w_color*s_col + ps.w_season*s_sea, 3)
                                            })
                                        except Exception as e:
                                            rows.append({"id": getattr(art,"id",None), "error": str(e)})

                                    if rows:
                                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                                else:
                                    st.caption("Aucune préférence fournie.")

                            with st.expander(" Explications de cette option"):
                                st.markdown(format_explanation_md(analysis_i))
                                with st.expander("Scores détaillés par item", expanded=False):
                                    show_item_scores_table(analysis_i)
                            if st.button(f"✨ Adopter cette tenue", key=f"adopt-{i}", use_container_width=True):
                                src = st.session_state.get("query_path")
                                view_items = []
                                for art in outfit.items:
                                    #p = getattr(art, "crop_path", None) or getattr(art, "image_path", None)
                                    p = _resolve_display_path(art)
                                    if p and src and os.path.abspath(p) == os.path.abspath(src):
                                        continue
                                    view_items.append({
                                        "path": p,
                                        "category": getattr(art, "category_name", "item"),
                                        "id": getattr(art, "id", None)
                                    })
                                st.session_state["last_outfit_view"] = {
                                    "items": view_items,
                                    "score": score,
                                    "pref": pref
                                }
                                st.success(" Tenue adoptée!")
                                st.rerun()

                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error(" Impossible de créer une tenue avec vos sélections actuelles.")

    # === TAB 3: GÉNÉRATEUR AVANCÉ ===
    with tab3:
        st.markdown("### Générateur de tenues multiples")
        st.markdown("Créez plusieurs tenues automatiquement avec des paramètres avancés.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Paramètres de génération")
            num_outfits = st.slider("Nombre de tenues à générer", 1, 10, 3, key="num_outfits_gen")
            topk_candidates = st.slider("Candidats par catégorie", 5, 30, 15, key="topk_gen")
            beam_size = st.slider("Largeur de recherche", 5, 30, 15, key="beam_gen")

            st.markdown("#### Catégories à inclure")
            available_slots = ["top", "bottom", "shoes", "outer", "bag"]
            selected_slots = []
            for slot in available_slots:
                if st.checkbox(f"Inclure {slot}",
                             value=slot in st.session_state.get("target_groups", []),
                             key=f"include_{slot}"):
                    selected_slots.append(slot)

        with col2:
            st.markdown("####  Règles de style")
            rule_limit_outer = st.checkbox("Limiter les vestes/manteaux à 1", value=True, key="rule_outer")
            rule_avoid_print = st.checkbox("Éviter les conflits d'imprimés", value=True, key="rule_print")

            st.markdown("####  Options avancées")
            use_seasonal = st.checkbox("Prioriser la cohérence saisonnière", value=True, key="seasonal_gen")
            use_color_harmony = st.checkbox("Optimiser l'harmonie des couleurs", value=True, key="color_gen")

            st.markdown("#### Résumé")
            st.info(f"""
            • **{num_outfits}** tenues à générer
            • **{len(selected_slots)}** catégories: {', '.join(selected_slots)}
            • **{topk_candidates}** candidats par catégorie
            • Règles actives: {sum([rule_limit_outer, rule_avoid_print, use_seasonal, use_color_harmony])}
            """)

        st.markdown("---")
        col_btn = st.columns([1, 2, 1])
        with col_btn[1]:
            if st.button("🚀 Générer les tenues",
                       key="generate_multiple",
                       type="primary",
                       use_container_width=True):
                if not selected_slots:
                    st.error("⚠️ Veuillez sélectionner au moins une catégorie.")
                else:
                    with st.spinner(f" Génération de {num_outfits} tenues personnalisées..."):
                        rules = {
                            "limit_outer": rule_limit_outer,
                            "avoid_print_clash": rule_avoid_print
                        }

                        outfits = system.get_recommendations_builder(
                            query_image_path=st.session_state["query_path"],
                            slots=selected_slots,
                            topk_per_slot=topk_candidates,
                            beam_size=beam_size,
                            max_outfits=num_outfits,
                            rules=rules,
                            preferences=st.session_state.get("user_prefs")
                        )

                    # Rerank préférences global
                    if outfits and st.session_state.get("user_prefs"):
                        outfits = apply_preference_rerank(outfits, st.session_state["user_prefs"], w_pref=st.session_state.get("pref_weight", 0.35))

                    if outfits:
                        st.success(f" {len(outfits)} tenues générées avec succès!")

                        st.markdown("###  Vos nouvelles tenues")

                        outfits_sorted = sorted(
                            outfits,
                            key=lambda x: getattr(x, "rank_score", getattr(x, "coherence_score", 0.0)),
                            reverse=True
                        )

                        for rank, outfit in enumerate(outfits_sorted, 1):
                          

                            col_rank, col_score, col_action = st.columns([2, 2, 1])

                            with col_rank:
                                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
                                st.markdown(f"## {medal} Tenue {rank}")

                            with col_score:
                                score = getattr(outfit, "coherence_score", 0)
                                pref = getattr(outfit, "preference_score", None)
                                if score > 0.8:
                                    st.success(f" Excellent: {score:.2f}/1.0")
                                elif score > 0.6:
                                    st.info(f" Bien: {score:.2f}/1.0")
                                else:
                                    st.warning(f" Correct: {score:.2f}/1.0")
                                if pref is not None:
                                    st.caption(f" Alignement préférences : {pref:.2f}")
                            analysis_g = _explain_outfit_safe(system, outfit)
                            with st.expander("Explications"):
                                st.markdown(format_explanation_md(analysis_g))
                                with st.expander("Scores détaillés par item"):
                                    show_item_scores_table(analysis_g)
                            with col_action:
                                if st.button("Adopter", key=f"adopt_gen_{rank}", use_container_width=True):
                                    src = st.session_state.get("query_path")
                                    view_items = []
                                    for art in outfit.items:
                                        #p = getattr(art, "crop_path", None) or getattr(art, "image_path", None)
                                        p = _resolve_display_path(art)
                                        if p and src and os.path.abspath(p) == os.path.abspath(src):
                                            continue
                                        view_items.append({
                                            "path": p,
                                            "category": getattr(art, "category_name", "item"),
                                            "id": getattr(art, "id", None)
                                        })
                                    st.session_state["last_outfit_view"] = {
                                        "items": view_items,
                                        "score": score,
                                        "pref": pref
                                    }
                                    st.balloons()
                                    st.success(" Tenue adoptée avec succès!")
                                    st.rerun()

                            item_cols = st.columns(len(outfit.items))
                            for col, art in zip(item_cols, outfit.items):
                                with col:
                                 
                                    path = _resolve_display_path(art)
                                    if path and os.path.exists(path):
                                        st.image(path, width=140)
                                    st.markdown(f"**{getattr(art, 'category_name', 'Item')}**")
                                    if hasattr(art, 'color_name') and getattr(art, 'color_name'):
                                        st.caption(f" {art.color_name}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                    else:
                        st.error(" Aucune tenue n'a pu être générée avec les paramètres actuels.")
                        st.info(" Essayez de réduire les contraintes ou d'augmenter le nombre de candidats.")

        st.markdown('</div>', unsafe_allow_html=True)
