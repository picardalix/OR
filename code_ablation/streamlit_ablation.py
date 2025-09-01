
import streamlit as st
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ablation Study — Fashion AI", page_icon="🧪", layout="wide")

st.title("🧪 Ablation Study (Components → Impact)")
st.caption("Compare l’apport de chaque brique : couleurs, saisons, style, tendance — par rapport à un baseline CLIP.")

results_path = st.text_input("Chemin du fichier de résultats (CSV)", value=str(Path(__file__).resolve().parent / "results.csv"))

@st.cache_data
def load_results(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

df = load_results(results_path)
if df.empty:
    st.warning("Aucun résultat trouvé. Lance d’abord runner.py pour générer results.csv.")
    st.stop()

st.subheader("Scores globaux")
st.dataframe(df, use_container_width=True)

metric = st.selectbox("Métrique", ["precision@k", "recall@k", "map@k", "ndcg@k", "fitb_acc"], index=3)

# Bar chart
fig = plt.figure()
plt.bar(df["variant"], df[metric])
plt.xticks(rotation=45, ha="right")
plt.ylabel(metric)
plt.title(f"Ablation — {metric}")
st.pyplot(fig)

# Delta vs reference
reference = st.selectbox("Référence pour le delta", sorted(df["variant"].unique()), index=list(df["variant"]).index("full") if "full" in list(df["variant"]) else 0)
ref_score = float(df[df["variant"] == reference][metric].values[0])

fig2 = plt.figure()
subset = df[df["variant"] != reference]
plt.bar(subset["variant"], ref_score - subset[metric])
plt.xticks(rotation=45, ha="right")
plt.ylabel(f"{reference} - variant ({metric})")
plt.title(f"Delta vs '{reference}' — {metric}")
st.pyplot(fig2)

# Shapley-like deltas
shapley_path = Path(results_path).with_suffix(".shapley.json")
if shapley_path.exists():
    st.subheader("Contribution (Shapley-like) — sur ndcg@k")
    data = json.loads(shapley_path.read_text(encoding="utf-8"))
    contrib = data.get("delta_vs_leave_one_out", {})
    if contrib:
        names = list(contrib.keys())
        vals = [contrib[n] for n in names]
        fig3 = plt.figure()
        plt.bar(names, vals)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Δ ndcg@k (full - leave-one-out)")
        plt.title("Importance relative des composants")
        st.pyplot(fig3)
    else:
        st.info("Pas de données Shapley-like disponibles.")
else:
    st.info("Fichier .shapley.json non trouvé (lance runner.py).")

st.divider()
st.caption("Conseil: garde les figures .png dans ablation_suite/plots pour les insérer directement dans le mémoire.")
