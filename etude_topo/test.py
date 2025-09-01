import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np

# Lecture des données
edges = pd.read_excel('influence_mode.xlsx', sheet_name='Edges_InfBrand')
coocc = pd.read_excel('influence_mode.xlsx', sheet_name='Cooccurrence', index_col=0)

# Configuration de la figure avec deux sous-graphiques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# ============ GRAPHIQUE 1 : RÉSEAU D'INFLUENCE ============
# Création du graphe
G = nx.Graph()
for _, row in edges.iterrows():
    G.add_edge(row['source_influencer'], row['target_brand'], weight=row['weight'])

# Positionnement avec plus d'espace entre les nœuds
pos = nx.spring_layout(G, k=1.5, iterations=50)

# Préparation des couleurs et tailles basées sur les poids
weights = [G[u][v]['weight'] for u, v in G.edges()]
norm = plt.Normalize(vmin=min(weights), vmax=max(weights))
edge_colors = cm.viridis(norm(weights))

# Distinction des types de nœuds (influenceurs vs marques)
influencers = set(edges['source_influencer'])
brands = set(edges['target_brand'])

# Couleurs des nœuds
node_colors = ['lightblue' if node in influencers else 'lightcoral' for node in G.nodes()]

# Dessin du graphe sur ax1
nx.draw_networkx_nodes(G, pos, 
                      node_color=node_colors,
                      node_size=500,
                      alpha=0.8,
                      ax=ax1)

nx.draw_networkx_edges(G, pos,
                      edge_color=edge_colors,
                      width=[w*2 for w in weights],
                      alpha=0.6,
                      ax=ax1)

nx.draw_networkx_labels(G, pos,
                       font_size=9,
                       font_weight='bold',
                       font_color='black',
                       ax=ax1)

# Ajout d'une colorbar pour les poids (CORRECTION DE L'ERREUR)
sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
sm.set_array([])
cbar1 = fig.colorbar(sm, ax=ax1, shrink=0.8)  # Utiliser fig.colorbar avec ax=ax1
cbar1.set_label('Poids des connexions', rotation=270, labelpad=20)

# Titre et légende pour le graphique 1
ax1.set_title("Réseau d'influence : Influenceurs → Marques", 
              fontsize=14, fontweight='bold')

# Légende pour les types de nœuds
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='lightblue', label='Influenceurs'),
                  Patch(facecolor='lightcoral', label='Marques')]
ax1.legend(handles=legend_elements, loc='upper right')

ax1.axis('off')

# ============ GRAPHIQUE 2 : HEATMAP DE COOCCURRENCE ============
# Création de la heatmap
sns.heatmap(coocc, 
           cmap='RdYlBu_r',
           center=0,
           annot=True,
           fmt='.2f',
           square=True,
           linewidths=0.5,
           cbar_kws={'shrink': 0.8, 'label': 'Coefficient de cooccurrence'},
           annot_kws={'size': 8, 'weight': 'bold'},
           ax=ax2)

# Rotation des labels
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=10)

# Titre et labels pour le graphique 2
ax2.set_title('Matrice de Cooccurrence entre Marques\n(Style & Collaborations)', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Marques', fontsize=12, fontweight='bold')
ax2.set_ylabel('Marques', fontsize=12, fontweight='bold')

# Ajustements finaux
plt.tight_layout()
plt.show()