import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import plotly.express as px

# ------------------------------------------------------
# Create output folder
# ------------------------------------------------------
output_folder = "dimensionality_reduction_outputs"
os.makedirs(output_folder, exist_ok=True)

# ------------------------------------------------------
# Load dataset (fixed warning)
# ------------------------------------------------------
df = pd.read_csv("data/processed/feature_engineered_crimes.csv", low_memory=False)

# Select numeric features
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Standardize
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# ------------------------------------------------------
# ---------------------- PCA ---------------------------
# ------------------------------------------------------
pca = PCA(n_components=3)
pca_components = pca.fit_transform(scaled_data)

df_pca = pd.DataFrame({
    "PC1": pca_components[:, 0],
    "PC2": pca_components[:, 1],
    "PC3": pca_components[:, 2],
})

variance_ratio = pca.explained_variance_ratio_
cum_variance = np.cumsum(variance_ratio)

# PCA variance plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 4), cum_variance, marker='o')
plt.title("PCA – Cumulative Variance Explained")
plt.savefig(f"{output_folder}/pca_variance_explained.png")
plt.close()

# PCA 2D plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x="PC1", y="PC2", data=df_pca, s=10)
plt.title("PCA – 2D Scatter")
plt.savefig(f"{output_folder}/pca_2d_scatter.png")
plt.close()

# PCA 3D interactive
fig = px.scatter_3d(df_pca, x="PC1", y="PC2", z="PC3", title="PCA 3D Scatter")
fig.write_html(f"{output_folder}/pca_3d_scatter.html")

# PCA feature importance
loading_scores = pd.Series(abs(pca.components_[0]), index=numeric_df.columns)
top_features = loading_scores.sort_values(ascending=False).head(5)

plt.figure(figsize=(8, 5))
top_features.plot(kind='bar')
plt.title("Top 5 PCA Feature Importances")
plt.savefig(f"{output_folder}/pca_top5_features.png")
plt.close()

# ------------------------------------------------------
# ---------------------- t-SNE -------------------------
# ------------------------------------------------------
sample_n = 10000  # FAST MODE
tsne_input = scaled_data[:sample_n]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_results = tsne.fit_transform(tsne_input)

df_tsne = pd.DataFrame({
    "TSNE1": tsne_results[:, 0],
    "TSNE2": tsne_results[:, 1]
})

plt.figure(figsize=(8, 6))
sns.scatterplot(x="TSNE1", y="TSNE2", data=df_tsne, s=10)
plt.title("t-SNE (10,000 sample)")
plt.savefig(f"{output_folder}/tsne_2d_scatter.png")
plt.close()

# ------------------------------------------------------
# ---------------------- UMAP --------------------------
# ------------------------------------------------------
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_results = umap_model.fit_transform(tsne_input)

df_umap = pd.DataFrame({
    "UMAP1": umap_results[:, 0],
    "UMAP2": umap_results[:, 1]
})

plt.figure(figsize=(8, 6))
sns.scatterplot(x="UMAP1", y="UMAP2", data=df_umap, s=10)
plt.title("UMAP (10,000 sample)")
plt.savefig(f"{output_folder}/umap_2d_scatter.png")
plt.close()

# ------------------------------------------------------
# -------------------- Save Summary --------------------
# ------------------------------------------------------
summary_text = f"""
STEP 5: Dimensionality Reduction Summary
----------------------------------------

PCA:
- 3 Components
- Variance PC1: {variance_ratio[0]:.2f}
- Variance PC2: {variance_ratio[1]:.2f}
- Variance PC3: {variance_ratio[2]:.2f}
- Total Variance: {cum_variance[2]:.2f}

Top 5 PCA Features:
{top_features.to_string()}

t-SNE:
- FAST MODE: Only first 10,000 samples used

UMAP:
- FAST MODE: Only first 10,000 samples used

All plots saved to: {output_folder}/
"""

with open(f"{output_folder}/step5_summary.txt", "w") as f:
    f.write(summary_text)

print("STEP 5 Completed ✔")
