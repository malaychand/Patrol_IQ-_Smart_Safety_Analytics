import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ==========================================================
# STEP 1 — LOAD DATA
# ==========================================================
csv_path = r"C:\Users\Lenovo\Desktop\Patrollq_proj3\data\processed\feature_engineered_crimes.csv"
df = pd.read_csv(csv_path, low_memory=False)

print("Loaded rows:", len(df))

# ==========================================================
# STEP 2 — SELECT FEATURES FOR CLUSTERING
# ==========================================================
required_cols = ["Latitude_norm", "Longitude_norm", "CrimeSeverity", "PrimaryType_Code"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

X = df[required_cols].values

# ==========================================================
# STEP 3 — PCA (3 Components)
# ==========================================================
pca = PCA(n_components=3)
pca_components = pca.fit_transform(X)
variance_ratio = pca.explained_variance_ratio_
explained_variance = sum(variance_ratio[:3])
reconstruction_error = 0.0  # Not computed
X_reduced = pca_components[:, :2]  # Use 2D PCA for clustering

# ==========================================================
# STEP 4 — Sample data for fast experiments
# ==========================================================
sample_size = 50000  # 50k rows instead of full 498k
if len(X_reduced) > sample_size:
    idx = np.random.choice(len(X_reduced), sample_size, replace=False)
    X_sample = X_reduced[idx]
else:
    X_sample = X_reduced

# ==========================================================
# STEP 5 — MLflow Setup
# ==========================================================
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Crime_Clustering")

# ==========================================================
# STEP 6 — Helper functions
# ==========================================================
def run_kmeans(X, k, explained_variance, reconstruction_error, log_model=True):
    with mlflow.start_run(run_name=f"KMeans_k{k}"):
        model = KMeans(n_clusters=k, random_state=42, n_init=10, algorithm='elkan')
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)

        # Log parameters
        mlflow.log_param("algorithm", "KMeans")
        mlflow.log_param("k", k)

        # Log metrics
        mlflow.log_metric("silhouette_score", sil)
        mlflow.log_metric("explained_variance", explained_variance)
        mlflow.log_metric("reconstruction_error", reconstruction_error)

        if log_model:
            mlflow.sklearn.log_model(model, "model")

        return model, sil

def run_dbscan(X, eps, min_samples, explained_variance, reconstruction_error, log_model=True):
    with mlflow.start_run(run_name=f"DBSCAN_eps{eps}"):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else -1

        mlflow.log_param("algorithm", "DBSCAN")
        mlflow.log_param("eps", eps)
        mlflow.log_param("min_samples", min_samples)
        mlflow.log_metric("silhouette_score", sil)
        mlflow.log_metric("explained_variance", explained_variance)
        mlflow.log_metric("reconstruction_error", reconstruction_error)

        if log_model:
            mlflow.sklearn.log_model(model, "model")

        return model, sil

# ==========================================================
# STEP 7 — Run experiments
# ==========================================================
results = []

print("Running KMeans on sample data...")
k_model, k_sil = run_kmeans(
    X_sample,
    k=5,
    explained_variance=explained_variance,
    reconstruction_error=reconstruction_error,
    log_model=False
)
results.append(("KMeans", k_model, k_sil))

print("Running DBSCAN on sample data...")
db_model, db_sil = run_dbscan(
    X_sample,
    eps=0.5,
    min_samples=10,
    explained_variance=explained_variance,
    reconstruction_error=reconstruction_error,
    log_model=False
)
results.append(("DBSCAN", db_model, db_sil))

# ==========================================================
# STEP 8 — Select best model
# ==========================================================
best_algo, best_model, best_sil = max(results, key=lambda x: x[2])
print("\n=============================")
print("BEST MODEL SELECTED")
print("=============================")
print(f"Algorithm: {best_algo}")
print(f"Silhouette Score: {best_sil}")

# ==========================================================
# STEP 9 — Register best model in MLflow
# ==========================================================
with mlflow.start_run(run_name="Best_Model_Registration") as run:
    mlflow.sklearn.log_model(best_model, "best_model")
    mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/best_model",
        name="Crime_Clustering_Model"
    )

print("\nBest model registered in MLflow Model Registry!")

# ==========================================================
# STEP 10 — Save detailed summary to text file
# ==========================================================
summary_text = f"""
CRIME CLUSTERING MODEL SUMMARY
==============================

Algorithms Tested:
-----------------
1. KMeans
   - Silhouette Score: {k_sil}
2. DBSCAN
   - Silhouette Score: {db_sil}

Best Model:
-----------
Algorithm: {best_algo}
Silhouette Score: {best_sil}

Reason for Selection:
---------------------
The best model was selected based on the **highest silhouette score**, 
which measures how well the clusters are separated and how cohesive 
each cluster is. A higher silhouette score indicates better-defined clusters. 

Model Details:
{best_model}

PCA Explained Variance (first 3 components): {explained_variance}
Reconstruction Error: {reconstruction_error}
"""

txt_path = r"C:\Users\Lenovo\Desktop\Patrollq_proj3\best_model_summary.txt"
with open(txt_path, "w") as f:
    f.write(summary_text)

print(f"\nModel summary saved to '{txt_path}'")
