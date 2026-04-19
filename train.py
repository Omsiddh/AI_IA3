"""
IA-III | Problem #71 — User Embedding Visualization
Dataset  : MovieLens 100k (grouplens.org)
Approach : SVD (Surprise) → User Embeddings → t-SNE 2D Map
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
import warnings
warnings.filterwarnings('ignore')

try:
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import cross_validate
    SURPRISE_AVAILABLE = True
except Exception:
    SURPRISE_AVAILABLE = False


def load_ratings() -> pd.DataFrame:
    """Load ratings from ratings.csv or raw MovieLens 100k files."""
    ratings_path = Path("ratings.csv")
    if ratings_path.exists():
        df_loaded = pd.read_csv(ratings_path)
        required_cols = {"user_id", "item_id", "rating"}
        if not required_cols.issubset(df_loaded.columns):
            raise ValueError("ratings.csv must contain columns: user_id, item_id, rating")
        return df_loaded[["user_id", "item_id", "rating"]]

    raw_path = Path("ml-100k") / "u.data"
    if raw_path.exists():
        df_loaded = pd.read_csv(
            raw_path,
            sep="\t",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        return df_loaded[["user_id", "item_id", "rating"]]

    raise FileNotFoundError(
        "Could not find ratings.csv or ml-100k/u.data in the current folder."
    )

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  MovieLens 100k — User Embedding Visualization")
print("=" * 55)

df = load_ratings()
df["user_id"] = pd.to_numeric(df["user_id"], errors="raise").astype(int)
df["item_id"] = pd.to_numeric(df["item_id"], errors="raise").astype(int)
df["rating"] = pd.to_numeric(df["rating"], errors="raise").astype(float)
print(f"\n[DATA] Ratings : {len(df):,}")
print(f"[DATA] Users   : {df['user_id'].nunique()}")
print(f"[DATA] Movies  : {df['item_id'].nunique()}")
print(f"[DATA] Rating Range : {df['rating'].min()} – {df['rating'].max()}")
print(f"[DATA] Avg Rating   : {df['rating'].mean():.2f}\n")

# ─────────────────────────────────────────────
# 2. TRAIN MATRIX FACTORIZATION MODEL
# ─────────────────────────────────────────────
print("[STEP 1] Training matrix factorization model...")

if SURPRISE_AVAILABLE:
    print("   Backend        : scikit-surprise (SVD)")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

    n_factors = 50
    svd = SVD(n_factors=n_factors, n_epochs=30, lr_all=0.005, reg_all=0.02, random_state=42)

    # Cross-validate to get RMSE/MAE
    cv_results = cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5, verbose=False)
    rmse_mean = float(cv_results["test_rmse"].mean())
    mae_mean = float(cv_results["test_mae"].mean())
    print(f"   Cross-Val RMSE : {rmse_mean:.4f} ± {cv_results['test_rmse'].std():.4f}")
    print(f"   Cross-Val MAE  : {mae_mean:.4f} ± {cv_results['test_mae'].std():.4f}")

    # Train on full dataset to extract embeddings
    trainset = data.build_full_trainset()
    svd.fit(trainset)

    # ─────────────────────────────────────────────
    # 3. EXTRACT USER EMBEDDINGS
    # ─────────────────────────────────────────────
    print("\n[STEP 2] Extracting user embeddings from SVD...")
    user_embeddings = svd.pu          # shape: (n_users, n_factors)
    user_biases = svd.bu              # shape: (n_users,)
    n_users_inner = user_embeddings.shape[0]
    print(f"   Embedding matrix shape: {user_embeddings.shape}")

    # Map inner user ids to original
    user_ids_raw = [int(trainset.to_raw_uid(i)) for i in range(n_users_inner)]
    metrics_rmse_name = "RMSE (CV)"
    metrics_mae_name = "MAE (CV)"

else:
    print("   Backend        : sklearn NMF fallback (surprise unavailable)")
    print("   Note           : Install Python 3.11/3.12 + scikit-surprise for Surprise-SVD backend")

    unique_users = np.sort(df["user_id"].unique())
    unique_items = np.sort(df["item_id"].unique())
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    item_to_idx = {m: i for i, m in enumerate(unique_items)}

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    n_users = len(unique_users)
    n_items = len(unique_items)
    train_matrix = np.zeros((n_users, n_items), dtype=float)

    for row in train_df.itertuples(index=False):
        train_matrix[user_to_idx[row.user_id], item_to_idx[row.item_id]] = row.rating

    n_factors = int(min(50, max(2, min(n_users, n_items) - 1)))
    nmf = NMF(n_components=n_factors, init="nndsvda", random_state=42, max_iter=400)
    user_embeddings = nmf.fit_transform(train_matrix)
    item_embeddings = nmf.components_

    test_u = test_df["user_id"].map(user_to_idx).to_numpy()
    test_i = test_df["item_id"].map(item_to_idx).to_numpy()
    test_true = test_df["rating"].to_numpy()
    test_pred = np.sum(user_embeddings[test_u] * item_embeddings[:, test_i].T, axis=1)
    test_pred = np.clip(test_pred, 1.0, 5.0)

    rmse_mean = float(np.sqrt(mean_squared_error(test_true, test_pred)))
    mae_mean = float(mean_absolute_error(test_true, test_pred))
    print(f"   Holdout RMSE   : {rmse_mean:.4f}")
    print(f"   Holdout MAE    : {mae_mean:.4f}")

    user_biases = user_embeddings.mean(axis=1) - user_embeddings.mean()
    user_ids_raw = [int(u) for u in unique_users]
    n_users_inner = user_embeddings.shape[0]
    print(f"   Embedding matrix shape: {user_embeddings.shape}")
    metrics_rmse_name = "RMSE (Holdout)"
    metrics_mae_name = "MAE (Holdout)"

# ─────────────────────────────────────────────
# 4. t-SNE DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────
print("\n[STEP 3] Running t-SNE (50D → 2D)...")
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(user_embeddings)

tsne = TSNE(
    n_components=2,
    perplexity=40,
    learning_rate=200,
    max_iter=1000,
    random_state=42,
    init="pca",
)
embeddings_2d = tsne.fit_transform(embeddings_scaled)
print(f"   t-SNE output shape: {embeddings_2d.shape}")

# ─────────────────────────────────────────────
# 5. K-MEANS CLUSTERING
# ─────────────────────────────────────────────
print("\n[STEP 4] K-Means clustering on embeddings...")
K = 6
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings_scaled)

sil_score = silhouette_score(embeddings_scaled, cluster_labels, sample_size=500, random_state=42)
print(f"   Clusters       : {K}")
print(f"   Silhouette Score : {sil_score:.4f}")

# ─────────────────────────────────────────────
# 6. USER STATS (for analysis)
# ─────────────────────────────────────────────
user_stats = df.groupby("user_id").agg(
    n_ratings=("rating", "count"),
    avg_rating=("rating", "mean"),
    rating_std=("rating", "std"),
).reset_index()

# ─────────────────────────────────────────────
# 7. SAVE RESULTS
# ─────────────────────────────────────────────
results = pd.DataFrame({
    "user_id"  : user_ids_raw,
    "tsne_x"   : embeddings_2d[:, 0],
    "tsne_y"   : embeddings_2d[:, 1],
    "cluster"  : cluster_labels,
    "user_bias": user_biases,
})
results["user_id"] = results["user_id"].astype(int)
user_stats["user_id"] = user_stats["user_id"].astype(int)
results = results.merge(user_stats, on="user_id", how="left")
results.to_csv("embeddings_2d.csv", index=False)

# Save metrics
metrics = {
    "Metric": [metrics_rmse_name, metrics_mae_name, "Silhouette Score", "n_users", "n_factors"],
    "Value" : [
        round(rmse_mean, 4),
        round(mae_mean, 4),
        round(sil_score, 4),
        n_users_inner,
        n_factors,
    ]
}
pd.DataFrame(metrics).to_csv("metrics.csv", index=False)

print("\n[SAVED] embeddings_2d.csv")
print("[SAVED] metrics.csv")
print("\n[DONE] Run visualize.py next.\n")
