"""
visualize.py — All plots for IA-III report and video
Run AFTER train.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')


def load_ratings() -> pd.DataFrame:
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

# ── Load results ──────────────────────────────────────────
df_emb  = pd.read_csv("embeddings_2d.csv")
df_rat  = load_ratings()
metrics = pd.read_csv("metrics.csv")

df_emb["user_id"] = pd.to_numeric(df_emb["user_id"], errors="coerce").astype("Int64")
df_rat["user_id"] = pd.to_numeric(df_rat["user_id"], errors="coerce").astype("Int64")
df_rat["item_id"] = pd.to_numeric(df_rat["item_id"], errors="coerce").astype("Int64")
df_rat["rating"] = pd.to_numeric(df_rat["rating"], errors="coerce")

print(f"Loaded {len(df_emb)} users with embeddings")

PALETTE = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#9B5DE5"]
BG      = "#0F172A"
GRID    = "#1E293B"
TEXT    = "#F1F5F9"
clusters = sorted(df_emb["cluster"].dropna().unique())
cluster_palette = {cluster_id: PALETTE[i % len(PALETTE)] for i, cluster_id in enumerate(clusters)}

rmse_row = metrics[metrics["Metric"].str.contains("RMSE", case=False, na=False)]
mae_row = metrics[metrics["Metric"].str.contains("MAE", case=False, na=False)]
sil_row = metrics[metrics["Metric"].str.contains("Silhouette", case=False, na=False)]
factors_row = metrics[metrics["Metric"].str.contains("n_factors", case=False, na=False)]
rmse_value = float(rmse_row["Value"].iloc[0]) if not rmse_row.empty else float("nan")
mae_value = float(mae_row["Value"].iloc[0]) if not mae_row.empty else float("nan")
sil_value = float(sil_row["Value"].iloc[0]) if not sil_row.empty else float("nan")
factors_value = int(float(factors_row["Value"].iloc[0])) if not factors_row.empty else 50

# ══════════════════════════════════════════════════════════
# PLOT 1 — Main t-SNE Map (colored by cluster)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

for cluster_id in clusters:
    mask = df_emb["cluster"] == cluster_id
    ax.scatter(
        df_emb.loc[mask, "tsne_x"],
        df_emb.loc[mask, "tsne_y"],
        c=cluster_palette[cluster_id],
        s=35, alpha=0.8, linewidths=0.3,
        edgecolors="white",
        label=f"Cluster {cluster_id + 1}",
    )

ax.set_title(
    "User Preference Map — t-SNE Visualization of MovieLens Embeddings\n"
    "SVD Matrix Factorization  |  50 latent factors  →  2D via t-SNE",
    fontsize=14, fontweight="bold", color=TEXT, pad=16
)
ax.set_xlabel("t-SNE Dimension 1", fontsize=11, color=TEXT)
ax.set_ylabel("t-SNE Dimension 2", fontsize=11, color=TEXT)
ax.tick_params(colors=TEXT)
ax.spines[["top","right","left","bottom"]].set_color(GRID)
ax.grid(color=GRID, linewidth=0.5, alpha=0.5)
legend = ax.legend(fontsize=10, framealpha=0.2, labelcolor=TEXT,
                   facecolor=GRID, edgecolor=GRID, loc="upper right")
plt.tight_layout()
plt.savefig("plot1_tsne_map.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("[SAVED] plot1_tsne_map.png")

# ══════════════════════════════════════════════════════════
# PLOT 2 — t-SNE colored by avg rating (heatmap style)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

sc = ax.scatter(
    df_emb["tsne_x"], df_emb["tsne_y"],
    c=df_emb["avg_rating"],
    cmap="RdYlGn", s=35, alpha=0.85,
    linewidths=0.3, edgecolors="white",
    vmin=1, vmax=5
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Average Rating Given by User", color=TEXT, fontsize=11)
cbar.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)

ax.set_title(
    "User Preferences — Colored by Average Rating\n"
    "Green = generous raters  |  Red = harsh raters",
    fontsize=13, fontweight="bold", color=TEXT, pad=14
)
ax.set_xlabel("t-SNE Dimension 1", fontsize=11, color=TEXT)
ax.set_ylabel("t-SNE Dimension 2", fontsize=11, color=TEXT)
ax.tick_params(colors=TEXT)
ax.spines[["top","right","left","bottom"]].set_color(GRID)
ax.grid(color=GRID, linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("plot2_tsne_avgrating.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("[SAVED] plot2_tsne_avgrating.png")

# ══════════════════════════════════════════════════════════
# PLOT 3 — t-SNE colored by number of ratings (activity)
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

sc = ax.scatter(
    df_emb["tsne_x"], df_emb["tsne_y"],
    c=df_emb["n_ratings"],
    cmap="plasma", s=35, alpha=0.85,
    linewidths=0.3, edgecolors="white",
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Number of Ratings (User Activity)", color=TEXT, fontsize=11)
cbar.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)

ax.set_title(
    "User Activity Map — Colored by Number of Ratings\n"
    "Bright = highly active users  |  Dark = casual users",
    fontsize=13, fontweight="bold", color=TEXT, pad=14
)
ax.set_xlabel("t-SNE Dimension 1", fontsize=11, color=TEXT)
ax.set_ylabel("t-SNE Dimension 2", fontsize=11, color=TEXT)
ax.tick_params(colors=TEXT)
ax.spines[["top","right","left","bottom"]].set_color(GRID)
ax.grid(color=GRID, linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("plot3_tsne_activity.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("[SAVED] plot3_tsne_activity.png")

# ══════════════════════════════════════════════════════════
# PLOT 4 — Cluster Analysis (box plots of avg rating per cluster)
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

# Ratings distribution per cluster
ax = axes[0]
ax.set_facecolor(BG)
cluster_groups = [
    df_emb[df_emb["cluster"] == cluster_id]["avg_rating"].dropna().values
    for cluster_id in clusters
]
bp = ax.boxplot(cluster_groups, patch_artist=True, medianprops=dict(color="white", linewidth=2))
for patch, cluster_id in zip(bp["boxes"], clusters):
    color = cluster_palette[cluster_id]
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
for element in ["whiskers", "caps", "fliers"]:
    for item in bp[element]:
        item.set_color(TEXT)
ax.set_title("Avg Rating Distribution per Cluster", fontsize=12, fontweight="bold", color=TEXT)
ax.set_xlabel("Cluster", fontsize=10, color=TEXT)
ax.set_ylabel("Average Rating", fontsize=10, color=TEXT)
ax.set_xticklabels([f"C{cluster_id + 1}" for cluster_id in clusters], color=TEXT)
ax.tick_params(colors=TEXT)
ax.spines[["top","right"]].set_color(GRID)
ax.spines[["left","bottom"]].set_color(GRID)
ax.grid(axis="y", color=GRID, linewidth=0.8)

# User count per cluster
ax = axes[1]
ax.set_facecolor(BG)
counts = df_emb["cluster"].value_counts().sort_index()
bar_labels = [f"Cluster {cluster_id + 1}" for cluster_id in counts.index]
bar_colors = [cluster_palette[cluster_id] for cluster_id in counts.index]
bars = ax.bar(bar_labels, counts.values,
              color=bar_colors, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(val), ha="center", va="bottom", fontsize=11,
            fontweight="bold", color=TEXT)
ax.set_title("User Count per Cluster", fontsize=12, fontweight="bold", color=TEXT)
ax.set_ylabel("Number of Users", fontsize=10, color=TEXT)
ax.tick_params(colors=TEXT)
ax.spines[["top","right"]].set_color(GRID)
ax.spines[["left","bottom"]].set_color(GRID)
ax.grid(axis="y", color=GRID, linewidth=0.8)

fig.suptitle("Cluster Analysis", fontsize=14, fontweight="bold", color=TEXT)
plt.tight_layout()
plt.savefig("plot4_cluster_analysis.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("[SAVED] plot4_cluster_analysis.png")

# ══════════════════════════════════════════════════════════
# PLOT 5 — Rating Distribution + Pipeline Summary
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

# Rating distribution
ax = axes[0]
ax.set_facecolor(BG)
rating_counts = df_rat["rating"].value_counts().sort_index()
bars = ax.bar(rating_counts.index, rating_counts.values,
              color=PALETTE[:5], edgecolor="white", linewidth=0.8, width=0.6)
for bar, val in zip(bars, rating_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f"{val:,}", ha="center", va="bottom", fontsize=10,
            fontweight="bold", color=TEXT)
ax.set_title("Rating Distribution in Dataset", fontsize=12, fontweight="bold", color=TEXT)
ax.set_xlabel("Star Rating", fontsize=10, color=TEXT)
ax.set_ylabel("Count", fontsize=10, color=TEXT)
ax.set_xticks([1,2,3,4,5])
ax.tick_params(colors=TEXT)
ax.spines[["top","right"]].set_color(GRID)
ax.spines[["left","bottom"]].set_color(GRID)
ax.grid(axis="y", color=GRID, linewidth=0.8)

# Metrics table
ax = axes[1]
ax.set_facecolor(BG)
ax.axis("off")
table_data = [
    ["Metric", "Value"],
    ["Dataset", "MovieLens 100k"],
    ["Total Ratings", f"{len(df_rat):,}"],
    ["Users", f"{df_rat['user_id'].nunique():,}"],
    ["Movies", f"{df_rat['item_id'].nunique():,}"],
    ["Latent Factors", f"{factors_value}"],
    ["RMSE", f"{rmse_value:.4f}"],
    ["MAE", f"{mae_value:.4f}"],
    ["t-SNE Perplexity", "40"],
    ["K-Means Clusters", f"{len(clusters)}"],
    ["Silhouette Score", f"{sil_value:.4f}"],
]
table = ax.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.4, 1.8)
for (r, c), cell in table.get_celld().items():
    cell.set_facecolor(GRID if r == 0 else BG)
    cell.set_text_props(color=TEXT, fontweight="bold" if r == 0 else "normal")
    cell.set_edgecolor(GRID)
ax.set_title("Experiment Summary", fontsize=12, fontweight="bold", color=TEXT, pad=20)

plt.tight_layout()
plt.savefig("plot5_summary.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("[SAVED] plot5_summary.png")

print("\n[ALL PLOTS SAVED] ✓")
