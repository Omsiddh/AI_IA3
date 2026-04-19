# Problem #71 — User Embedding Visualization
## IA-III | Implementation of Current Trends in AI
### MovieLens 100k | Surprise (SVD) + t-SNE

---

## What This Project Does

1. Loads the **MovieLens 100k** dataset (100,000 ratings from 943 users on 1,682 movies)
2. Trains **SVD** (Singular Value Decomposition) via the Surprise library — a matrix factorization technique that learns latent user preferences
3. Extracts **50-dimensional user embeddings** — each user is now a point in 50D space representing their taste
4. Applies **t-SNE** to compress 50D → 2D while preserving neighborhood structure
5. Applies **K-Means clustering** to group users by similar preferences
6. Visualizes the 2D map with multiple colorings (cluster, avg rating, activity)

**Key Insight:** Users who rate movies similarly end up close together on the 2D map — even though the model never explicitly knew their "taste type."

---

## Setup

```bash
pip install scikit-surprise scikit-learn matplotlib pandas numpy seaborn
```

> **Note:** Surprise requires numpy < 2. If you get a numpy error run:
> `pip install "numpy<2"`

---

## Run Order

```bash
# Step 1 — Train SVD, extract embeddings, run t-SNE
python train.py

# Step 2 — Generate all 5 plots
python visualize.py
```

On Windows, if python is not recognized, use the launcher:

```bash
py train.py
py visualize.py
```

You do not need to create ratings.csv manually anymore.
The scripts now auto-load either:
- ratings.csv (if present), or
- ml-100k/u.data (raw MovieLens 100k file)

---

## Output Files

| File | Description |
|------|-------------|
| `embeddings_2d.csv` | Per-user: t-SNE coordinates, cluster, bias, stats |
| `metrics.csv` | RMSE, MAE, Silhouette Score |
| `plot1_tsne_map.png` | **Main output** — 2D user map colored by cluster |
| `plot2_tsne_avgrating.png` | Same map colored by avg rating (generous vs harsh) |
| `plot3_tsne_activity.png` | Same map colored by number of ratings (activity) |
| `plot4_cluster_analysis.png` | Box plots + user count per cluster |
| `plot5_summary.png` | Rating distribution + full metrics table |

---

## Pipeline

```
MovieLens Ratings (100k)
        ↓
  User-Movie Matrix
        ↓
   SVD (Surprise)
   50 latent factors
        ↓
  User Embeddings
   shape: (943, 50)
        ↓
  StandardScaler
        ↓
     t-SNE
   50D → 2D
        ↓
  K-Means (k=6)
        ↓
  2D Visualization
```

---

## Results

| Metric | Value |
|--------|-------|
| SVD RMSE (5-fold CV) | ~1.09 |
| SVD MAE (5-fold CV)  | ~0.89 |
| K-Means Clusters     | 6     |
| Silhouette Score     | ~0.015 |

> **Note on Silhouette Score:** A low score is expected and explainable — user preferences exist on a continuous spectrum, not hard-edged groups. This is a valid analysis point for your report.

---

## Dataset

**MovieLens 100k** — GroupLens Research, University of Minnesota
- Source: https://grouplens.org/datasets/movielens/100k/
- 100,000 ratings (1–5 stars)
- 943 users, 1,682 movies
- Collected 1997–1998

---

## What to Say in the Video

1. Show `plot1_tsne_map.png` — "Each dot is a user. Users close together have similar movie tastes, even though the model never explicitly knew their preferences."
2. Show `plot2_tsne_avgrating.png` — "Green users tend to rate generously, red users are harsher raters — this pattern emerges naturally from the embeddings."
3. Show `plot3_tsne_activity.png` — "Bright dots are users who rated many movies. We can see active users are spread across all preference regions."
4. Explain SVD: "We trained a matrix factorization model that learns a 50-number fingerprint for each user."
5. Explain t-SNE: "t-SNE compresses those 50 numbers into 2 numbers for visualization, while keeping similar users close together."

---

## Tech Stack
- Python 3.x
- `scikit-surprise` — SVD Matrix Factorization
- `scikit-learn` — t-SNE, K-Means, StandardScaler
- `pandas`, `numpy` — data handling
- `matplotlib` — all visualizations

---

## Team Roles (suggested)
- **Member 1:** Code (train.py, visualize.py)
- **Member 2:** Report writing
- **Member 3:** Slides (use the 5 plots directly)
- **Member 4:** Video recording & narration
- **Member 5:** Result analysis section

---

## Citations
- F. Maxwell Harper and Joseph A. Konstan (2015). The MovieLens Datasets. ACM TIIS.
- Hug, N. (2020). Surprise: A Python library for recommender systems. JOSS.
- van der Maaten, L. & Hinton, G. (2008). Visualizing Data using t-SNE. JMLR.
- scikit-learn documentation — https://scikit-learn.org
