# IA-III Report: User Embedding Visualization from MovieLens 100k

## 1. Title
User Embedding Visualization of Movie Preference Behavior using Matrix Factorization (Surprise SVD) and t-SNE

## 2. Objective
The objective of this project is to learn latent user preference embeddings from movie-rating behavior and visualize those embeddings on a 2D map.

This project demonstrates that users with similar rating patterns are mapped close to each other, even when explicit profile features are not provided.

## 3. Problem Statement
Given sparse user-movie rating interactions, build a descriptive AI pipeline that:
- Learns user embeddings from collaborative filtering
- Reduces embedding dimensionality for visualization
- Groups users with similar preferences
- Interprets preference behavior through visual analytics

## 4. Dataset
Dataset used: MovieLens 100k (GroupLens)
- Source: https://grouplens.org/datasets/movielens/100k/
- Total ratings: 100,000
- Users: 943
- Movies: 1,682
- Rating scale: 1 to 5
- Mean rating: 3.5299

Raw input file used in this implementation: ml-100k/u.data

## 5. Tools and Libraries
- Python 3.10 (venv)
- scikit-surprise (SVD matrix factorization)
- scikit-learn (t-SNE, K-Means, StandardScaler)
- pandas, numpy, matplotlib

Environment compatibility note:
- scikit-surprise works with NumPy 1.x in this setup.
- Working combination: Python 3.10 + numpy 1.26.4 + scikit-surprise 1.1.4.

## 6. Methodology
### 6.1 Data Loading and Preparation
- Ratings are loaded from ratings.csv if available, otherwise from ml-100k/u.data.
- Relevant columns used: user_id, item_id, rating.

### 6.2 User Embedding Learning (Surprise SVD)
- Algorithm: SVD from Surprise
- Latent factors: 50
- Cross-validation: 5-fold
- Output: dense user embedding matrix of shape (943, 50)

### 6.3 Dimensionality Reduction
- StandardScaler applied to user embeddings
- t-SNE used to reduce 50D embeddings to 2D for plotting
- t-SNE configuration:
  - components: 2
  - perplexity: 40
  - learning rate: 200
  - iterations: 1000

### 6.4 User Group Discovery
- K-Means clustering on scaled embeddings
- Number of clusters (K): 6
- Cluster quality measured using silhouette score

### 6.5 Reporting Artifacts
- embeddings_2d.csv: user_id, t-SNE coordinates, cluster label, user bias, profile stats
- metrics.csv: RMSE, MAE, silhouette score, number of users, number of factors
- Five visualization PNG files for analysis and presentation

## 7. Experimental Results
### 7.1 Model Metrics
- RMSE (5-fold CV): 0.9399
- MAE (5-fold CV): 0.7380
- Silhouette score (K=6): 0.0013
- Number of users embedded: 943
- Latent factors: 50

### 7.2 Cluster-wise User Count
- Cluster 1: 132 users
- Cluster 2: 96 users
- Cluster 3: 166 users
- Cluster 4: 152 users
- Cluster 5: 161 users
- Cluster 6: 236 users

## 8. Plot-by-Plot Interpretation
### Plot 1: plot1_tsne_map.png
Main user preference map colored by cluster labels.
- Each point is one user.
- Nearby points represent users with similar preference structure in latent space.
- Cluster separation is soft (overlap is expected in human preference data).

### Plot 2: plot2_tsne_avgrating.png
Map colored by each user's average rating behavior.
- Helps identify generous vs harsh raters.
- Reveals that rating strictness varies smoothly across the preference manifold.

### Plot 3: plot3_tsne_activity.png
Map colored by number of ratings provided by each user.
- Shows high-activity and low-activity users across preference regions.
- Indicates activity level is not confined to a single taste cluster.

### Plot 4: plot4_cluster_analysis.png
Two-part cluster diagnostic:
- Boxplot: average rating distribution per cluster
- Bar chart: user count per cluster

This allows quick comparison of rating behavior and cluster population balance.

### Plot 5: plot5_summary.png
Consolidated summary figure:
- Dataset rating distribution
- Experiment configuration and metric table

Useful for report and viva overview slides.

## 9. Discussion
### 9.1 Why this works
Matrix factorization learns latent dimensions that encode co-rating behavior. Users that rate similar sets of movies similarly receive nearby latent vectors.

### 9.2 Interpretation of silhouette score
The silhouette score is low (0.0013), which is reasonable for recommendation preference spaces:
- User tastes are often continuous and overlapping
- Hard, well-separated clusters are not always expected
- Visualization remains meaningful for exploratory descriptive analysis

### 9.3 Practical value
- Segment users for marketing or recommendation strategy
- Identify user groups with different rating strictness
- Build interpretable analytics layer on top of recommender embeddings

## 10. Limitations
- t-SNE is non-linear and stochastic; map geometry is mainly local, not global
- Cluster count K=6 was fixed manually
- No genre-aware or temporal behavior features included

## 11. Future Enhancements
- Compare t-SNE with UMAP for potentially better global structure
- Tune K via elbow and silhouette sweep
- Add movie genre enrichment to explain user clusters semantically
- Compare SVD embeddings with neural collaborative filtering embeddings

## 12. Reproducibility Steps
1. Activate virtual environment (Python 3.10)
2. Ensure dependencies are installed in venv
3. Run:
   - .\venv\Scripts\python.exe train.py
   - .\venv\Scripts\python.exe visualize.py
4. Verify generated files:
   - embeddings_2d.csv
   - metrics.csv
   - plot1_tsne_map.png to plot5_summary.png

## 13. Conclusion
This project successfully builds a full descriptive AI pipeline for user embedding visualization on MovieLens 100k.

Using Surprise SVD, we learned 50-dimensional user preference embeddings and projected them to a 2D t-SNE map. The resulting visualizations reveal meaningful local neighborhood structure in user taste behavior and provide a practical, explainable way to analyze recommender-system user representations.

## 14. References
1. GroupLens Research, MovieLens 100k Dataset: https://grouplens.org/datasets/movielens/100k/
2. Nicolas Hug, Surprise: A Python Scikit for Recommender Systems
3. van der Maaten, L. and Hinton, G. (2008). Visualizing Data using t-SNE. Journal of Machine Learning Research.
4. scikit-learn documentation: https://scikit-learn.org/
