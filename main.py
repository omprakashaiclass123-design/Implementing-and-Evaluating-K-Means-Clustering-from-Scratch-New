import numpy as np
from sklearn.datasets import make_blobs
from kmeans_scratch import KMeansScratch
from silhouette import silhouette_score

X, y_true = make_blobs(n_samples=600, centers=5, n_features=2, random_state=42)

Ks = range(1, 11)
inertias = []

for k in Ks:
    km = KMeansScratch(k=k)
    km.fit(X)
    inertia = float(np.sum((X - km.centroids[km.labels_])**2))
    inertias.append(inertia)

with open("inertia_output.txt", "w") as f:
    for k, inertia in zip(Ks, inertias):
        f.write(f"K={k}, Inertia={inertia}\n")

# Removing K=1 for silhouette
use_Ks = range(2, 11)
sil_scores = []

for k in use_Ks:
    km = KMeansScratch(k=k)
    km.fit(X)
    sil = silhouette_score(X, km.labels_)
    sil_scores.append(sil)

best_k = use_Ks[np.argmax(sil_scores)]
best_sil = max(sil_scores)

with open("silhouette_output.txt", "w") as f:
    f.write(f"Best K based on Silhouette: {best_k}\n")
    f.write(f"Silhouette Score: {best_sil}\n")

with open("analysis.txt", "w") as f:
    f.write("Analysis of K-Means Results\n")
    f.write("---------------------------------------\n")
    f.write(f"Optimal K based on Silhouette: {best_k}\n")
    f.write(f"Final Silhouette Score: {best_sil}\n")
    f.write("Inertia values decrease as K increases, supporting the Elbow observation.\n")
    f.write("Higher Silhouette Scores indicate stronger, well-separated clusters.\n")
