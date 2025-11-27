
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans as SKKMeans
import matplotlib.pyplot as plt

class KMeansScratch:
    def __init__(self, k, max_iter=300, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        np.random.seed(42)
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iter):
            labels = self._assign_clusters(X)
            new_centroids = self._compute_centroids(X, labels)
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break
            self.centroids = new_centroids

        self.labels_ = self._assign_clusters(X)
        self.inertia_ = self._compute_inertia(X, self.labels_)

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

    def _compute_inertia(self, X, labels):
        return sum(np.sum((X[labels == i] - self.centroids[i])**2) for i in range(self.k))

if __name__ == "__main__":
    X, y_true = make_blobs(n_samples=600, centers=4, cluster_std=0.6, random_state=42)

    inertias = []
    for k in range(2, 11):
        km = KMeansScratch(k)
        km.fit(X)
        inertias.append(km.inertia_)

    optimal_k = np.argmin(np.gradient(np.gradient(inertias))) + 2

    final_model = KMeansScratch(optimal_k)
    final_model.fit(X)
    sil_scratch = silhouette_score(X, final_model.labels_)

    sk = SKKMeans(n_clusters=optimal_k, random_state=42).fit(X)
    sil_sklearn = silhouette_score(X, sk.labels_)

    print("Optimal K:", optimal_k)
    print("Silhouette (scratch):", sil_scratch)
    print("Silhouette (sklearn):", sil_sklearn)
