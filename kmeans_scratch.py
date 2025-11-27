import numpy as np

class KMeansScratch:
    def __init__(self, k=3, max_iters=300):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        n, d = X.shape
        rng = np.random.default_rng(42)
        self.centroids = X[rng.choice(n, self.k, replace=False)]

        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = []
            for i in range(self.k):
                pts = X[labels == i]
                if len(pts) > 0:
                    new_centroids.append(pts.mean(axis=0))
                else:
                    new_centroids.append(self.centroids[i])
            new_centroids = np.array(new_centroids)

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        self.labels_ = labels
