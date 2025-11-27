import numpy as np

def silhouette_score(X, labels):
    n = len(X)
    unique = np.unique(labels)
    scores = []

    for i in range(n):
        same_cluster = X[labels == labels[i]]
        other_clusters = [X[labels == uc] for uc in unique if uc != labels[i]]

        if len(same_cluster) > 1:
            a = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))
        else:
            a = 0

        b = min(np.mean(np.linalg.norm(c - X[i], axis=1)) for c in other_clusters)

        score = (b - a) / max(a, b) if max(a, b) > 0 else 0
        scores.append(score)

    return float(np.mean(scores))
