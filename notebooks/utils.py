import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from sklearn.datasets import make_blobs as sk_make_blobs
from sklearn.datasets import make_moons as sk_make_moons


make_blobs = partial(sk_make_blobs, n_samples=500, centers=5, cluster_std=1.95)
make_moons = partial(sk_make_moons, n_samples=500, noise=0.15)


def draw_clusters(X, y=None, centers=None, cluster_size=10000):
    _, ax = plt.subplots(figsize=(10, 10))
    
    if y is not None:
        cm = {c: f"C{c}" for c in np.unique(y)}
        colors = [cm[i] for i in y]
    else:
        cm = None
        colors = "b"
    
    ax.scatter(X[:,0], X[:,1], color=colors)

    if centers is not None:
        colors = [cm[i] for i in range(len(centers))] if cm else "b"
        ax.scatter(
            centers[:,0], centers[:,1], marker="o", 
            c=colors, alpha=0.25, s=cluster_size
        )
    
    ax.set_xlabel("$X_0$")
    ax.set_xticks([])
    ax.set_ylabel("$X_1$")
    ax.set_yticks([])
    
    return ax



