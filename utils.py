import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections.abc import Sequence
from matplotlib.figure import Figure


def plot_cluster_counts(labels: Sequence[int]) -> Figure:
    counts = pd.Series(labels).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values, edgecolor="black")

    ax.set_title("Cluster Size Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cluster Label", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    return fig


def visualize_clusters(
    X: np.ndarray,
    labels: Sequence[int],
    centers: np.ndarray
) -> Figure:

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, cluster in enumerate(unique_labels):
        mask = labels == cluster

        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=50,
            label=f"Cluster {cluster}",
            color=cmap(idx),
            edgecolor='k',
            alpha=0.7
        )

    # Centroids
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=200,
        marker='X',
        c='black',
        label='Centroids',
        linewidths=2
    )

    ax.set_title("Cluster Visualization", fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    return fig