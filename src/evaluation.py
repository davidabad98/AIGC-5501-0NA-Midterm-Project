import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


class ClusterEvaluator:
    """
    A class for evaluating and visualizing K-Means clustering results in NLP tasks.

    Methods:
    - compute_metrics: Calculates Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index.
    - plot_cluster_distribution: Displays a bar plot of cluster sizes.
    - get_top_words_per_cluster: Extracts the most important words per cluster.
    - plot_clusters_2d: Visualizes clusters using PCA or t-SNE.
    - save_results: Saves metrics and visualizations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_vector,
        vectorizer,
        save_path: str = "results",
    ):
        """
        Initializes the ClusterEvaluator with clustering data and vectorizer.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing clustering results.
        - text_vector (np.ndarray): The TF-IDF or vectorized text data.
        - vectorizer: The trained vectorizer (TF-IDF, Word2Vec, etc.).
        - save_path (str): Directory to save evaluation results.
        """
        self.df = df
        self.text_vector = text_vector
        self.vectorizer = vectorizer
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    from scipy.sparse import issparse

    def compute_metrics(self):
        """
        Computes key clustering metrics: Silhouette Score.
        """
        clusters = self.df["cluster"]

        if issparse(self.text_vector):
            text_vector = self.text_vector  # Keep it sparse
        else:
            text_vector = csr_matrix(self.text_vector)  # Convert if needed

        silhouette = silhouette_score(text_vector, clusters)

        metrics = {
            "Silhouette Score": silhouette,
        }

        print("Clustering Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        return metrics

    def plot_cluster_distribution(self):
        """
        Plots the distribution of cluster sizes.
        """
        plt.figure(figsize=(10, 5))
        sns.countplot(x=self.df["cluster"], palette="viridis")
        plt.title("Cluster Distribution")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=45)

        save_path = os.path.join(self.save_path, "cluster_distribution.png")
        plt.savefig(save_path)
        plt.show()

    def get_top_words_per_cluster(self, num_words=10):
        """
        Retrieves the most important words for each cluster using TF-IDF scores.

        Parameters:
        - num_words (int): Number of top words to retrieve per cluster.

        Returns:
        dict: A dictionary where keys are clusters and values are lists of top words.
        """
        feature_names = self.vectorizer.get_feature_names_out()
        cluster_centers = self.text_vector.toarray().mean(
            axis=0
        )  # Averaging TF-IDF scores per cluster

        top_words = {}
        for cluster in sorted(self.df["cluster"].unique()):
            # Get indices of top words for this cluster
            indices = np.argsort(cluster_centers)[::-1][:num_words]
            top_words[cluster] = [feature_names[i] for i in indices]

        # Save results
        words_path = os.path.join(self.save_path, "top_words_per_cluster.txt")
        with open(words_path, "w") as f:
            for cluster, words in top_words.items():
                f.write(f"Cluster {cluster}: {', '.join(words)}\n")

        return top_words

    def plot_clusters_2d(self, method="PCA"):
        """
        Projects high-dimensional clusters to 2D using PCA or t-SNE and visualizes them.
        """
        if method == "PCA":
            reducer = PCA(n_components=2)
        elif method == "t-SNE":
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        else:
            raise ValueError("Invalid method. Choose 'PCA' or 't-SNE'.")

        if issparse(self.text_vector):
            reduced_data = reducer.fit_transform(
                self.text_vector.toarray()
            )  # Only convert here
        else:
            reduced_data = reducer.fit_transform(self.text_vector)

        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            hue=self.df["cluster"],
            palette="viridis",
            alpha=0.7,
        )
        plt.title(f"Cluster Visualization using {method}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(title="Cluster")

        save_path = os.path.join(self.save_path, f"cluster_visualization_{method}.png")
        plt.savefig(save_path)
        plt.show()

    def save_results(self):
        """
        Calls all evaluation functions and saves results.
        """
        self.compute_metrics()
        self.plot_cluster_distribution()
        self.get_top_words_per_cluster()
        self.plot_clusters_2d(method="PCA")
        self.plot_clusters_2d(method="t-SNE")
