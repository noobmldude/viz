import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs

class KMeansVisualizer:
    def __init__(self, X, n_clusters=3, n_iterations=10):
        self.X = X
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.m, self.n_features = X.shape
        self.centroids = self._initialize_centroids()
        self.clusters = np.zeros(self.m)
        self.history = []

    def _initialize_centroids(self):
        random_indices = np.random.permutation(self.m)
        centroids = self.X[random_indices[:self.n_clusters]]
        return centroids

    def _assign_clusters(self):
        for i, point in enumerate(self.X):
            distances = np.linalg.norm(point - self.centroids, axis=1)
            self.clusters[i] = np.argmin(distances)

    def _update_centroids(self):
        for i in range(self.n_clusters):
            cluster_points = self.X[self.clusters == i]
            if len(cluster_points) > 0:
                self.centroids[i] = np.mean(cluster_points, axis=0)

    def train(self):
        for i in range(self.n_iterations):
            self.history.append((self.centroids.copy(), self.clusters.copy()))
            self._assign_clusters()
            self._update_centroids()

    def animate(self):
        fig, ax = plt.subplots()
        scatter = ax.scatter(self.X[:, 0], self.X[:, 1], c=self.clusters, cmap='viridis')
        centroid_scatter = ax.scatter([], [], c='red', marker='x', s=100, label='Centroids')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('K-Means Clustering')
        ax.legend()

        def init():
            scatter.set_offsets(np.empty((self.m, 2)))
            centroid_scatter.set_offsets(np.empty((self.n_clusters, 2)))
            return scatter, centroid_scatter

        def update(frame):
            centroids, clusters = self.history[frame]
            scatter.set_array(clusters)
            scatter.set_offsets(self.X)
            centroid_scatter.set_offsets(centroids)
            return scatter, centroid_scatter,

        anim = FuncAnimation(fig, update, frames=len(self.history),
                               init_func=init, blit=True, interval=500)
        plt.show()

def generate_data():
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
    return X

def run_kmeans():
    X = generate_data()
    kmeans_visualizer = KMeansVisualizer(X, n_clusters=3)
    kmeans_visualizer.train()
    kmeans_visualizer.animate()

if __name__ == '__main__':
    run_kmeans()
