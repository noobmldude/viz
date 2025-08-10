import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs
from animation_controller import AnimationController

class KMeansVisualizer:
    def __init__(self, X, n_clusters=3, n_iterations=10):
        self.X = X
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.m, self.n_features = X.shape
        self.centroids = self._initialize_centroids()
        self.clusters = np.zeros(self.m)
        self.history = []
        self.anim = None
        self.fig, self.ax = plt.subplots()
        self.scatter = self.ax.scatter(self.X[:, 0], self.X[:, 1], c=self.clusters, cmap='viridis')
        self.centroid_scatter = self.ax.scatter([], [], c='red', marker='x', s=100, label='Centroids')
        self.current_frame = 0
        self.paused = False

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
        self.history.append((self.centroids.copy(), self.clusters.copy())) # Add final state

    def _update_plot(self, frame):
        if not self.paused:
            self.current_frame = frame

        centroids, clusters = self.history[self.current_frame]
        self.scatter.set_array(clusters)
        self.centroid_scatter.set_offsets(centroids)
        return self.scatter, self.centroid_scatter,

    def animate(self):
        self.ax.set_xlabel('Feature 1')
        self.ax.set_ylabel('Feature 2')
        self.ax.set_title('K-Means Clustering (Space: Pause/Resume, Arrows: Step)')
        self.ax.legend()

        def init():
            self.scatter.set_offsets(self.X)
            self.centroid_scatter.set_offsets(np.empty((self.n_clusters, 2)))
            return self.scatter, self.centroid_scatter,

        self.anim = FuncAnimation(self.fig, self._update_plot, frames=len(self.history),
                                  init_func=init, blit=False, interval=500, repeat=False)

        self.controller = AnimationController(self.fig, self)
        plt.show()

    def toggle_pause(self, paused):
        self.paused = paused
        if self.paused:
            self.anim.event_source.stop()
        else:
            self.anim.event_source.start()

    def step_forward(self):
        self.current_frame = min(len(self.history) - 1, self.current_frame + 1)
        self._update_plot(self.current_frame)
        self.fig.canvas.draw_idle()

    def step_backward(self):
        self.current_frame = max(0, self.current_frame - 1)
        self._update_plot(self.current_frame)
        self.fig.canvas.draw_idle()

def generate_data():
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2)
    return X

def run_kmeans():
    X = generate_data()
    kmeans_visualizer = KMeansVisualizer(X, n_clusters=3)
    kmeans_visualizer.train()
    kmeans_visualizer.animate()

if __name__ == '__main__':
    run_kmeans()
