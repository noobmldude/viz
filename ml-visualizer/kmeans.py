import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs
from animation_controller import AnimationController

class KMeansVisualizer:
    def __init__(self, n_clusters=3, n_iterations=10):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.history = []
        self.anim = None
        self.fig, self.ax = plt.subplots()
        self.scatter = self.ax.scatter([], [], cmap='viridis')
        self.centroid_scatter = self.ax.scatter([], [], c='red', marker='x', s=100, label='Centroids')
        self.current_frame = 0
        self.paused = False

    def _generate_data(self):
        self.X, self.true_labels = make_blobs(n_samples=300, centers=self.n_clusters, n_features=2)

    def setup_and_train(self):
        self._generate_data()
        self.m, self.n_features = self.X.shape

        # Initialize centroids
        random_indices = np.random.permutation(self.m)
        self.centroids = self.X[random_indices[:self.n_clusters]]

        self.history = []
        for i in range(self.n_iterations):
            # Assign clusters
            clusters = np.zeros(self.m)
            for i_point, point in enumerate(self.X):
                distances = np.linalg.norm(point - self.centroids, axis=1)
                clusters[i_point] = np.argmin(distances)

            self.history.append((self.centroids.copy(), clusters.copy()))

            # Update centroids
            for i_cluster in range(self.n_clusters):
                cluster_points = self.X[clusters == i_cluster]
                if len(cluster_points) > 0:
                    self.centroids[i_cluster] = np.mean(cluster_points, axis=0)

        self.history.append((self.centroids.copy(), clusters.copy())) # Add final state

    def _update_plot(self, frame):
        if not self.paused:
            self.current_frame = frame

        centroids, clusters = self.history[self.current_frame]
        self.scatter.set_array(clusters)
        self.centroid_scatter.set_offsets(centroids)
        return self.scatter, self.centroid_scatter,

    def animate(self):
        self.setup_and_train()

        self.scatter.set_offsets(self.X)
        self.ax.set_xlabel('Feature 1')
        self.ax.set_ylabel('Feature 2')
        self.ax.set_title('K-Means Clustering (r: Refresh, Space: Pause, Arrows: Step)')
        self.ax.legend()

        def init():
            self.centroid_scatter.set_offsets(np.empty((self.n_clusters, 2)))
            return self.scatter, self.centroid_scatter,

        self.anim = FuncAnimation(self.fig, self._update_plot, frames=len(self.history),
                                  init_func=init, blit=False, interval=500, repeat=False)

        self.controller = AnimationController(self.fig, self)
        plt.show()

    def refresh(self):
        self.setup_and_train()
        self.current_frame = 0
        self.paused = False

        self.scatter.set_offsets(self.X)
        self.ax.set_xlim(self.X[:, 0].min() - 1, self.X[:, 0].max() + 1)
        self.ax.set_ylim(self.X[:, 1].min() - 1, self.X[:, 1].max() + 1)
        self.anim.frame_seq = range(len(self.history))

        if self.paused:
            self.anim.event_source.stop()
        else:
            if not self.anim.event_source._running:
                self.anim.event_source.start()

        self._update_plot(0)
        self.fig.canvas.draw_idle()

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

def run_kmeans():
    kmeans_visualizer = KMeansVisualizer(n_clusters=3)
    kmeans_visualizer.animate()

if __name__ == '__main__':
    run_kmeans()
