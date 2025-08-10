import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from animation_controller import AnimationController

class GradientDescentVisualizer:
    def __init__(self, learning_rate=0.001, n_iterations=10000):
        self.func = self.rosenbrock_func
        self.gradient = self.rosenbrock_gradient
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.history = []
        self.anim = None
        self.fig, self.ax = plt.subplots()
        self.path, = self.ax.plot([], [], 'r-o', markersize=4)
        self.current_frame = 0
        self.paused = False

    def setup_and_train(self):
        self.a = np.random.uniform(0.5, 1.5)
        self.b = np.random.uniform(50, 150)
        self.point = np.random.uniform(-2, 2, size=2)

        self.history = [self.point.copy()]
        for _ in range(self.n_iterations):
            grad = self.gradient(self.point[0], self.point[1], a=self.a, b=self.b)
            self.point -= self.learning_rate * grad
            self.history.append(self.point.copy())

    def _update_plot(self, frame):
        if not self.paused:
            self.current_frame = frame

        path_points = np.array(self.history[:self.current_frame+1])
        self.path.set_data(path_points[:, 0], path_points[:, 1])
        return self.path,

    def _draw_contour(self):
        x = np.linspace(-2, 2, 400)
        y = np.linspace(-1, 3, 400)
        X, Y = np.meshgrid(x, y)
        Z = self.func(X, Y, a=self.a, b=self.b)
        self.ax.contour(X, Y, Z, levels=np.logspace(0, 3.5, 20), cmap='viridis')

    def animate(self):
        self.setup_and_train()
        self._draw_contour()

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Gradient Descent (r: Refresh, Space: Pause, Arrows: Step)')

        def init():
            self.path.set_data([], [])
            return self.path,

        self.anim = FuncAnimation(self.fig, self._update_plot, frames=len(self.history),
                                  init_func=init, blit=False, interval=500, repeat=False)

        self.controller = AnimationController(self.fig, self)
        plt.show()

    def refresh(self):
        self.setup_and_train()
        self.current_frame = 0
        self.paused = False

        self.ax.clear()
        self._draw_contour()
        self.path, = self.ax.plot([], [], 'r-o', markersize=4)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Gradient Descent (r: Refresh, Space: Pause, Arrows: Step)')

        self.anim.frame_seq = iter(range(len(self.history)))

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

    @staticmethod
    def rosenbrock_func(x, y, a=1, b=100):
        """Rosenbrock function"""
        return (a - x)**2 + b * (y - x**2)**2

    @staticmethod
    def rosenbrock_gradient(x, y, a=1, b=100):
        """Gradient of the Rosenbrock function"""
        dx = -2 * (a - x) - 4 * b * x * (y - x**2)
        dy = 2 * b * (y - x**2)
        return np.array([dx, dy])

def run_gradient_descent():
    gd_visualizer = GradientDescentVisualizer()
    gd_visualizer.animate()

if __name__ == '__main__':
    run_gradient_descent()
