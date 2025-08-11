import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from animation_controller import AnimationController

class LinearRegressionVisualizer:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.history = []
        self.anim = None
        self.fig, self.ax = plt.subplots()
        self.scatter = self.ax.scatter([], [], label='Data')
        self.line, = self.ax.plot([], [], 'r-', label='Regression Line')
        self.current_frame = 0
        self.paused = False

    def _generate_data(self):
        self.X = 2 * np.random.rand(100, 1)
        self.y = 4 + 3 * self.X + np.random.randn(100, 1)

    def setup_and_train(self):
        self._generate_data()
        self.m = len(self.y)
        self.X_b = np.c_[np.ones((self.m, 1)), self.X]
        self.theta = np.random.randn(2, 1)
        self.history = []
        for iteration in range(self.n_iterations):
            gradients = 2/self.m * self.X_b.T.dot(self.X_b.dot(self.theta) - self.y)
            self.theta = self.theta - self.learning_rate * gradients
            self.history.append(self.theta.copy())

    def _update_plot(self, frame):
        if not self.paused:
            self.current_frame = frame

        theta = self.history[self.current_frame]
        y_predict = self.X_b.dot(theta)
        self.line.set_data(self.X, y_predict)
        return self.line,

    def animate(self):
        self.setup_and_train()

        self.scatter.set_offsets(np.c_[self.X, self.y])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('y')
        self.ax.set_title('Linear Regression (r: Refresh, Space: Pause, Arrows: Step)')
        self.ax.legend()
        self.ax.set_xlim(0, 2)
        self.ax.set_ylim(0, 15)

        def init():
            self.line.set_data([], [])
            return self.line,

        self.anim = FuncAnimation(self.fig, self._update_plot, frames=len(self.history),
                                  init_func=init, blit=False, interval=500, repeat=True)

        self.controller = AnimationController(self.fig, self)
        plt.show()

    def refresh(self):
        self.setup_and_train()
        self.current_frame = 0
        self.paused = False

        self.scatter.set_offsets(np.c_[self.X, self.y])
        self.anim.frame_seq = iter(range(len(self.history)))

        # Stop and restart the animation timer to ensure it's in a clean state
        self.anim.event_source.stop()
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

def run_linear_regression():
    lr_visualizer = LinearRegressionVisualizer()
    lr_visualizer.animate()

if __name__ == '__main__':
    run_linear_regression()
