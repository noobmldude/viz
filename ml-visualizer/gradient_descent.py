import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class GradientDescentVisualizer:
    def __init__(self, func, gradient, initial_point, learning_rate=0.1, n_iterations=100):
        self.func = func
        self.gradient = gradient
        self.point = np.array(initial_point, dtype=float)
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.history = [self.point.copy()]

    def optimize(self):
        for _ in range(self.n_iterations):
            grad = self.gradient(self.point[0], self.point[1])
            self.point -= self.learning_rate * grad
            self.history.append(self.point.copy())

    def animate(self):
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.func(X, Y)

        fig, ax = plt.subplots()
        ax.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='viridis')
        path, = ax.plot([], [], 'r-o', markersize=4)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Gradient Descent')

        def init():
            path.set_data([], [])
            return path,

        def update(frame):
            path_points = np.array(self.history[:frame+1])
            path.set_data(path_points[:, 0], path_points[:, 1])
            return path,

        anim = FuncAnimation(fig, update, frames=len(self.history),
                               init_func=init, blit=True, interval=50)
        plt.show()

def func_to_optimize(x, y):
    return x**2 + y**2

def gradient_of_func(x, y):
    return np.array([2*x, 2*y])

def run_gradient_descent():
    initial_point = [-9, 9]
    gd_visualizer = GradientDescentVisualizer(func_to_optimize, gradient_of_func, initial_point)
    gd_visualizer.optimize()
    gd_visualizer.animate()

if __name__ == '__main__':
    run_gradient_descent()
