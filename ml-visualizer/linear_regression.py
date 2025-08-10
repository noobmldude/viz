import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LinearRegressionVisualizer:
    def __init__(self, X, y, learning_rate=0.01, n_iterations=1000):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.m = len(y)
        self.X_b = np.c_[np.ones((self.m, 1)), self.X]  # Add x0 = 1 to each instance
        self.theta = np.random.randn(2, 1)  # Random initialization
        self.history = []

    def train(self):
        for iteration in range(self.n_iterations):
            gradients = 2/self.m * self.X_b.T.dot(self.X_b.dot(self.theta) - self.y)
            self.theta = self.theta - self.learning_rate * gradients
            self.history.append(self.theta.copy())

    def animate(self):
        fig, ax = plt.subplots()
        ax.scatter(self.X, self.y, label='Data')
        line, = ax.plot([], [], 'r-', label='Regression Line')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title('Linear Regression with Gradient Descent')
        ax.legend()

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            theta = self.history[frame]
            y_predict = self.X_b.dot(theta)
            line.set_data(self.X, y_predict)
            return line,

        anim = FuncAnimation(fig, update, frames=len(self.history),
                               init_func=init, blit=True, interval=20)
        plt.show()

def generate_data():
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y

def run_linear_regression():
    X, y = generate_data()
    lr_visualizer = LinearRegressionVisualizer(X, y)
    lr_visualizer.train()
    lr_visualizer.animate()

if __name__ == '__main__':
    run_linear_regression()
