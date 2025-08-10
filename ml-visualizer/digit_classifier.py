import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import urllib.request
import os
from animation_controller import AnimationController

class DigitClassifierVisualizer:
    def __init__(self, n_epochs=15):
        self.n_epochs = n_epochs
        self.history = []
        self.model = None # Will be initialized in setup_and_train

        self.fig, (self.ax_img, self.ax_bar) = plt.subplots(1, 2, figsize=(10, 5))
        self.img_display = self.ax_img.imshow(np.zeros((28, 28)), cmap='gray_r')
        self.bar_container = self.ax_bar.bar(range(10), np.ones(10) / 10)
        self.ax_bar.set_xticks(range(10))
        self.ax_bar.set_ylim(0, 1)
        self.ax_bar.set_ylabel('Probability')
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])

        self.anim = None
        self.current_frame = 0
        self.paused = False

    def _load_data_from_url(self):
        url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
        filepath = 'mnist.npz'
        if not os.path.exists(filepath):
            print("Downloading MNIST dataset...")
            urllib.request.urlretrieve(url, filepath)

        with np.load(filepath) as f:
            X_train, y_train = f['x_train'], f['y_train']
            X_test, y_test = f['x_test'], f['y_test']

        X_train = X_train.reshape(-1, 28 * 28) / 255.
        X_test = X_test.reshape(-1, 28 * 28) / 255.

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train.astype(str), y_test.astype(str)

    def setup_and_train(self):
        # Pick a random sample from the test set to visualize
        self.sample_index = np.random.randint(0, len(self.X_test))
        self.sample_image = self.X_test[self.sample_index]
        self.sample_label = self.y_test[self.sample_index]

        self.model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1, alpha=1e-4,
                                   solver='sgd', tol=1e-4, random_state=1,
                                   learning_rate_init=.1, verbose=False)
        self.history = []

        initial_probs = np.ones((1, 10)) / 10
        self.history.append(initial_probs[0])

        classes = np.unique(self.y_train)
        for epoch in range(self.n_epochs):
            self.model.partial_fit(self.X_train, self.y_train, classes=classes)
            probs = self.model.predict_proba([self.sample_image])
            self.history.append(probs[0])

    def _update_plot(self, frame):
        if not self.paused:
            self.current_frame = frame

        probs = self.history[self.current_frame]
        for bar, h in zip(self.bar_container, probs):
            bar.set_height(h)

        self.ax_img.set_title(f'Epoch {self.current_frame} / {self.n_epochs}')
        predicted_class = np.argmax(probs)
        self.ax_bar.set_title(f'Prediction: {predicted_class} (True: {self.sample_label})')

        return list(self.bar_container)

    def animate(self):
        self._load_data_from_url()
        self.setup_and_train()

        self.img_display.set_data(self.sample_image.reshape(28, 28))

        self.fig.suptitle('Digit Classification (r: Refresh, Space: Pause, Arrows: Step)', fontsize=16)

        def init():
            return self._update_plot(0)

        self.anim = FuncAnimation(self.fig, self._update_plot, frames=len(self.history),
                                  init_func=init, blit=False, interval=500, repeat=False)

        self.controller = AnimationController(self.fig, self)
        plt.show()

    def refresh(self):
        self.setup_and_train()
        self.current_frame = 0
        self.paused = False

        self.img_display.set_data(self.sample_image.reshape(28, 28))
        self.anim.frame_seq = iter(range(len(self.history)))

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

def run_digit_classifier():
    visualizer = DigitClassifierVisualizer()
    visualizer.animate()

if __name__ == '__main__':
    run_digit_classifier()
