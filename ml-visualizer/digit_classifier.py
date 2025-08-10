import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import urllib.request
import os
from animation_controller import AnimationController

class DigitClassifierVisualizer:
    def __init__(self, n_epochs=15):
        self.n_epochs = n_epochs
        self.history = []
        self.model = None # Will be initialized in setup_and_train

        self.fig, (self.ax_img, self.ax_hidden, self.ax_logits) = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 2]})

        # Input Image
        self.img_display = self.ax_img.imshow(np.zeros((28, 28)), cmap='gray_r')
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])
        self.ax_img.set_title('Input Image')

        # Hidden Layer Activations
        self.hidden_display = self.ax_hidden.imshow(np.zeros((5, 10)), cmap='viridis', aspect='auto')
        self.ax_hidden.set_xticks([])
        self.ax_hidden.set_yticks([])
        self.ax_hidden.set_title('Hidden Layer Activations')

        # Output Logits
        self.bar_container = self.ax_logits.bar(range(10), np.zeros(10))
        self.ax_logits.set_xticks(range(10))
        self.ax_logits.set_ylabel('Logit Value')
        self.ax_logits.set_title('Output Logits')
        self.prob_labels = [self.ax_logits.text(i, 0, '', ha='center', va='bottom', fontsize=8) for i in range(10)]

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

    def _manual_forward_pass(self, X_sample):
        # Check if model has been fitted at all
        if not hasattr(self.model, 'coefs_'):
            # For MNIST, there are 10 output classes (digits 0-9)
            return np.zeros(self.model.hidden_layer_sizes), np.zeros(10), np.ones(10) / 10

        # Manual forward pass to get internal states
        hidden_activations = X_sample.reshape(1, -1)
        # Layer 1 (Input -> Hidden)
        hidden_activations = hidden_activations @ self.model.coefs_[0] + self.model.intercepts_[0]
        # ReLU activation
        hidden_activations = np.maximum(0, hidden_activations)

        # Layer 2 (Hidden -> Output)
        logits = hidden_activations @ self.model.coefs_[1] + self.model.intercepts_[1]

        # Probabilities
        probabilities = softmax(logits)

        return hidden_activations[0], logits[0], probabilities[0]

    def setup_and_train(self):
        # Pick a random sample from the test set to visualize
        self.sample_index = np.random.randint(0, len(self.X_test))
        self.sample_image = self.X_test[self.sample_index]
        self.sample_label = self.y_test[self.sample_index]

        self.model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1, alpha=1e-4,
                                   solver='sgd', tol=1e-4, random_state=1,
                                   learning_rate_init=.1, verbose=False)
        self.history = []

        # Initial state before training
        initial_hidden, initial_logits, initial_probs = self._manual_forward_pass(self.sample_image)
        self.history.append((initial_hidden, initial_logits, initial_probs))

        # Train epoch by epoch (using partial_fit)
        classes = np.unique(self.y_train)
        for epoch in range(self.n_epochs):
            self.model.partial_fit(self.X_train, self.y_train, classes=classes)
            hidden, logits, probs = self._manual_forward_pass(self.sample_image)
            self.history.append((hidden, logits, probs))

    def _update_plot(self, frame):
        if not self.paused:
            self.current_frame = frame

        hidden_activations, logits, probabilities = self.history[self.current_frame]

        # Update Input Image
        self.img_display.set_data(self.sample_image.reshape(28, 28))

        # Update hidden layer heatmap
        self.hidden_display.set_data(hidden_activations.reshape(5, 10))
        self.hidden_display.set_clim(vmin=0, vmax=hidden_activations.max() + 1e-9)

        # Update logits bar chart and probability labels
        for i, (bar, h, prob) in enumerate(zip(self.bar_container, logits, probabilities)):
            bar.set_height(h)
            self.prob_labels[i].set_text(f'{prob:.2f}')
            # Position text above or below bar based on logit value
            y_pos = h + 0.1 * np.sign(h) if h != 0 else 0.1
            self.prob_labels[i].set_position((i, y_pos))

        # Adjust y-axis limits for logits
        min_logit, max_logit = logits.min(), logits.max()
        padding = (max_logit - min_logit) * 0.2 + 0.5 # Add a bit more padding for labels
        self.ax_logits.set_ylim(min_logit - padding, max_logit + padding)

        # Update titles
        self.ax_img.set_title(f'Input (True: {self.sample_label})')
        self.ax_hidden.set_title(f'Epoch {self.current_frame} / {self.n_epochs}')
        predicted_class = np.argmax(logits)
        self.ax_logits.set_title(f'Prediction: {predicted_class}')

        return [self.img_display, self.hidden_display] + list(self.bar_container) + self.prob_labels

    def animate(self):
        self._load_data_from_url()
        self.setup_and_train()

        self.fig.suptitle('Digit Classification (r: Refresh, Space: Pause, Arrows: Step)', fontsize=16)

        def init():
            # Return an empty list of artists, as _update_plot will draw everything.
            return []

        self.anim = FuncAnimation(self.fig, self._update_plot, frames=len(self.history),
                                  init_func=init, blit=False, interval=500, repeat=False)

        self.controller = AnimationController(self.fig, self)
        plt.show()

    def refresh(self):
        self.setup_and_train()
        self.current_frame = 0
        self.paused = False

        self.anim.frame_seq = iter(range(len(self.history)))

        # Stop and restart the animation timer to ensure it's in a clean state
        self.anim.event_source.stop()
        self.anim.event_source.start()

        # The animation loop will call _update_plot, so we don't need to call it manually.
        # self.fig.canvas.draw_idle() will be handled by the animation frame.

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
