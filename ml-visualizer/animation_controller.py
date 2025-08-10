class AnimationController:
    def __init__(self, fig, visualizer):
        self.fig = fig
        self.visualizer = visualizer
        self.paused = False
        self.current_frame = 0
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

    def on_press(self, event):
        if event.key == ' ':
            self.paused = not self.paused
            self.visualizer.toggle_pause(self.paused)
        elif event.key == 'right':
            if self.paused:
                self.visualizer.step_forward()
        elif event.key == 'left':
            if self.paused:
                self.visualizer.step_backward()
        elif event.key == 'r':
            self.visualizer.refresh()
