# display.py
import pygame
import numpy as np
import imageio
import os

class Display:
    _width = 800
    _height = 600
    _window = None
    _record_dir = "frames"

    @staticmethod
    def init(w, h, title="Software Renderer"):
        Display._width = w
        Display._height = h
        pygame.init()
        pygame.display.set_caption(title)
        Display._window = pygame.display.set_mode((w, h))
        if not os.path.exists(Display._record_dir):
            os.makedirs(Display._record_dir)

    @staticmethod
    def show(img: np.ndarray):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if img.shape[:2] != (Display._height, Display._width):
            raise ValueError(f"Image size mismatch: got {img.shape[:2]}, expected {(Display._height, Display._width)}")

        # Convert numpy array (H, W, 3) to surface
        surface = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        Display._window.blit(surface, (0, 0))
        pygame.display.flip()

    @staticmethod
    def record(t, img: np.ndarray):
        Display.show(img)
        filename = os.path.join(Display._record_dir, f"frame_{t:04d}.png")
        imageio.imwrite(filename, img)

if __name__ == "__main__":
    import time

    w, h = 400, 300
    Display.init(w, h)

    for t in range(10):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[20:80, :30, 0] = int(255 * (t / 10))  # Red channel increase
        # Display.record(t, img)
        Display.show(img)
        time.sleep(0.2)

    time.sleep(2)