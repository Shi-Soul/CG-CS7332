from dataclasses import dataclass
import numpy as np

from typing import List, Tuple
from scene import Camera
from illumination import IlluminatedScene

def get_image_size(camera: Camera)->Tuple[int, int]:
    resolution = camera.resolution
    fov = camera.fov
    aspect = camera.aspect
    height = 2 * resolution * np.tan(fov / 2)
    width = height * aspect
    breakpoint()
    return int(width), int(height)


def Rasterize(illuminated_scene: IlluminatedScene)->np.ndarray:
    empty_image = np.zeros((*get_image_size(illuminated_scene.camera), 3), dtype=np.uint8)
    return empty_image
    raise NotImplementedError("Rasterize is not implemented")