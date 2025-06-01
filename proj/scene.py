
from dataclasses import dataclass
import numpy as np

from typing import List
from obj import Object, Light

@dataclass
class Camera:
    viewpoint: np.ndarray
    direction: np.ndarray
    up: np.ndarray = np.array([0, 0, 1]) 
    fov: float # in degrees, vertical field of view, 0 < fov < 180
    aspect: float # width / height
    near: float # near clipping plane



@dataclass
class Scene:
    lights: List[Light]
    objects: List[Object]
    camera: Camera