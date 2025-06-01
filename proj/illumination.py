from dataclasses import dataclass
import numpy as np

from typing import List
from scene import Scene, Object, Light, Camera

@dataclass
class IlluinatedObject:
    object: Object
    vertex_colors: np.ndarray # [Nv, 3] We adopt Gouraud shading and Phong illumination model
    
@dataclass
class IlluminatedScene:
    objects: List[IlluinatedObject]
    camera: Camera


def phong_illuminate_object(object: Object, lights: List[Light], camera: Camera) -> IlluinatedObject:
    ...


def Illuminate(scene: Scene) -> IlluminatedScene:
    
    return IlluminatedScene(
        objects=[phong_illuminate_object(object, scene.lights, scene.camera) for object in scene.objects],
        camera=scene.camera
    )
