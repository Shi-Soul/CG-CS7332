from dataclasses import dataclass
import numpy as np
# from toolz import 

from typing import List, Tuple
from scene import Camera, Object
from illumination import IlluminatedObject, IlluminatedScene




def get_image_size(camera: Camera)->Tuple[int, int]:
    resolution = camera.resolution
    fov = camera.fov
    aspect = camera.aspect
    height = 2 * resolution * np.tan(fov / 2)
    width = height * aspect
    return int(width), int(height)



@dataclass
class CollapsedObject:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray


@dataclass
class CollapsedScene:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray
    camera: Camera


def CollapseObject(illuminated_object: IlluminatedObject)->CollapsedObject:
    mesh = illuminated_object.object.mesh
    return CollapsedObject(mesh.vertices, mesh.faces, illuminated_object.vertex_colors)


def CollapseScene(illuminated_scene: IlluminatedScene)->CollapsedScene:
    # Make all vertices and faces from different objects unified into one index system
    
    collapsed_objects = map(CollapseObject, illuminated_scene.objects)
    num_vertex_per_object = map(lambda x: x.vertices.shape[0], collapsed_objects)
    vertex_offset = np.cumsum(num_vertex_per_object)
    vertex_offset = np.insert(vertex_offset, 0, 0)[:-1]

    
    collapsed_vertices = np.concatenate(map(lambda x: x.vertices, collapsed_objects))
    collapsed_faces = np.concatenate(map(lambda x, y: x.faces + y, collapsed_objects, vertex_offset))
    collapsed_vertex_colors = np.concatenate(map(lambda x: x.vertex_colors, collapsed_objects))
    
    
    return CollapsedScene(collapsed_vertices, collapsed_faces, collapsed_vertex_colors, illuminated_scene.camera)
    




def Rasterize(illuminated_scene: IlluminatedScene)->np.ndarray:
    empty_image = np.zeros((*get_image_size(illuminated_scene.camera), 3), dtype=np.uint8)
    collapsed_scene = CollapseScene(illuminated_scene)

    raise NotImplementedError("Rasterize is not implemented")