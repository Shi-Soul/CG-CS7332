from dataclasses import dataclass
import numpy as np

from typing import List
from scene import Scene, Object, Light, Camera

@dataclass
class IlluminatedObject:
    object: Object
    vertex_colors: np.ndarray # [Nv, 3] We adopt Gouraud shading and Phong illumination model
    
@dataclass
class IlluminatedScene:
    objects: List[IlluminatedObject]
    camera: Camera

def phong_illuminate_object(object: Object, lights: List[Light], camera: Camera) -> IlluminatedObject:
    # Get mesh data
    mesh = object.mesh
    vertices = mesh.vertices
    vertex_normals = mesh.vertex_normals
    material = object.material
    
    # Calculate view direction for each vertex (from vertex to camera)
    view_dirs = camera.viewpoint - vertices
    view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=1, keepdims=True)
    
    # Initialize color components
    ambient = material.ka * material.color
    diffuse = np.zeros((len(vertices), 3))
    specular = np.zeros((len(vertices), 3))
    
    # Stack all light positions and colors
    light_positions = np.stack([light.position for light in lights])  # [Nl, 3]
    light_colors = np.stack([light.color * light.intensity for light in lights])  # [Nl, 3]
    
    # Calculate light directions and distances for all lights at once
    light_dirs = light_positions[None, :, :] - vertices[:, None, :]  # [Nv, Nl, 3]
    light_dists = np.linalg.norm(light_dirs, axis=2, keepdims=True)  # [Nv, Nl, 1]
    light_dirs = light_dirs / light_dists  # [Nv, Nl, 3]
    
    # Attenuation factor for all lights
    S = 1/(material.a + material.b * light_dists + material.c * light_dists ** 2)  # [Nv, Nl, 1]
    
    # Diffuse component for all lights
    diffuse_intensities = np.maximum(0, np.sum(vertex_normals[:, None, :] * light_dirs, axis=2, keepdims=True))  # [Nv, Nl, 1]
    diffuse = np.sum(material.kd * diffuse_intensities * light_colors[None, :, :] * S, axis=1)  # [Nv, 3]
    
    # Specular component for all lights
    reflect_dirs = 2 * np.sum(vertex_normals[:, None, :] * light_dirs, axis=2, keepdims=True) * vertex_normals[:, None, :] - light_dirs  # [Nv, Nl, 3]
    specular_intensities = np.maximum(0, np.sum(view_dirs[:, None, :] * reflect_dirs, axis=2, keepdims=True))  # [Nv, Nl, 1]
    specular = np.sum(material.ks * (specular_intensities ** material.n) * light_colors[None, :, :] * S, axis=1)  # [Nv, 3]
    
    # Combine components and apply material color
    vertex_colors = np.clip(ambient + diffuse + specular, 0, 1)
    
    return IlluminatedObject(object, vertex_colors)


def Illuminate(scene: Scene) -> IlluminatedScene:
    
    return IlluminatedScene(
        objects=[phong_illuminate_object(object, scene.lights, scene.camera) for object in scene.objects],
        camera=scene.camera
    )
