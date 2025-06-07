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
    
    # Initialize vertex colors array
    vertex_colors = np.zeros((len(vertices), 3))
    
    # Calculate view direction for each vertex (from vertex to camera)
    view_dirs = camera.viewpoint - vertices
    view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=1, keepdims=True)
    
    # For each vertex, calculate illumination from all lights
    for i, vertex in enumerate(vertices):
        normal = vertex_normals[i]
        view_dir = view_dirs[i]
        
        # Initialize color components
        ambient = material.ka * material.color
        diffuse = np.zeros(3)
        specular = np.zeros(3)
        
        # Calculate contribution from each light
        for light in lights:
            # Light direction (from vertex to light)
            light_dir = light.position - vertex
            light_dist = np.linalg.norm(light_dir)
            light_dir = light_dir / light_dist
            
            S = 1/(material.a  + material.b * light_dist + material.c * light_dist ** 2)
            
            # Diffuse component
            diffuse_intensity = max(0, np.dot(normal, light_dir))
            diffuse += material.kd * diffuse_intensity * light.color * light.intensity * S
            
            # Specular component
            reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir
            specular_intensity = max(0, np.dot(view_dir, reflect_dir))
            specular += material.ks * (specular_intensity ** material.n) * light.color * light.intensity * S
        
        # Combine components and apply material color
        vertex_colors[i] = np.clip(ambient + diffuse + specular, 0, 1)
    # breakpoint()
    
    return IlluminatedObject(object, vertex_colors)


def Illuminate(scene: Scene) -> IlluminatedScene:
    
    return IlluminatedScene(
        objects=[phong_illuminate_object(object, scene.lights, scene.camera) for object in scene.objects],
        camera=scene.camera
    )
