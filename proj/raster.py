from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
from scene import Camera, Object
from illumination import IlluminatedObject, IlluminatedScene

map_list = lambda *args: list(map(*args))


def GetImageSize(camera: Camera)->Tuple[int, int]:
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
    collapsed_objects = map_list(CollapseObject, illuminated_scene.objects)
    num_vertex_per_object = map_list(lambda x: x.vertices.shape[0], collapsed_objects)
    vertex_offset = np.cumsum(num_vertex_per_object)
    vertex_offset = np.insert(vertex_offset, 0, 0)[:-1]

    collapsed_vertices = np.concatenate(map_list(lambda x: x.vertices, collapsed_objects))
    collapsed_faces = np.concatenate(map_list(lambda x, y: x.faces + y, collapsed_objects, vertex_offset))
    collapsed_vertex_colors = np.concatenate(map_list(lambda x: x.vertex_colors, collapsed_objects))
    
    return CollapsedScene(collapsed_vertices, collapsed_faces, collapsed_vertex_colors, illuminated_scene.camera)


def ProjectScene(collapsed_scene: CollapsedScene)->np.ndarray:
    # Project the collapsed scene onto the image plane
    # Return the x, y, z coordinates of each projected vertex
    # Returned coordinates: [Nv, 3], int, [0->H-1, 0->W-1, 0->infinity], only z is world coordinate
    camera = collapsed_scene.camera
    vertices = collapsed_scene.vertices
    
    # Transform vertices to camera space
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = camera.position
    camera_to_world[:3, :3] = np.column_stack([np.cross(camera.up, camera.direction), camera.up, camera.direction])
    
    world_to_camera = np.linalg.inv(camera_to_world)
    
    # Add homogeneous coordinate
    vertices_h = np.column_stack([vertices, np.ones(len(vertices))])
    
    # Transform to camera space
    vertices_camera = (world_to_camera @ vertices_h.T).T[:, :3]
    
    # Project to image plane
    f = camera.resolution / np.tan(camera.fov / 2)
    vertices_projected = vertices_camera * f / vertices_camera[:, 2:]
    
    # Convert to image coordinates
    width, height = GetImageSize(camera)
    vertices_projected[:, 0] = vertices_projected[:, 0] + width / 2
    vertices_projected[:, 1] = vertices_projected[:, 1] + height / 2
    
    # Keep z for depth testing
    vertices_projected[:, 2] = vertices_camera[:, 2]
    
    return vertices_projected.astype(int)


def bounding_box(face: np.ndarray, projected_vertices: np.ndarray)->Tuple[Tuple[int, int], Tuple[int, int]]:
    # Return the bounding box of the face
    # Returned coordinates: (min_x, max_x), (min_y, max_y), int, [0->H-1, 0->W-1]
    face_projected_vertices = projected_vertices[face]
    min_x = np.min(face_projected_vertices[:, 0])
    max_x = np.max(face_projected_vertices[:, 0])
    min_y = np.min(face_projected_vertices[:, 1])
    max_y = np.max(face_projected_vertices[:, 1])
    
    return (min_x, max_x), (min_y, max_y)


def barycentric_coords(p: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # Calculate barycentric coordinates for point p relative to triangle (v0, v1, v2)
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0
    
    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)
    
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    return np.array([u, v, w])


def UpdateFace(face: np.ndarray, vertex_colors: np.ndarray, projected_vertices: np.ndarray, frame_buffer: np.ndarray, z_buffer: np.ndarray)->None:
    # Get face vertices and colors
    face_vertices = projected_vertices[face]
    face_colors = vertex_colors[face]
    
    # Get bounding box
    (min_x, max_x), (min_y, max_y) = bounding_box(face, projected_vertices)
    
    # Clip against screen boundaries
    min_x = max(0, min_x)
    max_x = min(frame_buffer.shape[1] - 1, max_x)
    min_y = max(0, min_y)
    max_y = min(frame_buffer.shape[0] - 1, max_y)
    
    # Rasterize each pixel in bounding box
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            p = np.array([x, y])
            
            # Calculate barycentric coordinates
            coords = barycentric_coords(p, face_vertices[0, :2], face_vertices[1, :2], face_vertices[2, :2])
            
            # Check if point is inside triangle
            if np.all(coords >= 0) and np.all(coords <= 1):
                # Interpolate z
                z = coords.dot(face_vertices[:, 2])
                
                # Depth test
                if z < z_buffer[y, x, 0]:
                    # Interpolate color
                    color = coords.dot(face_colors)
                    
                    # Update buffers
                    frame_buffer[y, x] = color
                    z_buffer[y, x, 0] = z


def DiscretizeFrameBuffer(frame_buffer: np.ndarray)->np.ndarray:
    # Return the discretized frame buffer
    # Returned coordinates: [H,W,3], uint8, [0->255, 0->255, 0->255]
    return (np.clip(frame_buffer, 0, 1) * 255).astype(np.uint8)


def Rasterize(illuminated_scene: IlluminatedScene)->np.ndarray:
    collapsed_scene = CollapseScene(illuminated_scene)
    projected_vertices = ProjectScene(collapsed_scene)
    
    image_size = GetImageSize(illuminated_scene.camera)
    frame_buffer = np.zeros((*image_size, 3)) # float, [H,W,3]
    z_buffer = np.full((*image_size, 1), np.inf) # float, [H,W,1]
    
    for face in collapsed_scene.faces:
        UpdateFace(face, collapsed_scene.vertex_colors, projected_vertices, frame_buffer, z_buffer)
    
    image = DiscretizeFrameBuffer(frame_buffer)
    
    return image