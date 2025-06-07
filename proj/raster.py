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
    height = 2 * resolution * np.tan(fov * np.pi / 360)
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
    camera_to_world[:3, 3] = camera.viewpoint
    
    # Normalize direction vector
    dir = camera.direction / np.linalg.norm(camera.direction)
    
    # Make up vector orthogonal to direction
    up = camera.up - np.dot(camera.up, dir) * dir
    up = up / np.linalg.norm(up)
    
    # Calculate cross product for right vector
    right = np.cross(up, dir)
    right = right / np.linalg.norm(right)
    
    # Build camera basis matrix
    camera_to_world[:3, :3] = np.column_stack([-right, -up, dir])
    
    world_to_camera = np.linalg.inv(camera_to_world)
    
    # Add homogeneous coordinate
    vertices_h = np.column_stack([vertices, np.ones(len(vertices))])
    
    # Transform to camera space
    vertices_camera = (world_to_camera @ vertices_h.T).T[:, :3]
    
    # Project to image plane
    f = camera.resolution / np.tan(camera.fov * np.pi / 360)
    vertices_projected = vertices_camera * f / vertices_camera[:, 2:]
    
    # Convert to image coordinates
    width, height = GetImageSize(camera)
    vertices_projected[:, 0] = vertices_projected[:, 0] + width / 2
    vertices_projected[:, 1] = vertices_projected[:, 1] + height / 2
    
    # Keep z for depth testing
    vertices_projected[:, 2] = vertices_camera[:, 2]
    
    return vertices_projected #.astype(int)


def bounding_box(face: np.ndarray, projected_vertices: np.ndarray)->Tuple[Tuple[int, int], Tuple[int, int]]:
    # Return the bounding box of the face
    # Returned coordinates: (min_x, max_x), (min_y, max_y), int, [0->H-1, 0->W-1]
    face_projected_vertices = projected_vertices[face]
    min_x = np.floor(np.min(face_projected_vertices[:, 0])).astype(int)
    max_x = np.ceil(np.max(face_projected_vertices[:, 0])).astype(int)
    min_y = np.floor(np.min(face_projected_vertices[:, 1])).astype(int)
    max_y = np.ceil(np.max(face_projected_vertices[:, 1])).astype(int)
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

def UpdateFace(face: np.ndarray, vertex_colors: np.ndarray, projected_vertices: np.ndarray, frame_buffer: np.ndarray, z_buffer: np.ndarray) -> None:
    # Get face vertices and colors
    face_vertices = projected_vertices[face]
    face_colors = vertex_colors[face]
    
    # Get bounding box
    (min_x, max_x), (min_y, max_y) = bounding_box(face, projected_vertices)
    
    # Clip against screen boundaries
    min_x = max(0, min(min_x, frame_buffer.shape[1] - 1))
    max_x = max(0, min(max_x, frame_buffer.shape[1] - 1))
    min_y = max(0, min(min_y, frame_buffer.shape[0] - 1))
    max_y = max(0, min(max_y, frame_buffer.shape[0] - 1))
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
    points = np.stack([x_coords, y_coords], axis=-1, dtype=np.int32)
    
    # Reshape points for vectorized barycentric calculation
    points_flat = points.reshape(-1, 2)
    
    # Calculate barycentric coordinates for all points
    v0v1 = face_vertices[1, :2] - face_vertices[0, :2]
    v0v2 = face_vertices[2, :2] - face_vertices[0, :2]
    v0p = points_flat - face_vertices[0, :2]
    
    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.einsum('ij,j->i', v0p, v0v1)
    d21 = np.einsum('ij,j->i', v0p, v0v2)
    
    denom = d00 * d11 - d01 * d01
    
    # Handle case where denom is close to zero
    if abs(denom) < 1e-10:
        return  # Skip this face as it's degenerate
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    coords = np.stack([u, v, w], axis=1)
    
    # Check which points are inside triangle
    mask = np.all(coords >= 0, axis=1) & np.all(coords <= 1, axis=1)
    
    if not np.any(mask): return
    
    # Get valid points and their coordinates
    valid_points = points_flat[mask]
    valid_coords = coords[mask]
    
    # Calculate z values for valid points
    z_values = np.dot(valid_coords, face_vertices[:, 2])
    
    # Calculate colors for valid points
    colors = np.dot(valid_coords, face_colors)
    
    # Get corresponding buffer indices
    y_indices = valid_points[:, 1].astype(int)
    x_indices = valid_points[:, 0].astype(int)
    
    # Update buffers where depth test passes
    depth_mask = z_values < z_buffer[y_indices, x_indices, 0].flatten()
    
    if not np.any(depth_mask): return
    
    valid_indices = np.where(depth_mask)[0]
    frame_buffer[y_indices[valid_indices], x_indices[valid_indices]] = colors[valid_indices]
    z_buffer[y_indices[valid_indices], x_indices[valid_indices], 0] = z_values[valid_indices]

def DiscretizeFrameBuffer(frame_buffer: np.ndarray)->np.ndarray:
    # Return the discretized frame buffer
    # Returned coordinates: [H,W,3], uint8, [0->255, 0->255, 0->255]
    return (np.clip(frame_buffer, 0, 1) * 255).astype(np.uint8)


def Rasterize(illuminated_scene: IlluminatedScene)->np.ndarray:
    collapsed_scene = CollapseScene(illuminated_scene)
    projected_vertices = ProjectScene(collapsed_scene)
    # print(projected_vertices)
        
    
    image_size = GetImageSize(illuminated_scene.camera)
    frame_buffer = np.zeros((image_size[1], image_size[0], 3)) # float, [H,W,3]
    z_buffer = np.full((image_size[1], image_size[0], 1), np.inf) # float, [H,W,1]
    
    
    for i, face in enumerate(collapsed_scene.faces):
        # if i == 33: 
        #     print(i)
        UpdateFace(face, collapsed_scene.vertex_colors, projected_vertices, frame_buffer, z_buffer)
            
    image = DiscretizeFrameBuffer(frame_buffer)
    return image