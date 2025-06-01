from dataclasses import dataclass
import numpy as np


@dataclass
class Light:
    position: np.ndarray  # [3] position in world space
    intensity: float      # light intensity
    color: np.ndarray    # [3] rgb color, 0 <= color <= 1


@dataclass
class Material:
    ka: float           # ambient coefficient
    kd: float           # diffuse coefficient  
    ks: float           # specular coefficient
    a: float            # phong exponent a
    b: float            # phong exponent b
    c: float            # phong exponent c
    color: np.ndarray   # [3] rgb color, 0 <= color <= 1


@dataclass
class Mesh:
    vertices: np.ndarray      # [Nv, 3] vertex positions
    faces: np.ndarray         # [Nf, 3] face vertex indices
    vertex_normals: np.ndarray  # [Nv, 3] vertex normals
    face_normals: np.ndarray    # [Nf, 3] face normals
    
    _str_: str = ""           # string representation
    
    def compute_normals(self):
        """Compute face and vertex normals for the mesh"""
        # Compute face normals
        self.face_normals = np.zeros((self.faces.shape[0], 3))
        for i, face in enumerate(self.faces):
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]] 
            v2 = self.vertices[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 0:  # Avoid division by zero
                self.face_normals[i] = normal / norm

        # Compute vertex normals by averaging adjacent face normals
        self.vertex_normals = np.zeros_like(self.vertices)
        for i in range(len(self.vertices)):
            adjacent_faces = [j for j, face in enumerate(self.faces) if i in face]
            if adjacent_faces:
                normals = self.face_normals[adjacent_faces]
                avg_normal = np.mean(normals, axis=0)
                norm = np.linalg.norm(avg_normal)
                if norm > 0:  # Avoid division by zero
                    self.vertex_normals[i] = avg_normal / norm

    def __str__(self):
        return self._str_


@dataclass
class Object:
    material: Material
    mesh: Mesh


def Triangle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Mesh:
    """Create a triangle mesh from three points"""
    mesh = Mesh(
        vertices=np.array([p1, p2, p3]),
        faces=np.array([[0, 1, 2]]),
        vertex_normals=None,
        face_normals=None,
        _str_=f"Triangle(p1={p1}, p2={p2}, p3={p3})"
    )
    mesh.compute_normals()
    return mesh


def Cube(center: np.ndarray, size: float) -> Mesh:
    """Create a cube mesh centered at center with given size"""
    half = size / 2
    x, y, z = center
    
    # Define 8 vertices
    vertices = np.array([
        [x - half, y - half, z - half],  # 0
        [x + half, y - half, z - half],  # 1
        [x + half, y + half, z - half],  # 2
        [x - half, y + half, z - half],  # 3
        [x - half, y - half, z + half],  # 4
        [x + half, y - half, z + half],  # 5
        [x + half, y + half, z + half],  # 6
        [x - half, y + half, z + half],  # 7
    ])
    
    # Define 12 triangles (6 faces)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # front
        [4, 5, 6], [4, 6, 7],  # back
        [0, 4, 7], [0, 7, 3],  # left
        [1, 5, 6], [1, 6, 2],  # right
        [0, 1, 5], [0, 5, 4],  # bottom
        [3, 2, 6], [3, 6, 7],  # top
    ])
    
    mesh = Mesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=None,
        face_normals=None,
        _str_=f"Cube(center={center}, size={size})"
    )
    mesh.compute_normals()
    return mesh


def Sphere(center: np.ndarray, radius: float, segments: int = 16) -> Mesh:
    """Create a sphere mesh using latitude-longitude method"""
    # Generate vertices
    vertices = []
    for i in range(segments + 1):
        phi = np.pi * i / segments
        for j in range(segments):
            theta = 2 * np.pi * j / segments
            x = center[0] + radius * np.sin(phi) * np.cos(theta)
            y = center[1] + radius * np.sin(phi) * np.sin(theta)
            z = center[2] + radius * np.cos(phi)
            vertices.append([x, y, z])
    vertices = np.array(vertices)

    # Generate faces
    faces = []
    for i in range(segments):
        for j in range(segments):
            p1 = i * segments + j
            p2 = i * segments + (j + 1) % segments
            p3 = (i + 1) * segments + j
            p4 = (i + 1) * segments + (j + 1) % segments
            faces.extend([[p1, p2, p3], [p2, p4, p3]])
    faces = np.array(faces)

    mesh = Mesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=None,
        face_normals=None,
        _str_=f"Sphere(center={center}, radius={radius}, segments={segments})"
    )
    mesh.compute_normals()
    return mesh


def Cylinder(center: np.ndarray, radius: float, height: float, segments: int = 16) -> Mesh:
    """Create a cylinder mesh"""
    # Generate vertices
    vertices = []
    # Bottom circle
    for i in range(segments):
        theta = 2 * np.pi * i / segments
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = center[2] - height/2
        vertices.append([x, y, z])
    # Top circle
    for i in range(segments):
        theta = 2 * np.pi * i / segments
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = center[2] + height/2
        vertices.append([x, y, z])
    # Add center points for caps
    vertices.append([center[0], center[1], center[2] - height/2])  # bottom center
    vertices.append([center[0], center[1], center[2] + height/2])  # top center
    vertices = np.array(vertices)

    # Generate faces
    faces = []
    bottom_center = len(vertices) - 2
    top_center = len(vertices) - 1
    # Side faces
    for i in range(segments):
        i2 = (i + 1) % segments
        faces.extend([
            [i, i2, i + segments],
            [i2, i2 + segments, i + segments]
        ])
    # Bottom cap
    for i in range(segments):
        i2 = (i + 1) % segments
        faces.append([bottom_center, i2, i])
    # Top cap
    for i in range(segments):
        i2 = (i + 1) % segments
        faces.append([top_center, i + segments, i2 + segments])
    faces = np.array(faces)

    mesh = Mesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=None,
        face_normals=None,
        _str_=f"Cylinder(center={center}, radius={radius}, height={height}, segments={segments})"
    )
    mesh.compute_normals()
    return mesh


def Cone(center, radius, height, segments=16) -> Mesh:
    mesh = Mesh()
    # Generate vertices
    vertices = []
    # Base circle
    for i in range(segments):
        theta = 2 * np.pi * i / segments
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = center[2] - height/2
        vertices.append([x, y, z])
    # Apex
    vertices.append([center[0], center[1], center[2] + height/2])
    # Base center
    vertices.append([center[0], center[1], center[2] - height/2])
    mesh.vertices = np.array(vertices)

    # Generate faces
    faces = []
    apex = len(vertices) - 2
    base_center = len(vertices) - 1
    # Side faces
    for i in range(segments):
        i2 = (i + 1) % segments
        faces.append([i, i2, apex])
    # Base cap
    for i in range(segments):
        i2 = (i + 1) % segments
        faces.append([base_center, i2, i])
    
    mesh.faces = np.array(faces)
    mesh._str_ = f"Cone(center={center}, radius={radius}, height={height}, segments={segments})"
    mesh.compute_normals()
    return mesh

def Polyhedron(vertices, faces) -> Mesh:
    mesh = Mesh()
    mesh.vertices = np.array(vertices)
    mesh.faces = np.array(faces)
    mesh._str_ = f"Polyhedron(vertices={len(vertices)}, faces={len(faces)})"
    mesh.compute_normals()
    return mesh
