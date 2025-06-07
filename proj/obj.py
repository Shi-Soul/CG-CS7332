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
    n: float            # shininess
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
        return self

    def __str__(self):
        return self._str_


@dataclass
class Object:
    material: Material
    mesh: Mesh


def Subdivide(mesh: Mesh, iterations: int = 1) -> Mesh:
    """Subdivide the mesh by splitting each triangle into 4 smaller triangles
    
    Args:
        mesh: Input mesh to subdivide
        iterations: Number of subdivision iterations to perform (default: 1)
    
    Returns:
        New mesh with subdivided geometry
    """
    current_mesh = mesh
    
    for _ in range(iterations):
        vertices = current_mesh.vertices
        faces = current_mesh.faces
        
        # Get all edges from faces
        edges = np.vstack([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]]
        ])
        edges = np.sort(edges, axis=1)  # Sort edges to ensure consistent ordering
        
        # Get unique edges and their indices
        unique_edges, edge_indices = np.unique(edges, axis=0, return_inverse=True)
        
        # Calculate midpoints for all unique edges
        edge_vertices = vertices[unique_edges]
        midpoints = np.mean(edge_vertices, axis=1)
        
        # Create mapping from original edges to new vertex indices
        edge_to_vertex = {tuple(edge): idx + len(vertices) 
                         for idx, edge in enumerate(unique_edges)}
        
        # Create new faces
        new_faces = []
        for face in faces:
            v1, v2, v3 = face
            # Get new vertices for each edge
            v12 = edge_to_vertex[tuple(sorted([v1, v2]))]
            v23 = edge_to_vertex[tuple(sorted([v2, v3]))]
            v31 = edge_to_vertex[tuple(sorted([v3, v1]))]
            
            # Create 4 new triangles
            new_faces.extend([
                [v1, v12, v31],
                [v12, v2, v23],
                [v31, v23, v3],
                [v12, v23, v31]
            ])
        
        # Create new mesh with subdivided geometry
        current_mesh = Mesh(
            vertices=np.vstack([vertices, midpoints]),
            faces=np.array(new_faces),
            vertex_normals=None,
            face_normals=None,
            _str_=f"Subdivided({current_mesh._str_})"
        )
    return current_mesh.compute_normals()

def Triangle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Mesh:
    """Create a triangle mesh from three points"""
    mesh = Mesh(
        vertices=np.array([p1, p2, p3]),
        faces=np.array([[0, 1, 2]]),
        vertex_normals=None,
        face_normals=None,
        _str_=f"Triangle(p1={p1}, p2={p2}, p3={p3})"
    )
    return mesh.compute_normals()


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
        [0, 2, 1], [0, 3, 2],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 4, 7], [0, 7, 3],  # back
        [1, 6, 5], [1, 2, 6],  # front
        [0, 1, 5], [0, 5, 4],  # left
        [3, 6, 2], [3, 7, 6],  # right
    ])
    
    mesh = Mesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=None,
        face_normals=None,
        _str_=f"Cube(center={center}, size={size})"
    )
    return mesh.compute_normals()


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
            faces.extend([[p1, p3, p2], [p2, p3, p4]])
            # faces.extend([[p1, p2, p3], [p2, p4, p3]])
    faces = np.array(faces)

    mesh = Mesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=None,
        face_normals=None,
        _str_=f"Sphere(center={center}, radius={radius}, segments={segments})"
    )
    return mesh.compute_normals()


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
    return mesh.compute_normals()

def Cone(center, radius, height, segments=16, vertical_segments=1) -> Mesh:
    # Generate vertices
    vertices = []
    
    # Generate vertices for each vertical level
    for v in range(vertical_segments + 1):
        # Calculate radius at this height (decreases linearly from base to apex)
        current_radius = radius * (1 - v/vertical_segments)
        current_height = center[2] - height/2 + (height * v/vertical_segments)
        
        # Generate circle at this height
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            x = center[0] + current_radius * np.cos(theta)
            y = center[1] + current_radius * np.sin(theta)
            z = current_height
            vertices.append([x, y, z])
    
    # Add base center
    vertices.append([center[0], center[1], center[2] - height/2])
    vertices = np.array(vertices)
    
    # Generate faces
    faces = []
    base_center = len(vertices) - 1
    
    # Generate side faces
    for v in range(vertical_segments):
        for i in range(segments):
            i2 = (i + 1) % segments
            # Get indices for current and next level
            curr_level = v * segments
            next_level = (v + 1) * segments
            
            # Create two triangles for each quad
            faces.extend([
                [curr_level + i, curr_level + i2, next_level + i],
                [curr_level + i2, next_level + i2, next_level + i]
            ])
    
    # Generate base cap
    for i in range(segments):
        i2 = (i + 1) % segments
        faces.append([base_center, i2, i])
    
    mesh = Mesh(
        vertices=vertices,
        faces=np.array(faces),
        vertex_normals=None,
        face_normals=None,
        _str_=f"Cone(center={center}, radius={radius}, height={height}, segments={segments}, vertical_segments={vertical_segments})"
    )
    return mesh.compute_normals()

def Polyhedron(vertices: np.ndarray, faces: np.ndarray) -> Mesh:
    """Create a polyhedron mesh from vertices and faces
    
    Args:
        vertices: Array of vertex positions [Nv, 3]
        faces: Array of face vertex indices [Nf, 3]
        
    Returns:
        Mesh object representing the polyhedron
    """
    return Mesh(
        vertices=np.array(vertices),
        faces=np.array(faces),
        vertex_normals=None,
        face_normals=None,
        _str_=f"Polyhedron(vertices={len(vertices)}, faces={len(faces)})"
    ).compute_normals()


def ConvexHull(points: np.ndarray) -> Mesh:
    """Create a convex hull mesh from a set of points
    
    Args:
        points: Array of point positions [Np, 3]
        
    Returns:
        Mesh object representing the convex hull
    """
    from scipy.spatial import ConvexHull as scipy_ConvexHull
    
    # Compute convex hull using scipy
    hull = scipy_ConvexHull(points)
    
    hull_origin = np.mean(points, axis=0)
    
    # Ensure faces have correct winding order
    faces = hull.simplices.copy()
    for i, face in enumerate(faces):
        # Get face vertices
        v0, v1, v2 = points[face]
        # Calculate face normal
        normal = np.cross(v1 - v0, v2 - v0)
        # Calculate vector from face center to origin
        center = (v0 + v1 + v2) / 3
        to_origin = center - hull_origin
        # If normal points inward, reverse face winding
        if np.dot(normal, to_origin) < 0:
            faces[i] = face[::-1]
    
    # Create mesh from hull vertices and faces
    return Mesh(
        vertices=points,
        faces=faces,
        vertex_normals=None,
        face_normals=None,
        _str_=f"ConvexHull(points={len(points)})"
    ).compute_normals()
