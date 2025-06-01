import numpy as np
import cv2
import math

class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def normalize(self):
        length = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        if length > 0:
            return Vector3(self.x / length, self.y / length, self.z / length)
        return self
    
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

class Light:
    def __init__(self, position, ambient, diffuse, specular, attenuation):
        self.position = position  # Vector3
        self.ambient = ambient    # (r,g,b)
        self.diffuse = diffuse    # (r,g,b)
        self.specular = specular  # (r,g,b)
        self.attenuation = attenuation  # (constant, linear, quadratic)

class Material:
    def __init__(self, ambient, diffuse, specular, shininess):
        self.ambient = ambient    # (r,g,b)
        self.diffuse = diffuse    # (r,g,b)
        self.specular = specular  # (r,g,b)
        self.shininess = shininess  # float

class Scene:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self.fov = 45  # 减小视场角
        self.near = 0.1
        self.far = 100.0
        
        # Camera setup - 调整相机位置
        self.camera_pos = Vector3(4, 3, 4)
        self.camera_target = Vector3(0, 0, 0)
        self.camera_up = Vector3(0, 1, 0)
        
        # Lights - 调整光照参数
        self.lights = [
            Light(
                Vector3(4, 4, 4),
                (0.4, 0.4, 0.4),  # 增加环境光
                (1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0),
                (1.0, 0.02, 0.002)  # 减小衰减
            ),
            Light(
                Vector3(-4, 3, -4),
                (0.3, 0.3, 0.4),
                (0.8, 0.8, 1.0),
                (0.8, 0.8, 1.0),
                (1.0, 0.02, 0.002)
            )
        ]
        
        # Materials - 调整材质参数
        self.materials = {
            'red': Material((0.4, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0), 8.0),
            'green': Material((0.0, 0.4, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 1.0), 8.0),
            'blue': Material((0.0, 0.0, 0.4), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0), 8.0),
            'yellow': Material((0.4, 0.4, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0), 8.0)
        }
        
        # Objects (will be populated with vertices and normals)
        self.objects = []
        
        # Depth buffer for z-buffering
        self.depth_buffer = None
        
    def calculate_phong_illumination(self, point, normal, material, view_dir):
        """Calculate Phong illumination for a point"""
        final_color = [0, 0, 0]
        normal = normal.normalize()
        view_dir = view_dir.normalize()
        
        for light in self.lights:
            # Calculate light direction and distance
            light_dir = light.position - point
            distance = light_dir.length()
            light_dir = light_dir.normalize()
            
            # Calculate attenuation
            att = (light.attenuation[0] + 
                  light.attenuation[1] * distance + 
                  light.attenuation[2] * distance * distance)
            attenuation = 1.0 / att if att > 0 else 1.0
            
            # Ambient
            ambient = [a * m for a, m in zip(light.ambient, material.ambient)]
            
            # Diffuse (Lambert's cosine law)
            diffuse_intensity = max(0, normal.dot(light_dir))
            diffuse = [d * m * diffuse_intensity for d, m in zip(light.diffuse, material.diffuse)]
            
            # Specular (Phong model)
            reflect_dir = normal * (2 * normal.dot(light_dir)) - light_dir
            specular_intensity = max(0, view_dir.dot(reflect_dir)) ** material.shininess
            specular = [s * m * specular_intensity for s, m in zip(light.specular, material.specular)]
            
            # Combine all components with attenuation
            for i in range(3):
                final_color[i] += (ambient[i] + diffuse[i] + specular[i]) * attenuation
        
        # Clamp colors to [0, 1]
        return [min(1.0, max(0.0, c)) for c in final_color]
    
    def generate_sphere(self, center, radius, material, resolution=32):
        """Generate sphere vertices and normals with triangles"""
        vertices = []
        normals = []
        indices = []
        
        # Generate vertices
        for i in range(resolution + 1):
            lat = math.pi * (-0.5 + float(i) / resolution)
            for j in range(resolution + 1):
                lon = 2 * math.pi * float(j) / resolution
                
                x = math.cos(lat) * math.cos(lon)
                y = math.cos(lat) * math.sin(lon)
                z = math.sin(lat)
                
                vertex = Vector3(
                    center.x + radius * x,
                    center.y + radius * y,
                    center.z + radius * z
                )
                normal = Vector3(x, y, z)
                
                vertices.append(vertex)
                normals.append(normal)
        
        # Generate triangle indices
        for i in range(resolution):
            for j in range(resolution):
                v1 = i * (resolution + 1) + j
                v2 = v1 + 1
                v3 = (i + 1) * (resolution + 1) + j
                v4 = v3 + 1
                
                indices.extend([(v1, v2, v3), (v2, v4, v3)])
        
        return {'vertices': vertices, 'normals': normals, 'indices': indices, 'material': material}
    
    def generate_cylinder(self, center, radius, height, material, resolution=32):
        """Generate cylinder vertices and normals with triangles"""
        vertices = []
        normals = []
        indices = []
        
        # Generate vertices for top and bottom circles
        for i in range(resolution + 1):
            angle = 2 * math.pi * i / resolution
            x = math.cos(angle)
            z = math.sin(angle)
            
            # Bottom circle
            vertices.append(Vector3(center.x + radius * x, center.y, center.z + radius * z))
            normals.append(Vector3(x, 0, z))
            
            # Top circle
            vertices.append(Vector3(center.x + radius * x, center.y + height, center.z + radius * z))
            normals.append(Vector3(x, 0, z))
        
        # Generate triangle indices for side faces
        for i in range(resolution):
            v1 = i * 2
            v2 = v1 + 1
            v3 = (i + 1) * 2
            v4 = v3 + 1
            
            indices.extend([(v1, v2, v3), (v2, v4, v3)])
        
        return {'vertices': vertices, 'normals': normals, 'indices': indices, 'material': material}
    
    def generate_cone(self, center, radius, height, material, resolution=32):
        """Generate cone vertices and normals with triangles"""
        vertices = []
        normals = []
        indices = []
        
        # Apex
        apex = Vector3(center.x, center.y + height, center.z)
        vertices.append(apex)
        normals.append(Vector3(0, 1, 0))  # Apex normal
        
        # Base circle
        base_vertices = []
        base_normals = []
        for i in range(resolution + 1):
            angle = 2 * math.pi * i / resolution
            x = math.cos(angle)
            z = math.sin(angle)
            
            vertex = Vector3(center.x + radius * x, center.y, center.z + radius * z)
            normal = Vector3(x, 0, z)
            
            base_vertices.append(vertex)
            base_normals.append(normal)
            vertices.append(vertex)
            normals.append(normal)
        
        # Generate triangle indices
        for i in range(resolution):
            # Side triangles
            indices.append((0, i + 1, i + 2))
            # Base triangles
            indices.append((1, i + 2, i + 1))
        
        return {'vertices': vertices, 'normals': normals, 'indices': indices, 'material': material}
    
    def generate_cube(self, center, size, material):
        """Generate cube vertices and normals with triangles"""
        vertices = []
        normals = []
        indices = []
        
        # Define the 8 vertices of the cube
        s = size / 2
        corners = [
            Vector3(-s, -s, -s), Vector3(s, -s, -s), Vector3(s, s, -s), Vector3(-s, s, -s),
            Vector3(-s, -s, s), Vector3(s, -s, s), Vector3(s, s, s), Vector3(-s, s, s)
        ]
        
        # Define the 6 faces with their normals
        faces = [
            ([0, 1, 2, 3], Vector3(0, 0, -1)),  # front
            ([4, 5, 6, 7], Vector3(0, 0, 1)),   # back
            ([0, 4, 7, 3], Vector3(-1, 0, 0)),  # left
            ([1, 5, 6, 2], Vector3(1, 0, 0)),   # right
            ([0, 1, 5, 4], Vector3(0, -1, 0)),  # bottom
            ([3, 2, 6, 7], Vector3(0, 1, 0))    # top
        ]
        
        vertex_count = 0
        for face_indices, normal in faces:
            # Add vertices for this face
            for idx in face_indices:
                vertex = Vector3(
                    center.x + corners[idx].x,
                    center.y + corners[idx].y,
                    center.z + corners[idx].z
                )
                vertices.append(vertex)
                normals.append(normal)
            
            # Add triangles for this face
            indices.extend([
                (vertex_count, vertex_count + 1, vertex_count + 2),
                (vertex_count, vertex_count + 2, vertex_count + 3)
            ])
            vertex_count += 4
        
        return {'vertices': vertices, 'normals': normals, 'indices': indices, 'material': material}
    
    def setup_scene(self):
        """Setup the scene with all objects"""
        # 调整对象位置和大小
        self.objects = [
            self.generate_sphere(Vector3(-2, 0, 0), 0.8, self.materials['red']),
            self.generate_cylinder(Vector3(2, 0, 0), 0.4, 1.6, self.materials['green']),
            self.generate_cone(Vector3(0, 0, -2), 0.8, 1.6, self.materials['blue']),
            self.generate_cube(Vector3(0, 0, 2), 1.6, self.materials['yellow'])
        ]
    
    def project_to_screen(self, point):
        """Project a 3D point to screen space with proper perspective"""
        # Transform point to camera space
        camera_space = point - self.camera_pos
        
        # Apply perspective projection
        if camera_space.z <= 0:
            return None, float('inf')
            
        # Calculate perspective projection
        f = 1.0 / math.tan(math.radians(self.fov / 2))
        aspect = self.width / self.height
        
        # 修改投影计算
        x = camera_space.x * f / (aspect * camera_space.z)
        y = camera_space.y * f / camera_space.z
        
        # 调整投影范围
        x = x * 0.5 + 0.5  # 映射到 [0,1]
        y = y * 0.5 + 0.5
        
        # Scale to screen coordinates
        screen_x = int(x * self.width)
        screen_y = int(y * self.height)
        
        return (screen_x, screen_y), camera_space.z
    
    def render_triangle(self, image, v1, v2, v3, n1, n2, n3, material, view_dir):
        """Render a triangle using scan-line algorithm"""
        # Get screen coordinates and depths
        p1, z1 = self.project_to_screen(v1)
        p2, z2 = self.project_to_screen(v2)
        p3, z3 = self.project_to_screen(v3)
        
        if p1 is None or p2 is None or p3 is None:
            return
            
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # Sort vertices by y-coordinate
        if y1 > y2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            z1, z2 = z2, z1
            v1, v2 = v2, v1
            n1, n2 = n2, n1
        if y1 > y3:
            x1, x3 = x3, x1
            y1, y3 = y3, y1
            z1, z3 = z3, z1
            v1, v3 = v3, v1
            n1, n3 = n3, n1
        if y2 > y3:
            x2, x3 = x3, x2
            y2, y3 = y3, y2
            z2, z3 = z3, z2
            v2, v3 = v3, v2
            n2, n3 = n3, n2
        
        # Calculate slopes
        if y2 != y1:
            slope1 = (x2 - x1) / (y2 - y1)
        else:
            slope1 = 0
        if y3 != y1:
            slope2 = (x3 - x1) / (y3 - y1)
        else:
            slope2 = 0
        if y3 != y2:
            slope3 = (x3 - x2) / (y3 - y2)
        else:
            slope3 = 0
        
        # Scan convert
        for y in range(max(0, y1), min(self.height, y3 + 1)):
            if y < y2:
                # First half
                x_start = int(x1 + (y - y1) * slope1)
                x_end = int(x1 + (y - y1) * slope2)
            else:
                # Second half
                x_start = int(x2 + (y - y2) * slope3)
                x_end = int(x1 + (y - y1) * slope2)
            
            if x_start > x_end:
                x_start, x_end = x_end, x_start
            
            for x in range(max(0, x_start), min(self.width, x_end + 1)):
                # Calculate barycentric coordinates
                area = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
                if area == 0:
                    continue
                    
                w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / area
                w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / area
                w3 = 1 - w1 - w2
                
                if w1 < 0 or w2 < 0 or w3 < 0:
                    continue
                
                # Interpolate depth
                z = w1 * z1 + w2 * z2 + w3 * z3
                
                # Check depth buffer
                if z >= self.depth_buffer[y, x]:
                    continue
                
                # Interpolate normal
                normal = (n1 * w1 + n2 * w2 + n3 * w3).normalize()
                
                # Interpolate position
                pos = v1 * w1 + v2 * w2 + v3 * w3
                
                # Calculate illumination
                color = self.calculate_phong_illumination(pos, normal, material, view_dir)
                
                # Update pixel
                image[y, x] = color
                self.depth_buffer[y, x] = z
    
    def render_frame(self, rotation_angle):
        """Render a single frame"""
        # Create image buffer and depth buffer
        image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.depth_buffer = np.full((self.height, self.width), float('inf'))
        
        # Calculate view direction
        view_dir = (self.camera_target - self.camera_pos).normalize()
        
        # Render each object
        for obj in self.objects:
            vertices = obj['vertices']
            normals = obj['normals']
            indices = obj['indices']
            material = obj['material']
            
            # 对每个顶点应用旋转
            rotated_vertices = [self.rotate_point(v, rotation_angle) for v in vertices]
            rotated_normals = [self.rotate_point(n, rotation_angle) for n in normals]
            
            # Render each triangle
            for idx1, idx2, idx3 in indices:
                self.render_triangle(
                    image,
                    rotated_vertices[idx1], rotated_vertices[idx2], rotated_vertices[idx3],
                    rotated_normals[idx1], rotated_normals[idx2], rotated_normals[idx3],
                    material,
                    view_dir
                )
        
        # Convert to 8-bit image
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    def rotate_point(self, point, angle):
        """Rotate a point around the Y axis"""
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        # 修正旋转矩阵
        x = point.x * cos_a - point.z * sin_a
        y = point.y  # Y坐标保持不变
        z = point.x * sin_a + point.z * cos_a
        
        return Vector3(x, y, z)
    
    def render_animation(self, output_file='output.mp4', fps=30, duration=5):
        """Render animation and save to video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (self.width, self.height))
        
        total_frames = fps * duration
        for frame in range(total_frames):
            rotation_angle = (frame / total_frames) * 360
            frame_image = self.render_frame(rotation_angle)
            out.write(frame_image)
            
            # Display progress
            if frame % fps == 0:
                print(f"Rendering frame {frame}/{total_frames}")
        
        out.release()
        print(f"Animation saved to {output_file}")

if __name__ == "__main__":
    # Create and setup scene
    scene = Scene()
    scene.setup_scene()
    
    # Render animation
    scene.render_animation()
