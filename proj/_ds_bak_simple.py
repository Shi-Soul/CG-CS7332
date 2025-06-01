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

class Scene:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self.fov = 45  # degrees
        self.near = 0.1
        self.far = 100.0
        
        # Camera setup
        self.camera_pos = Vector3(3, 3, 3)
        self.camera_target = Vector3(0, 0, 0)
        
        # Create a simple cube
        self.cube_vertices = self.generate_cube()
        
    def generate_cube(self):
        """Generate a simple cube centered at origin"""
        size = 1.0
        s = size / 2
        return [
            # Front face (red)
            Vector3(-s, -s, -s), Vector3(s, -s, -s), Vector3(s, s, -s), Vector3(-s, s, -s),
            # Back face (green)
            Vector3(-s, -s, s), Vector3(s, -s, s), Vector3(s, s, s), Vector3(-s, s, s),
            # Top face (blue)
            Vector3(-s, s, -s), Vector3(s, s, -s), Vector3(s, s, s), Vector3(-s, s, s),
            # Bottom face (yellow)
            Vector3(-s, -s, -s), Vector3(s, -s, -s), Vector3(s, -s, s), Vector3(-s, -s, s),
            # Right face (magenta)
            Vector3(s, -s, -s), Vector3(s, s, -s), Vector3(s, s, s), Vector3(s, -s, s),
            # Left face (cyan)
            Vector3(-s, -s, -s), Vector3(-s, s, -s), Vector3(-s, s, s), Vector3(-s, -s, s)
        ]
    
    def project_to_screen(self, point):
        """Simple perspective projection"""
        # Transform to camera space
        camera_space = point - self.camera_pos
        
        # Skip if behind camera
        if camera_space.z <= 0:
            return None
            
        # Simple perspective projection
        scale = 1.0 / camera_space.z
        x = camera_space.x * scale
        y = camera_space.y * scale
        
        # Map to screen coordinates
        screen_x = int((x + 1) * self.width / 2)
        screen_y = int((y + 1) * self.height / 2)
        
        return (screen_x, screen_y)
    
    def rotate_point(self, point, angle):
        """Rotate point around Y axis"""
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        x = point.x * cos_a - point.z * sin_a
        z = point.x * sin_a + point.z * cos_a
        
        return Vector3(x, point.y, z)
    
    def render_frame(self, rotation_angle):
        """Render a single frame"""
        # Create image buffer
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Rotate and project vertices
        projected_vertices = []
        for vertex in self.cube_vertices:
            rotated = self.rotate_point(vertex, rotation_angle)
            screen_pos = self.project_to_screen(rotated)
            if screen_pos is not None:
                projected_vertices.append(screen_pos)
        
        # Draw each face
        colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0)   # Cyan
        ]
        
        # Draw each face
        for i in range(0, len(projected_vertices), 4):
            if i + 3 >= len(projected_vertices):
                break
                
            # Get vertices for this face
            v1, v2, v3, v4 = projected_vertices[i:i+4]
            
            # Convert to numpy array for cv2
            pts = np.array([v1, v2, v3, v4], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw filled polygon
            cv2.fillPoly(image, [pts], colors[i//4])
            
            # Draw edges
            cv2.polylines(image, [pts], True, (255, 255, 255), 1)
        
        return image
    
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
    # Create and render scene
    scene = Scene()
    scene.render_animation() 