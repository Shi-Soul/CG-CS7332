import numpy as np
import cv2  # Make sure to install OpenCV
from scene import Scene, Camera, Light
from obj import *
from illumination import Illuminate
from raster import Rasterize, GetImageSize
from display import Display
from time import sleep

def main():
    material = Material(
        ka=0.15,
        kd=1.0,
        ks=0.8,
        n=10.0,
        a=1.0,
        b=0.2,
        c=0.2,
        # color=np.array([0.2, 0.2, 0.9]),
        color=np.array([0.9, 0.9, 0.9])
    )
    material_blue = Material(
        ka=0.15,
        kd=1.0,
        ks=0.8,
        n=10.0,
        a=1.0,
        b=0.2,
        c=0.2,
        color=np.array([0.2, 0.2, 0.8])
    )

    obj = Object(
        mesh=
            # Subdivide(
            #     ConvexHull(np.array([[0, 0, -0.5], [1, 0, -0.5], [0, 1, -0.5], [1, 1, -0.5], 
            #                          [0, 0, 0.1], [1, 0, 0.5], [0.3, 0.3, 0.4]])),
            # 4),
            # Cone(np.array([0, 0, 0]), 0.5, 1, 32, 8),
            # Subdivide(
            #     Cylinder(np.array([0, 0, 0]), 0.5, 1, 32),
            # 1),
            Sphere(np.array([0, 0, 0]), 0.5, 32),
            # Subdivide(
            #     Cube(np.array([0, 0, 0]), 1), 
            # 3),
        material=material
    )
    
    material_ground = Material(
        ka=0.15,
        kd=1.0,
        ks=0.8,
        n=10.0,
        a=1.0,
        b=0.2,
        c=0.2,
        color=np.array([0.8, 0.8, 0.8])
    )

    ground = Object(
        mesh=
            # Sphere(np.array([0, 0, 0]), 5, 8),
        Subdivide(
            Cylinder(np.array([0, 0, -0.5]), 3, 0.01, 32),
            1),
        material=material_ground
    )

    objs = [ground, obj]

    lights = [
        # Light(np.array([-0.7, -0.6, 0.8]), 1.0, np.array([0.8, 0, 0.0])),
        Light(np.array([-0.7, -0.6, 0.8]), 0.7, np.array([1, 1.0, 1.0])),
        Light(np.array([-0.2, +0.2, 0.6]), 1, np.array([1, 0.0, 0])),
    ]
    normalize = lambda x: x / np.linalg.norm(x)
    observer = normalize(np.array([-5, -3.5, 3])) * 3
    camera = Camera(observer, -observer, np.array([0, 0, 1]), 60, 1.5, 800)
    # Set up the scene
    scene = Scene(
        objects=objs,
        lights=lights,
        camera=camera  # We'll set the camera later
    )

    # Video settings
    record_type = 'display' # 'video' or 'display'
    frame_shape = GetImageSize(camera)
    frame_width, frame_height = frame_shape
    
    if record_type == 'display':
        Display.init(w=frame_width, h=frame_height)
    elif record_type == 'video':
        video_filename = 'rotation.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filename, fourcc, 30.0, (frame_width, frame_height))

    # Rotate the camera around the object
    T = 1/6  # Total duration in seconds
    # T = 4.0  # Total duration in seconds
    R = 1.0  # Number of rotations
    num_frames = int(T * 30)  # 30 FPS
    radius = 3.0  # Distance from the object
    for i in range(num_frames):
        # Print progress bar
        progress = (i + 1) / num_frames
        bar_length = 50
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f'\rProgress: [{bar}] {progress:.1%}', end='')


        # Change the core context of the scene
        # phase = ((2 * np.pi) * R * i / num_frames)
        # ratio = (np.sin(phase) + 1) / 2
        ratio = (i) / (num_frames-1)
        # material.ka = 0.15 + 0.35 * ratio  # Gradually increase ka from 0.15 to 0.5
        material.ks = ratio  # Gradually increase ka from 0.15 to 0.5
        
        # ratio = i / num_frames
        # lights[1].color = np.array([1-ratio, 0, ratio])
        
        # observer = np.array([radius * np.cos(phase), radius * np.sin(phase), 1.5])
        # camera = Camera(observer, -observer, np.array([0, 0, 1]), 60, 1.5, 800)
        # scene.camera = camera
        
        
        
        # Render the scene
        illuminated_scene = Illuminate(scene)
        image = Rasterize(illuminated_scene)

        if record_type == 'video':
            # Convert the image to BGR format for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            out.write(image_bgr)  # Write the frame to the video file
        elif record_type == 'display':
            Display.record(i, image)

    print()  # New line after progress bar
    if record_type == 'video':
        out.release()  # Finalize the video file
        print(f"Video saved as {video_filename}")

if __name__ == "__main__":
    main()