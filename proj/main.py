
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
        color=np.array([0.8, 0.8, 0.8])
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

    cube = Object(
        mesh=
            # Subdivide(
            #     ConvexHull(np.array([[0, 0, -0.5], [1, 0, -0.5], [0, 1, -0.5], [1, 1, -0.5], 
            #                          [0, 0, 0.1], [1, 0, 0.5], [0.3, 0.3, 0.4]])),
            # 4),
            Cone(np.array([0, 0, 0]), 0.5, 1, 32, 8),
            # Subdivide(
            #     Cylinder(np.array([0, 0, 0]), 0.5, 1, 32),
            # 1),
            # Sphere(np.array([0, 0, 0]), 0.5, 32),
            # Subdivide(
            #     Cube(np.array([0, 0, 0]), 1), 
            # 3),
        material=material
    )

    # light = Light(np.array([-0.7, -0.6, 0.8]), 1, np.array([1, 1, 1]))
    lights = [
        Light(np.array([-0.7, -0.6, 0.8]), 0.7, np.array([1, 1.0, 1.0])),
        Light(np.array([-0.4, +0.2, 0.5]), 1, np.array([1, 0.0, 0])),
    ]

    normalize = lambda x: x / np.linalg.norm(x)
    observer = normalize(np.array([-5, -3.5, 3])) * 2.5
    camera = Camera(observer, -observer, np.array([0, 0, 1]), 60, 1.5, 800)


    scene = Scene(
        objects=[cube],
        lights=lights,
        camera=camera
    )
    illuminated_scene = Illuminate(scene)
    image = Rasterize(illuminated_scene)

    image_size = GetImageSize(camera)
    Display.init(*image_size, title="Phong Illumination")
    Display.record(0, image)
    print("Done")


    sleep(3)
    
    
    
if __name__ == "__main__":
    main()
