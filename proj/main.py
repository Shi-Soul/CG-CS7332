
from scene import Scene, Camera, Light
from obj import *
from illumination import Illuminate
from raster import Rasterize, GetImageSize
from display import Display
from time import sleep

material = Material(
    ka=0.1,
    kd=0.5,
    ks=0.5,
    a=10,
    b=10,
    c=10,
    color=np.array([1, 1, 1])
)

cube = Object(
    mesh=Cube(np.array([0, 0, 0]), 1),
    material=material
)

light = Light(np.array([-2, -1.5, 1.2]), 1, np.array([1, 1, 1]))

camera = Camera(np.array([-3, -3, 3]), np.array([1, 1, -1]), np.array([0, 0, 1]), 52, 1.5, 800)

image_size = GetImageSize(camera)



scene = Scene(
    objects=[cube],
    lights=[light],
    camera=camera
)


illuminated_scene = Illuminate(scene)

image = Rasterize(illuminated_scene)

# image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)


Display.init(*image_size, title="Phong Illumination")
Display.show(image)
print("Done")


sleep(10)