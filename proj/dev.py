
from scene import Scene, Camera, Light
from obj import *
from illumination import Illuminate, IlluminatedObject, IlluminatedScene
from raster import Rasterize, GetImageSize
from display import Display
from time import sleep

material = Material(
    ka=0.1,
    kd=0.5,
    ks=0.5,
    n=2,
    a=1,
    b=1,
    c=1,
    color=np.array([1, 1, 1])
)

cube = Object(
    mesh=Triangle(np.array([2, -0.3, -0.3]), np.array([2, 0.5, 0]), np.array([2, 0, 0.5])),
    material=material
)

light = Light(np.array([0, 0, 0]), 1, np.array([1, 0.5, 1]))

camera = Camera(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 1]), 45, 1.5, 800)

image_size = GetImageSize(camera)

Display.init(*image_size, title="Phong Illumination")


scene = Scene(
    objects=[cube],
    lights=[light],
    camera=camera
)


illuminated_scene = Illuminate(scene)
# illuminated_scene = IlluminatedScene(
#     objects=[IlluminatedObject(object, np.array([0.3, 0.8, 0.5]).reshape(1,3).repeat(object.mesh.vertices.shape[0], axis=0)) for object in scene.objects],
#     camera=scene.camera
# )


image = Rasterize(illuminated_scene)



Display.record(0, image)

sleep(5)