

# Computer Graphics Rendering Project

This repository contains the source code for a basic software rendering pipeline written in Python. It was developed as part of the CS7332 Computer Graphics course project and implements a simplified model of the rendering process including scene management, rasterization, and illumination.

## Directory Structure

```
.
├── display.py        # GUI for displaying the rendered image
├── illumination.py   # Phong illumination model
├── main.py           # Entry point; sets up scene and triggers rendering
├── obj.py            # Defines mesh objects and lights
├── raster.py         # Triangle rasterization and Z-buffering
├── requirements.txt  # Python dependencies
└── scene.py          # Defines the structure and elements of the 3D scene
```

## Features

- Software rendering of 3D objects using the Phong lighting model.
- Support for ambient, diffuse, and specular reflectance.
- Z-buffer depth testing for correct visibility.
- Triangle rasterization with barycentric interpolation.
- OBJ file loading and transformation (translation, rotation, scaling).
- Basic scene management with customizable lighting and material properties.

## How to Run

1. **Install dependencies**  
   Use the provided `requirements.txt` to set up the Python environment:

    ```bash
    pip install -r requirements.txt
    ```

2. **Run the renderer**
   Run `main.py` to load the scene and render the output:

   ```bash
   python main.py
   ```

   This will generate rendered images and save them to the designated output path (configured inside `main.py`).
