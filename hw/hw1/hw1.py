import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def draw_quadratic(a, b, c, x_start, x_end):
    if x_start > x_end:
        x_start, x_end = x_end, x_start
    
    # Initial calculations
    x = x_start
    y = a * x**2 + b * x + c  # Starting point (only multiplication here)
    
    # Calculate initial delta and delta change
    delta = 2 * a * x + b
    delta_change = 2 * a
    
    # Store points for drawing
    points = []
    
    while x <= x_end:
        points.append((x, y))
        # Update y using incremental approach (no multiplication)
        y += delta
        delta += delta_change
        x += 1
    
    return points

def main():
    # Polynomial coefficients
    a = -0.1
    b = -3
    c = 2
    
    # Range of x values
    x_start = -10
    x_end = 10
    
    # Generate points using incremental algorithm
    points = draw_quadratic(a, b, c, x_start, x_end)
    
    # Set up pygame and OpenGL
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    # gluOrtho2D(x_start - 2, x_end + 2, min([a*x**2 + b*x + c for x in [x_start, x_end]]) - 5, max([a*x**2 + b*x + c for x in [x_start, x_end]]) + 5)
    gluOrtho2D(-13, 13, min(points, key=lambda x: x[1])[1] - 5, max(points, key=lambda x: x[1])[1] + 5)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBegin(GL_LINE_STRIP)
        for point in points:
            glVertex2f(point[0], point[1])
        glEnd()
        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()