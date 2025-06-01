import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

glutInit()
glutInitWindowSize(800, 600)
glutCreateWindow("Renderer")


def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
