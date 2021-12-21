"""import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file()
"""
import pygame
# import sys
pygame.init()
screen = pygame.display.set_mode((576, 576))
# screen = pygame.display.set_caption("Flappy Bird Game")
clock = pygame.time.Clock()

bg_surface = pygame.image.load('pythonProject1/2100.jpg')

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
           # sys.exit()
            quit()
    screen.blit(bg_surface, (0, 0))
    pygame.display.update()
    clock.tick(120)
