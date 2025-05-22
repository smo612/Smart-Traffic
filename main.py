# main.py

import pygame
from traffic_env import TrafficEnv
from config import *

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Smart Traffic Simulation")
    clock = pygame.time.Clock()

    env = TrafficEnv()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        env.update()
        env.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
