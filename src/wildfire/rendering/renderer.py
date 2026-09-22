import pygame
import numpy as np

class Renderer:
    def __init__(self, width: int, height: int, cell_size: int = 16):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        pygame.init()
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        pygame.display.set_caption("Wildfire Command Center")
        self.font = pygame.font.SysFont(None, 24)
        
    def render_terrain(self, terrain):
        """Renders the base terrain, mapping fuel/elevation to shades of green."""
        surface = pygame.Surface((self.width * self.cell_size, self.height * self.cell_size))
        
        # Precompute colors
        for x in range(self.width):
            for y in range(self.height):
                fuel = terrain.fuel[x, y]
                elevation = terrain.elevation[x, y]
                
                # Base color is a muddy green/brown
                # Higher fuel = greener, higher elevation = brighter
                r = int(50 + elevation * 50)
                g = int(100 + fuel * 100 + elevation * 30)
                b = int(30 + elevation * 30)
                
                color = (min(255, r), min(255, g), min(255, b))
                
                rect = (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(surface, color, rect)
                
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        
    def close(self):
        pygame.quit()

