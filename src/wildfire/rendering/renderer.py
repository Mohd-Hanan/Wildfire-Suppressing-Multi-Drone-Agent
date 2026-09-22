import pygame
import numpy as np

class Renderer:
    def __init__(self, width: int, height: int, cell_size: int = 16):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        pygame.init()
        # Create a modern dark-themed window
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        pygame.display.set_caption("Wildfire Command Center")
        self.font = pygame.font.SysFont("Courier", 18, bold=True)
        
    def render_terrain(self, terrain):
        """Renders the terrain with professional 3D hillshading and natural colors."""
        surface = pygame.Surface((self.width * self.cell_size, self.height * self.cell_size))
        
        # Calculate hillshade (simulating a sun shining from the Top-Left)
        # Gradient gives us the direction of the slope
        dy, dx = np.gradient(terrain.elevation)
        # Light vector (pointing down and right)
        light_dir = np.array([-1.0, -1.0])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        for x in range(self.width):
            for y in range(self.height):
                elev = terrain.elevation[x, y]
                fuel = terrain.fuel[x, y]
                
                # Base Biome Colors
                if elev < 0.3:
                    # Valley/Dense Forest (Dark Green)
                    base_r, base_g, base_b = 34, 100, 34
                elif elev < 0.7:
                    # Mid-altitude (Lighter Green / Shrubs)
                    base_r, base_g, base_b = 85, 130, 45
                else:
                    # High Altitude (Rocky/Brown)
                    base_r, base_g, base_b = 139, 115, 85
                    
                # Modify color based on how much fuel is present
                # More fuel = slightly richer green
                g = base_g + (fuel * 30)
                
                # Hillshading for 3D effect
                normal = np.array([-dx[x, y], -dy[x, y], 0.1])
                normal = normal / np.linalg.norm(normal)
                intensity = np.dot(normal[:2], light_dir)
                
                # Apply shadow or highlight
                shade = 1.0 + (intensity * 0.4) # +/- 40% brightness based on sun
                
                final_r = min(255, max(0, int(base_r * shade)))
                final_g = min(255, max(0, int(g * shade)))
                final_b = min(255, max(0, int(base_b * shade)))
                
                rect = (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(surface, (final_r, final_g, final_b), rect)
                
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        
    def close(self):
        pygame.quit()

