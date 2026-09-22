import pygame
import numpy as np

class Renderer:
    def __init__(self, width: int, height: int, cell_size: int = 16):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # HUD area on the right
        self.hud_width = 250 
        self.screen_width = (width * cell_size) + self.hud_width
        self.screen_height = height * cell_size
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Wildfire Command Center")
        
        self.font_title = pygame.font.SysFont("Trebuchet MS", 22, bold=True)
        self.font_body = pygame.font.SysFont("Trebuchet MS", 16)
        
        # Precompute the static terrain surface to save FPS during animation
        self.cached_terrain_surface = None
        
        # Wind particles for animation
        self.num_wind_particles = 40
        # Store positions as floats [x, y] in screen space
        self.wind_particles = np.random.rand(self.num_wind_particles, 2)
        self.wind_particles[:, 0] *= (self.width * self.cell_size)
        self.wind_particles[:, 1] *= (self.height * self.cell_size)
        
    def _cache_terrain(self, terrain):
        """Generates the high-res terrain surface exactly once."""
        from scipy.ndimage import zoom
        scale_factor = 4
        elev_hr = zoom(terrain.elevation, scale_factor, order=3)
        fuel_hr = zoom(terrain.fuel, scale_factor, order=3)
        
        dy, dx = np.gradient(elev_hr)
        light_dir = np.array([-1.0, -1.0])
        light_dir = light_dir / np.linalg.norm(light_dir)
        intensity = (-dx * light_dir[0]) + (-dy * light_dir[1])
        shade = 1.0 + (intensity * 1.5)
        
        r, g, b = np.zeros_like(elev_hr), np.zeros_like(elev_hr), np.zeros_like(elev_hr)
        
        m_water = elev_hr < 0.2
        m_low = (elev_hr >= 0.2) & (elev_hr < 0.45)
        m_high = (elev_hr >= 0.45) & (elev_hr < 0.7)
        m_mount = (elev_hr >= 0.7) & (elev_hr < 0.85)
        m_peak = elev_hr >= 0.85
        
        r[m_water], g[m_water], b[m_water] = 20, 75, 55
        r[m_low], g[m_low], b[m_low] = 40, 110, 45
        r[m_high], g[m_high], b[m_high] = 100, 120, 60
        r[m_mount], g[m_mount], b[m_mount] = 140, 110, 80
        r[m_peak], g[m_peak], b[m_peak] = 200, 200, 210
        
        g += (fuel_hr * 30)
        contours = (elev_hr % 0.1) < 0.01
        shade[contours] *= 0.6
        
        final_r = np.clip(r * shade, 0, 255).astype(np.uint8)
        final_g = np.clip(g * shade, 0, 255).astype(np.uint8)
        final_b = np.clip(b * shade, 0, 255).astype(np.uint8)
        
        rgb_array = np.stack((final_r, final_g, final_b), axis=-1)
        base_surface = pygame.surfarray.make_surface(rgb_array)
        self.cached_terrain_surface = pygame.transform.smoothscale(
            base_surface, 
            (self.width * self.cell_size, self.height * self.cell_size)
        )

    def render_world(self, world):
        """Renders the cached terrain and draws animated wind particles over it."""
        if self.cached_terrain_surface is None:
            self._cache_terrain(world.terrain)
            
        # 1. Draw Static Terrain
        self.screen.fill((25, 25, 30))
        self.screen.blit(self.cached_terrain_surface, (0, 0))
        
        # 2. Animate and Draw Wind Particles
        # Velocity in pixels per frame
        vx = world.wind.u * world.wind.speed * 8.0 
        vy = world.wind.v * world.wind.speed * 8.0
        
        map_pixel_width = self.width * self.cell_size
        map_pixel_height = self.height * self.cell_size
        
        for i in range(self.num_wind_particles):
            px, py = self.wind_particles[i]
            
            # Draw a faint white streak representing wind
            start_pos = (int(px), int(py))
            end_pos = (int(px + vx * 2), int(py + vy * 2))
            
            # Fade out particles based on a random length
            alpha = int(100 * world.wind.speed)
            streak_surface = pygame.Surface((abs(end_pos[0]-start_pos[0])+2, abs(end_pos[1]-start_pos[1])+2), pygame.SRCALPHA)
            pygame.draw.line(self.screen, (255, 255, 255, alpha), start_pos, end_pos, 2)
            
            # Move particle
            self.wind_particles[i, 0] += vx
            self.wind_particles[i, 1] += vy
            
            # Wrap around screen edges
            if self.wind_particles[i, 0] > map_pixel_width: self.wind_particles[i, 0] = 0
            if self.wind_particles[i, 0] < 0: self.wind_particles[i, 0] = map_pixel_width
            if self.wind_particles[i, 1] > map_pixel_height: self.wind_particles[i, 1] = 0
            if self.wind_particles[i, 1] < 0: self.wind_particles[i, 1] = map_pixel_height

        # 3. Draw HUD
        self._draw_hud(world)
        pygame.display.flip()
        
    def _draw_hud(self, world):
        hud_x = self.width * self.cell_size + 15
        
        title = self.font_title.render("COMMAND CENTER", True, (220, 220, 220))
        self.screen.blit(title, (hud_x, 20))
        
        pygame.draw.line(self.screen, (100, 100, 100), (hud_x, 50), (self.screen_width - 15, 50))
        
        wind_dir_str = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(((world.wind.direction + 22.5) % 360) / 45)]
        
        stats = [
            "STATUS: IDLE (No Fire)",
            f"Map Size: {self.width}x{self.height}",
            "",
            "ENVIRONMENT:",
            f"  Wind Spd: {world.wind.speed * 100:.0f}%",
            f"  Wind Dir: {wind_dir_str} ({world.wind.direction:.0f}°)",
            f"  Avg Elev: {world.terrain.elevation.mean():.2f}",
            "",
            "DRONE FLEET:",
            "  - Offline"
        ]
        
        for i, text in enumerate(stats):
            rendered = self.font_body.render(text, True, (170, 170, 180))
            self.screen.blit(rendered, (hud_x, 70 + (i * 25)))
            
    def close(self):
        pygame.quit()

