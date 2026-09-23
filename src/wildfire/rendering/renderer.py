import pygame
import numpy as np
import math

class Renderer:
    def __init__(self, width=48, height=48, cell_size=16):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        self.hud_width = 380
        self.screen_width = width * cell_size + self.hud_width
        self.screen_height = height * cell_size
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Wildfire Command Center")
        
        self.font_title = pygame.font.SysFont("Trebuchet MS", 22, bold=True)
        self.font_sub = pygame.font.SysFont("Trebuchet MS", 18, bold=True)
        self.font_body = pygame.font.SysFont("Trebuchet MS", 14)
        self.font_mono = pygame.font.SysFont("Courier New", 14, bold=True)
        
        self.cached_terrain_surface = None
        
        self.num_wind_particles = 40
        self.wind_particles = np.random.rand(self.num_wind_particles, 2)
        self.wind_particles[:, 0] *= (self.width * self.cell_size)
        self.wind_particles[:, 1] *= (self.height * self.cell_size)
        
    def _cache_terrain(self, terrain):
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
        if self.cached_terrain_surface is None:
            self._cache_terrain(world.terrain)
            
        from pygame.surfarray import pixels3d
        rgb_array = pixels3d(self.cached_terrain_surface).copy()
        
        moist_display = np.kron(world.terrain.moisture, np.ones((self.cell_size, self.cell_size)))
        fuel_display = np.kron(world.terrain.fuel, np.ones((self.cell_size, self.cell_size)))
        
        m_wet = moist_display >= 0.99
        m_fireline = fuel_display <= 0.01
        
        fire_display = np.kron(world.fire_manager.fire_map, np.ones((self.cell_size, self.cell_size), dtype=np.int8))
        
        m_unburned = fire_display == 0
        m_igniting = fire_display == 1
        m_burning = fire_display == 2
        m_smoldering = fire_display == 3
        m_burned = fire_display == 4
        
        rgb_array[m_wet & m_unburned] = [0, 255, 255]
        rgb_array[m_fireline & m_unburned] = [255, 50, 100]
        
        rgb_array[m_igniting] = [255, 220, 50]
        rgb_array[m_burning] = [255, 60, 20]
        rgb_array[m_smoldering] = [100, 50, 40]
        rgb_array[m_burned] = [30, 30, 30]
        
        fire_surface = pygame.surfarray.make_surface(rgb_array)
        self.screen.fill((20, 20, 24)) # Dark panel background
        self.screen.blit(fire_surface, (0, 0))
        
        vx = world.wind.u * world.wind.speed * 8.0 
        vy = world.wind.v * world.wind.speed * 8.0
        
        map_pixel_width = self.width * self.cell_size
        map_pixel_height = self.height * self.cell_size
        
        for i in range(self.num_wind_particles):
            px, py = self.wind_particles[i]
            start_pos = (int(px), int(py))
            end_pos = (int(px + vx * 2), int(py + vy * 2))
            alpha = int(100 * world.wind.speed)
            pygame.draw.line(self.screen, (255, 255, 255, alpha), start_pos, end_pos, 2)
            self.wind_particles[i, 0] += vx
            self.wind_particles[i, 1] += vy
            if self.wind_particles[i, 0] > map_pixel_width: self.wind_particles[i, 0] = 0
            if self.wind_particles[i, 0] < 0: self.wind_particles[i, 0] = map_pixel_width
            if self.wind_particles[i, 1] > map_pixel_height: self.wind_particles[i, 1] = 0
            if self.wind_particles[i, 1] < 0: self.wind_particles[i, 1] = map_pixel_height

        base_rect = (world.base_x * self.cell_size, world.base_y * self.cell_size, self.cell_size * 2, self.cell_size * 2)
        pygame.draw.rect(self.screen, (255, 255, 255), base_rect, 3)
        pygame.draw.line(self.screen, (255, 255, 255), (base_rect[0], base_rect[1]), (base_rect[0]+base_rect[2], base_rect[1]+base_rect[3]), 2)
        pygame.draw.line(self.screen, (255, 255, 255), (base_rect[0]+base_rect[2], base_rect[1]), (base_rect[0], base_rect[1]+base_rect[3]), 2)
        
        from wildfire.simulation.drone import DroneType
        
        # Draw Target Connection Lines FIRST (so they are under drones)
        for drone in world.drones:
            if not drone.active: continue
            target = getattr(drone, 'current_target', None)
            if target:
                cx = int((drone.x + 0.5) * self.cell_size)
                cy = int((drone.y + 0.5) * self.cell_size)
                tx = int((target[0] + 0.5) * self.cell_size)
                ty = int((target[1] + 0.5) * self.cell_size)
                color = (0, 200, 255) if drone.type == DroneType.WATER else (255, 100, 200)
                # Subtle dashed/alpha line can be hard in basic pygame, use thin solid line with lower alpha via surface
                line_surf = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
                pygame.draw.line(line_surf, (*color, 60), (cx, cy), (tx, ty), 1)
                self.screen.blit(line_surf, (0, 0))

        for drone in world.drones:
            if not drone.active: 
                continue 
                
            is_water = (drone.type == DroneType.WATER)
            color = (0, 150, 255) if is_water else (255, 50, 50)
            cx = int((drone.x + 0.5) * self.cell_size)
            cy = int((drone.y + 0.5) * self.cell_size)
            
            # Quadcopter Shape
            pygame.draw.circle(self.screen, color, (cx, cy), 4)
            o = 5 
            pygame.draw.circle(self.screen, (220, 220, 220), (cx-o, cy-o), 3)
            pygame.draw.circle(self.screen, (220, 220, 220), (cx+o, cy-o), 3)
            pygame.draw.circle(self.screen, (220, 220, 220), (cx-o, cy+o), 3)
            pygame.draw.circle(self.screen, (220, 220, 220), (cx+o, cy+o), 3)
            
            # Label near drone
            lbl = f"WAT-{drone.id}" if is_water else f"RET-{drone.id}"
            lbl_surf = self.font_body.render(lbl, True, (255, 255, 255))
            self.screen.blit(lbl_surf, (cx + 8, cy - 8))

        self._draw_hud(world)
        pygame.display.flip()
        
    def _draw_hud(self, world):
        hud_x = self.width * self.cell_size + 20
        y = 20
        
        def render_text(text, font, color, x_offset=0):
            nonlocal y
            surf = font.render(text, True, color)
            self.screen.blit(surf, (hud_x + x_offset, y))
            y += font.get_height()
            
        def render_progress(val, max_val, label):
            nonlocal y
            pct = val / max_val if max_val > 0 else 0
            bars = int(pct * 10)
            bar_str = "█" * bars + "░" * (10 - bars)
            surf = self.font_mono.render(f"{label} [{bar_str}] {val}/{max_val}", True, (200, 200, 200))
            self.screen.blit(surf, (hud_x + 10, y))
            y += self.font_mono.get_height()
            
        # Headers
        render_text("WILDFIRE COMMAND CENTER", self.font_title, (240, 240, 240))
        render_text("TESTING MODE", self.font_sub, (100, 255, 100))
        y += 4
        
        # Fire Status
        active_fires = np.sum(world.fire_manager.fire_map == 2)
        total_burned = np.sum(world.fire_manager.fire_map == 4)
        render_text("FIRE STATUS", self.font_sub, (200, 200, 200))
        render_text(f"ACTIVE FIRE CELLS: {active_fires}", self.font_body, (255, 100, 100), 10)
        render_text(f"TOTAL BURNED: {total_burned}", self.font_body, (150, 150, 150), 10)
        y += 4
        
        # Environment
        wind_dir_str = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(((world.wind.direction + 22.5) % 360) / 45)]
        render_text("ENVIRONMENT", self.font_sub, (200, 200, 200))
        render_text(f"Wind Direction: {wind_dir_str} ({world.wind.direction:.0f}*)", self.font_body, (200, 200, 200), 10)
        render_text(f"Wind Speed: {world.wind.speed * 100:.0f}%", self.font_body, (200, 200, 200), 10)
        render_text(f"Avg Elevation: {world.terrain.elevation.mean():.2f}", self.font_body, (200, 200, 200), 10)
        y += 4
        
        # Drones
        render_text("DRONE FLEET", self.font_sub, (200, 200, 200))
        from wildfire.simulation.drone import DroneType
        for drone in world.drones:
            y += 2
            is_water = drone.type == DroneType.WATER
            name = f"WAT-{drone.id}" if is_water else f"RET-{drone.id}"
            color = (100, 200, 255) if is_water else (255, 100, 100)
            
            render_text(name, self.font_sub, color)
            
            if drone.active:
                render_text("ACTIVE", self.font_body, (100, 255, 100), 10)
                render_progress(int(drone.battery), int(drone.max_battery), "Battery")
                render_progress(int(drone.payload), int(drone.max_payload), "Payload")
                
                target = getattr(drone, 'current_target', None)
                act_id = getattr(drone, 'current_action', 0)
                acts = ['STAY', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'WATER', 'RETARDANT']
                
                t_str = f"({target[0]},{target[1]})" if target else "None"
                d_str = f"{abs(drone.x - target[0]) + abs(drone.y - target[1])}" if target else "0"
                render_text(f"Target: {t_str} | Dist: {d_str}", self.font_body, (180, 180, 180), 10)
                render_text(f"Action: {acts[act_id]}", self.font_body, (180, 180, 180), 10)
            else:
                render_text("CRASHED", self.font_body, (255, 50, 50), 10)



        
    def close(self):
        pygame.quit()
