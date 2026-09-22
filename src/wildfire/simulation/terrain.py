import numpy as np
from scipy.ndimage import zoom

def generate_fractal_noise(width: int, height: int, rng: np.random.Generator, octaves=4, persistence=0.5, base_freq=4):
    """Generates beautiful multi-scale fractal noise (similar to Perlin noise)."""
    noise = np.zeros((width, height))
    amplitude = 1.0
    frequency = base_freq
    total_amplitude = 0.0
    
    for _ in range(octaves):
        # Generate low-res random grid
        base = rng.uniform(0, 1, (frequency, frequency))
        
        # Smoothly interpolate it up to the full map size using cubic interpolation
        zoom_x = width / frequency
        zoom_y = height / frequency
        scaled = zoom(base, (zoom_x, zoom_y), order=3)
        
        # Add it to the main noise
        noise += scaled[:width, :height] * amplitude
        total_amplitude += amplitude
        
        # Prepare for next octave (higher frequency, lower amplitude)
        amplitude *= persistence
        frequency *= 2
        
    # Normalize to [0, 1]
    return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

class Terrain:
    def __init__(self, width: int, height: int, seed: int = None):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        
        # 1. Generate beautiful, smooth elevation using fractal noise
        self.elevation = generate_fractal_noise(width, height, self.rng, base_freq=3)
        
        # 2. Calculate slope and aspect for 3D hillshading
        dx, dy = np.gradient(self.elevation)
        self.slope = np.sqrt(dx**2 + dy**2)
        s_max = self.slope.max()
        if s_max > 0:
            self.slope = self.slope / s_max
            
        # 3. Generate natural fuel (vegetation) clusters
        # Fuel often gathers in lower, flatter areas
        base_fuel = generate_fractal_noise(width, height, self.rng, base_freq=5)
        self.fuel = np.clip(base_fuel - (self.elevation * 0.3), 0, 1)
        
        # Normalize fuel and moisture, but compress them to [0.1, 0.8] 
        # so they don't accidentally trigger the drone drop visuals (which are 0.0 and 1.0)
        fuel_norm = (self.fuel - self.fuel.min()) / (self.fuel.max() - self.fuel.min() + 1e-8)
        self.fuel = fuel_norm * 0.7 + 0.1
        
        # 4. Moisture (valleys are wetter, peaks are drier)
        self.moisture = np.clip(1.0 - self.elevation + generate_fractal_noise(width, height, self.rng, base_freq=6)*0.2, 0, 1)
        moist_norm = (self.moisture - self.moisture.min()) / (self.moisture.max() - self.moisture.min() + 1e-8)
        self.moisture = moist_norm * 0.7 + 0.1
