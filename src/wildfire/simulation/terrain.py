import numpy as np

def generate_smooth_noise(width: int, height: int, scale: int, rng: np.random.Generator) -> np.ndarray:
    """Generates a 2D array of smooth noise using a simple box blur approach."""
    # Generate lower resolution noise
    low_res_w = max(1, width // scale)
    low_res_h = max(1, height // scale)
    base_noise = rng.uniform(0, 1, (low_res_w, low_res_h))
    
    # Upscale using Kronecker product for simplicity, or just numpy repeat and blur
    # Since we want smooth, we will repeat then convolve
    repeated = np.repeat(np.repeat(base_noise, scale, axis=0), scale, axis=1)
    
    # Crop to exact dimensions
    repeated = repeated[:width, :height]
    
    # Simple multi-pass box blur for smoothing
    smoothed = repeated.copy()
    passes = scale
    for _ in range(passes):
        # Shift and average
        up = np.roll(smoothed, 1, axis=0)
        down = np.roll(smoothed, -1, axis=0)
        left = np.roll(smoothed, 1, axis=1)
        right = np.roll(smoothed, -1, axis=1)
        smoothed = (smoothed + up + down + left + right) / 5.0
        
    # Normalize back to [0, 1]
    min_val = smoothed.min()
    max_val = smoothed.max()
    if max_val > min_val:
        smoothed = (smoothed - min_val) / (max_val - min_val)
    
    return smoothed

class Terrain:
    def __init__(self, width: int, height: int, seed: int = None):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        
        # Generate underlying fields [0, 1]
        self.elevation = generate_smooth_noise(width, height, scale=8, rng=self.rng)
        
        # Calculate slope magnitude based on elevation gradient
        dx, dy = np.gradient(self.elevation)
        self.slope = np.sqrt(dx**2 + dy**2)
        # Normalize slope
        s_max = self.slope.max()
        if s_max > 0:
            self.slope = self.slope / s_max
            
        self.fuel = generate_smooth_noise(width, height, scale=4, rng=self.rng)
        self.moisture = generate_smooth_noise(width, height, scale=6, rng=self.rng)

