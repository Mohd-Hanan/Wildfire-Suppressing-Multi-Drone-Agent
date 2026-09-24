import numpy as np

def calculate_wind_vector(speed, direction_degrees):
    # wind direction is where the wind is BLOWING TO (usually meteorological is blowing FROM, but let's assume standard vector)
    rad = np.radians(direction_degrees)
    wx = np.cos(rad) * speed
    wy = np.sin(rad) * speed
    return wx, wy

print(calculate_wind_vector(0.5, 90))
