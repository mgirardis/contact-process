import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def cmap_blue_deep():
    # Define the key colors (normalized RGB)
    key_colors = [
        [0.00, 0.75, 1.00],  # Capri (#00BFFF)
        [0.03, 0.56, 0.91],  # Blue Cola (#0790E8)
        [0.05, 0.38, 0.82],  # Denim (#0E61D1)
        [0.08, 0.20, 0.73],  # Persian Blue (#1432BA)
        [0.11, 0.01, 0.64],  # Neon Blue (#1B03A3)
    ]
    # Create a colormap object
    cmap = LinearSegmentedColormap.from_list("blue_deep", key_colors)
    return cmap

def cmap_red_standard():
    # Define the key colors (normalized RGB)
    key_colors = numpy.array([
        [255, 204, 204],  # Light Pink
        [255, 153, 153],  # Pink
        [255, 102, 102],  # Salmon
        [255, 51, 51],    # Bright Red
        [255, 0, 0],      # Red
    ],dtype=float)/255.0
    # Create the colormap
    cmap = LinearSegmentedColormap.from_list("red_standard", key_colors)
    return cmap