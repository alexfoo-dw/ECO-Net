"""Custom graph-based pixel segmentation."""
from .filter import *
from .segment_graph import *
import time

def segment(in_image, sigma, k, min_size, filename, four_connect=False):
    start_time = time.time()
    height, width, band = in_image.shape
    smoothed_image = np.zeros(shape=in_image.shape, dtype=float)
    if band == 3:
        for band_idx in range(band):
            smoothed_image[:, :, band_idx] = smooth(in_image[:, :, band_idx], sigma)
    else:
        smoothed_image = in_image
    edges_size = width * height * 4
    edges = np.zeros(shape=(edges_size, 3), dtype=object)
    num = 0
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int(y * width + (x + 1))
                edges[num, 2] = diff_general(smoothed_image, x, y, x + 1, y)
                num += 1
            if y < height - 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y + 1) * width + x)
                edges[num, 2] = diff_general(smoothed_image, x, y, x, y + 1)
                num += 1
            if not four_connect:
                if x < width - 1 and y < height - 1:
                    edges[num, 0] = int(y * width + x)
                    edges[num, 1] = int((y + 1) * width + (x + 1))
                    edges[num, 2] = diff_general(smoothed_image, x, y, x + 1, y + 1)
                    num += 1
                if x < width - 1 and y > 0:
                    edges[num, 0] = int(y * width + x)
                    edges[num, 1] = int((y - 1) * width + (x + 1))
                    edges[num, 2] = diff_general(smoothed_image, x, y, x + 1, y - 1)
                    num += 1
    u = segment_graph(width * height, num, edges, k)
    for i in range(num):
        a = u.find(edges[i, 0])
        b = u.find(edges[i, 1])
        if a != b and (u.size(a) < min_size or u.size(b) < min_size):
            u.join(a, b)
    num_cc = u.num_sets()
    output = np.zeros(shape=(height, width, 3))
    label = np.zeros(shape=(height, width))
    colors = np.zeros(shape=(height * width, 3))
    for i in range(height * width):
        colors[i, :] = random_rgb()
    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            label[y, x] = comp
    elapsed_time = time.time() - start_time
    return label
