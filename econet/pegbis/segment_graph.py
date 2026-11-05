"""Custom graph-based pixel segmentation."""
from .disjoint_set import *
import math
import numpy as np
import random

def segment_graph(num_vertices, num_edges, edges, c):
    edges[0:num_edges, :] = edges[edges[0:num_edges, 2].argsort()]
    u = universe(num_vertices)
    threshold = np.zeros(shape=num_vertices, dtype=float)
    for i in range(num_vertices):
        threshold[i] = get_threshold(1, c)
    for i in range(num_edges):
        pedge = edges[i, :]
        a = u.find(pedge[0])
        b = u.find(pedge[1])
        if a != b:
            if pedge[2] <= threshold[a] and pedge[2] <= threshold[b]:
                u.join(a, b)
                a = u.find(a)
                threshold[a] = pedge[2] + get_threshold(u.size(a), c)
    return u

def get_threshold(size, c):
    return c / size

def square(value):
    return value * value

def random_rgb():
    rgb = np.zeros(3, dtype=int)
    rgb[0] = random.randint(0, 255)
    rgb[1] = random.randint(0, 255)
    rgb[2] = random.randint(0, 255)
    return rgb

def diff(red_band, green_band, blue_band, x1, y1, x2, y2):
    result = math.sqrt(square(red_band[y1, x1] - red_band[y2, x2]) + square(green_band[y1, x1] - green_band[y2, x2]) + square(blue_band[y1, x1] - blue_band[y2, x2]))
    return result

def diff_general(image, x1, y1, x2, y2):
    result = math.sqrt(np.sum((image[y1, x1, :] - image[y2, x2, :]) ** 2, axis=-1))
    if image.shape[-1] > 3:
        result = result / image.shape[-1]
    return result
