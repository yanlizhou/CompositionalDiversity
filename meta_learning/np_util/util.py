import math
import numpy as np
from scipy.spatial.distance import jaccard

from ...bayes_utility.render import get_lines
from ...bayes_utility.render_util import shapes_in_exp
from ...gns_utility.render import make_binary_image

__all__ = [
    'dictionary', 'rotate_matrix', 'recenter',
    'hausdorff_distance', 'decode_part', 'pixel_distance'
]


"""Primitive dictionary"""


dictionary = shapes_in_exp[[1, 2, 3, 4, 5, 12, 14, 15, 18]]
dictionary = np.array([get_lines(p, side_len=32) for p in dictionary], dtype=np.float32)
dictionary -= dictionary.mean(1, keepdims=True)


"""Point transformations"""


def rotate_matrix(theta, degrees=False):
    theta = np.asarray(theta, dtype=np.float32)
    if degrees:
        theta = np.deg2rad(theta)
    vec = np.stack([theta, theta + math.pi / 2], -1)
    return np.stack([np.cos(vec), np.sin(vec)], -2)


def recenter(data, new_center=96, start_dim=0):
    dim = tuple(range(start_dim, data.ndim - 1))
    center = (data.min(dim, keepdims=True) + data.max(dim, keepdims=True)) / 2
    return data - center + new_center


"""Polygon distance utilities"""


def pixel_distance(vs, *vs_others):
    assert len(vs_others) >= 1

    # binary image for the target part (image1) and the other parts (image2)
    image1 = np.array(make_binary_image([vs])).view(np.uint8) != 0
    image2 = np.array(make_binary_image(vs_others)).view(np.uint8) != 0

    # jaccard distance
    return jaccard(image1.ravel(), image2.ravel())


def cdist(a, b):
    D = np.expand_dims(a, -2) - np.expand_dims(b, -3)
    return np.linalg.norm(D, axis=-1)


def hausdorff_distance(a, b, mode='mean'):
    D = cdist(a, b)  # [*, Na, Nb]
    min_a = np.amin(D, axis=-1)  # [*, Na]
    min_b = np.amin(D, axis=-2)  # [*, Nb]
    aggr = {'mean': np.mean, 'max': np.amax}.get(mode)
    val_a = aggr(min_a, axis=-1)
    val_b = aggr(min_b, axis=-1)

    return np.maximum(val_a, val_b)


"""Part transformations"""


def decode_part(pid, theta, loc, degrees=True):
    if isinstance(loc, np.ndarray):
        loc = np.expand_dims(loc, -2)
    return np.matmul(dictionary[pid], rotate_matrix(theta, degrees)) + loc
