import math
import functools
import numpy as np
from scipy.spatial.distance import jaccard
import torch

from ...utility.render import get_lines
from ...utility.render_util import shapes_in_exp
from ...utility2.render import make_binary_image

__all__ = [
    'dictionary', 'rotate_matrix', 'recenter', 'pixel_distance',
    'pixel_overlap', 'hausdorff_distance', 'hausdorff_distance_image',
    'decode_part', 'make_tensor_image', 'match_polygons'
]


"""Primitive dictionary"""

dictionary = [get_lines(shapes_in_exp[i], side_len=32) for i in (1,2,3,4,5,12,14,15,18)]
dictionary = torch.from_numpy(np.array(dictionary, dtype=np.float32))
dictionary -= dictionary.mean(1, keepdim=True)


"""Point transformations"""


def rotate_matrix(theta, degrees=False):
    theta = torch.as_tensor(theta, dtype=torch.float)
    if degrees:
        theta = torch.deg2rad(theta)
    vec = torch.stack([theta, theta + math.pi / 2], dim=-1)
    return torch.stack([torch.cos(vec), torch.sin(vec)], dim=-2)


def recenter(data, new_center=96, start_dim=0):
    dim = tuple(range(start_dim, data.dim() - 1))
    center = (data.amin(dim, keepdim=True) + data.amax(dim, keepdim=True)) / 2
    return data - center + new_center


"""Polygon distance utilities"""


def pixel_distance(vs, *vs_others):
    assert len(vs_others) >= 1

    # format inputs for PIL
    to_cpu = lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x
    vs = to_cpu(vs)
    vs_others = list(map(to_cpu, vs_others))

    # binary image for the target part (image1) and the other parts (image2)
    image1 = np.asarray(make_binary_image([vs]))
    image2 = np.asarray(make_binary_image(vs_others))

    # jaccard distance
    return jaccard(image1.ravel(), image2.ravel())


def pixel_overlap(vs, *vs_others):
    return 1 - pixel_distance(vs, *vs_others)


def hausdorff_distance(a, b, mode='mean'):
    """Hausdorff distance formula

    For each point contained in polygon a, compute the distance to
    the nearest point in polygon b. Then aggregate by taking the mean
    (or max) over all points.

    The metric is computed in both directions (a->b, b->a) and
    the maximum is returned.
    """
    D = torch.cdist(a, b)  # [*, Na, Nb]
    min_a = D.amin(-1)  # [*, Na]
    min_b = D.amin(-2)  # [*, Nb]
    if mode == 'mean':
        return torch.max(min_a.mean(-1), min_b.mean(-1))
    elif mode == 'max':
        return torch.max(min_a.amax(-1), min_b.amax(-1))
    else:
        raise ValueError(f'{mode} is not a valid value for mode')


@functools.lru_cache(maxsize=10)
def _pixel_distance_grid(imsize):
    height, width = imsize
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    grid = torch.stack([x, y], -1)
    grid = grid.view(-1, 2).float()
    return torch.cdist(grid, grid)


def hausdorff_distance_image(a, b, mode='mean'):
    """Alternative to the standard 'hausdorff_distance' function

    In this variant, the inputs are represented as binary images,
    where "true" indicates an active pixel in the polygon. This version
    can handle batch inputs.
    """
    assert a.shape == b.shape
    assert a.dtype == b.dtype == torch.bool

    D = _pixel_distance_grid(a.shape[-2:]).to(a.device)
    a, b = a.flatten(-2), b.flatten(-2)
    a_mask, b_mask = ~a, ~b

    if a.dim() > 1:
        for _ in range(a.dim() - 1):
            D = D.unsqueeze(0)
        D = D.expand(a.shape + (a.shape[-1],))

    D = D.masked_fill(a_mask.unsqueeze(-1), float('inf'))
    D = D.masked_fill(b_mask.unsqueeze(-2), float('inf'))
    min_a = D.amin(-1)
    min_b = D.amin(-2)

    if mode == 'mean':
        val_a = min_a.masked_fill_(a_mask, 0.).sum(-1) / a.float().sum(-1)
        val_b = min_b.masked_fill_(b_mask, 0.).sum(-1) / b.float().sum(-1)
    elif mode == 'max':
        val_a = min_a.masked_fill_(a_mask, float('-inf')).amax(-1)
        val_b = min_b.masked_fill_(b_mask, float('-inf')).amax(-1)
    else:
        raise ValueError(f'{mode} is not a valid value for mode')

    return torch.maximum(val_a, val_b)


"""Part transformations"""


def decode_part(pid, theta, loc, degrees=True):
    if torch.is_tensor(loc):
        loc = loc.unsqueeze(-2)
    return torch.matmul(dictionary[pid], rotate_matrix(theta, degrees)) + loc


"""Miscelaneous"""

from ...utility2.render import make_image


def make_tensor_image(polygons, prim_ids, imsize=(80,80)):
    image = make_image(polygons, prim_ids).resize(imsize)
    image = torch.tensor(np.asarray(image), dtype=torch.float) / 255
    return image


def match_polygons(target_image, polygons, prim_ids):
    images = target_image.new_zeros((4,) + target_image.shape)
    images[0] = make_tensor_image(polygons, prim_ids)

    polygons_0 = recenter(polygons, new_center=0)
    for i in [1, 2, 3]:
        polygons_f = recenter(polygons_0 @ rotate_matrix(i * math.pi / 2))
        images[i] = make_tensor_image(polygons_f, prim_ids)

    errors = torch.abs(images - target_image).flatten(1).sum(1)
    best_ix = errors.argmin()
    if errors[best_ix] > 300:
        best_ix = 0

    return images[best_ix]

