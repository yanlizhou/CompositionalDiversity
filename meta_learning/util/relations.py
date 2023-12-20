import warnings
import copy
import math
import torch

from . import dictionary, decode_part, rotate_matrix, pixel_distance, recenter
from .fit_angle import fit_angle

__all__ = [
    'attach_part', '_extract', 'extract_relation', 'decode_relations'
]


def _rotate_and_shift(source, attachment):
    """rotate & shift points in 'source' to match 'attachment'"""

    # grid of possible rotation values [-pi, -pi/2, 0, pi/2]
    theta = torch.linspace(0, 2*math.pi, 5, device=source.device)[:-1]
    source = source.expand(4, -1, -1)
    attachment = attachment.expand(4, -1, -1)

    # solve for optimal location at each rotation
    W = rotate_matrix(theta)  # [rot, 2, 2]
    source_rot = torch.bmm(source, W)  # [rot, vert, 2]
    locs = (attachment - source_rot).mean(-2)  # [rot, 2]

    # compute loss per rotation
    output = source_rot + locs.unsqueeze(-2)  # [rot, vert, 2]
    losses = (output - attachment).square().sum([-2,-1])  # [rot]
    ix = losses.argmin()

    return theta[ix], locs[ix]


def attach_part(source, v_source, attachment, v_attachment, distance_fn=pixel_distance):
    source = source - source.mean(0)

    side_source = source[v_source:v_source+2]
    side_attachment = attachment[v_attachment:v_attachment+2]

    theta, loc = _rotate_and_shift(side_source, side_attachment)

    source = torch.mm(source, rotate_matrix(theta)) + loc

    # check if a 180-degree rotation leads to less overlap
    center = side_attachment.mean(0)
    rotate_mat = rotate_matrix(math.pi).to(source.device)
    source_rot = torch.mm(source - center, rotate_mat) + center
    if distance_fn(source_rot, attachment) > distance_fn(source, attachment):
        source = source_rot

    return source


def extract_relation(source, previous):
    """
    source: [verts, 2]
    previous: [n, verts, 2]
    """
    source_sides = source.unfold(-2, size=2, step=1).transpose(-1, -2)  # [sides_s, size, 2]
    previous_sides = previous.unfold(-2, size=2, step=1).transpose(-1, -2)   # [n, sides_p, size, 2]

    diff_0 = (previous_sides.unsqueeze(-3) - source_sides).pow(2).sum([-1,-2])
    diff_1 = (previous_sides.unsqueeze(-3) - source_sides.flip([-2])).pow(2).sum([-1,-2])
    diff = torch.min(diff_0, diff_1)  # [n, sides_p, sides_s]

    prev_ix, prev_side, source_side = [u[0].item() for u in torch.where(diff == diff.min())]

    return prev_ix, prev_side, source_side


def _extract(source, previous):
    warnings.warn('_extract is deprecated. Use extract_relation instead',
                  DeprecationWarning, stacklevel=2)
    return extract_relation(source, previous)


def decode_relations(prim_ids, relations, angle=0, centered=True, reorient=True):
    npart = len(prim_ids)
    assert prim_ids.shape == (npart,)
    assert relations.shape == (npart, 3)
    angle = int(angle)
    assert angle in [0, 90, 180, 270]

    polygons = torch.zeros(npart, 7, 2)
    for v, (cv, (u, su, sv)) in enumerate(zip(prim_ids, relations)):
        xv = dictionary[cv]
        if v == 0:
            if angle != 0:
                xv = xv @ rotate_matrix(angle, degrees=True)
            xv = recenter(xv).round_()
        else:
            xv = attach_part(xv, sv, polygons[u], su).round_()

        if reorient:
            theta, loc = fit_angle(xv, cv)
            xv = decode_part(cv, theta, loc, degrees=False).round_()

        polygons[v] = xv

    if centered:
        polygons = recenter(polygons).round_()

    return polygons


"""
Functions from here down are only for reference and are not part
of the `util` module
"""

def _extract_relations(ex):
    vertices = torch.zeros_like(ex.vertices)
    for i in range(len(ex)):
        vertices[i] = decode_part(ex.prim_ids[i], ex.thetas[i], ex.locations[i])

    relations = [None]
    for i in range(1, len(ex)):
        r = extract_relation(vertices[i], vertices[:i])
        relations.append(r)

    ex = copy.deepcopy(ex)
    setattr(ex, 'relations', relations)

    return ex


def _construct_exemplar(pids, relations, first_part=None):
    # first part
    if first_part is None:
        first_part = dictionary[pids[0]] + 96
    parts = [first_part]

    # remaining parts
    for i in range(1, len(pids)):
        next_part = dictionary[pids[i]]
        prev_part_ix, prev_side, next_side = relations[i]
        prev_part = parts[prev_part_ix]
        parts += [attach_part(next_part, next_side, prev_part, prev_side)]

    parts = torch.stack(parts)
    parts -= (parts.amax(dim=(0,1)) + parts.amin(dim=(0,1))) / 2
    parts += 96

    return parts