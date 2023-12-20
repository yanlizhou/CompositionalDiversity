import numpy as np

from .util import rotate_matrix, pixel_distance, dictionary, recenter, decode_part
from .fit_angle import fit_angle


__all__ = ['extract_relation', 'attach_part', 'decode_relations']


def unfold(x: np.ndarray, dim: int, size: int) -> np.ndarray:
    """Equivalent to pytorch's x.unfold(dim, size, step=1)"""
    return np.lib.stride_tricks.sliding_window_view(x, size, dim)


def extract_relation(input: np.ndarray, targets: np.ndarray):
    """
    input: [V, 2]
    targets: [N, V, 2]
    """
    s_input = unfold(input, -2, 2)  # [Si, 2, sz]

    s_targets = unfold(targets, -2, 2)
    s_targets = np.expand_dims(s_targets, -3)  # [N, St, 1, 2, sz]

    dA = np.sum((s_targets - s_input)**2, axis=(-1,-2))
    dB = np.sum((s_targets - s_input[..., ::-1])**2, axis=(-1,-2))
    d = np.minimum(dA, dB)  # [n, St, Si]

    # target_ix, target_side, input_side
    return np.unravel_index(d.argmin(), d.shape)


def _rotate_and_shift(source, attachment):
    """rotate & shift points in 'source' to match 'attachment'"""

    # grid of possible rotation values [-pi, -pi/2, 0, pi/2]
    theta = np.linspace(0, 2*np.pi, 5, dtype=np.float32)[:-1]
    source = np.broadcast_to(source[None], (4,) + source.shape)
    attachment = np.broadcast_to(attachment[None], (4,) + attachment.shape)

    # solve for optimal location at each rotation
    source_rot = source @ rotate_matrix(theta)  # [rot, vert, 2]
    locs = (attachment - source_rot).mean(-2)  # [rot, 2]

    # compute loss per rotation
    output = source_rot + locs[:,None]  # [rot, vert, 2]
    losses = np.sum((output - attachment)**2, axis=(1,2))  # [rot]
    ix = losses.argmin()

    return theta[ix], locs[ix]


def attach_part(source, v_source, attachment, v_attachment, distance_fn=pixel_distance):
    source = source - source.mean(0)

    side_source = source[v_source:v_source+2]
    side_attachment = attachment[v_attachment:v_attachment+2]

    theta, loc = _rotate_and_shift(side_source, side_attachment)

    source = np.dot(source, rotate_matrix(theta)) + loc

    # check if a 180-degree rotation leads to less overlap
    center = side_attachment.mean(0)
    rotate_mat = rotate_matrix(np.pi)
    source_rot = np.dot(source - center, rotate_mat) + center
    if distance_fn(source_rot, attachment) > distance_fn(source, attachment):
        source = source_rot

    return source


def decode_relations(prim_ids, relations, angle=0, centered=True, reorient=True):
    npart = len(prim_ids)
    assert prim_ids.shape == (npart,)
    assert relations.shape == (npart, 3)
    angle = int(angle)
    assert angle in [0, 90, 180, 270]

    polygons = np.zeros((npart, 7, 2), np.float32)
    for v, (cv, (u, su, sv)) in enumerate(zip(prim_ids, relations)):
        xv = dictionary[cv]
        if v == 0:
            if angle != 0:
                xv = xv @ rotate_matrix(angle, degrees=True)
            xv = recenter(xv).round()
        else:
            xv = attach_part(xv, sv, polygons[u], su).round()

        if reorient:
            theta, loc = fit_angle(xv, cv)
            xv = decode_part(cv, theta, loc, degrees=False).round()

        polygons[v] = xv

    if centered:
        polygons = recenter(polygons).round(out=polygons)

    return polygons