import copy
import math
import torch

from . import rotate_matrix, dictionary, hausdorff_distance

__all__ = [
    'match_vertices', 'fit_angle'
]


def tensor_round(x, n=0):
    return torch.round(x * 10**n) / 10**n


def match_vertices(source, target):
    # remove repeated vertex
    source, target = [elt[..., 1:, :] for elt in (source, target)]

    # solve for best roll value
    target_rolls = [torch.roll(target, i, dims=-2) for i in range(target.shape[-2])]
    target_rolls = torch.stack(target_rolls, dim=-3)  # [*, rolls, verts, 2]
    diff = target_rolls - source.unsqueeze(-3)  # [*, rolls, verts, 2]
    diff = torch.linalg.norm(diff.flatten(-2), dim=-1)  # [*, rolls]
    roll_idx = diff.argmin(-1)  # [*]

    # select rolls by index using torch.gather
    bshape = roll_idx.shape
    roll_idx = roll_idx.view(bshape + (1, 1)).expand(bshape + target_rolls.shape[-2:])  # [*, verts, 2]
    target = target_rolls.gather(-3, roll_idx.unsqueeze(-3)).squeeze(-3)

    # add repeated vertex
    target = torch.cat([target[..., -1:, :], target], dim=-2)

    return target


def fit_angle(target, pid):
    """
    `target` has shape [*, vertices, 2] and `pid` shape [*],
    where * is an arbitrary batch shape
    """
    source = dictionary[pid]
    target = torch.as_tensor(target, dtype=torch.float)

    # standardize batch inputs
    assert source.shape == target.shape
    batch_shape = source.shape[:-2]
    source = source.view(-1, *source.shape[-2:])  # [N,V,2]
    target = target.view(-1, *target.shape[-2:])  # [N,V,2]

    # grid of possible rotation values (in radians)
    theta = torch.linspace(0, 2*math.pi, 5)[:-1]
    source = source.unsqueeze(1).expand(-1, 4, -1, -1)  # [N, R, V, 2]
    target = target.unsqueeze(1).expand(-1, 4, -1, -1)  # [N, R, V, 2]

    # solve for optimal location at each rotation
    W = rotate_matrix(theta)  # [R, 2, 2]
    source_rot = torch.matmul(source, W)  # [N, R, V, 2]
    target_matched = match_vertices(source_rot, target)  # [N, R, V, 2]
    locs = (target_matched - source_rot).mean(-2)  # [N, R, 2]

    # compute loss per rotation
    output = source_rot + locs.unsqueeze(-2)  # [N, R, V, 2]
    losses = hausdorff_distance(output, target)  # [N, R]
    losses = tensor_round(losses, 2)
    ix = losses.argmin(-1)  # [N]

    # select optimal theta/locs
    theta_out = theta[ix]   # [N]
    locs_out = locs.gather(-2, ix.view(-1,1,1).expand(-1,1,2)).squeeze(-2)  # [N, 2]

    return theta_out.view(batch_shape), locs_out.view(batch_shape + (2,))


"""
Functions from here down are only for reference and are not part
of the `util` module
"""

def _transform_exemplar(ex):
    thetas = torch.zeros(len(ex))
    locs = torch.zeros(len(ex), 2)
    for i, (target, pid) in enumerate(zip(ex.vertices, ex.prim_ids)):
        thetas[i], locs[i] = fit_angle(target, pid)

    ex = copy.deepcopy(ex)
    setattr(ex, 'thetas', torch.rad2deg(thetas))
    setattr(ex, 'locs', locs)

    return ex

