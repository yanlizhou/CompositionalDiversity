import math
import torch

from . import rotate_matrix, hausdorff_distance

__all__ = ['attach_part_batched', 'extract_relation_batched']


"""
attach_part (batched)
"""


def _rotate_and_shift(x_next, x_prev):
    """rotate & shift points in 'x_next' to match 'x_prev'"""
    batch_shape = x_next.shape[:-2]
    x_next = x_next.view(-1, *x_next.shape[-2:])  # [N,V,2]
    x_prev = x_prev.view(-1, *x_prev.shape[-2:])  # [N,V,2]

    # grid of rotation values
    theta = torch.linspace(0, 2*math.pi, 5, device=x_next.device)[:-1]  # [R]
    x_next = x_next.unsqueeze(1).expand(-1,4,-1,-1)  # [N,R,V,2]
    x_prev = x_prev.unsqueeze(1).expand(-1,4,-1,-1)  # [N,R,V,2]

    # solve optimal location at each rotation
    W = rotate_matrix(theta)  # [R,2,2]
    x_next_rot = torch.matmul(x_next, W)  # [N,R,V,2]
    locs = (x_prev - x_next_rot).mean(-2)  # [N,R,2]

    # compute loss per rotation
    output = x_next_rot + locs.unsqueeze(-2)  # [N,R,V,2]
    losses = (output - x_prev).square().sum([-2,-1])  # [N,R]
    ix = losses.argmin(-1)  # [N]

    theta = theta[ix]
    locs = locs.gather(-2, ix.view(-1,1,1).expand(-1,-1,2)).squeeze(-2)

    return theta.view(batch_shape), locs.view(batch_shape + (2,))


def _gather_side(x, v):
    """x: [*,V,2],  v: [*]"""
    v = v.view(v.shape + (1,1)).expand(v.shape + (1,2))
    side = torch.cat([x.gather(-2, v), x.gather(-2, v+1)], dim=-2)
    return side


def attach_part_batched(x_next, v_next, x_prev, v_prev, distance_fn=hausdorff_distance):
    """Batch variant of the `attach_part` function

    x_next, x_prev: [*,V,2]
    v_next, v_prev: [*]

    TODO: set distance_fn=pixel_distance as default
    """
    v_next = torch.as_tensor(v_next, dtype=torch.long, device=x_next.device)
    v_prev = torch.as_tensor(v_prev, dtype=torch.long, device=x_next.device)
    x_next = x_next - x_next.mean(-2, keepdim=True)

    side_next = _gather_side(x_next, v_next)  # [*,Z,2]
    side_prev = _gather_side(x_prev, v_prev)  # [*,Z,2]
    theta, loc = _rotate_and_shift(side_next, side_prev)  # [*], [*,2]

    # [*,V,2] @ [*,2,2] + [*,1,2]
    x_next = torch.matmul(x_next, rotate_matrix(theta)) + loc.unsqueeze(-2)

    # check if a 180-degree rotation leads to less overlap
    center = side_prev.mean(-2, keepdim=True)
    rotate_mat = rotate_matrix(math.pi).to(x_next.device)
    x_next_rot = torch.matmul(x_next - center, rotate_mat) + center
    improved = distance_fn(x_next_rot, x_prev) > distance_fn(x_next, x_prev)  # [*]
    x_next[improved] = x_next_rot[improved]

    return x_next



"""
extract_relation (batched)
"""


def extract_relation_batched(x_next, x_prev):
    """Batch variant of the _extract function

    x_next: [*,V,2]
    x_prev: [*,N,V,2]
    """
    assert x_next.dim() >= 2
    assert x_prev.dim() >= 3
    assert x_prev.shape[:-3] == x_next.shape[:-2]

    x_prev = x_prev.unfold(-2, 2, step=1).unsqueeze(-3)                # [*,N,Sa,1,2,Z]
    x_next = x_next.unfold(-2, 2, step=1).unsqueeze(-4).unsqueeze(-4)  # [*,1,1,Sb,2,Z]

    # [*,N,Sa,1,2,Z] - [*,1,1,Sb,2,Z]
    diff_0 = (x_prev - x_next).pow(2).sum([-1,-2])
    diff_1 = (x_prev - x_next.flip([-1])).pow(2).sum([-1,-2])
    diff = torch.min(diff_0, diff_1)  # [*,N,Sa,Sb]

    Sa, Sb = diff.shape[-2:]
    index = diff.flatten(-3).argmin(-1)  # [*]
    prev_ix = index.div(Sa * Sb, rounding_mode='floor')
    index = index % (Sa * Sb)
    prev_side = index.div(Sb, rounding_mode='floor')
    next_side = index % Sb

    return prev_ix, prev_side, next_side