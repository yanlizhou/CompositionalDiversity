import numpy as np

from .util import dictionary, rotate_matrix, hausdorff_distance


__all__ = [
    'match_vertices', 'fit_angle'
]



def gather(input, index, dim):
    index = np.expand_dims(index, dim)
    output = np.take_along_axis(input, index, dim)
    return np.squeeze(output, dim)


def match_vertices(source, target):
    # remove repeated vertex
    source = source[..., 1:, :]
    target = target[..., 1:, :]

    # solve for best roll value
    target_rolls = [np.roll(target, i, -2) for i in range(target.shape[-2])]
    target_rolls = np.stack(target_rolls, -3)          # [*, R, V, 2]
    diff = target_rolls - np.expand_dims(source, -3)   # [*, R, V, 2]
    diff = diff.reshape(diff.shape[:-2] + (-1,)) # [*, R, V*2]
    diff = np.linalg.norm(diff, axis=-1)  # [*, R]
    roll_idx = diff.argmin(-1)  # [*]

    # select rolls via torch.gather
    bshape = roll_idx.shape
    roll_idx = roll_idx.reshape(bshape + (1, 1))
    roll_idx = np.broadcast_to(roll_idx, bshape + target_rolls.shape[-2:])  # [*, verts, 2]
    target = gather(target_rolls, roll_idx, -3)

    # add repeated vertex
    target = np.concatenate([target[..., -1:, :], target], -2)

    return target


def fit_angle(target, pid):
    source = dictionary[pid]
    target = np.asarray(target, dtype=np.float32)

    # standardize batch inputs
    assert source.shape == target.shape
    batch_shape = source.shape[:-2]
    source = source.reshape((-1,) + source.shape[-2:])  # [N,V,2]
    target = target.reshape((-1,) + target.shape[-2:])  # [N,V,2]

    # grid of possible rotation values (in radians)
    theta = np.linspace(0, 2*np.pi, 5, dtype=np.float32)[:-1]
    new_shape = (source.shape[0], 4) + source.shape[1:]
    source = np.broadcast_to(source[:,None], new_shape)  # [N, R, V, 2]
    target = np.broadcast_to(target[:,None], new_shape)  # [N, R, V, 2]

    # solve for optimal location at each rotation
    W = rotate_matrix(theta)  # [R, 2, 2]
    source_rot = source @ W  # [N, R, V, 2]
    target_matched = match_vertices(source_rot, target)  # [N, R, V, 2]
    locs = (target_matched - source_rot).mean(-2)  # [N, R, 2]

    # compute loss per rotation
    output = source_rot + np.expand_dims(locs, -2)  # [N, R, V, 2]
    losses = hausdorff_distance(output, target)  # [N, R]
    losses = np.round(losses, 2)
    ix = losses.argmin(-1)  # [N]

    # select optimal theta/locs
    theta_out = theta[ix]   # [N]

    ix = np.broadcast_to(np.expand_dims(ix, 1), (ix.shape[0], 2))
    locs_out = gather(locs, ix, -2)

    return theta_out.reshape(batch_shape), locs_out.reshape(batch_shape + (2,))