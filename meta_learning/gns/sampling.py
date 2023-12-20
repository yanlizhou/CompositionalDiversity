import warnings
import torch

from ..util import dictionary, recenter, attach_part, pixel_overlap


def sample_attachment(part_source, *parts_attach, max_iter=50):
    """sample an attachment specification for the current part
    given the previous parts
    """
    ix = torch.randint(len(parts_attach), size=())
    part_attach = parts_attach[ix]

    for _ in range(max_iter):
        # for both the current part and attachment part, randomly
        # select a side for attachment
        side_attach, side_source = torch.randint(6, size=(2,))
        part_source_ = attach_part(part_source, side_source, part_attach, side_attach)
        if pixel_overlap(part_source_, *parts_attach) < 0.0123:
            break
    else:
        warnings.warn('Failed to sample a valid attachment')

    return part_source_, side_source, side_attach


def sample_exemplar(num_parts=9):
    pids = []
    vertices = []

    def recenter_list(vs):
        return list(recenter(torch.stack(vs)))

    def finalize_sample():
        return torch.stack(vertices), torch.stack(pids)

    # sample first part
    s1 = torch.randint(num_parts, size=())
    part1 = dictionary[s1]
    pids.append(s1)
    vertices.append(part1)
    vertices = recenter_list(vertices)

    if torch.rand(size=()) < (1/3):
        return finalize_sample()

    # sample second part and attachment specs
    s2 = torch.randint(num_parts, size=())
    part2, _, _ = sample_attachment(dictionary[s2], *vertices)
    pids.append(s2)
    vertices.append(part2)
    vertices = recenter_list(vertices)

    if torch.rand(size=()) < (1/2):
        return finalize_sample()

    # sample third part and attachment specs
    s3 = torch.randint(num_parts, size=())
    part3, _, _ = sample_attachment(dictionary[s3], *vertices)
    pids.append(s3)
    vertices.append(part3)
    vertices = recenter_list(vertices)

    return finalize_sample()

