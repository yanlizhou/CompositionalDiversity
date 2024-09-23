from collections import namedtuple
import random
import numpy as np
import torch

from ...gns_utility.render import make_image, render_from_string
from ..concepts import MetaConcept
from ..util import decode_part, recenter
from .fit_angle import fit_angle
from .relations import _extract


Exemplar = namedtuple('Exemplar', ['canvases', 'prim_ids', 'relations', 'angle'])


def token_getter(token_str, primitives):
    vertices, prim_ids = render_from_string(token_str, primitives)
    vertices = torch.as_tensor(vertices, dtype=torch.float)
    prim_ids = torch.as_tensor(prim_ids, dtype=torch.long)
    npart = len(vertices)

    # render partial canvases
    canvases = np.zeros((npart+1, 193, 193, 3), dtype=np.uint8)
    for i in range(1, npart + 1):
        canvases[i] = np.asarray(make_image(recenter(vertices[:i]), prim_ids[:i]))
    canvases = torch.tensor(canvases, dtype=torch.float) / 255

    # compute the angle + location of each part
    thetas, locs = fit_angle(vertices, prim_ids)
    thetas = torch.rad2deg(thetas)
    angle = thetas[0]

    # extract symbolic relations for each part
    vertices = decode_part(prim_ids, thetas, locs)
    relations = [None] + [_extract(vertices[i], vertices[:i]) for i in range(1, npart)]

    return Exemplar(canvases, prim_ids, relations, angle)


class Dataset:
    def __init__(self, num_concepts, k_support, k_target=1, seed=None):
        if seed is not None:
            # random seed for reproducibility
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.num_concepts = num_concepts
        self.k_support = k_support
        self.k_target = k_target
        k_total = k_support + k_target

        meta_concept = MetaConcept(getter=token_getter)
        self.concepts = []
        self.data = []
        for _ in range(num_concepts):
            concept = []
            while len(concept) < k_total:
                concept = meta_concept.sample()
            self.concepts.append(concept)
            idx = torch.randperm(len(concept))[:k_total]
            self.data.append([concept[i] for i in idx])

    def __len__(self):
        return self.num_concepts

    def __getitem__(self, ix):
        concept = self.data[ix]

        def get_full_image(exemplar):
            """Get the full image (the final canvas) from an exemplar"""
            canvases = exemplar[0]
            return canvases[-1]

        idx = torch.randperm(self.k_support + self.k_target)
        support = torch.stack([get_full_image(concept[i]) for i in idx[:self.k_support]])
        targets = [concept[i] for i in idx[self.k_support:]]

        return support, targets



# -------------------------------------------------------------
#      Collate function utilities (i.e. code to process
#                  a batch of inputs)
# -------------------------------------------------------------
from collections import OrderedDict
import torch.nn.functional as F
from ptkit.utils import pad_sequence


def collate_targets(exemplars):
    # un-zip the exemplars into unique lists per variable
    canvas, pid, relation, theta = list(zip(*exemplars))
    relation = [torch.tensor([(0,0,0) if ri is None else ri for ri in r], dtype=torch.long)
                for r in relation]
    lengths = torch.tensor([elt.size(0) for elt in pid], dtype=torch.long)

    # pad all sequences to fixed length
    canvas, pid, relation = [pad_sequence(v, max_len=4) for v in (canvas, pid, relation)]

    # convert theta into a categorical variable
    theta = torch.stack(theta)
    theta = (theta + 180) / 90
    assert torch.all((theta % 1) == 0)

    # process images: permute channels and then downsize to (80,80)
    if canvas.shape[-1] == 3:
        canvas = canvas.permute(0,1,4,2,3).contiguous()
    if canvas.shape[-2:] != (80, 80):
        canvas = F.interpolate(canvas.flatten(0,1), size=(80, 80)).view(*canvas.shape[:2], 3, 80, 80)

    action = OrderedDict()
    action['pid'] = pid
    action['theta'] = theta.long()
    action['prev_ix'] = relation[..., 0]
    action['prev_side'] = relation[..., 1]
    action['curr_side'] = relation[..., 2]

    return canvas, action, lengths


def collate_fn(data, multi_target=True):
    support, targets = list(zip(*data))

    ### collate support inputs
    support = torch.stack(support, dim=1)  # [kshot, batch, height, width, 3]
    # process images: permute channels and then downsize to (80,80)
    support = support.permute(0,1,4,2,3).contiguous()
    support = F.interpolate(support.flatten(0, 1), size=(80, 80)).view(*support.shape[:2], 3, 80, 80)

    ### collate targets
    if not multi_target:
        targets = [tgt[0] for tgt in targets]
        canvas, action, lengths = collate_targets(targets)
    else:
        canvas, action, lengths = list(zip(*map(collate_targets, targets)))
        canvas = torch.stack(canvas, dim=2)
        lengths = torch.stack(lengths, dim=1)
        keys = action[0].keys()
        values = list(zip(*[a.values() for a in action]))
        action = OrderedDict([(k, torch.stack(v, dim=-1)) for k, v in zip(keys, values)])

    return support, canvas, action, lengths