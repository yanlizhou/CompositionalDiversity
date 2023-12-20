from functools import partial
from PIL import Image, ImageDraw
import numpy as np
import torch

from ...utility2.render import prim_color, render_from_string
from ..concepts import Concept as _Concept
from ..concepts import MetaConcept as _MetaConcept
from .fit_angle import fit_angle


def render_canvases(polygons, prim_ids):
    image = Image.new('RGB', (193, 193), color=(0,0,0))
    drawer = ImageDraw.Draw(image)
    canvases = np.zeros((len(polygons) + 1, 193, 193, 3), dtype=np.float32)
    for i, poly in enumerate(polygons):
        drawer.polygon([tuple(tup) for tup in poly],
                       outline='#ffffff',
                       fill=prim_color(prim_ids[i]))
        canvases[i+1] = np.array(image, dtype=np.float32)

    return canvases / 255


class Exemplar(object):
    def __init__(self, canvases, vertices, prim_ids, thetas, locations):
        self.canvases = torch.as_tensor(canvases, dtype=torch.float)
        self.vertices = torch.as_tensor(vertices, dtype=torch.float)
        self.prim_ids = torch.as_tensor(prim_ids, dtype=torch.long)
        self.thetas = torch.as_tensor(thetas, dtype=torch.float)
        self.locations = torch.as_tensor(locations, dtype=torch.float)

    def __len__(self):
        return len(self.prim_ids)


def token_getter(token_str, primitives):
    vertices, prim_ids = render_from_string(token_str, primitives)

    # render all intermediate canvases
    canvases = render_canvases(vertices, prim_ids)

    # compute the angle + location of each part
    thetas = torch.zeros(len(vertices), dtype=torch.float)
    locs = torch.zeros(len(vertices), 2, dtype=torch.float)
    for i, (vs, pid) in enumerate(zip(vertices, prim_ids)):
        thetas[i], locs[i] = fit_angle(vs, pid)
    thetas = torch.rad2deg(thetas)

    return Exemplar(canvases, vertices, prim_ids, thetas, locs)


Concept = partial(_Concept, getter=token_getter)
MetaConcept = partial(_MetaConcept, getter=token_getter)
