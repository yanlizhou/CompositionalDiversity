import sys
if 'matplotlib' not in sys.modules:
    import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image, ImageDraw

from ..utility.render import get_lines, all_possible_shapes
from ..utility.render_util import generate_all_attachments, shapes_in_exp

PRIMITIVES = shapes_in_exp[[1, 2, 3, 4, 5, 12, 14, 15, 18]]


def rotate_matrix(degrees):
    theta = math.radians(degrees)
    return np.array([
        [math.cos(theta), math.cos(theta + math.pi / 2)],
        [math.sin(theta), math.sin(theta + math.pi / 2)]
    ], dtype=np.float32)


def prim_color(pid):
    c = plt.cm.Set1((int(pid) + 1) / 10)
    c = tuple(int(x*255) for x in c[:3])
    return c


def shapes_to_vertices(shapes, angle=None, side_len=32):
    """Convert primitive representation (shapes) to polygon vertices"""

    # merge sub-part polygons into one polygon for the whole part
    polys = [get_lines(shape, side_len) for shape in shapes]
    polys = np.array(polys, dtype=np.float32)  # [parts, vertices, 2]

    # re-center the coordinates
    polys -= (polys.max((0,1)) + polys.min((0,1))) / 2
    if angle is not None:
        polys = np.matmul(polys, rotate_matrix(angle))
    polys += 3 * side_len

    return polys


def make_image(polys, prim_ids, size=(193,193), lw=1):
    """Render the image

    For consistency with `shapes_to_vertices` we
    should have size = side_len * 6 + 1
    """
    img = Image.new('RGB', size, color=(0,0,0))
    drawer = ImageDraw.Draw(img)
    for poly, pid in zip(polys, prim_ids):
        poly = [tuple(tup) for tup in poly]
        if lw == 1:
            drawer.polygon(poly, fill=prim_color(pid), outline='white')
        else:
            drawer.polygon(poly, fill=prim_color(pid))
            drawer.line(poly, fill='white', width=lw)

    return img


def make_binary_image(polys, size=(193,193)):
    """Alternate renderer that creates a binary image"""
    img = Image.new('1', size)
    drawer = ImageDraw.Draw(img)
    for poly in polys:
        drawer.polygon([tuple(tup) for tup in poly], fill=True)

    return img


def make_image_yanli(polygons, prim_ids, primitives, size=(193,193)):
    """Alternate color image renderer using Yanli's color scheme

    4 unique colors are used for the 4 unique primitives in "primitives".
    Each part is color-coded based on the order its PID appears in
    the "primitives" set.
    """
    colors = ['#bee37f','#d5a4de','#fcce8d','#a6d4ff']
    primitives = [int(elt) for elt in primitives]
    assert 0 < len(primitives) <= 4

    img = Image.new('RGB', size, color='#ffffff')
    drawer = ImageDraw.Draw(img)
    for poly, pid in zip(polygons, prim_ids):
        cid = primitives.index(int(pid))  # color ID
        drawer.polygon([tuple(xy) for xy in poly],
                       outline='#ffffff',
                       fill=colors[cid])

    return img


def render_shapes(t, prims):
    if len(t)==7:
        if t[4].isnumeric():
            s1 = prims[ord(t[0])-65]
            s2 = prims[ord(t[1])-65]
            s3 = prims[ord(t[3])-65]
            s4 = prims[ord(t[5])-65]
            temp1 = generate_all_attachments(PRIMITIVES[s1],PRIMITIVES[s2])
            att1  = int(t[2]) % temp1[1]
            temp2 = generate_all_attachments(temp1[0][att1],PRIMITIVES[s3])
            att2  = int(t[4]) % temp2[1]
            temp3 = generate_all_attachments(temp2[0][att2],PRIMITIVES[s4])
            att3  = int(t[6]) % temp3[1]
            shape_set = temp3[0][att3]

        else:
            s1 = prims[ord(t[0])-65]
            s2 = prims[ord(t[1])-65]
            s3 = prims[ord(t[3])-65]
            s4 = prims[ord(t[4])-65]
            temp1 = generate_all_attachments(PRIMITIVES[s1],PRIMITIVES[s2])
            temp2 = generate_all_attachments(PRIMITIVES[s3],PRIMITIVES[s4])
            att1  = int(t[2]) % temp1[1]
            att2  = int(t[5]) % temp2[1]
            temp3 = generate_all_attachments(temp1[0][att1],temp2[0][att2])
            att3  = int(t[6]) % temp3[1]
            shape_set = temp3[0][att3]

        prim_ids = (s1, s2, s3, s4)

    elif len(t) == 5:
        s1 = prims[ord(t[0]) - 65]
        s2 = prims[ord(t[1]) - 65]
        att1 = int(t[2]) % len(all_possible_shapes[s1][s2].keys())
        s3 = prims[ord(t[3]) - 65]
        att2 = int(t[4]) % len(all_possible_shapes[s1][s2][att1][s3].keys())
        shape_set = all_possible_shapes[s1][s2][att1][s3][att2]
        prim_ids = (s1, s2, s3)
    elif len(t) == 3:
        s1 = prims[ord(t[0]) - 65]
        s2 = prims[ord(t[1]) - 65]
        att1 = int(t[2]) % len(all_possible_shapes[s1][s2].keys())
        shape_set = all_possible_shapes[s1][s2][att1]['#']
        prim_ids = (s1, s2)
    elif len(t) == 1:
        s1 = prims[ord(t) - 65]
        shape_set = all_possible_shapes[s1]['#']
        prim_ids = (s1,)
    else:
        raise ValueError("Bad token string (t = '{}')".format(t))

    shape_set = shape_set.reshape(-1,4,3).astype(np.float32)
    prim_ids = np.array(prim_ids, np.int64)

    return shape_set, prim_ids


def render_shapes2(t, prims):
    shapes = all_possible_shapes
    prim_ids = []
    att = ''
    for x in t + '#':
        if x.isdigit():
            att += x
        else:
            if att != '':
                shapes = shapes[int(att) % len(shapes)]
                att = ''
            if x == '#':
                break
            s = prims[ord(x) - 65]
            prim_ids.append(s)
            shapes = shapes[s]

    if isinstance(shapes, dict):
        shapes = shapes['#']

    shapes = shapes.reshape(-1,4,3).astype(np.float32)
    prim_ids = np.array(prim_ids, np.int64)

    return shapes, prim_ids


def render_from_string(token_str, prims, with_image=False, v2=False):
    t, angle = token_str.split('+')[:2]
    if v2:
        shape_set, prim_ids = render_shapes2(t, prims)
    else:
        shape_set, prim_ids = render_shapes(t, prims)
    vertices = shapes_to_vertices(shape_set, int(angle))

    if with_image:
        image = make_image(vertices, prim_ids)
        return image, vertices, prim_ids

    return vertices, prim_ids


def make_grid(shapes, center=False):
    shapes = shapes.astype(np.uint8, order='F').reshape(-1,3)
    if center:
        xy = shapes[:,1:]
        xy += 3 - (xy.min(0) + xy.max(0)) // 2

    grid = np.zeros((7,7,4), dtype=np.uint8)
    c, x, y = shapes.T
    grid[y, x, c] = 1

    img = make_grid_image(shapes)

    return grid, img


def get_triangle(c, size=10):
    tri = np.tri(size)
    if c == 0:
        return tri
    elif c == 1:
        return tri[::-1]
    elif c == 2:
        return tri.T
    elif c == 3:
        return tri[:, ::-1]
    raise ValueError('triangle ID must be one of [0,1,2,3]')


def make_grid_image(shapes, size=10, center=False):
    shapes = shapes.astype(np.uint8, order='F').reshape(-1,3)
    if center:
        xy = shapes[:,1:]
        xy += 3 - (xy.min(0) + xy.max(0)) // 2

    img = np.zeros((7,7,size,size))
    for ci, xi, yi in zip(*shapes.T):
        tri = get_triangle(ci, size)
        np.maximum(img[yi, xi], tri, out=img[yi, xi])

    img = img.transpose(0,2,1,3).reshape(7*size, 7*size)

    return img




