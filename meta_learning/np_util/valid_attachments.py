from tqdm import tqdm
import numpy as np

from ...utility2.render import make_image
from . import recenter, dictionary, attach_part


def image_overlap(xu, xv, cu, cv):
    image_a = np.array(make_image([xu], [cu]))
    image_a = (image_a > 0).any(2) & (image_a.astype(int).sum(2) < 700)

    image_b = np.array(make_image([xv], [cv]))
    image_b = (image_b > 0).any(2) & (image_b.astype(int).sum(2) < 700)

    overlap = np.sum(image_a & image_b, dtype=float)

    return overlap / 2145


def valid_attachments(progress=False):
    overlap = np.zeros((9,9,6,6), np.float32)
    buf = np.zeros((2,7,2), np.float32)
    with tqdm(total=overlap.size, ncols=80, disable=not progress) as pb:
        for cu in range(9):
            xu_ = recenter(dictionary[cu]).round()
            for cv in range(9):
                xv_ = dictionary[cv]
                for su in range(6):
                    for sv in range(6):
                        buf[0] = xu_
                        buf[1] = attach_part(xv_, sv, xu_, su)
                        buf += 96 - (buf.min((0,1)) + buf.max((0,1))) / 2
                        buf = buf.round(out=buf)
                        overlap[cu, cv, su, sv] = image_overlap(buf[0], buf[1], cu, cv)
                        pb.update(1)

    valid = overlap < 0.0114

    return valid, overlap


def load_valid_attachments(progress=True):
    import os
    import svc

    file = os.path.join(os.path.dirname(svc.__file__), 'data', 'valid_attachments.npy')

    if os.path.exists(file):
        valid = np.load(file)
    else:
        print('valid_attachments.npy not found. Computing valid tensor...', flush=True)
        valid, _ = valid_attachments(progress=progress)
        np.save(file, valid)

    return valid
