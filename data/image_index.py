import os
import pickle
from collections import defaultdict

from svc.utility.render import all_possible_shapes

_index_path = os.path.join(os.path.dirname(__file__), 'image_index.pkl')


class TokenImageIndex:
    def __init__(self):
        with open(_index_path, 'rb') as f:
            self._index = pickle.load(f)

        inverse_index = defaultdict(list)
        for key, val in self._index.items():
            inverse_index[val].append(key)
        self.inverse = {k:sorted(v) for k,v in inverse_index.items()}

    def __getitem__(self, key):
        if isinstance(key[0], str):
            assert isinstance(key, tuple) and len(key) == 2
            token_str, prims = key
            key = self.encode_key(token_str, prims)
        return self._index[key]

    def _check_key(self, key):
        key = tuple(int(x) for x in key)
        # assert len(key) >= 2
        assert len(key) in {2, 4, 6}
        return key

    def _check_prims(self, prims):
        prims = [int(p) for p in prims]
        assert 0 < len(prims) <= 4
        return prims

    def equivalents(self, key, skip_self=False):
        key = self._check_key(key)
        image_id = self._index[key]
        others = self.inverse[image_id]
        if skip_self:
            others = [k for k in others if k != key]
        return others

    def equivalent_strings(self, token_str, prims, skip_self=False):
        key = self.encode_key(token_str, prims)
        equiv_keys = self.equivalents(key, skip_self=skip_self)
        equiv_strings = [self.decode_key(k, prims) for k in equiv_keys]
        return equiv_strings

    def encode_key(self, token_str, prims):
        prims = self._check_prims(prims)
        t, angle = token_str.split('+')

        shapes = all_possible_shapes
        key = []
        att = ''
        for x in t + '#':
            if x.isdigit():
                att += x
                continue

            if att != '':
                att = int(att) % len(shapes)
                key.append(att)
                shapes = shapes[att]
                att = ''
            if x == '#':
                break
            p = prims[ord(x) - 65]
            key.append(p)
            shapes = shapes[p]

        key.append(int(angle))

        return self._check_key(key)

    def decode_key(self, key, prims):
        key = self._check_key(key)
        prims = self._check_prims(prims)
        letters = ['A', 'B', 'C', 'D']

        t = letters[prims.index(key[0])]
        for i, x in enumerate(key[1:-1]):
            if i % 2 == 0:
                # primitive
                t += letters[prims.index(x)]
            else:
                # attachment
                t += str(x)

        token_str = t + '+' + str(key[-1])

        return token_str