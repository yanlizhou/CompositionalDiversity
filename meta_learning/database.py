"""
A database class for indexing into a discrete set of possible
tokens (i.e. possible primitive combinations)
"""
import functools
from svc.utility.render import all_possible_shapes


class Meta(type):
    def __getitem__(self, key):
        data = all_possible_shapes
        for k in key:
            data = data[k]
        if len(key) in {1, 3}:
            data = data['#']
        return data


class database(metaclass=Meta):

    @classmethod
    def keys(cls, npart=(1,2,3)):
        if isinstance(npart, (list, tuple)):
            return functools.reduce(lambda a, b: a + cls._keys(b), npart, ())
        return cls._keys(npart)

    @staticmethod
    @functools.lru_cache(maxsize=3)
    def _keys(npart):
        npart = int(npart)
        assert npart in {1, 2, 3}

        options = lambda d: filter(lambda x: x != '#', d.keys())

        keys = []
        for s1 in options(all_possible_shapes):
            if npart == 1:
                keys += [(s1,)]
                continue
            for s2 in options(all_possible_shapes[s1]):
                for att1 in options(all_possible_shapes[s1][s2]):
                    if npart == 2:
                        keys += [(s1,s2,att1)]
                        continue
                    for s3 in options(all_possible_shapes[s1][s2][att1]):
                        for att2 in options(all_possible_shapes[s1][s2][att1][s3]):
                            keys += [(s1,s2,att1,s3,att2)]

        return tuple(keys)

    @staticmethod
    def prim_ids(key):
        size = len(key)
        if size == 1:
            return [key[0]]
        elif size == 3:
            return [key[0], key[1]]
        elif size == 5:
            return [key[0], key[1], key[3]]
        raise ValueError