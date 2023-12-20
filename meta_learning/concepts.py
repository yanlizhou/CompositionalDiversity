import math
import random
import numpy as np

from ..utility.grammar import grammar as base_grammar
from ..utility.hypothesis import ShapeHypothesis
from ..utility2.render import render_from_string


def default_getter(token_str, primitives):
    image, verts, prim_ids = render_from_string(token_str, primitives, with_image=True)
    image = np.array(image, dtype=np.float32) / 255  # [0, 1] pixel range
    return image


class Concept(object):
    def __init__(self, grammar, t, primitives, getter=default_getter):
        self.hypothesis = ShapeHypothesis(grammar, t)
        is_valid = lambda token: len(token.split('+')[0]) in [1,3,5] and not token.startswith('p')
        self.exemplars = sorted(filter(is_valid, self.hypothesis()))
        self.primitives = primitives
        self.getter = getter

    def __len__(self):
        return len(self.exemplars)

    def __getitem__(self, ix):
        return self.getter(self.exemplars[ix], self.primitives)

    def __repr__(self):
        return 'Concept(\n    rule: {},\n    primitives: {}\n)'.format(
            self.hypothesis.value, self.primitives)

    def sample(self):
        ix = np.random.choice(len(self))
        return self[ix]

    def score(self, *args):
        """uniform distribution p(c) = 1/N, log(p(c)) = -log(N)"""
        loglike = - math.log(len(self))
        return loglike


class MetaConcept(object):
    def __init__(self, grammar=base_grammar, num_primitives=9, getter=default_getter):
        self.grammar = grammar
        self.num_primitives = num_primitives
        # the number of possible 4-length primitive sets that we can
        # select (assuming that order matters)
        self.num_possible_primsets = math.factorial(num_primitives) / math.factorial(num_primitives - 4)
        self.getter = getter

    def sample(self):
        t = self.grammar.generate()
        primitives = random.sample(range(self.num_primitives), 4)
        concept = Concept(self.grammar, t, primitives, self.getter)
        return concept

    def score(self, concept):
        """p(type) = p(rule) * p(prim_set)"""
        loglike = self.grammar.log_probability(concept.hypothesis.value)
        loglike -= math.log(self.num_possible_primsets)
        return loglike
