import torch

from ..bayes_utility.grammar import grammar
from .concepts import MetaConcept


class DataLoader(object):
    """Data loader for training a model with "K-shot" meta-learning epsiodes.

    Args:
        num_concepts (int): the total number of concepts in the dataset
        batch_size (int): the number of concepts in each batch
        k_shot (int): the number of training exemplars per episode
        static (bool): whether to use a fixed set of concepts or to generate
            new concepts for every batch
        transform (callable): optional post-processing transform to apply to
            each batch.
    """

    def __init__(self, num_concepts, batch_size, k_shot, static=False,
                 transform=None, return_concepts=False):
        self.meta_concept = MetaConcept(grammar, num_primitives=9)
        self.num_concepts = num_concepts
        self.batch_size = batch_size
        self.k_shot = k_shot
        self.static = static
        self.transform = transform
        self.return_concepts = return_concepts
        if static:
            self.concepts = [self.sample_concept() for _ in range(num_concepts)]

    def sample_concept(self):
        """sample a new concept with at least k_shot+1 exemplars"""
        concept = self.meta_concept.sample()
        while len(concept) <= self.k_shot:
            concept = self.meta_concept.sample()
        return concept

    def __iter__(self):
        self.i = 0
        if self.static:
            self.perm = torch.randperm(self.num_concepts)
        return self

    def __next__(self):
        if self.i + self.batch_size > self.num_concepts:
            raise StopIteration()

        if self.static:
            concepts = [self.concepts[i] for i in self.perm[self.i:self.i + self.batch_size]]
        else:
            concepts = [self.sample_concept() for _ in range(self.batch_size)]

        self.i += self.batch_size

        inputs = []
        target = []
        for concept in concepts:
            # select k_shot+1 exemplars uniformly at random
            idx = torch.randperm(len(concept))[:self.k_shot+1]
            inputs.append([concept[i] for i in idx[:self.k_shot]])
            target.append(concept[idx[self.k_shot]])

        if self.transform is not None:
            inputs, target = self.transform(inputs, target)

        if self.return_concepts:
            return inputs, target, concepts

        return inputs, target