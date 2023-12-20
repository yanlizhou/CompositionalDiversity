import copy
import math
import random
# import numpy as np
from scipy.special import logsumexp
from LOTlib3.BVRuleContextManager import BVRuleContextManager
from LOTlib3.FunctionNode import NodeSamplingException
# from LOTlib3.Subtrees import least_common_difference
from LOTlib3.Grammar import Grammar
from LOTlib3.FunctionNode import FunctionNode


def lambda_one(*args):
    return 1


def least_common_difference(t1, t2):
    """
        For a pair of trees, find the nodes (one in each tree) defining
        the root of their differences. Return None for identical trees.
    """
    if t1 == t2:
        return None, None

    if t1.args is None or t2.args is None:
        return t1, t2

    # if these rules look the same, counting terminals and nonterminal kids
    if t1.get_rule_signature() == t2.get_rule_signature():

        # first check if any strings (non functions nodes) below differ
        for a, b in zip(t1.argNonFunctionNodes(), t2.argNonFunctionNodes()):
            if a != b:
                return t1, t2 # if so, t1 and t2 differ

        diff = None
        argzip = zip(t1.argFunctionNodes(), t2.argFunctionNodes())
        argzip = filter(lambda tup: tup[0] != tup[1], argzip)
        for arg12 in argzip:
            if diff is None:
                diff = arg12
            else:
                return t1, t2

        return least_common_difference(*diff)

    return t1, t2


class ProposalFailedException(Exception):
    """Raised when we have a proposal that can't succeed"""
    pass


class RegenerationProposer:

    @classmethod
    def propose(cls, h):
        try:
            value, fb = cls.proposal_content(h.grammar, h.value)
        except ProposalFailedException:
            return cls.propose(h)
        return h.__copy__(value=value), fb

    @classmethod
    def proposal_content(self, grammar, tree, resample_prob=lambda_one):
        """
        Parameters
        ----------
        grammar: Grammar
            Prior distribution over hypotheses (expressed as a PCFG)
        tree: FunctionNode
            Current hypothesis; e.g. a sample from the prior
        resample_prob: Callable
            TODO
        """
        t = self.propose_tree(grammar, tree, resample_prob)
        f_a = self.proposal_probability(grammar, tree, t, resample_prob)
        f_b = self.proposal_probability(grammar, t, tree, resample_prob)
        return t, f_a - f_b

    @staticmethod
    def propose_tree(grammar, t, resample_prob=lambda_one):
        """Propose, returning the new tree"""
        t = copy.copy(t)
        try:
            # try to sample a subnode
            # n, lp = t.sample_subnode(resampleProbability=resample_prob)
            n, lp = sample_subnode(t, resample_prob)
        except NodeSamplingException:
            # when no nodes can be sampled
            # print('exception encountered!')
            raise ProposalFailedException

        # In the context of the parent, resample n according to the
        # grammar. recurse_up in order to add all the parent's rules
        with BVRuleContextManager(grammar, n.parent, recurse_up=True):
            n.setto(grammar.generate(n.returntype))

        return t

    @staticmethod
    def proposal_probability(grammar, t1, t2, resample_prob=lambda_one, recurse=True):
        """
        NOTE: This is not strictly necessary since we don't actually
        have to sum over trees if we use an auxiliary variable argument.
        But this fits nicely with the other proposers and is not much slower.
        """
        sampling_log_probability = t1.sampling_log_probability
        if resample_prob is lambda_one:
            _sample_lp = - math.log(sum(1. for _ in t1))
            sampling_log_probability = lambda node: _sample_lp

        def score(n1, n2):
            lp_of_choosing_node = sampling_log_probability(n1)
            with BVRuleContextManager(grammar, n2.parent, recurse_up=True):
                # TODO: 90 - 130 us per hit
                #lp_of_generating_tree = grammar.log_probability(n2)
                lp_of_generating_tree = log_probability(grammar, n2)
            return lp_of_choosing_node + lp_of_generating_tree

        # TODO: 660 us per hit
        node1, node2 = least_common_difference(t1, t2)

        if node1 is None:
            # There are no paths up the tree.
            # This means any node in the tree could have been regenerated
            return logsumexp([score(n, n) for n in t1])

        if not recurse:
            return score(node1, node2)

        lps = []
        while node1:
            lps.append(score(node1, node2))
            node1, node2 = node1.parent, node2.parent

        return logsumexp(lps)


def log_probability(grammar, t):
    """
    Returns the log probability of t under the prior (grammar)
    """
    assert isinstance(t, FunctionNode)

    rule_sig = t.get_rule_signature()
    lp = None
    Z = 0.
    for r in grammar.get_rules(t.returntype):
        Z += r.p
        if lp is None and r.get_rule_signature() == rule_sig:
            lp = math.log(r.p)
    assert lp is not None
    lp -= math.log(Z)

    with BVRuleContextManager(grammar, t):
        for arg in t.argFunctionNodes():
            lp += log_probability(grammar, arg)

    return lp


def sample_subnode(self, resample_prob=None):
    """Sample a subnode at random.

    We return a sampled tree and the log probability of sampling it
    """
    # if resample_prob is lambda_one:
    #     subnodes = [t for t in self]
    #     N = len(subnodes)
    #     if N <= 0:
    #         raise NodeSamplingException
    #     i = random.randint(0, N - 1)
    #     return subnodes[i], -math.log(N)

    Z = 0.
    probs = []
    subnodes = []
    for t in self:
        p = resample_prob(t)
        Z += p
        subnodes.append(t)
        probs.append(p)

    if Z <= 0:
        raise NodeSamplingException

    # now select a random number (giving a random node)
    r = random.random() * Z
    for t, p in zip(subnodes, probs):
        r -= p
        if r <= 0:
            return t, math.log(p) - math.log(Z)

    raise Exception('An error occured in sample_subnode')


