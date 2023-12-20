# import heapq
import copy
import math
import random
import multiprocessing as mp
from functools import partial
from scipy.special import logsumexp

from LOTlib3.Grammar import Grammar
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Miscellaneous import attrmem
from LOTlib3.Eval import TooBigException
from LOTlib3.FunctionNode import FunctionNode
from LOTlib3.TopN import TopN
from LOTlib3.Samplers.MetropolisHastings import MetropolisHastingsSampler


def check_grammar(grammar=None):
    if grammar is None:
        from svc.utility.grammar import grammar
    assert isinstance(grammar, Grammar)
    return grammar


def valid_tkn(tkn, max_parts=3):
    # return not tkn.startswith('p')
    # return not tkn.startswith('p') and len(tkn.split('+')[0]) in [1,3,5]
    if tkn.startswith('p'):
        return False
    num_parts = sum(not x.isnumeric() for x in tkn.split('+')[0])
    return num_parts <= max_parts


class ShapeHypothesis(LOTHypothesis):
    # If a token not in the hypothesis, it still has (1 - alpha)
    # probability of being generated.
    def __init__(self, grammar=None, value=None, max_level=3, alpha=0.99, **kwargs):
        super().__init__(
            grammar=grammar, value=value, display="lambda : %s", **kwargs)
        self.max_level = max_level
        self.alpha = alpha

    def __call__(self, *args, **kwargs):
        try:
            all_tokens = super().__call__()
        except TooBigException:
            return set()
        except OverflowError:
            if not kwargs.get('catch_overflow', False):
                raise
            return set()

        return set(tkn for tkn in all_tokens if valid_tkn(tkn, self.max_level))

    def __len__(self):
        return len(self())

    def sample(self, n=None):
        all_tokens = sorted(self())
        if n is None:
            return random.choice(all_tokens)
        return random.choices(all_tokens, k=n)

    def log_prob(self, data):
        temp = self.likelihood_temperature
        self.likelihood_temperature = 1
        if isinstance(data, str):
            data = [data]
        ll = self.compute_likelihood(data)
        self.likelihood_temperature = temp
        return ll

    @attrmem('prior')
    def compute_prior(self):
        """Compute the log of the prior probability.
        """
        if self.value.count_subnodes() > self.maxnodes:
            return float('-inf')
        if len(self()) == 0:
            return float('-inf')
        return self.grammar.log_probability(self.value) / self.prior_temperature

    @attrmem('likelihood')
    def compute_likelihood(self, data, **kwargs):
        """Likelihood of specified data being produced by this hypothesis.
        """
        # make sure we're not evaluating a single string
        assert not isinstance(data, str)

        all_tokens = self(catch_overflow=True)
        ll = sum(self.compute_single_likelihood(tk, all_tokens) for tk in data)
        return ll / self.likelihood_temperature

    def compute_single_likelihood(self, token, all_tokens=None):
        if all_tokens is None:
            all_tokens = self(catch_overflow=True)

        assert isinstance(token, str)
        p = (1 - self.alpha) / 1000
        if token in all_tokens:
            p += self.alpha / len(all_tokens)

        return math.log(p)


def sample_init(tokens, grammar, n=1, trials=1000):
    top = TopN(n)
    for i in range(trials):
        value = grammar.generate()
        score = ShapeHypothesis(grammar, value).compute_posterior(tokens)
        top.add(value, score)

    top = list(reversed(list(top)))

    if n == 1:
        return top[0]

    return top


def lambda_one(x):
    return 1.


def mcmc_chain(tokens, h0=None, grammar=None, top_n=10, max_iter=100000,
               verbose=False, print_freq=100, anneal_schedule=lambda_one):
    grammar = check_grammar(grammar)
    if h0 is None:
        h0 = sample_init(tokens, grammar)
    if isinstance(h0, FunctionNode):
        h0 = ShapeHypothesis(grammar, h0)
    assert isinstance(h0, ShapeHypothesis)

    sampler = MetropolisHastingsSampler(h0, tokens, steps=max_iter,
                                        acceptance_temperature=anneal_schedule(0))

    max_digits = len(str(max_iter))
    niter = 0
    top = TopN(top_n)
    for h in sampler:
        niter += 1
        top.add(h)
        sampler.acceptance_temperature = anneal_schedule(niter)
        if verbose and niter % print_freq == 0:
            print('iter: {:{m}d}'.format(niter, m=max_digits), end='    ')
            # log-joint
            print(f'logp(h,D): {h.posterior_score:8.3f}', end='    ')
            # Lower bound on log-marginal (ELBO: evidence lower bound)
            log_marginal = logsumexp([h.posterior_score for h in top])
            print(f'logp(D): {log_marginal:8.3f}')

    return top


def sample_posterior(tokens, grammar=None, top_n=10, num_chains=1,
                     max_iter=100000, parallel=False, verbose=False,
                     print_freq=100, anneal_schedule=lambda_one):
    if num_chains == 1:
        return mcmc_chain(
            tokens, grammar=grammar, top_n=top_n, max_iter=max_iter,
            verbose=verbose, print_freq=print_freq,
            anneal_schedule=anneal_schedule)

    grammar = check_grammar(grammar)

    # sample unique initialization for each chain
    initializations = sample_init(tokens, grammar, n=num_chains)

    # run all chains in parallel
    run_chain = partial(mcmc_chain, tokens, top_n=top_n,
                        max_iter=max_iter, verbose=verbose,
                        print_freq=print_freq, anneal_schedule=anneal_schedule)
    if parallel:
        inputs = [(h, copy.deepcopy(grammar)) for h in initializations]
        with mp.Pool() as p:
            chain_results = p.starmap(run_chain, inputs)
    else:
        chain_results = [run_chain(h, grammar) for h in initializations]

    # consolidate best results
    # top = heapq.merge(*chain_results, key=lambda h: h.posterior_score)[-top_n:]
    top = TopN(top_n)
    for chain in chain_results:
        for h in chain:
            top.add(h)

    return top

