# import heapq
import copy
import multiprocessing as mp
from ..bayes_utility.hypothesis import ShapeHypothesis
from functools import partial
from scipy.special import logsumexp

from LOTlib3.Grammar import Grammar
from LOTlib3.FunctionNode import FunctionNode
from LOTlib3.TopN import TopN
from LOTlib3.Samplers.MetropolisHastings import MetropolisHastingsSampler


def check_grammar(grammar=None):
    if grammar is None:
        from svc.utility.grammar import grammar
    assert isinstance(grammar, Grammar)
    return grammar


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

