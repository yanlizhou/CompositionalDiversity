import math
import random

from .proposers import RegenerationProposer


class MetropolisHastingsSampler:
    """A class to implement MH sampling.

    Parameters
    ----------
    current_sample : Hypothesis
        Initial hypothesis
    data : TODO
        TODO
    steps : int
        Number of steps to generate before stopping.
    proposer : function
        Defaultly this calls the sample Hypothesis's propose() function. If
        provided, it should return a *new copy* of the object
    skip : int
        Throw out this many samples each time MHSampler yields a sample.
    prior_temperature : float
        How much do we weight our prior relative to likelihood?
    likelihood_temperature : float
        How much do we weight our likelihood relative to prior?
    acceptance_temperature : float
        This weights the probability of accepting proposals.
    shortcut_likelihood : bool
        If true, we allow for short-cut evaluation of the likelihood, rejecting when we can if the ll
        drops below the acceptance value
    """
    def __init__(self,
                 current_sample,
                 data,
                 steps=int(1e6),
                 proposer=RegenerationProposer.propose,
                 skip=0,
                 prior_temperature=1.,
                 likelihood_temperature=1.,
                 acceptance_temperature=1.,
                 #shortcut_likelihood=True
                 ):
        self.current_sample = current_sample
        self.data = data
        self.steps = steps
        self.proposer = proposer
        self.skip = skip
        self.prior_temperature = prior_temperature
        self.likelihood_temperature = likelihood_temperature
        self.acceptance_temperature = acceptance_temperature
        #self.shortcut_likelihood = shortcut_likelihood
        self.was_accepted = None
        self.current_sample = current_sample
        if self.current_sample is not None:
            self.current_sample.compute_posterior(self.data)
        self.samples_yielded = 0
        self.acceptance_count = 0
        self.proposal_count = 0
        self.posterior_calls = 0

    def compute_posterior(self, h, data, shortcut=float('-inf')):
        """
        A wrapper for hypothesis.compute_posterior(data) that can be
        overwritten in fancy subclassses.
        """
        self.posterior_calls += 1
        return h.compute_posterior(data, shortcut=shortcut)

    def acceptance_ratio(self):
        """The proportion of proposals that have been accepted"""
        if self.proposal_count == 0:
            return float('nan')
        return self.acceptance_count / self.proposal_count

    def mh_acceptance(self, cur, prop, fb):
        """
        Returns whether to accept the proposal, while handling weird corner
        cases for computing MH acceptance ratios.

        Parameters
        ----------
        cur : float
            The current sample's posterior score
        prop : float
            The proposal's posterior score
        fb : float
            The forward-backward ratio
        """
        if math.isnan(prop) or prop == float('-inf') or fb == float('inf'):
            # never accept
            return False

        # If we get infs or are in a stupid state, let's just sample from
        # the prior so things don't get crazy
        if math.isnan(cur) or (cur == float('-inf') and prop == float('-inf')):
            # Just choose at random
            #   we can't sample priors since they may be -inf both
            r = - math.log(2.)
        else:
            r = (prop - cur - fb) / self.acceptance_temperature

        return r >= 0 or random.random() < math.exp(r)

    def __iter__(self):
        return self

    def __next__(self):
        """Generate another sample."""
        if self.samples_yielded >= self.steps:
            raise StopIteration

        self.samples_yielded += 1

        for _ in range(self.skip + 1):
            # todo: refactor (90% overall time)
            self.proposal, fb = self.proposer(self.current_sample)

            # store current and proposed points
            x = self.current_sample
            x_prop = self.proposal
            assert x_prop is not x and x_prop.value is not x.value

            # Call myself so memoized subclasses can override
            # todo: refactor (10% overall time)
            _ = self.compute_posterior(x_prop, self.data)

            # Note: It is important that we re-compute from the temperature
            # since these may be altered externally
            cur = x.prior / self.prior_temperature + \
                  x.likelihood / self.likelihood_temperature
            prop = x_prop.prior / self.prior_temperature + \
                   x_prop.likelihood / self.likelihood_temperature

            self.was_accepted = self.mh_acceptance(cur, prop, fb)

            if self.was_accepted:
                self.current_sample = x_prop
                self.acceptance_count += 1

            self.proposal_count += 1

        return self.current_sample