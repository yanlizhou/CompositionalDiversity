from math import log
#from .grammar import *
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis, Infinity
from LOTlib3.Miscellaneous import attrmem
from LOTlib3.Eval import TooBigException

class ShapeHypothesis(LOTHypothesis):

    def __init__(self, grammar=None, value=None, max_level=3, **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, value=value, display="lambda : %s", **kwargs)
        self.max_level = max_level

    @attrmem('prior')
    def compute_prior(self):
        """Compute the log of the prior probability."""

        # Compute this hypothesis prior
        if self.value.count_subnodes() > self.maxnodes:
            return -Infinity
        elif len(self()) == 0:
            return -Infinity
        else:
            # If all those checks pass, just return the tree log prob
            return self.grammar.log_probability(self.value) / self.prior_temperature

    @attrmem('likelihood')
    def compute_likelihood(self, data, **kwargs):
        """Likelihood of specified data being produced by this hypothesis.
        If datum item not in set, it still has (1 - alpha) likelihood of being generated.
        Args:
            data (FunctionData): this is the data; we only use data.output
            update_post (bool): boolean -- do we update posterior?
        """
        try:
            cached_set = self()      # Set of numbers corresponding to this hypothesis
        except OverflowError:
            cached_set = set()       # If our hypothesis call blows things up

        return sum([self.compute_single_likelihood(datum, cached_set) for datum in data]) / self.likelihood_temperature

    def compute_single_likelihood(self, d, cached_set=None):
        # the likelihood of getting all of these data points

        assert cached_set is not None, "***  Require precomputation of the hypothesis' set in compute_likelihood"
        assert len(d.input) == 0, "*** Required input is [] to use this implementation (functions are thunks)"

        ll = 0.0

        # Must sum over all elements in the set
        for di in d.output:
            if len(cached_set) > 0:
                ll += log(d.alpha*(di in cached_set)/len(cached_set) + (1.-d.alpha) / 1000)
            else:
                ll += log((1.-d.alpha)/1000)

        return ll
    


    def compute_predictive_probability(self, datum, posterior, ALPHA=0.9, BETA=0.5):
        
        try:
            cached_set = self()      # Set of numbers corresponding to this hypothesis
        except OverflowError:
            cached_set = set()       # If our hypothesis call blows things up
        
        if len(cached_set) > 0:
            
            ll = (1 - ALPHA) * BETA + ALPHA * (datum in cached_set)

        else:
            ll = (1 - ALPHA) * BETA

                    
            
        return ll*posterior 



    def __call__(self, *args, **kwargs):
        # Sometimes self.value has too many nodes
        try:
            value_set = LOTHypothesis.__call__(self)
        except TooBigException:
            value_set = set()

#         if ~isinstance(value_set, set):
#             value_set = set()

        return value_set

