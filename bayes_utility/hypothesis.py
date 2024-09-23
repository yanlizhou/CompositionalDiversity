import math
import random
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Miscellaneous import attrmem
from LOTlib3.Eval import TooBigException

def valid_tkn(tkn, max_parts=3):
    if tkn.startswith('p'):
        return False
    num_parts = sum(not x.isnumeric() for x in tkn.split('+')[0])
    return num_parts <= max_parts

class ShapeHypothesis(LOTHypothesis):
    # If a token not in the hypothesis, it still has (1 - alpha)
    # probability of being generated.
    def __init__(self, grammar=None, value=None, max_level=3, alpha=0.999, **kwargs):
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


    

    # def compute_predictive_probability(self, datum, posterior, ALPHA=0.9, BETA=0.5):
        
    #     try:
    #         cached_set = self()      # Set of numbers corresponding to this hypothesis
    #     except OverflowError:
    #         cached_set = set()       # If our hypothesis call blows things up
        
    #     if len(cached_set) > 0:
            
    #         ll = (1 - ALPHA) * BETA + ALPHA * (datum in cached_set)

    #     else:
    #         ll = (1 - ALPHA) * BETA
               
            
    #     return ll*posterior 




