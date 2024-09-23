import os
import pickle
import re
import warnings
import functools
import math
import random
import numpy as np
from scipy.special import logsumexp, log_softmax
from collections import Counter
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Miscellaneous import attrmem
from LOTlib3.Eval import TooBigException
from LOTlib3.FunctionNode import BVUseFunctionNode
from LOTlib3.DataAndObjects import FunctionData

from ..bayes_utility.render import all_possible_shapes
from ..bayes_utility.grammar import all_rot_, has_, mystr_


if os.path.exists('/Users/rfeinman'):
    ROOT = '/Users/rfeinman/NYU/PhdThesis/GNS Modeling/' \
           'Structured Visual Concepts/Yanli Stuff/fitted_bayes_model'
elif os.path.exists('/home/yz1349'):
    ROOT = '/misc/vlgscratch4/LakeGroup/Yanli/SVC/fitted_bayes_model'
else:
    raise Exception('root path must be provided.')

with open(os.path.join(ROOT, 'token_to_img_dict.pkl'), 'rb') as f:
    token_to_img_dict = pickle.load(f)


zoo_exp1 = [['A+0'],
           ['A+0', 'A+0', 'A+0'],
           ['A+0', 'A+90'],
           ['A+0','B+0','C+0'],
           ['AB0+0'],
           ['AB0+0', 'AB0+0', 'AB0+0'],
           ['AA0+0'],
           ['AB0+0', 'AC0+0'],
           ['AA0+0', 'AB0+0'],
           ['AB0+0', 'AB1+0'],
           ['AB0+0', 'AB0+90']]

zoo_exp2_3 =  [['A+0','A+90','A+270'],
               ['AB0+0','AB1+90','AB2+270'],
               ['AB0+0','AB0+90','AB0+270'],
               ['AB0+0','AA0+90','AC0+270'],
               ['AA0+0', 'BB0+0', 'CC0+0'],
               ['A+0','AA3+0','AA2A0+0'],
               ['A+0','AD0+90','AB1C0+0'],
               ['AB1B0+0','AC3C2+90','DD3A3+0'],
               ['AA0B0+0','AA0C0+270','AA0D0+90'],
               ['AA0C3+0','AA1B3+90','AA2D0+270']]

zoo_exp2_6 =  [['A+0','A+90','A+270','A+90','A+0','A+270'],
               ['AB0+0','AB1+90','AB2+270','AB2+0','AB0+90','AB1+0'],
               ['AB0+0','AB0+90','AB0+270','AB0+0','AB0+270','AB0+90'],
               ['AB0+0','AA0+90','AC0+270','AA2+270','AD0+0','AB1+0'],
               ['AA0+0','BB0+0','CC0+0','BB2+0','AA2+90','CC0+270'],
               ['A+0','AA3+0','AA2A0+0','A+270','AA1A0+0','AA1+90'],
               ['A+0','AD0+90','AB1C0+0','A+90','AB0+270','AC1D0+90'],
               ['AB1B0+0','AC3C2+90','DD3A3+0','BB3A2+0','AD3D2+180','AC3C3+0'],
               ['AA0B0+0','AA0C0+270','AA0D0+90','AA0B1+90','AA0C0+0','AA0D0+180'],
               ['AA0C3+0','AA1B3+90','AA2D0+270','AA3D2+90','AA2B1+90','AA0C3+0']]

zoo_types = zoo_exp2_3 + zoo_exp2_6 + zoo_exp1
PRIMS = ['A', 'B', 'C', 'D']

ALL_TOKEN_STR = set()
for p in PRIMS:
    ALL_TOKEN_STR.update(all_rot_(has_(18, mystr_(p))))
token_to_ind = {token:i for i, token in enumerate(sorted(ALL_TOKEN_STR))}


def toindex(tokens):
    out = np.zeros(len(token_to_ind), dtype=np.float32)
    for token in tokens:
        out[token_to_ind[token]] += 1
    return out


@functools.lru_cache(maxsize=1024)
def evaluate_tokens(value):
    return set() if value.count_nodes() > 25 else eval(str(value))

@functools.lru_cache(maxsize=1024)
def get_all_tokens(value, prims):
    tokens = evaluate_tokens(value)

    out = set()
    for tk in tokens:
        letters = "".join(filter(str.isalpha, tk))

        if len(letters) == 1:
            out.add(tk)

        elif len(letters) == 2:
            x, angle = tk.split('+')[:2]
            p1 = prims[PRIMS.index(letters[0])]
            p2 = prims[PRIMS.index(letters[1])]
            att = int(x[2:])

            if att not in all_possible_shapes[p1][p2]:
                att %= len(all_possible_shapes[p1][p2])

            out.add(f'{letters[0]}{letters[1]}{att}+{angle}')

        elif len(letters) == 3:
            x, angle = tk.split('+')[:2]
            p1 = prims[PRIMS.index(letters[0])]
            p2 = prims[PRIMS.index(letters[1])]
            p3 = prims[PRIMS.index(letters[2])]
            if x[3] == letters[2]:
                att1 = int(x[2])
                att2 = int(x[4:])
            else:
                att1 = int(x[2:4])
                att2 = int(x[5:])

            if att1 not in all_possible_shapes[p1][p2]:
                att1 %= len(all_possible_shapes[p1][p2])
            if att2 not in all_possible_shapes[p1][p2][att1][p3]:
                att2 %= len(all_possible_shapes[p1][p2][att1][p3])

            out.add(f'{letters[0]}{letters[1]}{att1}{letters[2]}{att2}+{angle}')

    return sorted(out)


# ----------------------------------------------------------------------


class ShapeHypothesis(LOTHypothesis):
    def __init__(self,
                 grammar, value, prims,
                 token_set=None,
                 prior_temperature=1.0,
                 likelihood_temperature=1.0):
        super().__init__(grammar, value,
                         display="lambda : %s",
                         prior_temperature=prior_temperature,
                         likelihood_temperature=likelihood_temperature)
        self.prims = prims
        if token_set is None:
            token_set = get_all_tokens(value, tuple(prims))
        self._all_tokens = token_set
        self._all_tokens_index = toindex(self._all_tokens)
        self._is_empty = np.sum(self._all_tokens_index) == 0

        imgs = token_to_img_dict[self.prims]
        self.N_TOKENS = len(set(imgs.values()))
        self.h_size = len(set(imgs[tkn] for tkn in self._all_tokens))

    @attrmem('prior')
    def compute_prior(self):
        if self.h_size == 0:
            return float('-inf')
        ll = self.grammar.log_probability(self.value)
        return ll / self.prior_temperature

    @attrmem('likelihood')
    def compute_likelihood(self, data, **kwargs):
        ll = sum(map(self.compute_single_likelihood, data))
        return ll / self.likelihood_temperature

    def compute_single_likelihood(self, d):
        assert len(d.input) == 0
        ll = 0.
        for di in d.output:
            p = (1. - d.alpha) / self.N_TOKENS
            if not self._is_empty:
                dot = np.dot(self._all_tokens_index, toindex([di]))
                p += d.alpha * dot / self.h_size
            ll += math.log(p)
        return ll

    def __call__(self, *args, **kwargs):
        try:
            value_set = super().__call__()
        except (TooBigException, OverflowError):
            value_set = set()
        return value_set

    def sample(self, n=1, weights=None):
        if n == 1:
            return random.choice(self._all_tokens)
        return random.choices(self._all_tokens, k=n, weights=weights)

    def log_prob(self, token, alpha):
        p = (1 - alpha) / self.N_TOKENS
        if token in self._all_tokens:
            p += alpha / self.h_size
        return math.log(p)


class ShapeHypothesisDMPrior(ShapeHypothesis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts, self.next_free_idx = self.create_counts()

    def create_counts(self):
        grammar_rules = [r for r in self.grammar]
        next_free_idx = Counter()
        sig2idx = dict()

        for r in grammar_rules:
            nt, s = r.nt, r.get_rule_signature()
            if s not in sig2idx:
                sig2idx[s] = next_free_idx[nt]
                next_free_idx[nt] += 1

        counts = {nt: np.zeros(next_free_idx[nt]) for nt in next_free_idx.keys()}
        nodes = [n for n in self.value]
        for n in nodes:
            nt, s = n.returntype, n.get_rule_signature()
            if not isinstance(n, BVUseFunctionNode):
                counts[nt][sig2idx[s]] += 1
            else:
                if len(counts[nt]) == next_free_idx[nt]:
                    counts[nt] = np.append(counts[nt], 1)
                else:
                    counts[nt][-1] += 1

        return counts, next_free_idx

    @attrmem('prior')
    def compute_prior(self):
        """Compute the log of the prior probability."""
        if self.h_size == 0:
            return float('-inf')

        # from scipy.special import gamma
        # log_multi_beta = lambda x: np.log(np.prod(gamma(x)) / gamma(np.sum(x)))
        from scipy.special import gammaln
        log_multi_beta = lambda x: np.sum(gammaln(x)) - gammaln(np.sum(x))

        ll = 0.
        for nt in self.counts.keys():
            betas = np.array([r.p for r in self.grammar.get_rules(nt)])
            if len(self.counts[nt]) != self.next_free_idx[nt]:
                betas = np.append(betas, 1.0)
            ll += log_multi_beta(betas + self.counts[nt]) - log_multi_beta(betas)

        return ll / self.prior_temperature


class PosteriorPredictive:
    def __init__(self, grammar, h_values, prims, n_att=18, zoo=None, trial=None,
                 h_token_sets=None, pt=1.0, lt=1.0, dm_prior=False):
        if zoo is None:
            assert trial is not None, "Either 'zoo' or 'trial' must be provided"
            zoo = zoo_types[trial]
        self.zoo = zoo
        self.grammar = grammar
        self.prims = prims
        self.n_att = n_att

        assert not dm_prior
        assert len(h_values) >= 1
        hypotheses = []
        for i, value in enumerate(h_values):
            h = ShapeHypothesis(grammar, value, prims,
                                token_set=None if h_token_sets is None else h_token_sets[i],
                                prior_temperature=pt,
                                likelihood_temperature=lt)
            h.compute_posterior([FunctionData(input=[], output=self.zoo, alpha=1-1e-8)])
            hypotheses.append(h)
        self.hypotheses  = sorted(hypotheses, key=lambda h: -h.posterior_score)
        self.log_weights = log_softmax([h.posterior_score for h in self.hypotheses])
        self.N_TOKENS = self.hypotheses[0].N_TOKENS

    def get_valid_tokens(self, topn=None):
        if topn is None:
            topn = len(self.hypotheses)
        return list(set(x for h in self.hypotheses[:topn] for x in h._all_tokens))

    def get_all_tokens(self):
        valid = has_(self.n_att, PRIMS)

        out = set()
        for tk in valid:
            # letters = "".join(re.split("[^a-zA-Z]*", tk))
            letters = "".join(filter(str.isalpha, tk))

            if len(letters) == 1:
                out.add(tk)

            elif len(letters) == 2:
                p1 = self.prims[PRIMS.index(letters[0])]
                p2 = self.prims[PRIMS.index(letters[1])]
                att = int(tk[2:])

                if att in all_possible_shapes[p1][p2]:
                    out.add(tk)

            elif len(letters) == 3:
                p1 = self.prims[PRIMS.index(letters[0])]
                p2 = self.prims[PRIMS.index(letters[1])]
                p3 = self.prims[PRIMS.index(letters[2])]

                if tk.split('+')[0][3] == letters[2]:
                    att1 = int(tk[2])
                    att2 = int(tk[4:])
                else:
                    att1 = int(tk[2:4])
                    att2 = int(tk[5:])

                if att1 in all_possible_shapes[p1][p2]:
                    if att2 in all_possible_shapes[p1][p2][att1][p3]:
                        out.add(tk)

        out = all_rot_(out)

        return sorted(out)

    @property
    def all_tokens(self):
        if not hasattr(self, '_all_tokens'):
            self._all_tokens = self.get_all_tokens()
        return self._all_tokens

    def sample(self, n=1, ALPHA=1.0):
        weights = np.exp(self.log_weights)
        samples = []
        for i in range(n):
            if random.random() < ALPHA:
                h = random.choices(self.hypotheses, weights, k=1)[0]
                token = h.sample()
            else:
                token = random.choice(self.all_tokens)
            samples.append(token)

        return samples

    def log_prob(self, samples, ALPHA=1.0, image=True):
        if isinstance(samples, str):
            return self.log_prob([samples], ALPHA, image)

        for tkn in samples:
            if tkn not in token_to_ind:
                warnings.warn(f"Unknown token encountered by {type(self).__name__}.")
                return - np.inf

        num_tokens = self.N_TOKENS if image else len(token_to_img_dict[self.prims])
        base_prob = (1 - ALPHA) / num_tokens
        scores = np.full((len(samples), len(self.hypotheses)), base_prob)
        # tvec = toindex(samples)
        # for j, h in enumerate(self.hypotheses):
        #     h_size = h.h_size if image else h._all_tokens_index.sum()
        #     c = np.dot(h._all_tokens_index, tvec)
        #     scores[:int(c), j] += ALPHA / h_size
        ti = np.array([token_to_ind[tkn] for tkn in samples], dtype=int)
        for j, h in enumerate(self.hypotheses):
            h_size = h.h_size if image else h._all_tokens_index.sum()
            scores[:, j] += h._all_tokens_index[ti] * (ALPHA / h_size)

        scores = np.log(scores, out=scores)
        scores += self.log_weights
        scores = logsumexp(scores, axis=1)
        loglikelihood = np.sum(scores)

        return loglikelihood



if __name__ == '__main__':
    from ..bayes_utility.grammar import get_grammar

    topn = 500  # number of top hypotheses to use

    thetas = {
        'EXPR':   [0.00331080, 0.00217693, 0.99451227],
        'NOMAP':  [0.29927610, 0.70072391],
        'ATT_SP': [0.89487270, 0.10512730],
        'FUNC':   [0.81556999, 0.18443001],
        'SET':    [0.99927838, 0.00072162],
        'L_INV':  [0.00008303, 0.99991697],
    }
    alpha = 0.90336106
    pt = 2.34237384
    lt = 12.00470250

    print('initializing grammar...')
    grammar = get_grammar(thetas, n_att=18)

    print('loading data...')
    with open(os.path.join(ROOT, 'trial_data.pkl'), 'rb') as f:
        trial_data = pickle.load(f)

    with open(os.path.join(ROOT, 'reweighted_h.pkl'), 'rb') as f:
        h_values = pickle.load(f)

    print('initializing posterior predictive...')
    _, zoo_id, tokens, prims, seed = trial_data[5]
    h_vals = h_values[5][:topn]
    pp = PosteriorPredictive(grammar, h_vals, prims, trial=zoo_id, pt=pt, lt=lt)

    print('sampling from posterior predictive...')
    random.seed(293581)
    samples = pp.sample(20, ALPHA=alpha)
    print(f'sample:\n{samples}')
    lp = pp.log_prob(samples, ALPHA=alpha)
    print(f'logprob: {lp:.4f}')
