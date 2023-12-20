import re
import numpy as np
from LOTlib3.Grammar import Grammar
from itertools import combinations_with_replacement, product
from LOTlib3.Miscellaneous import qq

from .mixin import GrammarMixin


all_prims = ['A','B','C','D']


# ---------------------------------------------------------------------
#                    Primitive functions
# ---------------------------------------------------------------------

def custom_sort_(x):
    singles = [item for item in x if len(item) == 1]
    non_singles = [item for item in x if len(item) != 1]
    return sorted(non_singles)+sorted(singles)


def rot_(x, a):
    try:
        return x+'+'+str(a)
    except:
        return set()


def all_rot_(x):
    if isinstance(x, set):
        out = []
        for item in x:
            out += [item+'+'+str(angle) for angle in [0,90,180,270]]
        return set(out)
    else:
        return {x+'+'+str(angle) for angle in [0,90,180,270]}


def att_parts_(n_att, *x):
    x = custom_sort_(x)
    k = sum(list(map(lambda x: 1*(len(x)>1), x)))
    if k>1:
        return set()
    elif k==1:
        if len(x)>2:
            return set()
        else:
            return {x[0]+x[1]+str(att) for att in range(n_att)}
    else:
        if len(x) > 3:
            return set()
        elif len(x)==2:
            return {x[0]+x[1]+str(att) for att in range(n_att)}
        elif len(x)==3:
            out = []
            combos = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
            for c in combos:
                for att2 in range(n_att):
                    out.extend([x[c[0]]+x[c[1]]+str(att1)+x[c[2]]+str(att2) for att1 in range(n_att)])
            return list(set(out))


def att_(n_att,*x):
    out = []
    parts = []
    for item in x:
        if isinstance(item, set):
            parts.append(item)
        else:
            parts.append({item})
    for tup in product(*parts):
        out.extend(att_parts_(n_att,*tup))
    return set(out)


def atts_(x1, x2, x3):
    for item in [x1, x2]:
        if isinstance(item, set):
            return set()
    x = custom_sort_([x1, x2])
    k = sum(list(map(lambda x: len(x), x)))
    if k < 5:
        return x[0] + x[1] + str(x3)
    else:
        return set()


def Sigma_():
    return set(['A','B','C','D'])


def mymapset_(f,A):
    out = []
    for a in A:
        temp = f(a)
        if isinstance(temp, set):
            temp = list(temp)
        else:
            temp = [temp]
        out.extend(temp)
    return set(out)


def myset_(*args):
    if len(args) == 1 and isinstance(args[0], set):
        return args[0]
    else:
        out = set()
        try:
            for a in args:
                out.add(a)
            return out
        except:
            return out


def _has_(n_att, x):
    out = set()
    for part in x:
        temp = []
        if len(part)==1:
            temp.append(part)
            for part2 in Sigma_():
                temp.extend(list(att_(n_att,part,part2)))
            for combo in list(combinations_with_replacement(Sigma_(),3)):
                if part in combo:
                    temp.extend(list(att_(n_att,*combo)))
        elif len(part)==3:
            temp.append(part)
            for part2 in Sigma_():
                temp.extend(list(att_(n_att,part,part2)))
        elif len(part)==5:
            temp.append(part)
        temp = set(temp)

        if len(out)> 0:
            out = out.intersection(temp)
        else:
            out = temp
    return out


def has_(n_att,x):
    out = set()
    for item in x:
        if len(out)>0:
            out = out.union(_has_(n_att,item))
        else:
            out = _has_(n_att,item)
    return out


def _only_(n_att,x):
    def helper(x, l):
        for item in l:
            if item in x:
                return False
        return True

    temp = _has_(n_att, x)

    if len(temp) == 0:
        return set()
    k = np.sum(list((map(lambda i: len(i)>1, x))))


    if k > 1:
        return set()
    elif k == 1:
        if len(x) > 2:
            return set()
        elif len(x) == 2:
            out = []
            tempsigma = Sigma_().difference(custom_sort_(x)[1])

            for item in temp:
                templist = item.split(custom_sort_(x)[0])
                k2 = np.sum(list((map(lambda i: helper(i, tempsigma), templist))))
                if k2 == len(templist):
                    out.append(item)
            return set(out)

        else:
            return set(x)
    else:
        out = []

        for item in temp:
            k = 0
            for char in Sigma_().difference(set(x)):
                if char not in item:
                    k+=1
                if k == len(Sigma_().difference(set(x))):
                    out.append(item)

        return set(out)


def only_(n_att,x):
    out = set()
    for item in x:
        if len(out)>0:
            out = out.union(_only_(n_att,item))
        else:
            out = _only_(n_att,item)
    return out


def mydiff_(S,p):
    if p == set():
        return S
    else:
        return S.difference({p})


def myor_(a,b):
    if isinstance(a, set):
        if isinstance(b, set):
            return set()
        else:
            return b

    if isinstance(b, set):
        return a

    if a == b:
        return a

    if re.search(r"U", b):
        return '('+a+'U'+b[1:-1]+')'
    else:
        return '('+a+'U'+b+')'


def myand_(a,b):
    if isinstance(a, set):
        if isinstance(b, set):
            return set()
        else:
            return b

    if isinstance(b, set):
        return a

    if a == b:
        return a

    if re.search(r"U", b):
        temp = b[1:-1].split("U")
        out = myand_(a,temp[0])
        for item in temp[1:]:
            out = out+'U'+myand_(a,item)
        return '('+out+')'
    elif re.search(r"N", b):
        return '('+a+'N'+b[1:-1]+')'
    else:
        return '('+a+'N'+b+')'


def push(obj, l, depth):
    while depth:
        l = l[-1]
        depth -= 1

    l.append(obj)


def parse_parentheses(s):
    groups = []
    depth = 0

    try:
        for char in s:
            if char == '(':
                push([], groups, depth)
                depth += 1
            elif char == ')':
                depth -= 1
            else:
                push(char, groups, depth)
    except IndexError:
        raise ValueError('Parentheses mismatch')

    if depth > 0:
        raise ValueError('Parentheses mismatch')
    else:
        return groups


def parse_list(l, out):
    sstr = ''

    for item in l:
        if isinstance(item, list):
            parse_list(item, out)
        else:
            sstr+=item
    sstr = sstr.split('U')
    if len(sstr)>0:
        for j in sstr:
            if len(j)>0:
                out.append(j.split('N'))


def mystr_(s):
    l = parse_parentheses(s)
    out = []
    parse_list(l, out)
    return out


# ---------------------------------------------------------------------
#                    Grammar class
# ---------------------------------------------------------------------


class SVCGrammarV2(Grammar, GrammarMixin):

    primitive_funcs = [
        custom_sort_, rot_, all_rot_, att_parts_, att_, atts_, Sigma_,
        mymapset_, myset_, _has_, _only_, has_, only_, mydiff_, myor_, myand_,
        push, parse_parentheses, parse_list, mystr_
    ]

    def __init__(self, thetas=None, n_att=4):
        super().__init__(start='EXPR')

        if thetas is None:
            # default thetas
            thetas = {
                'EXPR': [1., 1., 1.],
                'NOMAP': [1., 1.],
                'ATT_SP': [1., 1.],
                'FUNC': [1., 1.],
                'SET': [2., 1.],
                'L_INV': [1., 1.]
            }

        # variable number of primitives
        self.add_rule('EXPR', 'all_rot_(%s)', ['L_INV'], thetas['EXPR'][0])
        # concepts with lambda variables
        self.add_rule('EXPR', 'all_rot_(%s)', ['MAP'], thetas['EXPR'][1])
        # concepts without lambda variables
        self.add_rule('EXPR', '', ['NOMAP'], thetas['EXPR'][2])

        # attachment invariant
        self.add_rule('NOMAP', 'all_rot_(%s)', ['ATTACH_ALL'], thetas['NOMAP'][0])
        # attachment specific
        self.add_rule('NOMAP', '', ['ATT_SP'], thetas['NOMAP'][1])

        # rotation invariant
        self.add_rule('ATT_SP', 'all_rot_(%s)', ['FIXED'], thetas['ATT_SP'][0])
        # rotation specific
        self.add_rule('ATT_SP', 'myset_(rot_(%s, %s))', ['FIXED','ANGLE'],  thetas['ATT_SP'][1])

        self.add_rule('ATTACH_ALL', 'att_(%s, %s, %s)', ['N_ATT', 'FIXED','STR'], 1.0)
        self.add_rule('STR', '%s, %s', ['FIXED', 'STR'], 1.0)
        self.add_rule('STR', '%s', ['FIXED'], 10.0)

        self.add_rule('FIXED', '', ['PRIM'], 5.0)
        self.add_rule('FIXED', '', ['ATTACH*'], 1.0)

        self.add_rule('MAP', 'mymapset_(%s, %s)', ['LAMBDAFUNC', 'SET'], 1.0)
        self.add_rule('LAMBDAFUNC', 'lambda', ['FUNC'], 1.0,
                      bv_type='PART', bv_prefix='p')

        self.add_rule('FUNC', '', ['ATTACH_ALL*'], thetas['FUNC'][0])
        self.add_rule('FUNC', '', ['PART'], thetas['FUNC'][1])

        self.add_rule('ATTACH_ALL*', 'att_(%s, %s, %s)', ['N_ATT', 'PART','STR2'], 1.0)
        self.add_rule('STR2', '%s, %s', ['PART', 'STR2'], 1.0)
        self.add_rule('STR2', '%s', ['PART'], 10.0)

        self.add_rule('PART', '', ['FIXED'], 1.0)

        self.add_rule('SET', 'Sigma_()', None, thetas['SET'][0])
        self.add_rule('SET', 'mydiff_(%s, %s)', ['SET', 'PART'], thetas['SET'][1])

        # concept that requires having some set of parts
        self.add_rule('L_INV', 'has_(%s, mystr_(%s))', ['N_ATT','STR3'], thetas['L_INV'][0])
        # concept that requires only some set of parts
        self.add_rule('L_INV', 'only_(%s, mystr_(%s))', ['N_ATT','STR3'], thetas['L_INV'][1])

        self.add_rule('STR3', '%s(%s, %s)', ['OPT', 'FIXED', 'STR3'], 1.0)
        self.add_rule('STR3', '%s', ['PART'], 10.0)

        self.add_rule('ATTACH*', 'atts'+'_(%s, %s, %s)', ['FIXED','FIXED','ATT_N'], 1.0)

        self.add_rule('OPT', 'myor_',  None, 1.0)
        self.add_rule('OPT', 'myand_', None, 1.0)

        for n in range(n_att):
            self.add_rule('ATT_N', str(n) , None, 1.0)

        for angle in [0,90,180,270]:
            self.add_rule('ANGLE', str(angle) , None, 1.0)

        # shape primatives
        for n in range(4):
            self.add_rule('PRIM', qq(all_prims[n]), None, 1.0)

        self.add_rule('N_ATT', str(n_att), None, 1.0)


        self._register_primitives()