from LOTlib3.Grammar import Grammar
from itertools import combinations_with_replacement, product
from LOTlib3.Miscellaneous import qq

from .mixin import GrammarMixin


all_prims = ['A','B','C','D']
pk = 50


# ---------------------------------------------------------------------
#                    Primitive functions
# ---------------------------------------------------------------------

def custom_sort_(x):
    singles = [item for item in x if len(item) == 1]
    non_singles = [item for item in x if len(item) != 1]
    return sorted(non_singles) + sorted(singles)


def rot0_(x):
    try:
        return x + '+0'
    except:
        return set()


def rot90_(x):
    try:
        return x+'+90'
    except:
        return set()


def rot180_(x):
    try:
        return x+'+180'
    except:
        return set()


def rot270_(x):
    try:
        return x+'+270'
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


def att_parts_(*x):
    x = custom_sort_(x)
    k = sum(list(map(lambda x: 1*(len(x)>1), x)))
    if k>1:
        return set()
    elif k==1:
        if len(x)>2:
            return set()
        else:
            return {x[0]+x[1]+str(att) for att in range(4)}
    else:
        if len(x) > 3:
            return set()
        elif len(x)==2:
            return {x[0]+x[1]+str(att) for att in range(4)}
        elif len(x)==3:
            out = []
            combos = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
            for c in combos:
                for att2 in range(4):
                    out.extend([x[c[0]]+x[c[1]]+str(att1)+x[c[2]]+str(att2) for att1 in range(4)])
            return list(set(out))


def att_(*x):
    out = []
    parts = []
    for item in x:
        if isinstance(item, set):
            parts.append(item)
        else:
            parts.append({item})
    for tup in product(*parts):
        out.extend(att_parts_(*tup))
    return set(out)


def att0_(*x):
    for item in x:
        if isinstance(item, set):
            return set()
    x = custom_sort_(x)
    k = sum(list(map(lambda x: len(x), x)))
    if k<5:
        return x[0]+x[1]+'0'
    else:
        return set()


def att1_(*x):
    for item in x:
        if isinstance(item, set):
            return set()
    x = custom_sort_(x)
    k = sum(list(map(lambda x: len(x), x)))
    if k<5:
        return x[0]+x[1]+'1'
    else:
        return set()


def att2_(*x):
    for item in x:
        if isinstance(item, set):
            return set()
    x = custom_sort_(x)
    k = sum(list(map(lambda x: len(x), x)))
    if k<5:
        return x[0]+x[1]+'2'
    else:
        return set()


def att3_(*x):
    for item in x:
        if isinstance(item, set):
            return set()
    x = custom_sort_(x)
    k = sum(list(map(lambda x: len(x), x)))
    if k<5:
        return x[0]+x[1]+'3'
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


def has_(x):
    out = []
    for part in x:
        if len(part)==1:
            out.append(part)
            for part2 in Sigma_():
                out.extend(list(att_(part,part2)))
            for combo in list(combinations_with_replacement(Sigma_(),3)):
                if part in combo:
                    out.extend(list(att_(*combo)))
        elif len(part)==3:
            out.append(part)
            for part2 in Sigma_():
                out.extend(list(att_(part,part2)))
        elif len(part)==5:
            out.append(part)
    if len(out) > 0:
        out.extend('p'+str(i) for i in range(pk))
    return set(out)


def only_(x):
    out = []
    for part in x:
        out.append(part)
    for combo in list(combinations_with_replacement(x,2)):
        out.extend(list(att_(*combo)))
    for combo in list(combinations_with_replacement(x,3)):
        out.extend(list(att_(*combo)))
    if len(out) > 0:
        out.extend('p'+str(i) for i in range(pk))
    return set(out)


def mydiff_(S,p):
    if p == set():
        return S
    else:
        return S.difference({p})


# ---------------------------------------------------------------------
#                    Grammar class
# ---------------------------------------------------------------------


class SVCGrammar(Grammar, GrammarMixin):

    primitive_funcs = [
        custom_sort_, rot0_, rot90_, rot180_, rot270_, all_rot_, att_parts_,
        att_, att0_, att1_, att2_, att3_, Sigma_, mymapset_, myset_, has_,
        only_, mydiff_
    ]

    def __init__(self):
        super().__init__(start='EXPR')

        # specific number of primitives
        self.add_rule('EXPR', '', ['L_SP'],  5.0)
        # variable number of primitives
        self.add_rule('EXPR', 'all_rot_(%s)', ['L_INV'], 1.0)

        # concepts with lambda variables
        self.add_rule('L_SP', 'all_rot_(%s)', ['MAP'], 1.0)
        # concepts without lambda variables
        self.add_rule('L_SP', '', ['NOMAP'], 1.0)

        # attachment invariant
        self.add_rule('NOMAP', 'all_rot_(%s)', ['ATTACH_ALL'], 1.0)
        # attachment specific
        self.add_rule('NOMAP', '', ['ATT_SP'], 1.0)

        # rotation invariant
        self.add_rule('ATT_SP', '', ['ROT_INV'], 1.0)
        # rotation specific
        self.add_rule('ATT_SP', '', ['ROTATE'], 1.0)

        self.add_rule('ROT_INV', 'all_rot_(%s)', ['FIXED'], 1.0)

        self.add_rule('ATTACH_ALL', 'att_(%s, %s)', ['PART','STR'], 1.0)
        self.add_rule('STR', '%s, %s', ['PART', 'STR'], 1.0)
        self.add_rule('STR', '%s', ['PART'], 10.0)

        self.add_rule('PART', '', ['FIXED'], 1.0)

        self.add_rule('FIXED', '', ['PRIM'], 10.0)
        self.add_rule('FIXED', '', ['ATTACH*'], 1.0)

        self.add_rule('MAP', 'mymapset_(%s, %s)', ['LAMBDAFUNC', 'SET'], 1.0)
        self.add_rule('LAMBDAFUNC', 'lambda', ['FUNC'], 1.0,
                      bv_type='PART', bv_prefix='p')

        self.add_rule('FUNC', '', ['ATTACH_ALL'], 1.0)
        self.add_rule('FUNC', '', ['PART'], 1.0)

        self.add_rule('SET', 'Sigma_()', None, 1.0)
        self.add_rule('SET', 'mydiff_(%s, %s)', ['SET', 'PART'], 1.0)

        # concept that requires having some set of parts
        self.add_rule('L_INV', 'has_(myset_(%s))', ['SET2'], 1.0)
        # concept that requires only some set of parts
        self.add_rule('L_INV', 'only_(myset_(%s))', ['SET2'], 1.0)

        self.add_rule('SET2', '%s', ['PART'], 10.0)
        self.add_rule('SET2', '%s, %s', ['PART', 'SET2'], 1.0)

        for i, n in enumerate([0,90,180,270]):
            self.add_rule('ROTATE', f'myset_(rot{n}_(%s))', ['FIXED'], 1.0)

        for n in range(4):
            self.add_rule('ATTACH*', f'att{n}_(%s, %s)', ['FIXED','FIXED'], 1.0)

        # shape primatives
        for n in range(4):
            self.add_rule('PRIM', qq(all_prims[n]), None, 1.0)

        self._register_primitives()


