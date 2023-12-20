import re
import numpy as np
from LOTlib3.Grammar import Grammar
from LOTlib3.Eval import register_primitive
from itertools import combinations_with_replacement, product, permutations
from LOTlib3.Miscellaneous import qq

################################### Define Grammar ######################################
# all_prims = ['A','B','C','D']
# grammar = Grammar(start='EXPR')

# grammar.add_rule('EXPR', '', ['L_SP'],  5.0) # specific number of primitives
# grammar.add_rule('EXPR', 'all_rot_(%s)', ['L_INV'], 1.0) # variable number of primitives

# grammar.add_rule('L_SP', 'all_rot_(%s)', ['MAP'], 1.0) # concepts with lambda variables
# grammar.add_rule('L_SP', '', ['NOMAP'], 1.0) # concepts without lambda variables

# grammar.add_rule('NOMAP', 'all_rot_(%s)', ['ATTACH_ALL'], 1.0) # attachment invariant
# grammar.add_rule('NOMAP', '', ['ATT_SP'], 1.0) # attachment specific

# grammar.add_rule('ATT_SP', '', ['ROT_INV'], 1.0) # rotation invariant
# grammar.add_rule('ATT_SP', '', ['ROTATE'], 1.0) # rotation specific

# grammar.add_rule('ROT_INV', 'all_rot_(%s)', ['FIXED'], 1.0)

# grammar.add_rule('ATTACH_ALL', 'att_(%s, %s)', ['PART','STR'], 1.0)
# grammar.add_rule('STR', '%s, %s', ['PART', 'STR'], 1.0)
# grammar.add_rule('STR', '%s', ['PART'], 10.0)

# grammar.add_rule('PART', '', ['FIXED'], 1.0)

# grammar.add_rule('FIXED', '', ['PRIM'], 10.0)
# grammar.add_rule('FIXED', '', ['ATTACH*'], 1.0)

# grammar.add_rule('MAP', 'mymapset_(%s, %s)', ['LAMBDAFUNC', 'SET'], 1.0)
# grammar.add_rule('LAMBDAFUNC', 'lambda', ['FUNC'], 1.0, bv_type='PART', bv_prefix='p')

# grammar.add_rule('FUNC', '', ['ATTACH_ALL'], 1.0)
# grammar.add_rule('FUNC', '', ['PART'], 1.0)

# grammar.add_rule('SET', 'Sigma_()', None, 1.0)
# grammar.add_rule('SET', 'mydiff_(%s, %s)', ['SET', 'PART'], 1.0)

# grammar.add_rule('L_INV', 'has_(myset_(%s))', ['SET2'], 1.0)  # concept that requires having some set of parts
# grammar.add_rule('L_INV', 'only_(myset_(%s))', ['SET2'], 1.0) # concept that requires only some set of parts

# grammar.add_rule('SET2', '%s', ['PART'], 10.0)
# grammar.add_rule('SET2', '%s, %s', ['PART', 'SET2'], 1.0)

# for i,n in enumerate([0,90,180,270]):
#     grammar.add_rule('ROTATE', 'myset_(rot'+str(n)+'_(%s))' , ['FIXED'], 1.0)

# for n in range(4):
#     grammar.add_rule('ATTACH*', 'att'+str(n)+'_(%s, %s)', ['FIXED','FIXED'], 1.0) 

# # shape primatives
# for n in range(4):
#     grammar.add_rule('PRIM', qq(all_prims[n]), None, 1.0)

################################### Define Grammar ######################################

def get_grammar(thetas, n_att = 4):
    all_prims = ['A','B','C','D']
    grammar = Grammar(start='EXPR')

    grammar.add_rule('EXPR', 'all_rot_(%s)', ['L_INV'], thetas['EXPR'][0]) # variable number of primitives
    grammar.add_rule('EXPR', 'all_rot_(%s)', ['MAP'], thetas['EXPR'][1]) # concepts with lambda variables
    grammar.add_rule('EXPR', '', ['NOMAP'], thetas['EXPR'][2]) # concepts without lambda variables

    grammar.add_rule('NOMAP', 'all_rot_(%s)', ['ATTACH_ALL'], thetas['NOMAP'][0]) # attachment invariant
    grammar.add_rule('NOMAP', '', ['ATT_SP'], thetas['NOMAP'][1]) # attachment specific

    grammar.add_rule('ATT_SP', 'all_rot_(%s)', ['FIXED'], thetas['ATT_SP'][0]) # rotation invariant
    grammar.add_rule('ATT_SP', 'myset_(rot_(%s, %s))', ['FIXED','ANGLE'],  thetas['ATT_SP'][1]) # rotation specific

    grammar.add_rule('ATTACH_ALL', 'att_(%s, %s, %s)', ['N_ATT', 'FIXED','STR'], 1.0)
    grammar.add_rule('STR', '%s, %s', ['FIXED', 'STR'], 1.0)
    grammar.add_rule('STR', '%s', ['FIXED'], 10.0)
    

    grammar.add_rule('FIXED', '', ['PRIM'], 5.0)
    grammar.add_rule('FIXED', '', ['ATTACH*'], 1.0)

    grammar.add_rule('MAP', 'mymapset_(%s, %s)', ['LAMBDAFUNC', 'SET'], 1.0)
    grammar.add_rule('LAMBDAFUNC', 'lambda', ['FUNC'], 1.0, bv_type='PART', bv_prefix='p')

    grammar.add_rule('FUNC', '', ['ATTACH_ALL*'], thetas['FUNC'][0])
    grammar.add_rule('FUNC', '', ['PART'], thetas['FUNC'][1])
    
    grammar.add_rule('ATTACH_ALL*', 'att_(%s, %s, %s)', ['N_ATT', 'PART','STR2'], 1.0)
    grammar.add_rule('STR2', '%s, %s', ['PART', 'STR2'], 1.0)
    grammar.add_rule('STR2', '%s', ['PART'], 10.0)

    grammar.add_rule('PART', '', ['FIXED'], 1.0)

    grammar.add_rule('SET', 'Sigma_()', None, thetas['SET'][0])
    grammar.add_rule('SET', 'mydiff_(%s, %s)', ['SET', 'PART'], thetas['SET'][1])

    grammar.add_rule('L_INV', 'has_(%s, mystr_(%s))', ['N_ATT','STR3'], thetas['L_INV'][0])  # concept that requires having some set of parts
    grammar.add_rule('L_INV', 'only_(%s, mystr_(%s))', ['N_ATT','STR3'], thetas['L_INV'][1]) # concept that requires only some set of parts

    grammar.add_rule('STR3', '%s(%s, %s)', ['OPT', 'FIXED', 'STR3'], 1.0)
    grammar.add_rule('STR3', '%s', ['FIXED'], 10.0)
    
    grammar.add_rule('ATTACH*', 'atts'+'_(%s, %s, %s)', ['FIXED','FIXED','ATT_N'], 1.0) 
    
    grammar.add_rule('OPT', 'myor_',  None, 1.0) 
    grammar.add_rule('OPT', 'myand_', None, 1.0) 
    
    for n in range(n_att):
        grammar.add_rule('ATT_N', str(n) , None, 1.0)
    
    for angle in [0,90,180,270]:
        grammar.add_rule('ANGLE', str(angle) , None, 1.0)

    # shape primatives
    for n in range(4):
        grammar.add_rule('PRIM', qq(all_prims[n]), None, 1.0)
    
    grammar.add_rule('N_ATT', str(n_att), None, 1.0)
    
    return grammar
    


################################# Register primitives ###################################

def custom_sort_(x):
    singles = [item for item in x if len(item) == 1]
    non_singles = [item for item in x if len(item) != 1]
    return sorted(non_singles)+sorted(singles)
register_primitive(custom_sort_)
#########################################

def rot_(x, a):
    try:
        return x+'+'+str(a)
    except:
        return set()
register_primitive(rot_)

#########################################

def all_rot_(x):
    if isinstance(x, set):
        out = []
        for item in x:
            out += [item+'+'+str(angle) for angle in [0,90,180,270]]
        return set(out)
    else:
        return {x+'+'+str(angle) for angle in [0,90,180,270]}

register_primitive(all_rot_)  

#########################################

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

register_primitive(att_parts_)   
#########################################

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

register_primitive(att_)    

#########################################

def atts_(x1,x2,x3):
    for item in [x1,x2]:
        if isinstance(item, set):
            return set()
    x = custom_sort_([x1,x2])
    k = sum(list(map(lambda x: len(x), x)))
    if k<5:
        return x[0]+x[1]+str(x3)
    else:
        return set()

register_primitive(atts_) 

#########################################

def Sigma_():
    return set(['A','B','C','D'])
register_primitive(Sigma_) 

#########################################

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
register_primitive(mymapset_) 

#########################################

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
register_primitive(myset_) 


#########################################

def has_(n_att,x):
    out = set()
    for item in x:
        if len(out)>0:
            out = out.union(_has_(n_att,item))
        else:
            out = _has_(n_att,item)
    return out
register_primitive(has_) 

#########################################

def only_(n_att,x):
    out = set()
    for item in x:
        if len(out)>0:
            out = out.union(_only_(n_att,item))
        else:
            out = _only_(n_att,item)
    return out
register_primitive(only_) 

#########################################

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
#     if len(out) > 0:
#         out.extend('p'+str(i) for i in range(pk))
    return out

register_primitive(_has_) 
    
#########################################
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
    
register_primitive(_only_)  

#########################################
def mydiff_(S,p):
    if p == set():
        return S
    else:
        return S.difference({p})
register_primitive(mydiff_) 

#########################################
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
    
register_primitive(myor_) 

#########################################
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
#         return '(('+a+' AND '+temp[0]+') OR ('+a+' AND '+temp[1]+'))'
        out = myand_(a,temp[0])
        for item in temp[1:]:
            out = out+'U'+myand_(a,item)
        return '('+out+')'
    elif re.search(r"N", b):
        return '('+a+'N'+b[1:-1]+')'
    else:
        return '('+a+'N'+b+')'

    
register_primitive(myand_) 
#########################################

def push(obj, l, depth):
    while depth:
        l = l[-1]
        depth -= 1

    l.append(obj)
    
register_primitive(push) 

#########################################

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
    
register_primitive(parse_parentheses) 

#########################################

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
                
register_primitive(parse_list) 

#########################################

def mystr_(s):
    l = parse_parentheses(s)
    out = []
    parse_list(l, out)
    return out
                
register_primitive(mystr_) 
#########################################

# pk = 50

# def custom_sort_(x):
#     singles = [item for item in x if len(item) == 1]
#     non_singles = [item for item in x if len(item) != 1]
#     return sorted(non_singles)+sorted(singles)
# register_primitive(custom_sort_)
# #########################################

# def rot0_(x):
#     try:
#         return x+'+0'
#     except:
#         return set()
# register_primitive(rot0_)

# #########################################

# def rot90_(x):
#     try:
#         return x+'+90'
#     except:
#         return set()
# register_primitive(rot90_)

# #########################################    

# def rot180_(x):
#     try:
#         return x+'+180'
#     except:
#         return set()
# register_primitive(rot180_)

# #########################################

# def rot270_(x):
#     try:
#         return x+'+270'
#     except:
#         return set()

# register_primitive(rot270_)

# #########################################

# def all_rot_(x):
#     if isinstance(x, set):
#         out = []
#         for item in x:
#             out += [item+'+'+str(angle) for angle in [0,90,180,270]]
#         return set(out)
#     else:
#         return {x+'+'+str(angle) for angle in [0,90,180,270]}

# register_primitive(all_rot_) 

# #########################################
# def all_att_(x,y):
#     return [x+y+str(att) for att in range(4)]

# register_primitive(all_att_) 

# def att_parts_(*x):
#     x = custom_sort_(x)
# #     k = sum(list(map(lambda x: 1*(len(x)>1), x)))
#     k = sum(list(map(lambda x: len("".join(re.findall("[A-Z]+", x))), x)))
#     if k>4:
#         return set()
#     elif k==4:
#         if len(x) == 2:
#             return list(set(all_att_(x[0],x[1])))
#         elif len(x) == 3:
#             out = []
#             for i in range(1,3):
#                 temp = all_att_(x[0],x[i])
#                 for s in temp:
#                     out.extend(all_att_(s,x[3-i]))
#             return list(set(out))
#         elif len(x) == 4:
#             out = []
#             combos = [x for x in list(permutations([0,1,2,3])) if x[0]<x[1]]
#             for c in combos:
#                 temp1 = all_att_(x[c[0]],x[c[1]])
#                 for s1 in temp1:
#                     temp2 = all_att_(s1,x[c[2]])
#                     for s2 in temp2:
#                         out.extend(all_att_(s2,x[c[3]]))
#             return list(set(out))
#         else:
#             return set()                               
#     elif k==3:
#         if len(x)==2:
#             return list(set(all_att_(x[0],x[1])))
#         elif len(x) == 3:
#             out = []
#             combos = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
#             for c in combos:
#                 temp1 = all_att_(x[c[0]],x[c[1]])      
#                 for s1 in temp1:
#                     out.extend(all_att_(s1,x[c[2]]))
#             return list(set(out))
#         else:
#             return set()                               
                                   
#     else:
#         return list(set(all_att_(x[0],x[1])))

# register_primitive(att_parts_) 

# #########################################

# def att_(*x):
#     out = []
#     parts = []
#     for item in x:
#         if isinstance(item, set):
#             parts.append(item)
#         else:
#             parts.append({item})
#     for tup in product(*parts):
#         out.extend(att_parts_(*tup))
#     return set(out)

# register_primitive(att_)          
# #########################################

# def att0_(*x):
#     for item in x:
#         if isinstance(item, set):
#             return set()
#     x = custom_sort_(x)
#     k = sum(list(map(lambda x: len(x), x)))
#     if k<7:
#         return x[0]+x[1]+"0"
#     else:
#         return set()

# register_primitive(att0_) 
# #########################################

# def att1_(*x):
#     for item in x:
#         if isinstance(item, set):
#             return set()
#     x = custom_sort_(x)
#     k = sum(list(map(lambda x: len(x), x)))
#     if k<7:
#         return x[0]+x[1]+"1"
#     else:
#         return set()

# register_primitive(att1_) 
# #########################################

# def att2_(*x):
#     for item in x:
#         if isinstance(item, set):
#             return set()
#     x = custom_sort_(x)
#     k = sum(list(map(lambda x: len(x), x)))
#     if k<7:
#         return x[0]+x[1]+"2"
#     else:
#         return set()

# register_primitive(att2_) 
# #########################################

# def att3_(*x):
#     for item in x:
#         if isinstance(item, set):
#             return set()
#     x = custom_sort_(x)
#     k = sum(list(map(lambda x: len(x), x)))
#     if k<7:
#         return x[0]+x[1]+"3"
#     else:
#         return set()
    
# register_primitive(att3_) 
# #########################################

# def Sigma_():
#     return set(['A','B','C','D'])
# register_primitive(Sigma_) 

# #########################################

# def mymapset_(f,A):
#     out = []
#     for a in A:
#         temp = f(a)
#         if isinstance(temp, set):
#             temp = list(temp)
#         else:
#             temp = [temp]
#         out.extend(temp)
#     return set(out)
# register_primitive(mymapset_) 

# #########################################

# def myset_(*args):
#     if len(args) == 1 and isinstance(args[0], set):
#         return args[0]
#     else:
#         out = set()
#         try:
#             for a in args:
#                 out.add(a)
#             return out
#         except:
#             return out
# register_primitive(myset_) 
# #########################################

# # def all_combo_(x):
# #     out = []
# #     for tup in [(x,x), (x,x,x)]:
# #         out.extend(att_parts_(*tup))
# #     out = set(out)
# #     out = out.union({x})
# #     return out
        
# # register_primitive(all_combo_) 
# #########################################

# def has_(x):
#     out = []
#     for part in x:
#         if len(part)==1:
#             out.append(part)
#             for part2 in Sigma_():
#                 out.extend(list(att_(part,part2)))
#             for combo in list(combinations_with_replacement(Sigma_(),3)):
#                 if part in combo:
#                     out.extend(list(att_(*combo)))
#             for combo in list(combinations_with_replacement(Sigma_(),4)):
#                 if part in combo:
#                     out.extend(list(att_(*combo)))
#         elif len(part)==3:
#             out.append(part)
#             for part2 in Sigma_():
#                 out.extend(list(att_(part,part2)))
#             for combo in list(combinations_with_replacement(Sigma_(),2)):
#                 out.extend(list(att_(part,combo[0],combo[1])))            
#         elif len(part)==5:
#             out.append(part)
#             for part2 in Sigma_():
#                 out.extend(list(att_(part,part2)))
#         elif len(part)==7:
#             out.append(part)
#     if len(out) > 0:
#         out.extend('p'+str(i) for i in range(pk))
#     return set(out)

# register_primitive(has_) 
    
# #########################################
# def only_(x):
#     out = []
#     for part in x:
#         out.append(part)
#     for combo in list(combinations_with_replacement(x,2)):
#         out.extend(list(att_(*combo)))
#     for combo in list(combinations_with_replacement(x,3)):
#         out.extend(list(att_(*combo)))
#     for combo in list(combinations_with_replacement(x,4)):
#         out.extend(list(att_(*combo)))
#     if len(out) > 0:    
#         out.extend('p'+str(i) for i in range(pk))
#     return set(out)
# register_primitive(only_) 

# #########################################
# def mydiff_(S,p):
#     if p == set():
#         return S
#     else:
#         return S.difference({p})
# register_primitive(mydiff_) 

