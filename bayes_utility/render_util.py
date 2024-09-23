import numpy as np
from itertools import combinations
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def map_seed_to_shapeid(seed_id):
    if seed_id == 0:
        return {0:[10,6],1:[11,19],2:[9,12],3:[5,10,3,6],
                4:[10,17,16],5:[16,13,3],6:[12,13,10],7:[7,18,20],
                8:[16,18,5],9:[8,3,6],10:[18,13,10],11:[4,16,13]}
    elif seed_id == 1:
        return {0:[1,5],1:[20,2],2:[7,6],3:[16,20,15,3],
                4:[7,2,10],5:[2,15,8],6:[0,9,18],7:[1,8,15],
                8:[19,12,18],9:[0,12,13],10:[10,7,6],11:[20,19,4]}
    elif seed_id == 2:
        return {0:[6,7],1:[1,19],2:[4,19],3:[11,2,14,18],
                4:[6,20,16],5:[8,0,19],6:[20,8,12],7:[11,9,14],
                8:[11,14,17],9:[12,8,4],10:[16,8,1],11:[11,12,20]}
    elif seed_id == 3:
        return {0:[10,13],1:[9,17],2:[1,10],3:[14,7,17,0],
                4:[7,19,20],5:[20,18,9],6:[7,6,17],7:[9,7,19],
                8:[17,10,20],9:[9,2,20],10:[20,12,11],11:[2,1,5]}
    elif seed_id == 4:
        return {0:[7,11],1:[11,8],2:[15,18],3:[10,13,18,7],
                4:[9,13,19],5:[9,4,7],6:[7,20,10],7:[16,11,1],
                8:[8,13,16],9:[5,0,14],10:[9,3,4],11:[7,20,14]}

def shift(raw):
    xs = [cord[1] for cord in raw]
    ys = [cord[2] for cord in raw]
    y_shift = -min(ys)
    x_shift = -min(xs)
    translated = raw.copy()
    for cord in translated:
        cord[1] += x_shift
        cord[2] += y_shift
        
    return translated

def render_vertices(cord):
    shape_id = cord[0]
    if shape_id == 0:
        vs = [(0,32),(0,0),(32,32)]
    elif shape_id == 1:
        vs = [(0,32),(0,0),(32,0)]
    elif shape_id == 2:
        vs = [(0,0),(32,0),(32,32)]
    elif shape_id == 3:
        vs = [(32,0),(32,32),(0,32)]
    vs = [(tup[0]+np.round(cord[1])*32,tup[1]+np.round(cord[2])*32) for tup in vs]
    return vs
    
def visualize_shapes(filled, color = [(113, 159, 235),(84, 146, 247)]):
    img = Image.new('RGB', (96,96), color=(0, 0, 0))
    drawer = ImageDraw.Draw(img)
    for shape in filled:
        vs = render_vertices(shape)
        drawer.polygon(vs, outline = color[0], fill=color[1])
    return img

def if_attached(tri1,tri2):
    x = tri1[1]
    y = tri1[2]
    if tri1[0] == 0:
        legals = [(2,x,y),(2,x,y+1),(2,x-1,y),(1,x,y+1),(3,x-1,y)]
    elif tri1[0] == 1:
        legals = [(3,x,y),(3,x,y-1),(3,x-1,y),(0,x,y-1),(2,x-1,y)]
    elif tri1[0] == 2:
        legals = [(0,x,y),(0,x,y-1),(0,x+1,y),(1,x+1,y),(3,x,y-1)]
    elif tri1[0] == 3:
        legals = [(1,x,y),(1,x+1,y),(1,x,y+1),(2,x,y+1),(0,x+1,y)]
    if tuple(tri2) in legals:
        return True       
    return False

def check_overlap(pairs):
    all_tris = np.array([[pair[0],pair[1]] for pair in pairs])
    all_tris = all_tris.reshape(-1,3)
    uniques = np.unique(all_tris,axis=0)
    combos = list(combinations(uniques,2))
    for combo in combos:
        s1 = combo[0]
        s2 = combo[1]
        if s1[1]==s2[1] and s1[2]==s2[2]:
            if np.abs(s1[0]-s2[0])!=2:
                return True
    return False
def get_combos(shape):
    connections = []
    tri_pairs = list(combinations(shape,2))
    for pair in tri_pairs:
        if if_attached(pair[0],pair[1]):
            connections.append(pair)
    return connections

def check_if_legal(shape):
    connections = []
    tri_pairs = list(combinations(shape,2))
    for pair in tri_pairs:
        if if_attached(pair[0],pair[1]):
            connections.append(pair)
    if len(connections)==3:
        if check_overlap(connections) is False:
            return True
    return False

def rotate_shape(shape,angle):
    ids = shape[:,0]
    vertices = np.array(shape[:,1:])*100.0
    vertices -= vertices.mean(axis=0)
    vertices = vertices+[1,1]
    theta = np.radians(angle)
    R = np.array([[np.cos(theta),np.sin(theta)],
                  [-np.sin(theta),np.cos(theta)]])
    new_vertices= np.dot(vertices,R)
    new_vertices= np.around(new_vertices, decimals=0)
    new_vertices= new_vertices/100
    if angle == 0:
        new_ids = ids
    elif angle == 90:
        new_ids = ids+1
    elif angle == 180:
        new_ids = ids+2
    elif angle == 270:
        new_ids = ids+3
    new_ids = np.mod(new_ids,4)
    
    rotated = np.zeros(shape.shape)
    for i in range(len(shape)):
        rotated[i,0] = new_ids[i]
        rotated[i,1:] = new_vertices[i]

    return shift(rotated)

def check_if_rotated_copy(shape1,shape2):
    angles = [0,90,180,270]
    for ang in angles:
        rotated_shape1 = rotate_shape(shape1,ang)
        if np.allclose(rotated_shape1,shape2):
            return True
    return False

def find_all_edges(shape):
    edges = []
    for i,tri in enumerate(shape):
        if tri[0] == 0:
            edges.append((0,tri[1],tri[2],i))
            edges.append((1,tri[1],tri[2],i))
            edges.append((4,tri[1],tri[2],i))
        elif tri[0] == 1:
            edges.append((1,tri[1],tri[2],i))
            edges.append((2,tri[1],tri[2],i))
            edges.append((5,tri[1],tri[2],i))
        elif tri[0] == 2:
            edges.append((2,tri[1],tri[2],i))
            edges.append((3,tri[1],tri[2],i))
            edges.append((4,tri[1],tri[2],i))
        elif tri[0] == 3:
            edges.append((0,tri[1],tri[2],i))
            edges.append((3,tri[1],tri[2],i))
            edges.append((5,tri[1],tri[2],i))
    attached = []
    edge_pairs = list(combinations(edges,2))
    for pair in edge_pairs:
        if pair[0][0] in [4,5] and pair[1][0] in [4,5]:
            if pair[0][0] == pair[1][0]:
                if pair[0][1]==pair[1][1] and pair[0][2]==pair[1][2]:
                    attached.append(pair[0])
                    attached.append(pair[1])
                    continue
        elif pair[0][1]-pair[1][1]==-1 and pair[0][2]-pair[1][2]==0:
            if (pair[0][0]==3 and pair[1][0]==1):
                attached.append(pair[0])
                attached.append(pair[1])
                continue
        elif pair[0][1]-pair[1][1]==1 and pair[0][2]-pair[1][2]==0:
            if (pair[0][0]==1 and pair[1][0]==3):
                attached.append(pair[0])
                attached.append(pair[1])
                continue
        elif pair[0][1]-pair[1][1]==0 and pair[0][2]-pair[1][2]==-1:
            if (pair[0][0]==0 and pair[1][0]==2):
                attached.append(pair[0])
                attached.append(pair[1])
                continue
        elif pair[0][1]-pair[1][1]==0 and pair[0][2]-pair[1][2]==1:
            if (pair[0][0]==2 and pair[1][0]==0):
                attached.append(pair[0])
                attached.append(pair[1])

    return list(set(edges)-set(attached))

def attach_cords(shape1,shape2,edge1,edge2):
    x_ori = edge2[1]
    y_ori = edge2[2]
    
    if edge1[0]==0:
        x_tar = edge1[1]
        y_tar = edge1[2]+1
    elif edge1[0]==1:
        x_tar = edge1[1]-1
        y_tar = edge1[2]
    elif edge1[0]==2:
        x_tar = edge1[1]
        y_tar = edge1[2]-1
    elif edge1[0]==3:
        x_tar = edge1[1]+1
        y_tar = edge1[2]
    elif edge1[0]==4:
        x_tar = edge1[1]
        y_tar = edge1[2]
    elif edge1[0]==5:
        x_tar = edge1[1]
        y_tar = edge1[2]
    
    x_shift = x_tar-x_ori
    y_shift = y_tar-y_ori
    
    shifted_shape2 = shape2.copy()
    for tri in shifted_shape2:
        tri[1] += x_shift
        tri[2] += y_shift
    attached = np.concatenate((shape1, shifted_shape2))
    return shift(attached)

def if_legal_combo(attached):
    all_pairs = list(combinations(attached,2))
    for pair in all_pairs:
        if pair[0][1]==pair[1][1] and pair[0][2]==pair[1][2]:
            if pair[0][0] == 0:
                if pair[1][0] != 2:
                    return False
            elif pair[0][0] == 1:
                if pair[1][0] != 3:
                    return False
            elif pair[0][0] == 2:
                if pair[1][0] != 0:
                    return False
            elif pair[0][0] == 3:
                if pair[1][0] != 1:
                    return False
    return True

# def visualize_combos(filled, color = [[(113, 159, 235),(84, 146, 247)],
#     [(250, 225, 157),(235, 186, 52)],
#     [(181, 255, 223),(61, 235, 159)]]):
#     img = Image.new('RGB', (192,192), color=(0, 0, 0))
#     drawer = ImageDraw.Draw(img)
#     for i,shape in enumerate(filled):
#         vs = render_vertices(shape)
#         if i<4:
#             drawer.polygon(vs, outline = color[0][0], fill=color[0][1])
#         elif i<8:
#             drawer.polygon(vs, outline = color[1][0], fill=color[1][1])
#         else:
#             drawer.polygon(vs, outline = color[2][0], fill=color[2][1])
#     return img

def visualize_combos(filled, color = [[(113, 159, 235),(84, 146, 247)],
                                    [(250, 225, 157),(235, 186, 52)],
                                    [(181, 255, 223),(61, 235, 159)],[(245, 224, 255),(197, 110, 240)]]):
    img = Image.new('RGB', (193,193), color=(0, 0, 0))
    drawer = ImageDraw.Draw(img)

    xs = []
    ys = []
    vss = []
    for i,shape in enumerate(filled):
        vs = render_vertices(shape) 
        vss.append(vs)
        x, y = zip(*vs)
        xs += x
        ys += y
        
    x_center = (np.max(xs) - np.min(xs))/2+np.min(xs)
    y_center = (np.max(ys) - np.min(ys))/2+np.min(ys)
    shift = np.array([96-x_center,96-y_center])    


    for i,vs in enumerate(vss):
        if i<4:
            drawer.polygon([tuple(tup) for tup in np.array(vs)+shift], outline = color[0][0], fill=color[0][1])
        elif i<8:
            drawer.polygon([tuple(tup) for tup in np.array(vs)+shift], outline = color[1][0], fill=color[1][1])
        elif i<12:
            drawer.polygon([tuple(tup) for tup in np.array(vs)+shift], outline = color[2][0], fill=color[2][1])
        else:
            drawer.polygon([tuple(tup) for tup in np.array(vs)+shift], outline = color[3][0], fill=color[3][1])
    
    return img

def check_for_dup(all_attachments):
    if len(all_attachments)<1:
        unique_attachments = []
    else:
        unique_attachments = [all_attachments[0]]
        for item1 in all_attachments:
            k = len(unique_attachments)
            c=0
            for item2 in unique_attachments:
                if np.allclose(np.array(visualize_combos(item1)),np.array(visualize_combos(item2))):
                    break
                c += 1
            if c == k:
                unique_attachments.append(item1)
    return unique_attachments

def check_for_rotated(all_attachments):
    if len(all_attachments)<1:
        unique_attachments = []
    else:
        unique_attachments = [all_attachments[0]]
        for item1 in all_attachments[1:]:
            k = len(unique_attachments)
            c=0
            for item2 in unique_attachments:
                a = 0
                for rot in range(1,4):
                    if np.sum(np.array(visualize_combos(rotate_shape(item1,0)))-
                                  np.array(visualize_combos(rotate_shape(item2,rot*90))))/255<1500:
                        break
                    a += 1
                if a == 3:
                    c+=1
            if c == k:
                unique_attachments.append(item1)
    return unique_attachments

def check_for_mirror(all_attachments):
    if len(all_attachments)<1:
        unique_attachments = []
    else:
        unique_attachments = [all_attachments[0]]
        for item1 in all_attachments[1:]:
            k = len(unique_attachments)
            c=0
            for item2 in unique_attachments:
                if np.allclose(np.array(visualize_combos(item1).transpose(Image.FLIP_LEFT_RIGHT)),np.array(visualize_combos(item2))):
                    break
                c += 1
            if c == k:
                unique_attachments.append(item1)
    return unique_attachments
    

def generate_all_attachments(shape1,shape2):

    all_attachments = []
    shape1_edges = find_all_edges(shape1)
    shape2_edges = find_all_edges(shape2)
    for edge1 in shape1_edges:
        for edge2 in shape2_edges:
            if edge1[0] in [0,1,2,3] and edge2[0] in [0,1,2,3]:
                target = np.mod(edge1[0]+2,4)
                rotation = np.mod(target-edge2[0]+4,4)
                candidate = rotate_shape(shape2,rotation*90)    
                rotated_edge2 = (np.mod(edge2[0]+rotation,4),
                                 candidate[edge2[3],1],
                                 candidate[edge2[3],2],edge2[3])
                attached = attach_cords(shape1,candidate,edge1,rotated_edge2)
                if if_legal_combo(attached):
                    all_attachments.append(attached)
            elif edge1[0] in [4,5] and edge2[0] in [4,5]:
                if edge1[0] == edge2[0]:
                    rotations = [0,2]
                else:
                    rotations = [1,3]
                for rot in rotations:
                    candidate = rotate_shape(shape2,rot*90)
                    rotated_edge2 = (np.mod(edge2[0]+rot,2)+4,
                                 candidate[edge2[3],1],
                                 candidate[edge2[3],2],edge2[3])
                    attached = attach_cords(shape1,candidate,edge1,rotated_edge2)
                    if if_legal_combo(attached):
                        all_attachments.append(attached)
    all_attachments = check_for_dup(all_attachments)
    all_attachments = check_for_rotated(all_attachments)
    all_attachments = check_for_mirror(all_attachments)


    return all_attachments, len(all_attachments)

def token_toset(token_str):
    temp = token_str.split('+')
    if len(temp[0])==3:
        return [ord(temp[0][0])-65,ord(temp[0][1])-65,int(temp[0][2])-1,int(temp[1])]
    else:
        return [ord(temp[0][0])-65,None,None,int(temp[1])]
# def hypo_tostring(hypo_set):
#     if hypo_set[1] is None:
#         return str(hypo_set[0])+"+"+str(hypo_set[4])
#     return str(hypo_set[0])+"+"+str(hypo_set[1])+"+"+str(hypo_set[2])+"+"+str(hypo_set[3])+'+'+str(hypo_set[4])

def hypo_tostring(hypo_set):
    if hypo_set[1] is None:
        return str(hypo_set[0])+"+"+str(hypo_set[4])
    return str(hypo_set[0])+"+"+str(hypo_set[1])+"+"+str(hypo_set[2])+"+"+str(hypo_set[3]).zfill(2)+'+'+str(hypo_set[4])

def to_shape_set(ind_list):
    return [shapes_in_exp[ind] for ind in ind_list]

def show_images(imgs):
    n_col = 5
    n_row = int(np.ceil(len(imgs)/5))
    fig,ax = plt.subplots(n_row,n_col,figsize=(n_col*3,n_row*3))
    k = 0
    for row in range(n_row):
        for col in range(n_col):
            if k<len(imgs):
                temp = visualize_combos(imgs[k])
                ax[row,col].imshow(np.array(temp))
                # Hide the right and top spines
                ax[row,col].spines['right'].set_visible(False)
                ax[row,col].spines['top'].set_visible(False)
                ax[row,col].spines['bottom'].set_visible(False)
                ax[row,col].spines['left'].set_visible(False)
                
                # Only show ticks on the left and bottom spines
                ax[row,col].yaxis.set_ticks_position('left')
                ax[row,col].xaxis.set_ticks_position('bottom')

                ax[row,col].set_title(k)
                ax[row,col].set_aspect('equal', 'box')

            else:
                ax[row,col].axis('off')
            k += 1
    plt.show()
    

all_combo = []
for i in range(3):
    for j in range(3):
        for k in range(4):
            all_combo.append([k,i,j])
all_combo = np.array(list(combinations(all_combo,4)))
shifted_all_combo = [shift(combo) for combo in all_combo]
mask = [check_if_legal(combo) for combo in shifted_all_combo]
shifted_all_combo = np.array(shifted_all_combo)
legal_combos = shifted_all_combo[mask]

ids = [8,15,100,83,101,57,29,82,56,50,9,10,70,69,87,86,46,24,66,182,136,33]
shapes_in_exp=legal_combos[ids]