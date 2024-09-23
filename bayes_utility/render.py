import os
import math
import numpy as np
from PIL import Image, ImageDraw
import pickle

from .render_util import shapes_in_exp


primitives = shapes_in_exp[[1,2,3,4,5,12,14,15,18]]


shapes_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'all_possible_shapes.pkl')
with open(shapes_path, 'rb') as f:
    all_possible_shapes = pickle.load(f)


def render_vertices(cord,side_len):
    shape_id = cord[0]
    if shape_id == 0:
        vs = [(0,side_len),(0,0),(side_len,side_len)]
    elif shape_id == 1:
        vs = [(0,side_len),(0,0),(side_len,0)]
    elif shape_id == 2:
        vs = [(0,0),(side_len,0),(side_len,side_len)]
    elif shape_id == 3:
        vs = [(side_len,0),(side_len,side_len),(0,side_len)]
    vs = [(tup[0]+cord[1]*side_len,tup[1]+cord[2]*side_len) for tup in vs]
    return vs


def get_lines(shape_set,side_len):
    
    def get_current_side(sides,current_v):
        for s in sides:
            if s[0]==current_v:
                return s
            
    v_set = []
    tri_set  = {}
    for i,tri in enumerate(shape_set):
        vs = render_vertices(tri,side_len)
        v_set += vs
        tri_set[i] = vs
            
    
    v_dict = {vert:i for i,vert in enumerate(set(v_set))}
    v_dict_rev = {i:vert for i,vert in enumerate(set(v_set))}
    
    sides = []
    for i in tri_set:
        tri = tri_set[i]
        for j in range(3):
            if j < 2:
                sides.append((v_dict[tri[j]],v_dict[tri[j+1]]))
            else:
                sides.append((v_dict[tri[j]],v_dict[tri[0]]))
    sorted_sides = [tuple(sorted(tup)) for tup in sides]      
    single_sides = [tup for tup in sides if sorted_sides.count(tuple(sorted(tup)))==1]
    
    poly_vs = [v_dict_rev[0]]
    for i in range(len(single_sides)):
        if i == 0:
            current_side = get_current_side(single_sides,0)           
            sind = current_side[1]
        else:
            current_side = get_current_side(single_sides,sind)  
            sind = current_side[1]
        poly_vs.append(v_dict_rev[sind])  
        
    return np.array(poly_vs).astype('int64')

def rotate_figure(vs,angle):
    theta = np.radians(angle)
    R = np.array([[np.cos(theta),-np.sin(theta)],
                  [np.sin(theta),np.cos(theta)]])
    return np.around(np.dot(vs,R),-1).astype('int64')

def calc_rotate_angle(base_angle, rotate_angle):
    return (base_angle+rotate_angle)%360


def make_stim(shape_set, base_angle, rotate_angle, side_len=20): 
    
    xs = []
    ys = []
    poly_x = []
    poly_y = []
    polys  = []
    angle  = calc_rotate_angle(base_angle, rotate_angle)
    for prim in shape_set:
        poly_vs = get_lines(prim, side_len)  
        
        poly_vs = rotate_figure(poly_vs,angle)
        x, y = zip(*poly_vs)
        xs += x
        ys += y
        poly_x.append(x)
        poly_y.append(y)
        polys.append(poly_vs)
    
    x_shift = 0-np.min(xs)
    y_shift = 0-np.min(ys)
    
    shift = [x_shift, y_shift]

    for i in range(len(polys)):
        for j in range(len(polys[i])):
            polys[i][j][0] = polys[i][j][0]+shift[0]
            polys[i][j][1] = polys[i][j][1]+shift[1]   
    
    polys = np.array(polys).astype('int64')

    return polys

def tokenstr_to_set(token_str, prims):
    temp = token_str.split('+')
    if len(temp[0])==5:
        s1 = prims[ord(temp[0][0])-65]
        s2 = prims[ord(temp[0][1])-65]
        att1 = int(temp[0][2])%len(all_possible_shapes[s1][s2].keys())
        s3 = prims[ord(temp[0][3])-65]
        att2 = int(temp[0][4])%len(all_possible_shapes[s1][s2][att1][s3].keys())
        ang = int(temp[1])
        return [(s1,s2,att1,s3,att2),ang]
    elif len(temp[0])==3:
        s1 = prims[ord(temp[0][0])-65]
        s2 = prims[ord(temp[0][1])-65]
        att1 = int(temp[0][2])%len(all_possible_shapes[s1][s2].keys())
        ang = int(temp[1])
        return [(s1,s2,att1),ang]
    else:
        s1 = prims[ord(temp[0])-65]
        ang = int(temp[1])
        return (s1,), ang

def render_from_string(token_str, prims, if_stim = True):
    
    ids, ang = tokenstr_to_set(token_str, prims)
    if len(ids) == 5:
        shape_set = all_possible_shapes[ids[0]][ids[1]][ids[2]][ids[3]][ids[4]]
        shape_set = [shape_set[:4], shape_set[4:8], shape_set[8:]]

    elif len(ids) == 3:
        shape_set = all_possible_shapes[ids[0]][ids[1]][ids[2]]['#']
        shape_set = [shape_set[:4], shape_set[4:]]
    else:
        shape_set = all_possible_shapes[ids[0]]['#']
        shape_set = [shape_set]
    if if_stim:
        out = make_stim(shape_set, 0, ang, side_len=20)
    else:
        out = make_image(shape_set, token_str, ang)

    return out, shape_set

def get_points(shape):
    string = ""
    for tup in shape:
        string+= str(tup[0])+' '+str(tup[1])+', '
    
    return string[:-2]

def str_to_primid(token_str):
    
    temp = token_str.split('+')
    
    if len(temp[0])==5:
        s1 = ord(temp[0][0])-65
        s2 = ord(temp[0][1])-65
        s3 = ord(temp[0][3])-65

        return (s1,s2,s3)
    
    elif len(temp[0])==3:
        s1 = ord(temp[0][0])-65
        s2 = ord(temp[0][1])-65

        return (s1,s2)
    
    else:
        s1 = ord(temp[0])-65
        return (s1,)
    
def make_image(shape_set, token_str, rotate_angle, base_angle=0, side_len=60, imgsize=200): 
    
    colors = ['#bee37f','#d5a4de','#fcce8d','#a6d4ff']
#     colors = ['#fa2f4a','#ffe838','#51a1ed']
    ids = str_to_primid(token_str)
    xs = []
    ys = []
    poly_x = []
    poly_y = []
    polys  = []
    angle  = calc_rotate_angle(base_angle, rotate_angle)
    
    for prim in shape_set:
        poly_vs = get_lines(prim, side_len)  
        poly_vs = rotate_figure(poly_vs,angle)
        x, y = zip(*poly_vs)
        xs += x
        ys += y
        poly_x.append(x)
        poly_y.append(y)
        polys.append(poly_vs)

    shifted = []
    for i,poly in enumerate(polys):
        temp = back_to_origin(poly,polys)
        shifted.append(temp)

    maxx = []
    maxy = []
    for i,poly in enumerate(shifted):
        xs = [tup[0] for tup in poly]
        ys = [tup[1] for tup in poly]
        maxx.append(np.max(xs))
        maxy.append(np.max(ys))
    img_x = np.max(maxx)
    img_y = np.max(maxy)
    img = Image.new('RGB', (int(img_x),int(img_y)), color='#ffffff')
    drawer = ImageDraw.Draw(img)

    #shifted  = np.array(polys).astype('int64')
    shifted = np.array(shifted).astype('int64')
    
    for i, poly in enumerate(shifted):
        drawer.polygon([tuple(tup) for tup in poly], outline = '#ffffff', fill=colors[ids[i]])

    return img

def str_to_primid(token_str):
    
    temp = token_str.split('+')
    
    if len(temp[0])==5:
        s1 = ord(temp[0][0])-65
        s2 = ord(temp[0][1])-65
        s3 = ord(temp[0][3])-65

        return (s1,s2,s3)
    
    elif len(temp[0])==3:
        s1 = ord(temp[0][0])-65
        s2 = ord(temp[0][1])-65

        return (s1,s2)
    
    else:
        s1 = ord(temp[0])-65
        return (s1,)

def rotatePoint(centerPoint,point,angle):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
    temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
    return temp_point

def back_to_origin(points,all_points):
    xs = []
    ys = []
    for pts in all_points:
        xs.extend([pt[0] for pt in pts])
        ys.extend([pt[1] for pt in pts])
    x_shift = min(xs)
    y_shift = min(ys)
    shifted_points = []
    for i in range(len(points)):
        shifted_points.append((points[i][0]-x_shift,points[i][1]-y_shift))
    return shifted_points 

def get_shape(point_str):
    shape = []
    points = point_str.split(',')
    for pt in points:
        pt = pt.split(' ')
        if len(pt) == 3:
            shape.append([int(pt[1]),int(pt[2])])
        else:
            shape.append([int(pt[0]),int(pt[1])])
    
    return np.array(shape).astype('int64')


def make_image_from_data(polys, angles, x_shift, y_shift, colors = None, if_draw = False): 
    
    rotated_polys = []
    for i,poly in enumerate(polys):
        xs = [tup[0] for tup in poly]
        ys = [tup[1] for tup in poly]
        rotated_polys.append([rotatePoint([np.min(xs),np.min(ys)],tup,angles[i]) for tup in poly])
    
    for i in range(len(rotated_polys)):
        for j in range(len(polys[i])):
            rotated_polys[i][j] = rotated_polys[i][j][0]+x_shift[i],rotated_polys[i][j][1]+y_shift[i]
    img = ''

    shifted_and_rotated = []
    for i,poly in enumerate(rotated_polys):
        poly = back_to_origin(poly,rotated_polys)
        shifted_and_rotated.append(poly)
    
    if if_draw:
        maxx = []
        maxy = []
        for i,poly in enumerate(shifted_and_rotated):
            xs = [tup[0] for tup in poly]
            ys = [tup[1] for tup in poly]
            maxx.append(np.max(xs))
            maxy.append(np.max(ys))
        img_x = np.max(maxx)
        img_y = np.max(maxy)
        img = Image.new('RGB', (int(img_x),int(img_y)), color='#ffffff')
        drawer = ImageDraw.Draw(img)

        for i,poly in enumerate(shifted_and_rotated):
            drawer.polygon([tuple(tup) for tup in poly], outline = '#ffffff', fill=colors[i])

    return shifted_and_rotated, img

def process_canvas(row, raw_data):
    # if row['n_example']==3:
    #     raw_data = raw_zoo_data_3
    # else:
    #     raw_data = raw_zoo_data_6
    colors = ['#bee37f','#d5a4de','#fcce8d','#a6d4ff']
    row_prim_data = pd.DataFrame(raw_data[str(int(row['seed_id']))]['all_prim_data'][str(int(row['zoo_id']))])
    polys = []
    angles = []
    x_shift = []
    y_shift = []
    c = []
    for prim in row['canvasdata']:
        prim_id = prim['id']
        polys.append(get_shape(row_prim_data['points'][row_prim_data['id']==prim_id].iloc[0])*3)
        if 'r' in prim['transformstr']:
            angle = prim['transformstr'].split('r')
            angle = angle[1].split(',')[0]
            angles.append(int(angle))
        else:
            angles.append(0)
        shift = prim['transformstr'].split('t')
        shift = shift[1].split(',')
        x_shift.append(int(shift[0]))
        if 'r' in prim['transformstr']:
            y_shift.append(int(shift[1].split('r')[0]))
        else:
            y_shift.append(int(shift[1].split('s')[0]))
        c.append(colors[int(prim_id.split('prim')[1])])
    coords, img = make_image_from_data(polys, angles, x_shift, y_shift, c, if_draw=True)
    return img
