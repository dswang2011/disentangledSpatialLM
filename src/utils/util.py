import os
from math import sqrt
from typing import Tuple
import torch
import math
import numpy as np

def read_pairs(file_path, delimiter='\t'):
    paths, labels = [],[]
    with open(file_path, 'r', encoding='utf8') as fr:
        data = fr.readlines()
    for line in data:
        strs = line.split(delimiter)
        if len(strs)<2: continue
        paths.append(strs[0])
        labels.append(strs[1].strip())
    return paths, labels

def write_line(file_path, content):
    with open(file_path, 'a', encoding='utf8') as fw:
        fw.write(content+'\n')

def read_lines(file_path):
    res = []
    with open(file_path, 'r', encoding='utf8') as fr:
        data = fr.readlines()
    for line in data:
        res.append(line.strip())
    return res


def create_folder(folder_path : str) -> None:
    """create a folder if not exists
    Args:
        folder_path (str): path
    """
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    return


def fully_connected(ids):
    u,v = [],[]
    for id in ids:
        u.extend([id for i in range(len(ids)) if i!=id])
        v.extend([i for i in range(len(ids)) if i!=id])
    return u,v


def polar(rect_src : list, rect_dst : list) -> Tuple[int, int]:
    """Compute distance and angle from src to dst bounding boxes (poolar coordinates considering the src as the center)
    Args:
        rect_src (list) : source rectangle coordinates
        rect_dst (list) : destination rectangle coordinates
    
    Returns:
        tuple (ints): distance and angle
    """
    
    # check relative position
    left = (rect_dst[2] - rect_src[0]) <= 0 # left-top point
    bottom = (rect_src[3] - rect_dst[1]) <= 0   # 
    right = (rect_src[2] - rect_dst[0]) <= 0
    top = (rect_dst[3] - rect_src[1]) <= 0
    
    vp_intersect = (rect_src[0] <= rect_dst[2] and rect_dst[0] <= rect_src[2]) # True if two rects "see" each other vertically, above or under
    hp_intersect = (rect_src[1] <= rect_dst[3] and rect_dst[1] <= rect_src[3]) # True if two rects "see" each other horizontally, right or left
    rect_intersect = vp_intersect and hp_intersect 

    # center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)

    # # evaluate reciprocal position
    # sc = center(rect_src)
    # ec = center(rect_dst)
    # new_ec = (ec[0] - sc[0], ec[1] - sc[1])
    new_ec = (rect_dst[0]-rect_src[0], rect_dst[1]-rect_src[1])
    angle = int(math.degrees(math.atan2(new_ec[1], new_ec[0])))%360
    
    if rect_intersect:
        return 0, angle
    elif top and left:
        a, b = (rect_dst[2] - rect_src[0]), (rect_dst[3] - rect_src[1])
        return int(sqrt(a**2 + b**2)), angle
    elif left and bottom:
        a, b = (rect_dst[2] - rect_src[0]), (rect_dst[1] - rect_src[3])
        return int(sqrt(a**2 + b**2)), angle
    elif bottom and right:
        a, b = (rect_dst[0] - rect_src[2]), (rect_dst[1] - rect_src[3])
        return int(sqrt(a**2 + b**2)), angle
    elif right and top:
        a, b = (rect_dst[0] - rect_src[2]), (rect_dst[3] - rect_src[1])
        return int(sqrt(a**2 + b**2)), angle
    elif left:
        return (rect_src[0] - rect_dst[2]), angle
    elif right:
        return (rect_dst[0] - rect_src[2]), angle
    elif bottom:
        return (rect_dst[1] - rect_src[3]), angle
    elif top:
        return (rect_src[1] - rect_dst[3]), angle


def to_bin(dist :int, angle : int, b=8) -> torch.Tensor:
    """ Discretize the space into equal "bins": return a distance and angle into a number between 0 and 1.
    Args:
        dist (int): distance in terms of pixel, given by "polar()" util function
        angle (int): angle between 0 and 360, given by "polar()" util function
        b (int): number of bins, MUST be power of 2
    
    Returns:
        torch.Tensor: new distance and angle (binary encoded)
    """
    def isPowerOfTwo(x):
        return (x and (not(x & (x - 1))) )

    # dist
    assert isPowerOfTwo(b)
    m = max(dist) / b   # maximum bin size
    new_dist = []
    for d in dist:
        bin = int(d / m)    # which bin it falls at
        if bin >= b: bin = b - 1
        bin = [int(x) for x in list('{0:0b}'.format(bin))]
        while len(bin) < sqrt(b): bin.insert(0, 0)
        new_dist.append(bin)
        print(bin)
    
    # angle
    amplitude = 360 / b
    new_angle = []
    for a in angle:
        bin = (a - amplitude / 2) 
        bin = int(bin / amplitude)
        bin = [int(x) for x in list('{0:0b}'.format(bin))]
        while len(bin) < sqrt(b): bin.insert(0, 0)
        new_angle.append(bin)

    return torch.cat([torch.tensor(new_dist, dtype=torch.float32), torch.tensor(new_angle, dtype=torch.float32)], dim=1)




def _fully_connected_matrix(size,bboxs):
    pair_lookup = {}
    for i in range(len(bboxs)):
        for j in range(len(bboxs)):
            box1, box2 = normalize_bbox(bboxs[i],size),normalize_bbox(bboxs[j],size)  # normalized boxes;
            dist,angle = polar(box1, box2)
            direct = angle //45 
            pair_lookup[(i,j)] = (dist,direct)
    return pair_lookup
def _eight_neibs(idx, N,pair_lookup):
    direct2near = {}    # idx : (dist, direct, index)
    for neib_idx in range(N):
        if idx == neib_idx: continue  # skip itself
        dist,direct = pair_lookup[(idx,neib_idx)]
        if direct in direct2near.keys():
            if dist<direct2near[direct][0]:
                direct2near[direct] = (dist, direct, neib_idx)
        else:
            direct2near[direct] = (dist, direct, neib_idx)
    return direct2near

def rolling_neibor_matrix(size, bboxs):
    pair_lookup = _fully_connected_matrix(size, bboxs)
    u,v = [],[]
    # dists, directs = [],[]
    edge_index = []
    edge_attr = []
    for idx in range(len(bboxs)):
        direct2near = _eight_neibs(idx, len(bboxs), pair_lookup)
        for direct, (dist,direct,neib_idx) in direct2near.items():
            u.append(idx)
            v.append(neib_idx)
            # dist.append(dist)
            # directs.append(direct)
            edge_attr.append([dist,direct])
    edge_index = [u,v]
    return edge_index, edge_attr




def KNN(size : tuple, bboxs : list, k = 8):
    """ Given a list of bounding boxes, find for each of them their k nearest ones.
    Args:
        size (tuple) : width and height of the image
        bboxs (list) : list of bounding box coordinates
        k (int) : k of the knn algorithm
    
    Returns:
        u, v (lists) : lists of indices
        e_features: [num_edge, 2] -> list of features [dist,angle]
    """

    # for all nodes in a graph/doc
    edges = []
    edge_attr = []
    width, height = size[0], size[1]
    
    # creating projections
    vertical_projections = [[] for i in range(width)]
    horizontal_projections = [[] for i in range(height)]
    for node_index, bbox in enumerate(bboxs):
        for hp in range(bbox[0], bbox[2]):
            if hp >= width: hp = width - 1
            vertical_projections[hp].append(node_index)
        for vp in range(bbox[1], bbox[3]):
            if vp >= height: vp = height - 1
            horizontal_projections[vp].append(node_index)
    
    def bound(a, ori=''):
        if a < 0 : return 0
        elif ori == 'h' and a > height: return height
        elif ori == 'w' and a > width: return width
        else: return a

    for node_index, node_bbox in enumerate(bboxs):
        neighbors = [] # collect list of neighbors
        window_multiplier = 2 # how much to look around bbox
        wider = (node_bbox[2] - node_bbox[0]) > (node_bbox[3] - node_bbox[1]) # if bbox wider than taller
        
        ### finding neighbors ###
        while(len(neighbors) < k and window_multiplier < 100): # keep enlarging the window until at least k bboxs are found or window too big
            vertical_bboxs = []
            horizontal_bboxs = []
            neighbors = []
            
            if wider:
                h_offset = int((node_bbox[2] - node_bbox[0]) * window_multiplier/4)
                v_offset = int((node_bbox[3] - node_bbox[1]) * window_multiplier)
            else:
                h_offset = int((node_bbox[2] - node_bbox[0]) * window_multiplier)
                v_offset = int((node_bbox[3] - node_bbox[1]) * window_multiplier/4)
            
            window = [bound(node_bbox[0] - h_offset),
                    bound(node_bbox[1] - v_offset),
                    bound(node_bbox[2] + h_offset, 'w'),
                    bound(node_bbox[3] + v_offset, 'h')] 
            
            [vertical_bboxs.extend(d) for d in vertical_projections[window[0]:window[2]]]
            [horizontal_bboxs.extend(d) for d in horizontal_projections[window[1]:window[3]]]
            
            for v in set(vertical_bboxs):
                for h in set(horizontal_bboxs):
                    if v == h: neighbors.append(v)
            
            window_multiplier += 1 # enlarge the window
        
        ### finding k nearest neighbors ###
        neighbors = list(set(neighbors))
        if node_index in neighbors:
            neighbors.remove(node_index)

        neighbors_distances, neibors_angles = [],[]
        for n in neighbors:
            dist,angle = polar(node_bbox, bboxs[n])
            neighbors_distances.append(dist)
            neibors_angles.append(angle)

        for sd_num, sd_idx in enumerate(np.argsort(neighbors_distances)):        
            if sd_num < k:
                if [node_index, neighbors[sd_idx]] not in edges and [neighbors[sd_idx], node_index] not in edges:
                    edges.append([neighbors[sd_idx], node_index])
                    edges.append([node_index, neighbors[sd_idx]])

                    # add features
                    edge_attr.append([neighbors_distances[sd_idx], 360-neibors_angles[sd_idx]])
                    edge_attr.append([neighbors_distances[sd_idx], neibors_angles[sd_idx]])
            else: break

        edge_index = [[e[0] for e in edges], [e[1] for e in edges]]
    return edge_index, edge_attr


from PIL import Image
import json

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def get_line_bbox(bboxs):
    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
    return bbox


# one doc per img (not multiple pages)
def _generate_examples(self, base_dir):
    # logger.info("⏳ Generating examples from = %s", filepath)
    ann_dir = os.path.join(base_dir, "annotations")
    img_dir = os.path.join(base_dir, "images")
    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        tokens = []
        bboxes = []
        ner_tags = []
        seg_ids = []

        file_path = os.path.join(ann_dir, file)
        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)
        image_path = os.path.join(img_dir, file)
        image_path = image_path.replace("json", "png")
        image, size = load_image(image_path)
        seg_id = 0
        for item in data["form"]:
            cur_line_bboxes = []
            words, label = item["words"], item["label"]
            words = [w for w in words if w["text"].strip() != ""]
            if len(words) == 0:
                continue
            if label == "other":
                for w in words:
                    tokens.append(w["text"])
                    seg_ids.append(seg_id)
                    ner_tags.append("O")
                    cur_line_bboxes.append(normalize_bbox(w["box"], size))
            else:
                tokens.append(words[0]["text"])
                seg_ids.append(seg_id)
                ner_tags.append("B-" + label.upper())
                cur_line_bboxes.append(normalize_bbox(words[0]["box"], size))
                for w in words[1:]:
                    tokens.append(w["text"])
                    seg_ids.append(seg_id)
                    ner_tags.append("I-" + label.upper())
                    cur_line_bboxes.append(normalize_bbox(w["box"], size))
            cur_line_bboxes = get_line_bbox(cur_line_bboxes)
            bboxes.extend(cur_line_bboxes)
            seg_id +=1
        yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags,
                        "image": image,
                        "seg_ids":seg_ids}


if __name__ == '__main__':
    edge_index, edge_attr = rolling_neibor_matrix([100,100], [[0,0,1,1],[10,10,10,10]])
    print(edge_index)
    print(edge_attr)
