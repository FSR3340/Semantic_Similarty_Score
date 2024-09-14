import numpy as np
import open_clip
import os
import pickle
import json
import torch
import cv2
import time
from utils.relation import relation_classes
from PIL import Image
from scipy.optimize import linear_sum_assignment
from utils.semantic_importance import get_semantic_imp

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#读取图像文件
def load_dict_from_file(filename):
    with open(filename, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary

#读取json文件
def read_json(file_path):
    with open(file_path, 'rb') as f:
        data = json.load(f)
    return data

#保存结果文件
def save_dict_to_file(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#clip计算文本相似度

device = 'cuda:0'
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="checkpoints/open_clip_pytorch_model.bin", device=device)
tokenizer = open_clip.get_tokenizer('ViT-H-14')

def get_text_feature(text_content):
    text = tokenizer([text_content]).to(device)
    text_features = clip_model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#计算所有标签的CLip数值
relation_features = []
for relation in relation_classes:
    relation_features.append(get_text_feature(relation))

#计算标签相似度矩阵
R = np.zeros((len(relation_classes), len(relation_classes)))
for relation1 in range(len(relation_classes)):
    for relation2 in range(len(relation_classes)):
        R[relation1, relation2] = (relation_features[relation1] @ relation_features[relation2].T).item()
        
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#使用 Kuhn-Munkres 算法求解二部图的最大权重匹配

def maximum_weight_independent_set(matrix, output=False):
    n, m = matrix.shape
    weights = -matrix  
    row_indices, col_indices = linear_sum_assignment(weights)
    #selected_points = set(zip(row_indices, col_indices))
    max_weight = -weights[row_indices, col_indices].sum().item()
    
    return max_weight#, selected_points

def find_mask_center(mask):
    # 获取掩码数组的形状
    H, W = mask.shape
    # 找到所有为True的元素的索引
    true_indices = np.where(mask)
    # 计算索引的平均值
    center_h = np.mean(true_indices[0]) / H
    center_w = np.mean(true_indices[1]) / W

    return [center_h, center_w]

def calc_dis(pointA, pointB):
    x1 = pointA[0]
    y1 = pointA[1]
    x2 = pointB[0]
    y2 = pointB[1]
    return 1.0-((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))/2.0

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
#知识图谱建图

class Graph:
    def __init__(self, img_path, num_nodes, relation_matrixs, score_matrixs, masks):
        
        self.img_path = img_path
        self.masks = masks
        
        self.num_nodes = num_nodes
        self.relation_matrixs = relation_matrixs
        self.score_matrixs = score_matrixs
   
        self.node_imp = get_semantic_imp(img_path, masks, score_matrixs)
        self.clip_features, self.location = self.get_clip_features()
     
    def get_clip_features(self):
        node_features = []
        location = []
        for index in range(self.num_nodes):
            image = Image.open(self.img_path)
            mask = self.masks[index]
            location.append(find_mask_center(mask))
            
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            min_row, max_row = np.where(rows)[0][[0, -1]]
            min_col, max_col = np.where(cols)[0][[0, -1]]
            sub_image = image.crop((min_col, min_row, max_col + 1, max_row + 1))
            img = clip_preprocess(sub_image).unsqueeze(0).to(device)
            img_features = clip_model.encode_image(img)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            node_features.append(img_features)
        
        return node_features, location

    def get_neighbors(self, node):
        neighbors = []
        sum_imp = 0.0
        relation_row = self.relation_matrixs[node]
        for i, relation in enumerate(relation_row):
            if i != node and relation != -1:
                neighbors.append((i, relation))
                sum_imp = sum_imp + self.node_imp[i]
        return neighbors, sum_imp
        
    def output(self):
        print(f"node nums: {self.num_nodes}")
        print(f"node imp: {self.node_imp}")
        for node in range(self.num_nodes):
            print(f"node {node}:")
            relation_row = self.relation_matrixs[node]
            for i, relation in enumerate(relation_row):
                if i != node and relation != -1:
                    print(f"--- {relation_classes[relation]} --->  node {i}   :   {self.score_matrixs[node, i]}")
        print("")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------

def calc_result(Matrix, graph1, graph2):
    L = Matrix.copy()
    for node1 in range(graph1.num_nodes):
        L[node1, :] = L[node1, :] * graph1.node_imp[node1] 
    score_1 = maximum_weight_independent_set(L, output=True)
    for node1 in range(graph1.num_nodes):
        L[node1, :] = L[node1, :] / graph1.node_imp[node1]
    for node2 in range(graph2.num_nodes):
        L[:, node2] = L[:, node2] * graph2.node_imp[node2]
    score_2 = maximum_weight_independent_set(L, output=True)
    score_ite = (score_1*graph1.num_nodes + score_2*graph2.num_nodes) / (graph1.num_nodes+graph2.num_nodes)
    return score_ite

def inference(path1, path2):

    dict1 = load_dict_from_file(path1['dict_path'])
    dict2 = load_dict_from_file(path2['dict_path'])

    graph1 = Graph(path1['img_path'], dict1['num_nodes'], dict1['relation_matrixs'], dict1['score_matrixs'], dict1['mask'])
    graph2 = Graph(path2['img_path'], dict2['num_nodes'], dict2['relation_matrixs'], dict2['score_matrixs'], dict2['mask'])

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------
    # 得到初始点对相似度矩阵
    theta = 0.2
    L_origin = np.zeros((graph1.num_nodes, graph2.num_nodes))
    for node1 in range(graph1.num_nodes):
        for node2 in range(graph2.num_nodes):
            L_origin[node1, node2] = (1-theta)*(graph1.clip_features[node1] @ graph2.clip_features[node2].T).item() + theta*calc_dis(graph1.location[node1], graph2.location[node2]) 

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------
    # 计算两张整图的相似度

    image1 = clip_preprocess(Image.open(graph1.img_path)).unsqueeze(0).to(device)
    image2 = clip_preprocess(Image.open(graph2.img_path)).unsqueeze(0).to(device)

    image_features1 = clip_model.encode_image(image1)
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    image_features2 = clip_model.encode_image(image2)
    image_features2 /= image_features2.norm(dim=-1, keepdim=True)
    score_total = (image_features1 @ image_features2.T).item()

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------
    # 迭代计算点对相似度矩阵

    #算法迭代次数
    iter_nums = 7
    #节点差异与关系差异求和时的权重 
    alpha = 0.25
    # 原节点差异与新差异的迭代比率   
    beta = 0.05 
    L = L_origin.copy()
    L_new = np.zeros((graph1.num_nodes, graph2.num_nodes))
    for iter_index in range(iter_nums):
        for node1 in range(graph1.num_nodes):
            neighbors1, node1_sum_imp = graph1.get_neighbors(node1)
            for node2 in range(graph2.num_nodes): 
                neighbors2, node2_sum_imp = graph2.get_neighbors(node2)
                if(len(neighbors1) == 0 and len(neighbors2) == 0):
                    L_new[node1, node2] = (1-beta)*L[node1, node2] + beta*1.0
                    continue
                L_nei = np.zeros((len(neighbors1), len(neighbors2)))
                for i, (neighbor1, relation_index1) in enumerate(neighbors1):
                    relation_label1 = relation_classes[relation_index1]
                    for j, (neighbor2, relation_index2) in enumerate(neighbors2):
                        relation_label2 = relation_classes[relation_index2]
                        L_nei[i, j] = (1-alpha)*L[neighbor1, neighbor2] + alpha*R[relation_index1, relation_index2] 
                
                for i, (neighbor1, relation_index1) in enumerate(neighbors1):
                    L_nei[i, :] = L_nei[i, :] * graph1.node_imp[neighbor1] / node1_sum_imp
                score_1 = maximum_weight_independent_set(L_nei)
                
                for i, (neighbor1, relation_index1) in enumerate(neighbors1):
                    L_nei[i, :] = L_nei[i, :] / graph1.node_imp[neighbor1] * node1_sum_imp
                
                for j, (neighbor2, relation_index2) in enumerate(neighbors2):
                    L_nei[:, j] = L_nei[:, j] * graph2.node_imp[neighbor2] / node2_sum_imp
                
                score_2 = maximum_weight_independent_set(L_nei)
                
                score_avg = (score_1*graph1.node_imp[node1] + score_2*graph2.node_imp[node2]) / (graph1.node_imp[node1] + graph2.node_imp[node2])
                
                L_new[node1, node2] = (1-beta)*L[node1, node2] + beta*score_avg
        
        L = L_new
    score = (1-0.05)*calc_result(L, graph1, graph2) + 0.05*score_total

    return score

def main():
    folder1 = ["./dataset/origin"]
    folder2 = ["./dataset/recover"]
    nums = 50

    semantic_avg = 0.0
    with torch.no_grad():
        semantic_re = []
        for folder2 in folder2s:
            semantic_avg = 0.0
            for index in range(nums):
                #print(time.time())
                file_path = str(index+1).zfill(4) + ".jpg.json"
                img_path = str(index+1).zfill(4) + ".jpg"
                path_source = {'dict_path': os.path.join(folder1, file_path), 'img_path': os.path.join(folder1, img_path)}          
                path_jpeg = {'dict_path': os.path.join(folder2, file_path), 'img_path': os.path.join(folder2, img_path)}
                semantic = inference(path_source, path_jpeg)  
                semantic_avg = semantic_avg + semantic
                print(index, semantic)
            print(semantic_avg / nums)
            semantic_re.append(semantic_avg / nums) 
        print(semantic_re)
main()
