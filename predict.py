import sys
sys.path.append('.')

import pickle
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from utils import iou, sort_and_deduplicate, relation_classes, MLP, show_anns, show_mask
import torch

from ram_train_eval import RamModel,RamPredictor
from mmengine.config import Config

def save_dict_to_file(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 512
hidden_size = 256
num_classes = 56

# load sam model
sam = build_sam(checkpoint="./checkpoints/sam_vit_h_4b8939.pth").to(device)
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

# load ram model
model_path = "./checkpoints/ram_epoch12.pth"
config = dict(
    model=dict(
        pretrained_model_name_or_path='bert-base-uncased',
        load_pretrained_weights=False,
        num_transformer_layer=2,
        input_feature_size=256,
        output_feature_size=768,
        cls_feature_size=512,
        num_relation_classes=56,
        pred_type='attention',
        loss_type='multi_label_ce',
    ),
    load_from=model_path,
)
config = Config(config)

class Predictor(RamPredictor):
    def __init__(self,config):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._build_model()

    def _build_model(self):
        self.model = RamModel(**self.config.model).to(self.device)
        if self.config.load_from is not None:
            self.model.load_state_dict(torch.load(self.config.load_from, map_location=self.device))
        self.model.train()

model = Predictor(config)


def relate_anything(image_path):
    input_image = Image.open(image_path).convert("RGB")    
    w = input_image.width
    h = input_image.height
    if w > 800:
        input_image.thumbnail((800, 800*h/w))

    image = np.array(input_image)
    #print(image.shape)
    sam_masks = mask_generator.generate(image)
    filtered_masks = sort_and_deduplicate(sam_masks)
    
    feat_list = []
    
    if len(filtered_masks) > 512:
        filtered_masks = filtered_masks[:512]
    
    for fm in filtered_masks:
        feat = torch.Tensor(fm['feat']).unsqueeze(0).unsqueeze(0).to(device)
        feat_list.append(feat)
    feat = torch.cat(feat_list, dim=1).to(device)
    #print(feat.shape)
    
    if feat.shape[1] == 1:
        masks = []
        masks.append(filtered_masks[0]['segmentation'])
        relation_matrix = np.full((1, 1), -1)
        score_matrix = np.full((1, 1), -1.0)
        graph = {'num_nodes': 1, 'relation_matrixs': relation_matrix, 'score_matrixs': score_matrix, 'mask': masks}     
        return graph, input_image
    
    matrix_output, rel_triplets = model.predict(feat)
    
    # 置信度阈值
    k = min(len(rel_triplets), 20)
    for i, rel in enumerate(rel_triplets):
        if i < k:
            continue
        s,o,r = int(rel[0]),int(rel[1]),int(rel[2])
        if matrix_output[0][r][s][o] < 0.05:
            break
        k = i
    
    # 节点映射
    all_indices = set()
    for triplet in rel_triplets[:k]:
        all_indices.add(triplet[0])
        all_indices.add(triplet[1])
    sorted_indices = sorted(all_indices)
    node_nums = len(sorted_indices)
    index_map = {}
    for i, index in enumerate(sorted_indices):
        index_map[index] = i
    
    # 获得掩码
    masks = []
    for index in range(node_nums):
        masks.append(filtered_masks[sorted_indices[index]]['segmentation'])
    
    # 获得关系矩阵
    relation_matrix = np.full((node_nums, node_nums), -1)
    score_matrix = np.full((node_nums, node_nums), -1.0)
    for rel in reversed(rel_triplets[:k]):
        s,o,r = index_map[int(rel[0])], index_map[int(rel[1])], int(rel[2])
        relation_matrix[s, o] = r
        score_matrix[s, o] = matrix_output[0][r][sorted_indices[s]][sorted_indices[o]]
    
    graph = {'num_nodes': node_nums, 'relation_matrixs': relation_matrix, 'score_matrixs': score_matrix, 'mask': masks}        
    #print(graph)
    
    return graph, input_image

def get_all_jpg_files(folder_path):
    jpg_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):    
                #print(file)
                #if int(file[-7:-4]) > 200:
                #    continue
                file_path = os.path.join(root, file)
                jpg_files.append(file_path)
    sorted_jpg_files = sorted(jpg_files)
    return sorted_jpg_files

def save_dict_to_file(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

def main():
    
    with torch.no_grad():
        folder_path = './dataset'
        jpg_files = get_all_jpg_files(folder_path)
        
        print(len(jpg_files))
        cnt = 0
        
        for file_path in jpg_files:
            cnt = cnt + 1
            graph, img = relate_anything(file_path)
            save_dict_to_file(graph, file_path+'.json')
            print(f'{cnt} : graph from [{file_path}] has been saved into [{file_path}.json]')

if __name__ == '__main__':
    main()

