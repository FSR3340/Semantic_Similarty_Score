import argparse
import pickle
import cv2
import numpy as np
import math

def load_dict_from_file(filename):
    with open(filename, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary

def get_semantic_imp(img_path, masks, score_matrixs, k1=2.5, k2=3, k3=0.8): 
    
    if len(masks) == 1:
        return [1.0]
    
    image = cv2.imread(img_path)
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    semantic_imp = np.zeros(len(masks))
    
    
    for index, mask in enumerate(masks):
        #mask_img = image.copy()
        #mask_img[mask==False] = 0
        #cv2.imwrite(img_path+str(index)+'.jpg', mask_img)
        #cv2.imwrite(img_path+str(index)+'.jpg', mask)
        semantic_imp[index] = pow(saliencyMap[mask].sum().item(), 1.0/k1)
    semantic_imp = semantic_imp / semantic_imp.sum().item()

    
    relation_imp = np.zeros(len(masks))
    for i in range(len(masks)):
        for j in range(len(masks)):
            if i!=j and score_matrixs[i, j] > 0:
                relation_imp[i] = relation_imp[i] + score_matrixs[i, j]*semantic_imp[i]/(semantic_imp[i]+semantic_imp[j])
                relation_imp[j] = relation_imp[j] + score_matrixs[i, j]*semantic_imp[j]/(semantic_imp[i]+semantic_imp[j])
    for index in range(len(masks)):
        relation_imp[index] = pow(relation_imp[index], 1.0/k2)
    relation_imp = relation_imp / relation_imp.sum().item()
    
    imp = k3*semantic_imp + (1-k3)*relation_imp
    imp = imp / imp.sum().item()
    
    return imp
