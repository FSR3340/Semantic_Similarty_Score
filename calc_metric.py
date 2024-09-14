import os
import pickle
import json
import torch
import argparse
from utils.metric import calc_metric

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
parser = argparse.ArgumentParser(description='choice of metrics')
parser.add_argument('-m', '--metric', type=str, help='name of the chosen metric')


def main():
    args = parser.parse_args()
    metric = args.metric
    folder2s = ["/home/fsr/Dataloader/extra/recon_01", "/home/fsr/Dataloader/extra/recon_08"]
    with torch.no_grad():
        metric_re = []
        folder1 = "/home/fsr/Dataloader/LSCI/origin"
        for folder2 in folder2s:
            metric_avg = 0.0
            for index in range(50):
                img_path = str(index+1).zfill(4) + ".jpg"
                img_path1 = os.path.join(folder1, img_path)
                img_path2 = os.path.join(folder2, img_path)
                metric_score = calc_metric(img_path1, img_path2, metric)          
                print(f"{metric} : {metric_score}")
                metric_avg = metric_avg + metric_score
            metric_re.append(metric_avg / 50)
        print(f'{metric} : {metric_re}')

main()
