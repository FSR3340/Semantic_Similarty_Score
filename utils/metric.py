import numpy as np
import open_clip
import os
import torch
import torchvision.transforms as transforms
from vit_pytorch import ViT
from PIL import Image
from torchvision.transforms.functional import to_tensor
import random
import open_clip
from skimage.metrics import structural_similarity as calc_ssim
from pytorch_msssim import ms_ssim as calc_msssim
import cv2
import lpips

# 固定 Python 的随机种子
random_seed = 42
random.seed(random_seed)
# 固定 NumPy 的随机种子
np.random.seed(random_seed)
# 固定 PyTorch 的随机种子
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------  
def mse(img_path1, img_path2):
    image1 = cv2.imread(img_path1).astype(float)
    image2 = cv2.imread(img_path2).astype(float)
    mse_score = np.mean((image1 - image2) ** 2)
    return mse_score
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------  
def psnr(img_path1, img_path2):
    image1 = cv2.imread(img_path1).astype(float)
    image2 = cv2.imread(img_path2).astype(float)
    mse_score = np.mean((image1 - image2) ** 2)
    max_pixel = 255.0
    psnr_score = 10 * np.log10((max_pixel ** 2) / mse_score)
    return psnr_score
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------  
def ssim(img_path1, img_path2):
    image1 = cv2.imread(img_path1)
    image2 = cv2.imread(img_path2)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    ssim_score = calc_ssim(image1, image2, multichannel=True)
    return ssim_score
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------      
def ms_ssim(img_path1, img_path2):
    image1 = Image.open(img_path1)
    image2 = Image.open(img_path2)
    
    if image1.width <= 160:
        image1 = image1.resize((200, image1.height))
        image2 = image2.resize((200, image2.height))
    if image1.height <= 160:
        image1 = image1.resize((image1.width, 200))
        image2 = image2.resize((image2.width, 200))
    #print(image1.width, image1.height)
    
    image1 = to_tensor(image1).unsqueeze(0)
    image2 = to_tensor(image2).unsqueeze(0)
    msssim_score = calc_msssim(image1, image2, data_range=1.0, size_average=True).item()
    return msssim_score
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------     
loss_fn = lpips.LPIPS(net='alex').cuda()
lpips_preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
def calc_lpips(img_path1, img_path2, net='alex', use_gpu = True):
    image1 = Image.open(img_path1).convert('RGB')
    image2 = Image.open(img_path2).convert('RGB')
    image1 = lpips_preprocess(image1).unsqueeze(0).cuda()
    image2 = lpips_preprocess(image2).unsqueeze(0).cuda()
    lpips_score = loss_fn.forward(image1, image2).item()
    return lpips_score
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------  
vit_model = ViT(
    image_size = 512,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
).cuda().eval()
vit_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
def RViTScore(A, B, dim=1024):
    n = len(A)
    m = len(B)
    score = 0
    for i in range(n):
        max_product = torch.dot(A[i], B[0])
        for j in range(1, m):
            product = torch.dot(A[i], B[j])
            max_product = max(max_product, product)
        score += max_product
    return score / (n*dim)

def PViTScore(A, B, dim=1024):
    n = len(A)
    m = len(B)
    score = 0
    for j in range(m):
        max_product = torch.dot(A[0], B[j])
        for i in range(1, n):
            product = torch.dot(A[i], B[j])
            max_product = max(max_product, product)
        score += max_product
    return score / (m*dim)  

def calc_ViTScore(A, B):
    RViTScore_value = RViTScore(A, B)
    PViTScore_value = PViTScore(A, B)
    numerator = 2 * RViTScore_value * PViTScore_value
    denominator = RViTScore_value + PViTScore_value
    return numerator / denominator

def vitscore(img_path1, img_path2):
    image1 = vit_transform(Image.open(img_path1)).unsqueeze(0).cuda()
    image2 = vit_transform(Image.open(img_path2)).unsqueeze(0).cuda()
    img1_feature = vit_model(image1) 
    img2_feature = vit_model(image2)
    vit_score = calc_ViTScore(img1_feature, img2_feature).item()
    return vit_score
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------  
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="checkpoints/open_clip_pytorch_model.bin", device='cuda:0')
tokenizer = open_clip.get_tokenizer('ViT-H-14')
def clipscore(img_path1, img_path2):
    image1 = clip_preprocess(Image.open(img_path1)).unsqueeze(0).to('cuda:0')
    image2 = clip_preprocess(Image.open(img_path2)).unsqueeze(0).to('cuda:0')
    image_features1 = clip_model.encode_image(image1)
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    image_features2 = clip_model.encode_image(image2)
    image_features2 /= image_features2.norm(dim=-1, keepdim=True)
    clip_score = (image_features1 @ image_features2.T).item()
    return clip_score
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
def calc_metric(img_path1, img_path2, metric):
    if metric == "mse":
        return mse(img_path1, img_path2)
    if metric == "psnr":
        return psnr(img_path1, img_path2)
    if metric == "ssim":
        return ssim(img_path1, img_path2)
    if metric == "ms_ssim":
        return ms_ssim(img_path1, img_path2)
    if metric == "lpips":
        return calc_lpips(img_path1, img_path2)
    if metric == "vitscore":
        return vitscore(img_path1, img_path2)
    if metric == "clipscore":
        return clipscore(img_path1, img_path2)
