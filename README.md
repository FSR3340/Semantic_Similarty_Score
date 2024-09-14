# Semantic_Similarty_Score

---

This is a demo of SeSS (Semantic Similarity Score) proposed in paper ["Semantic Similarity Score for Measuring Visual Similarity at Semantic Level"](https://arxiv.org/abs/2406.03865).

SeSS is a novel image similarity score for measuring differences between images at semantic level. This metric is based on Scene Graph Generation and graph matching techniques, transforming image similarity scores into graph matching scores. By manually annotating thousands of image pairs, we fine-tuned the hyperparameters within SeSS to align it more closely with human semantic perception. 

The performance of SeSS has been tested across various image datasets and specific IoT visual tasks. Experimental results demonstrate the effectiveness of SeSS in measuring differences in semantic-level information between images, making it a valuable tool for evaluating visual semantic communication systems.

The demo includes the technique of [RelateAnything](https://github.com/Luodian/RelateAnything) which combines the Meta's [Segment-Anything](https://segment-anything.com/) model with the ECCV'22 paper: [Panoptic Scene Graph Generation](https://psgdataset.org/). 

## Setup

##### 1.Environment Settings

To set up the environment, we use Conda to manage dependencies:

```bash
conda env create -f environment.yml
```

##### 2.Download model checkpoints

Download the pretrained model

1.SAM:[link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
2.RAM:[link](https://1drv.ms/u/s!AgCc-d5Aw1cumQapZwcaKob8InQm?e=qyMeTS)
3.Openclip:[link](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin)

And put these pretrained models into **"./checkpoints"**.

4.Bert:[link](https://huggingface.co/google-bert/bert-base-uncased/tree/main)

And put these files into **"./bert-base-uncased"**.

##### 3.Images preparing for evaluation

To perform image semantic difference evaluation, you first need to place the original images and the transmitted images in the respective directories **"./dataset/origin"** and **"./dataset/recover"**. Name the images sequentially as **"0001.jpg", "0002.jpg", ..., "xxxx.jpg"**.

##### 4.Run the demo

SeSS is a two-step process. 

First, run 
```
python predict.py
```
this program will generate a JSON file with the same name for each image, containing the object relationship network matrix for that image. 

Then, run 
```
python demo.py
```
to obtain the average image semantic difference score from the two image folders.

## Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```BibTex
@article{fan2024semantic,
  title={Semantic Similarity Score for Measuring Visual Similarity at Semantic Level},
  author={Fan, Senran and Bao, Zhicheng and Dong, Chen and Liang, Haotai and Xu, Xiaodong and Zhang, Ping},
  journal={arXiv preprint arXiv:2406.03865},
  year={2024}
}
```
