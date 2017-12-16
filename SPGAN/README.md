# -- This version is not fully edited and will be updated soon --
# SPGAN

----------


**Paper**: Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification (https://arxiv.org/pdf/1711.07027.pdf)

Tensorflow implementation of Similarity Preserving cycleconsistent Generative Adversarial Network ([SPGAN](https://arxiv.org/pdf/1711.07027.pdf)), mostly modified from https://github.com/LynnHo/CycleGAN-Tensorflow-PyTorch-Simple. Thank you for your kindly attention.

----------

## Visual examples of image-image translation
![](./pics/fig1.PNG)

# Prerequisites
- tensorflow r1.0
- python 2.7

# Usage
```
cd SPGAN-master
```

## Download Datasets
- Download the Market-1501 dataset and DukeMTMC-reID:

> Market-1501:  http://liangzheng.com.cn/Project/project_reid.html 
> 
> DukeMTMC-reID: https://github.com/layumi/DukeMTMC-reID_evaluation

- Put the bounding_box_train of Duke and bounding_box_train of Market to ./Datasets/market2duke/ :
![](./pics/fig2.PNG)
## Train Example
```bash
python train.py --dataset=market2duke --gpu_id=0
```

## Test Example
```bash
python test.py --dataset=market2duke
```
