# Learning-via-Translation
Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification (https://arxiv.org/pdf/1711.07027.pdf) 

----------
### Framework Overview
![](./pics/fig1.PNG)
Learning via translation for domain adaptation in person re-ID consists of two steps:
**1. Source-target image translation**
The first step is to translate the annotated dataset from source domain to target domain in an unsupervised manner.

For more reference, you can find our modified training code and generating code in ./SPGAN. We wrote a detailed README. If you still has some question, feel free to contact me (dengwj16@gmail.com).
**2. Feature learning**
With the translated dataset that contain labels, feature learning methods are applied to train re-ID models.
