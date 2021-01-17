---
title: "CovidPrognosis"
excerpt: "COVID patient deterioration prediction based on X-ray images. We achieve this via MoCo-based self-supervised representation learning and multi-image prediction."
---

I am a maintainer of [CovidPrognosis](https://github.com/facebookresearch/CovidPrognosis), which contains code for reproducing the paper, [COVID-19 Deterioration Prediction via Self-Supervised Representation Learning and Multi-Image Prediction](https://arxiv.org/abs/2101.04909). For this project we pretrained models for X-ray image classification by using the [Momentum Contrast method](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html) on large, non-COVID public X-ray datasets. We then fine-tuned them on anonymized COVID X-ray data shared with our group at FAIR by the NYU School of Medicine. As part of the project we open-sourced the MoCo pretrained models.