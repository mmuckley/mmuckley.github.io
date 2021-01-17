---
title: "DlDeGibbs"
excerpt: "Gibbs artifact and noise removal using deep learning. Models pretrained via aggressive image augmentation on ImageNet."
---

I developed [DlDeGibbs](https://doi.org/10.1002/mrm.28395) ([repository](https://github.com/mmuckley/dldegibbs)), a method for Gibbs and noise removal in diffusion MRI using deep learning. One challenge of this project is that it's hard to collect a lot of diffusion images. Rather than overfit to a small training set, we applied aggressive augmentation and Gibbs simulation to ImageNet images and verified that the models could transfer to real-world diffusion data.