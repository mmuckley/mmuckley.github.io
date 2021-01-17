---
title: "dldegibbs"
excerpt: "Gibbs artifact and noise removal for diffusion MRI using deep learning. Models pretrained via aggressive image augmentation on ImageNet."
link: https://github.com/mmuckley/dldegibbs
---

I am the lead developer and maintainer of [dldegibbs]((https://github.com/mmuckley/dldegibbs)), a repository for reproducing the paper [Training a neural network for Gibbs and noise removal in diffusion MRI](https://doi.org/10.1002/mrm.28395). One challenge of this project is that it's hard to collect a lot of diffusion images. Rather than overfit to a small training set, we applied aggressive augmentation and Gibbs simulation to ImageNet images and verified that the models could transfer to real-world diffusion data.
