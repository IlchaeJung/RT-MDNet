## RT-MDNet: Real-Time Multi-Domain Convolutional Neural Network Tracker

Created by [Ilchae Jung](cvlab.postech.ac.kr/~chey0313), [Jeany Son](cvlab.postech.ac.kr/~jeany), [Mooyeol Baek](cvlab.postech.ac.kr/~mooyeol), and [Bohyung Han](cvlab.snu.ac.kr/~bhhan) 

### Introduction
RT-MDNet is the real-time extension of [MDNet](http://cvlab.postech.ac.kr/research/mdnet/) and is the state-of-the-art real-time tracker.
Detailed description of the system is provided by our [paper](https://arxiv.org/pdf/1808.08834.pdf)

### Citation
If you're using this code in a publication, please cite our paper.

	@InProceedings{rtmdnet,
	author = {Jung, Ilchae and Son, Jeany and Baek, Mooyeol and Han, Bohyung},
	title = {Real-Time MDNet},
	booktitle = {The IEEE European Conference on Computer Vision (ECCV)},
	month = {Sept},
	year = {2018}
	}
  
### License
This software is being made available for research purpose only.
Check LICENSE file for details.

### System Requirements

This code is tested on 64 bit Linux (Ubuntu 14.04 LTS).

**Prerequisites** 
  0. MATLAB (tested with R2014a)
  0. MatConvNet (tested with version 1.0-beta10, included in this repository)
  0. For GPU support, a GPU (~2GB memory) and CUDA toolkit according to the [MatConvNet installation guideline](http://www.vlfeat.org/matconvnet/install/) will be needed.
0. Pre-requisite \newline
0.1. Pre-training Datasets 
ImageNet-Vid: 

1. Run test

2. Train model

3. Pre-trained model

