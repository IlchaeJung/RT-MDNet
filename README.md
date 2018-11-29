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

This code is tested on 64 bit Linux (Ubuntu 16.04 LTS).

**Prerequisites** 
  0. PyTorch (>= 0.2.1)
  0. For GPU support, a GPU (~2GB memory for test) and CUDA toolkit.
  0. Training Dataset (ImageNet-Vid) if needed.
  
### Online Tracking

**Pretrained Models**
If you only run the tracker, you can use the pretrained models: 
[RT-MDNet-ImageNet-pretrained](dropbox)

**Demo**

### Learning RT-MDNet
**Preparing Datasets**
  0. If you download ImageNet-Vid dataset, you run 'modules/prepro_data_imagenet.py' to parse meta-data from dataset. After that, 'imagenet_refine.pkl' is generized.
  0. type the path of 'imagenet_refine.pkl' in 'train_mrcnn.py'
**Demo**
  0. Run 'train_mrcnn.py' after hyper-parameter tuning suitable to the capacity of your system.
  
