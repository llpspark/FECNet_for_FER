# A Pytorch mplementation for FECNet 

- A Pytorch implementation of the paper: "A Compact Embedding for Facial Expression Similarity". 
- Note: data selection and net structure have some difference from original paper.   

# Pipeline

* Data Preparation
* DataSet download:
  - Download image lists from: [here](https://ai.google/tools/datasets/google-facial-expression/)
  - Run  this command to download images from csv(place these images under "datas/train/" folder): python download_image.py
  - Run this command to extract trainable csv data: python export_train_label.py
* Train
  * Run this command under root dir to train the net: python train_fecnet.py

# Reference

* https://github.com/GerardLiu96/FECNet
* https://github.com/tbmoon/facenet
