# -*- coding: UTF-8 -*-
'''
@Project ：CRNN
@File    ：config.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch

# ===generator===
root_path = "文件根目录(绝对路径)"
training_path = "训练集文档路径(绝对路径)"
validate_path = "验证集文档路径(绝对路径)"
Epoches = 100
cropped_size = (100, 32)  # set heigth to 32, width to 100
batch_size = 32
mask_num = 31

# ===training===
target_size = 7
logit_size = 21
per_sample_interval = 200
device = torch.device('cuda') if torch.cuda.is_available() else None
learning_rate = 1e-4
weight_decay = 5e-4
ckpt_path = '.\\saved\\checkpoint'
font_path = '.\\font\\simhei.ttf'
sample_path = '.\\sample\\Batch{:d}.jpg'
load_ckpt = False
