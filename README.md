## PyTorch实现Convolutional Recurrent Neural Network implemented by Multihead Attention
---

## 目录
1. [所需环境 Environment](#所需环境)
2. [模型结构 Structure](#模型结构)
3. [注意事项 Cautions](#注意事项)
4. [文件下载 Download](#文件下载)
5. [训练步骤 How2train](#训练步骤)
6. [预测效果 predict](#位置编码)
7. [参考资料 Reference](#参考资料)

## 所需环境
1. Python3.7
2. PyTorch>=1.7.0+cu110
3. numpy==1.19.5
4. Pillow==8.2.0
5. CUDA 11.0+  

## 模型结构
CRNN  
<p align="center">
----------------------------------------------------------------  
<p align="center">
        Layer (type)               Output Shape         Param #  
<p align="center">
================================================================  
<p align="center">
            Conv2d-1          [-1, 64, 32, 100]           1,792  
<p align="center">
         MaxPool2d-2           [-1, 64, 16, 50]               0  
<p align="center">
            Conv2d-3          [-1, 128, 16, 50]          73,856  
<p align="center">
         MaxPool2d-4           [-1, 128, 8, 25]               0  
<p align="center">
            Conv2d-5           [-1, 256, 8, 25]         295,168  
<p align="center">
       BatchNorm2d-6           [-1, 256, 8, 25]             512  
<p align="center">
            Conv2d-7           [-1, 256, 8, 25]         590,080  
<p align="center">
         MaxPool2d-8           [-1, 256, 5, 24]               0  
<p align="center">
            Conv2d-9           [-1, 512, 5, 24]       1,180,160  
<p align="center">
      BatchNorm2d-10           [-1, 512, 5, 24]           1,024  
<p align="center">
           Conv2d-11           [-1, 512, 5, 24]       2,359,808  
<p align="center">
        MaxPool2d-12           [-1, 512, 3, 23]               0  
<p align="center">
           Conv2d-13           [-1, 512, 1, 21]       2,359,808  
<p align="center">
      BatchNorm2d-14           [-1, 512, 1, 21]           1,024  
<p align="center">
           Linear-15              [-1, 21, 512]         262,656  
<p align="center">
           Linear-16              [-1, 21, 512]         262,656  
<p align="center">
           Linear-17              [-1, 21, 512]         262,656 
<p align="center"> 
           Linear-18              [-1, 21, 512]         262,656  
<p align="center">
        LayerNorm-19              [-1, 21, 512]           1,024  
<p align="center">
MultiHeadAttention-20  [[-1, 21, 512], [-1, 21, 21]]          0  
<p align="center">
           Linear-21              [-1, 21, 512]         262,656  
<p align="center">
           Linear-22              [-1, 21, 512]         262,656  
<p align="center">
           Linear-23              [-1, 21, 512]         262,656  
<p align="center">
           Linear-24              [-1, 21, 512]         262,656  
<p align="center">
        LayerNorm-25              [-1, 21, 512]           1,024  
<p align="center">
MultiHeadAttention-26  [[-1, 21, 512], [-1, 21, 21]]          0  
<p align="center">
           Linear-27               [-1, 21, 66]          33,858  
<p align="center">
================================================================  
<p align="center">
Total params: 9,000,386  
<p align="center">
Trainable params: 9,000,386  
<p align="center">
Non-trainable params: 0  
<p align="center">
----------------------------------------------------------------  
<p align="center">
Input size (MB): 0.04  
<p align="center">
Forward/backward pass size (MB): 65.34 
<p align="center"> 
Params size (MB): 34.33  
<p align="center">
Estimated Total Size (MB): 99.72 
<p align="center"> 
----------------------------------------------------------------  

## 注意事项  
** 此项目使用CCPD2019数据集，专门针对车牌的单行定长文本进行识别，如有多行可变长文本需求，请自主更改或关注作者并发件至m13541280433@163.com详谈 **  
1. 确保图像高、宽尺寸不一致，将特征压缩至高、宽的其中一维(视场景而定)
2. 原始CRNN使用BiLSTM堆叠信息量，此处改用MultiHeadAttention压缩信息量
3. 可设置sequence_mask，增强字符逻辑推理
4. 为维持图像的尺寸比例，增加了灰条处理
5. 加入权重正则化操作，防止过拟合
6. 使用CTC Loss降低端对端文本识别中重复、空白字符对真实字符造成的影响
7. 训练过程中的文本识别准确率低于真实准确率

## 文件下载    
https://github.com/detectRecog/CCPD
将train.txt、val.txt路径放置于config.py即可。

## 训练步骤
运行train.py即可开始训练。  

## 预测效果
sample1  
![image]()  

sample2  
![image]() 
sample3  
![image]()  

## 参考资料  
https://arxiv.org/pdf/1507.05717.pdf  
