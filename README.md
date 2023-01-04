# Deep Learning Image Classification Models Based CNN or Attention
This project organizes classic Neural Networks based convolution or attention mechanism, in order to solve image classification.

This readme is introduced in Chinese (including most of the comments in the code). Please translate it into English if necessary.

## 1. Introduction

以下是本项目支持的模型列表，包含了自 AelxNet 以来经典的深度学习图像分类模型，大部分模型是基于卷积神经网络的，也有一部分是基于注意力机制的。 博客链接是对模型的介绍，会持续更新...

在项目目录中，模型的搭建代码在classic_models文件夹中；**所有的模型训练代码和推理代码都是共用的，只有模型搭建代码不同**，训练代码有三个不同的版本：
- train_sample.py是最简单的实现，必须掌握，以下两个版本看个人需求。
- train.py是简单升级版的实现，具体改进的地方见train.py脚本中顶部的注释。
- train_distrubuted.py支持多gpu分布式训练。  

最后，test.py是推理脚本，用于使用训练好的模型。dataload中是数据集加载代码；utils是封装各种功能的包，包括学习策略，训练和验证，分布式初始化，可视化等等。建议先学习掌握classic_models，train_sample.py和test.py这三部分，其他部分用到的时候再学习。

## 2. Dataset And Project 
本项目是使用python语言基于pytorch深度学习框架编写的。

默认的数据集是花朵数据集，此数据集包含五种不同种类的花朵图像，用于训练的图像有3306张，用于验证的图像有364张。下载链接如下：https://pan.baidu.com/s/1EhPMVLOQlLNN55ndrLbh4Q 
提取码：7799 。

**下载完成后，记得在训练和推理代码中，将数据集加载的路径修改成自己电脑中下载存储的路径。**

数据集图像展示如下： 
<div align="center">
  <img src="https://user-images.githubusercontent.com/102544244/192847344-958812cc-0988-4fa4-a458-ed842c41b8d2.png"  alt="Dataset show" width="700"/>
</div>
  
 
开启模型的训练只需要在IDE中执行train_sample.py脚本即可；或者在终端执行命令行`python train_sample.py` 训练的log打印示例如下：
<div align="center">
  <img src="https://user-images.githubusercontent.com/102544244/192849338-d7297768-88d4-40f8-83b6-79962ace7fd4.png"  alt="training log" width="600"/>
</div>
 
将模型用于推理只需要在IDE中执行test.py脚本即可；或者在终端执行命令行`python test.py` 给一张向日葵的图像，模型的输出结果示例结果如下：：

<div align="center">
  <img src="https://user-images.githubusercontent.com/102544244/192850216-f9ebf217-97f9-4c87-a5e5-4c1e032f436b.png"  alt="infer show" width="400"/>
</div>
 

## 3. Methods And Papers
- 1. AlexNet        
Blog Introduction Link: [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E4%BA%8Calexnet)

- 2. ZFNet      
Blog Introduction Link:  [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E4%B8%89zfnet)

- 3. VggNet  
Blog Introduction Link: [博文传送门]
(http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%9B%9Bvgg)

4. GoogleNet  
Blog Introduction Link: [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E4%BA%94googlenet)

5. ResNet  
Blog Introduction Link: [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%85%ADresnet)

6. DenseNet  
Blog Introduction Link: [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E4%B8%83densenet)

7. SENet  
Blog Introduction Link: [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%85%ABsenet)

8. MobileNet  
Blog Introduction Link: [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E4%B9%9Dmobilenet%E7%B3%BB%E5%88%97v1v1v3)

9. ShuffleNet  
Blog Introduction Link: [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%8D%81shufflenet%E7%B3%BB%E5%88%97v1v2)

10. EfficientNet  
Blog Introduction Link:  [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%8D%81%E4%B8%80eifficientnet%E7%B3%BB%E5%88%97%E7%B3%BB%E5%88%97v1v2)

11. Vision_Transformer  
Blog Introduction Link:  [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%8D%81%E4%BA%8Cvisiontransformer)

12. Swin_Transformer  
Blog Introduction Link:  [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%8D%81%E4%B8%89swintransformer)

13. Visual Attention Network 
Blog Introduction Link:  [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%8D%81%E4%BA%8Cvisiontransformer)

14. ConvNeXt
Blog Introduction Link:  [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%8D%81%E4%BA%94convnext)

15. MLP-Mixer
Blog Introduction Link:  [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%8D%81%E5%85%ADmlp-mixer)

16. AS-MLP
Blog Introduction Link:  [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%8D%81%E4%B8%83as-mlp)

17. ConvMixer
Blog Introduction Link:  [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%8D%81%E5%85%ABconvmixer)

18. MetaFormer
Blog Introduction Link:  [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%8D%81%E4%B9%9Dmetaformer)
