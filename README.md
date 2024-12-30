# Deep Learning Image Classification Models Based CNN or Attention
This project organizes classic Neural Networks based convolution or attention mechanism, in order to solve image classification.

This readme is introduced in Chinese (including most of the comments in the code). Please translate it into English if necessary.

## 1. Introduction

本项目包含了自 AelxNet 以来经典的深度学习图像分类模型，大部分模型是基于卷积神经网络的，也有一部分是基于注意力机制的。 博客链接是对模型的介绍，会持续更新...

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
以下是本项目支持的模型列表
1. **[AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)**
   - *ImageNet Classification with Deep Convolutional Neural Networks*

2. **[ZFNet]([https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Zhang_Visualizing_Deep_Learning_2013_CVPR_paper.pdf](https://arxiv.org/abs/1311.2901))**
   - *Visualizing and Understanding Convolutional Networks*

3. **[VGGNet](https://arxiv.org/abs/1409.1556)**
   - *Very Deep Convolutional Networks for Large-Scale Image Recognition*

4. **[GoogleNet (Inception v1)](https://arxiv.org/abs/1409.4842)**
   - *Going Deeper with Convolutions*

5. **[ResNet](https://arxiv.org/abs/1512.03385)**
   - *Deep Residual Learning for Image Recognition*

6. **[DenseNet](https://arxiv.org/abs/1608.06993)**
   - *Densely Connected Convolutional Networks*

7. **[SENet](https://arxiv.org/abs/1709.01507)**
   - *Squeeze-and-Excitation Networks*

8. **[MobileNet](https://arxiv.org/abs/1704.04861)**
   - *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*

9. **[ShuffleNet](https://arxiv.org/abs/1707.01083)**
   - *ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices*

10. **[EfficientNet](https://arxiv.org/abs/1905.11946)**
    - *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*

11. **[RepVGG](https://arxiv.org/abs/2101.03697)**
    - *RepVGG: Making VGG-style ConvNets Great Again*

12. **[Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)**
    - *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*

13. **[Swin Transformer](https://arxiv.org/abs/2103.14030)**
    - *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*

14. **[Visual Attention Network](https://arxiv.org/abs/2012.08776)**
    - *Visual Attention Network for Efficient Image Recognition*

15. **[ConvNeXt](https://arxiv.org/abs/2201.03545)**
    - *A ConvNet for the 2020s*

16. **[MLP-Mixer](https://arxiv.org/abs/2105.01601)**
    - *MLP-Mixer: An All-MLP Architecture for Vision*

17. **[AS-MLP](https://arxiv.org/abs/2208.11842)**
    - *AS-MLP: Adaptive Selective MLP for Efficient Vision*

18. **[ConvMixer](https://arxiv.org/abs/2112.10752)**
    - *Patches Are All You Need?*

19. **[MetaFormer](https://arxiv.org/abs/2201.09588)**
    - *MetaFormer is Actually What You Need for Vision*

---

