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
- 1. AlexNet        
Blog Introduction Link: [博文传送门](https://www.aideeplearning.cn/alexnet%ef%bc%9a%e6%b7%b1%e5%ba%a6%e5%ad%a6%e4%b9%a0%e5%b4%9b%e8%b5%b7%e7%9a%84%e6%a0%87%e5%bf%97/)

- 2. ZFNet      
Blog Introduction Link:  [博文传送门](https://www.aideeplearning.cn/vggnet%ef%bc%9a%e5%8d%b7%e7%a7%af%e5%8e%9f%e7%90%86%e7%9a%84%e6%b7%b1%e5%ba%a6%e8%a7%a3%e6%9e%90/)

- 3. VggNet  
Blog Introduction Link:   [博文传送门](https://www.aideeplearning.cn/vggnet-%e6%8e%a2%e7%b4%a2%e6%b7%b1%e5%ba%a6%e7%9a%84%e5%8a%9b%e9%87%8f/))

- 4. GoogleNet  
Blog Introduction Link: [博文传送门](https://www.aideeplearning.cn/googlenet%ef%bc%9a%e6%8e%a2%e7%b4%a2%e5%ae%bd%e5%ba%a6%e7%9a%84%e5%8a%9b%e9%87%8f/))

- 5. ResNet  
Blog Introduction Link: [博文传送门](https://www.aideeplearning.cn/resnet%ef%bc%9a%e7%a5%9e%e6%9d%a5%e4%b9%8b%e8%b7%af/))

- 6. DenseNet  
Blog Introduction Link: [博文传送门](https://www.aideeplearning.cn/densenet%ef%bc%9a%e7%89%b9%e5%be%81%e5%a4%8d%e7%94%a8%e7%9c%9f%e9%a6%99/))

- 7. SENet  
Blog Introduction Link: [博文传送门](https://www.aideeplearning.cn/senet%ef%bc%9a%e9%80%9a%e9%81%93%e7%bb%b4%e5%ba%a6%e7%9a%84%e6%b3%a8%e6%84%8f%e5%8a%9b%e6%9c%ba%e5%88%b6/))

- 8. MobileNet  
Blog Introduction Link: [博文传送门](https://www.aideeplearning.cn/mobilenet%ef%bc%9a%e8%bd%bb%e9%87%8f%e5%8c%96%e6%a8%a1%e5%9e%8b/))

- 9. ShuffleNet  
Blog Introduction Link: [博文传送门](https://www.aideeplearning.cn/shufflenet%e8%bd%bb%e9%87%8f%e5%8c%96%e7%bd%91%e7%bb%9c/))

- 10. EfficientNet  
Blog Introduction Link:  [博文传送门](https://www.aideeplearning.cn/efficientnet%e8%bd%bb%e9%87%8f%e5%8c%96%e7%bd%91%e7%bb%9c/))

- 11. RepVGG  
Blog Introduction Link:  [博文传送门](https://www.aideeplearning.cn/repvgg%ef%bc%9a%e6%96%b0%e5%9e%8b%e5%8d%b7%e7%a7%af%e7%a5%9e%e7%bb%8f%e7%bd%91%e7%bb%9c%e6%9e%b6%e6%9e%84/)

- 11. Vision_Transformer  
Blog Introduction Link:  [博文传送门](https://www.aideeplearning.cn/vit%ef%bc%9a%e8%a7%86%e8%a7%89transformer/))

- 12. Swin_Transformer  
Blog Introduction Link:  [博文传送门](https://www.aideeplearning.cn/swin-transformer%ef%bc%9a%e7%aa%97%e5%8f%a3%e5%8c%96%e7%9a%84transformer/)

- 13. Visual Attention Network 
Blog Introduction Link:  [博文传送门](https://www.aideeplearning.cn/van%ef%bc%9a%e5%9f%ba%e4%ba%8e%e5%8d%b7%e7%a7%af%e5%ae%9e%e7%8e%b0%e7%9a%84%e6%b3%a8%e6%84%8f%e5%8a%9b/)

- 14. ConvNeXt
Blog Introduction Link:  [博文传送门](https://www.aideeplearning.cn/convnext%ef%bc%9a%e5%8d%b7%e7%a7%af%e4%b8%8e%e8%ae%be%e8%ae%a1%e7%ad%96%e7%95%a5%e7%9a%84%e6%96%b0%e7%af%87%e7%ab%a0/)

- 15. MLP-Mixer
Blog Introduction Link:  [博文传送门](https://www.aideeplearning.cn/wp-admin/post.php?post=3328&action=edit)

- 16. AS-MLP
Blog Introduction Link:  [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%8D%81%E4%B8%83as-mlp)

- 17. ConvMixer
Blog Introduction Link:  [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%8D%81%E5%85%ABconvmixer)

- 18. MetaFormer
Blog Introduction Link:  [博文传送门](http://124.220.164.99:8090/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B9%8B%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%8D%81%E4%B9%9Dmetaformer)
