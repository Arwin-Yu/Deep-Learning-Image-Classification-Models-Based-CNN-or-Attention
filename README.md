# Deep-Learning-Models-For-Classification
This readme is introduced in Chinese (including most of the comments in the code). Please translate it into English if necessary.
## Introduction

以下是本项目支持的模型列表，包含了自AelxNet以来经典的深度学习分类模型，大部分模型是基于卷积神经网络的，也有一部分是基于注意力机制的。  

模型代码在classic_models文件夹中。博客链接是对模型的介绍，有一些正在编写，会持续更新...

This project organizes classic classification Neural Networks based  convolution or attention mechanism:

1. AlexNet        
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123839843  

2. ZFNet          
Blog Introduction Link: WRINTING

3. VggNet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123716634  

4. GoogleNet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123717625  

5. ResNet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123739792  

6. DenseNet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123765554  

7. MobileNet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123793236  

8. ShuffleNet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123797686  

9. SENet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123848298  

10. Vision_Transformer  
Blog Introduction Link: WRINTING  

11. Swin_Transformer  
Blog Introduction Link: WRINTING  

12. EfficientNet  
Blog Introduction Link: https://blog.csdn.net/qq_39297053/article/details/123804502  

13. ConvNeXt  
Blog Introduction Link: WRINTIN  

14. MLP-mixer  
Blog Introduction Link: WRINTING... ...  

In additon, I write training and inference python scripts for image classification task.
train.py 

## configures
本项目是使用python语言基于pytorch深度学习框架编写的。
此外，我写了三个训练脚本用于模型的训练，默认的数据集是花朵数据集，此数据集包含五种不同种类共三千多张花朵图像，下载链接：链接：https://pan.baidu.com/s/1EhPMVLOQlLNN55ndrLbh4Q 
提取码：7799 。如要使用，请指定参超到数据集地址/flower（eg： --data_path /.../.../.../flower）

三个训练脚本中，train_sample.py是最简单的实现；train.py是升级版的实现，具体改进点见train.py脚本中的注释; train_distrubuted.py支持多gpu分布式训练。  

最后，test.py是推理脚本。dataload中是数据集加载代码；utils是封装的功能包，包括学习策略，训练和验证，分布式初始化，可视化等等。
