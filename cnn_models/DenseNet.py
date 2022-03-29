from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F 

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        # 1. Dense layer的结构参照了ResNetV2的结构，BN->ReLU->Conv
        # 2. 与ResNet的bottleneck稍有不同的是，此处仅做两次conv（1*1conv,3*3conv)，不需要第三次1*1conv将channel拉升回去
        # 3. 由于Dense block中Tensor的channel数是随着Dense layer不断增加的,所以Dense layer设计的就很”窄“（channel数很小，固定为growth_rate)，每层仅学习很少一部分的特征
        # 4. add.module 等效于 self.norm1 = nn.Batchnorm2d(num_input_features)
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,  kernel_size=1, stride=1, bias=False)),
        
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        # 每一Dense layer结束后dropout丢弃的feature maps的比例
        self.drop_rate = drop_rate
 
    def forward(self, x):
        # 调用所有add_module方法添加到sequence的模块的forward函数。
        new_features = super(_DenseLayer, self).forward(x)
        # 若设置了dropout丢弃比例，则按比例”丢弃一部分的features“（将该部分features置为0），channel数仍为growth_rate
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        # 最后将新生成的featrue map和输入的feature map在channel维度上concat起来
        # 1.不需要像ResNet一样将x进行变换使得channel数相等, 因为DenseNet 3*3conv stride=1 不会改变Tensor的h,w，并且最后是channel维度上的堆叠而不是相加
        # 2.原文中提到的内存消耗多也是因为这步，在一个block中需要把所有layer生成的feature都保存下来
        return torch.cat([x, new_features], 1)
        
        
# Dense block其实就是多个Dense layer的叠加，需要注意的就是两个layer连接处的input_features值是逐渐增加growth_rate
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            # 由于一个DenseBlock中，每经过一个layer，宽度（channel）就会堆叠增加growth_rate，所以仅需要改变num_input_features即可
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

# Transition layer（过渡层）, 两个Dense block间加入的Transition层起到了两个作用： 防止features数无限增大，进一步压缩数据; 下采样，降低feature map的分辨率
class _Transition(nn.Sequential): 
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        # 作用1：即使每一层Dense layer都采取了很小的growth_rate，但是堆叠之后channel数难免会越来越大, 所以需要在每一个Dense block之后接transition层用1*1conv将channel再拉回到一个相对较低的值（一般为输入的一半）
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        # 作用2：用average pooling改变图像分辨率，下采样
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        
        super(DenseNet, self).__init__() 
        # First convolution
        # 和ResNet一样，先通过7*7的卷积，将分辨率从224*224->112*112
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
 
        # Each denseblock
        num_features = num_init_features
        # 读取每个Dense block层数的设定
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            # 第四个Dense block后不再连接Transition层
            if i != len(block_config) - 1:
                # 此处可以看到，默认过渡层将channel变为原来输入的一半
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        
        # Final batch norm 
        # Final global average pool
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avgpool5', nn.AvgPool2d(kernel_size=7, stride=1))
 
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
 
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
 
    def forward(self, x):
        features = self.features(x) 
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out