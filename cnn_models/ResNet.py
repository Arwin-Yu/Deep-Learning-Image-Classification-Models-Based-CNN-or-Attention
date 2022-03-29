from turtle import forward
from numpy import identity
import torch
import torch.nn as nn 
from torch.nn import functional as F
from custom_layers.CustomLayers import ConvActivation, ConvBNActivation, ConvBatchNormalization

class SmallResidual(nn.Module):
    expansion = 1    
    def __init__(self, input_channels, output_channels, stride=1, downsample=None, **kwargs):
        super().__init__()
        self.conv1 = ConvBNActivation(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = ConvBatchNormalization(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample= downsample        
    def forward(self, x): 
        indentity = x
        if self.downsample is not None:
            indentity = self.downsample(x)
        # F.ReLU()是函数调用，一般使用在foreward函数里。而nn.ReLU()是模块调用，一般在定义网络层的时候使用。
        out = self.conv1(x)
        out = self.conv2(out)
        out += indentity
        out = F.relu(out, True) 
        return out
 
class BigResidual(nn.Module):
    expansion = 4
    # groups是组卷积； 用于实现ResNeXt
    def __init__(self, input_channels, output_channels, stride=1, downsample=None, groups=1, width_per_group=64, **kwargs):
        super().__init__()
        width = int(output_channels*(width_per_group/64))*groups
        
        self.conv1 = ConvBNActivation(input_channels=input_channels, output_channels=width, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNActivation(input_channels=width, output_channels=width, kernel_size=3, stride=stride, padding=1, groups=groups)
        self.conv3 = ConvBatchNormalization(input_channels=width, output_channels=output_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.downsample = downsample
    
    def forward(self, x):
        indentity = x
        if self.downsample is not None:
            indentity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + indentity
        out = F.relu(out, True)
        return out
class ResNet(nn.Module):
    
    def __init__(self, which_residual, num_blocks, num_classes=None, include_top=None, groups=1, width_per_groups=64):
        super().__init__()
        self.include_top = include_top
        self.in_nc = 64
        self.groups=groups
        self.width_per_groups = width_per_groups
        
        self.conv1 = nn.Conv2d(3, self.in_nc, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_nc)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.Block1 = self._make_Block(which_residual, 64, num_blocks[0])
        self.Block2 = self._make_Block(which_residual, 128, num_blocks[1], stride=2)
        self.Block3 = self._make_Block(which_residual, 256, num_blocks[2], stride=2)
        self.Block4 = self._make_Block(which_residual, 512, num_blocks[3], stride=2)
        
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512*which_residual.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.Block4(x)
        
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x,start_dim=1)
            x = self.fc(x)
        return x
    
    def _make_Block(self, which_residual, channel, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_nc != channel * which_residual.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_nc, channel * which_residual.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * which_residual.expansion))
        
        Block = []
        Block.append(which_residual(self.in_nc,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups = self.groups,
                            width_per_group=self.width_per_groups))
        self.in_nc = channel * which_residual.expansion
        
        for _ in range(1, num_block):
            Block.append(which_residual(self.in_nc, channel, groups=self.groups, width_per_group=self.width_per_groups))
        return nn.Sequential(*Block)
 

def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(SmallResidual, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(BigResidual, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(BigResidual, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(BigResidual, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(BigResidual, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

