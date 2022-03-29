from collections import OrderedDict
from functools import partial 
from typing import Callable
from unittest import result

import torch.nn as nn
import torch
from torch import Tensor 
from custom_layers.CustomLayers import ConvBNActivation, SqueezeExcitation
from custom_layers.CustomMethod import DropPath
class MBconv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, expand_ratio, stride, se_ratio, drop_ratio, norm_layer):
        super().__init__()
        if stride not in [1,2]:
            raise ValueError('illegal stride value')
        activate_layer = nn.SiLU
        expanded_channels = input_channels * expand_ratio
        # 在EfficientNetV2中，MBConv中不存在expansion=1的情况(V1存在)
        assert expand_ratio != 1
        
        # point-wise expansion
        self.expand_conv = ConvBNActivation(input_channels, expanded_channels, kernel_size=1, stride=1, padding=0, norm_layer=norm_layer, activation_layer=activate_layer)
        # depth-wise conv
        self.dw_conv = ConvBNActivation(expanded_channels, expanded_channels, kernel_size, stride, groups=expanded_channels, norm_layer=norm_layer, activation_layer=activate_layer)
        # SE-block
        self.se_block = SqueezeExcitation(expanded_channels, input_channels, scale_ratio=4) if se_ratio > 0 else nn.Identity()
        # point-wise conv
        self.pw_conv = ConvBNActivation(expanded_channels, output_channels, kernel_size=1, stride=1, padding=0, norm_layer=norm_layer, activation_layer=nn.Identity)
        
        self.output_channels = output_channels
        self.drop_ratio = drop_ratio
        self.has_shortcut = (stride == 1 and input_channels == output_channels)
        # dropPath
        if self.drop_ratio > 0 and self.has_shortcut:
            self.dropout = DropPath(drop_ratio)
        
    def forward(self, x):
        x = self.expand_conv(x)
        x = self.dw_conv(x)
        x = self.se_block(x)
        x = self.pw_conv(x)
        
        if self.has_shortcut:
            if self.drop_ratio > 0:
                result = self.dropout(x)
            result += x
        else:
            result = x
        return result

class FusedMBConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, expand_ratio, stride, se_ratio, drop_ratio, norm_layer):
        super().__init__()
        assert stride in [1, 2]
        assert se_ratio == 0

        self.has_shortcut = ( stride==1 and input_channels == output_channels)
        self.drop_rate = drop_ratio
        self.has_expansion = expand_ratio != 1 
        activate_layer = nn.SiLU
        expanded_c = input_channels * expand_ratio
        
        # 只有当expand ratio不等于1时才有expand conv
        if self.has_expansion:
            # Expansion convolution
            self.expand_conv = ConvBNActivation(input_channels, expanded_c, kernel_size, stride, norm_layer=norm_layer, activation_layer=activate_layer)
            self.project_conv = ConvBNActivation(expanded_c, output_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Identity)
        else:
            self.project_conv = ConvBNActivation(input_channels, output_channels, kernel_size, stride, norm_layer=norm_layer, activation_layer=activate_layer)

        self.out_channels = output_channels

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_ratio
        if self.has_shortcut and drop_ratio > 0:
            self.dropout = DropPath(drop_ratio)
    
    def forward(self, x:Tensor):
        if self.has_expansion:
            reslut = self.expand_conv(x)
            result = self.project_conv(reslut)
        else:
            result = self.project_conv(x)
        
        if self.has_shortcut:
            if self.drop_rate>0:
                reslut = self.dropout(result)
            result += x
        return result

class EfficientNetV2(nn.Module):
    def __init__(self,
                 model_cnf: list,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2):
        super(EfficientNetV2, self).__init__()

        for cnf in model_cnf:
            assert len(cnf) == 8

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        stem_filter_num = model_cnf[0][4]

        self.stem = ConvBNActivation(3,
                              stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)  # 激活函数默认是SiLU

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []
        for cnf in model_cnf:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBconv
            for i in range(repeats):
                blocks.append(op(kernel_size=cnf[1],
                                 input_channels=cnf[4] if i == 0 else cnf[5],
                                 output_channels=cnf[5],
                                 expand_ratio=cnf[3],
                                 stride=cnf[2] if i == 0 else 1,
                                 se_ratio=cnf[-1],
                                 drop_ratio=drop_connect_rate * block_id / total_blocks,
                                 norm_layer=norm_layer))
                block_id += 1
        self.blocks = nn.Sequential(*blocks)

        head_input_c = model_cnf[-1][-3]
        head = OrderedDict()

        head.update({"project_conv": ConvBNActivation(head_input_c,
                                               num_features,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})  # 激活函数默认是SiLU

        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})

        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
        head.update({"classifier": nn.Linear(num_features, num_classes)})

        self.head = nn.Sequential(head)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) :
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)

        return x


def efficientnetv2_s(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 300, eval_size: 384

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 4],
                    [9, 3, 1, 6, 128, 160, 1, 4],
                    [15, 3, 2, 6, 160, 256, 1, 4]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.2)
    return model


def efficientnetv2_m(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 4],
                    [14, 3, 1, 6, 160, 176, 1, 4],
                    [18, 3, 2, 6, 176, 304, 1, 4],
                    [5, 3, 1, 6, 304, 512, 1, 4]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.3)
    return model


def efficientnetv2_l(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 4],
                    [19, 3, 1, 6, 192, 224, 1, 4],
                    [25, 3, 2, 6, 224, 384, 1, 4],
                    [7, 3, 1, 6, 384, 640, 1, 4]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4)
    return model
        
    