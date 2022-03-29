from typing import List

import torch
from torch import Tensor
import torch.nn as nn 
from custom_layers.CustomLayers import ConvBatchNormalization, ConvBNActivation
from custom_layers.CustomMethod import channel_shuffle



class ShuffleResidual(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        
        if stride not in [1,2]:
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = output_channels //2
        assert output_channels % 2 ==0
        # 当stride为1时，input_channel应该是branch_features的两倍, python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride !=1) or (input_channels == branch_features <<1)
        
        if self.stride == 2:
            self.branch1 = nn.Sequential(
                # depth-wise conv and bn
                ConvBatchNormalization(input_channels, input_channels, kernel_size=3, stride=self.stride, padding=1, groups=input_channels),
                # point-wise conv and bn
                ConvBNActivation(input_channels, branch_features, kernel_size=1, stride=1, padding=0)           
            )
        else:
            self.branch1 = nn.Sequential()
        
        input_c = input_channels if self.stride >1 else branch_features
        self.branch2 = nn.Sequential(
            # point-wise conv
            ConvBNActivation(input_channels=input_c, output_channels=branch_features, kernel_size=1, stride=1, padding=0),
            # depth-wise conv
            ConvBatchNormalization(input_channels=branch_features, output_channels=branch_features, kernel_size=3, stride=self.stride, padding=1, groups=branch_features),
            # point-wise conv
            ConvBNActivation(input_channels=branch_features, output_channels=branch_features, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x:Tensor):
        if self.stride == 1:
            x1 , x2 = x.chunk(2, dim=1)
            x1 = x1
            x2 =  self.branch2(x2)
            out = torch.cat((x1, x2), dim=1)
        else:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            out = torch.cat((x1, x2), dim=1)
        
        out = channel_shuffle(out, 2)
        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats: List[int], stages_out_channels:List[int], num_classes:int, shuffle_residual = ShuffleResidual):
        super(ShuffleNetV2, self).__init__()
        
        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels
        
        # input RGB images
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        
        self.conv1 =  ConvBNActivation(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False)
     
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)        
        self.stage2 = nn.Sequential
        self.stage3 = nn.Sequential
        self.stage4 = nn.Sequential
        
        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [shuffle_residual(input_channels, output_channels, 2)]
            for i in range(repeats -1):
                seq.append(shuffle_residual(output_channels, output_channels,1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        
        output_channels = self._stage_out_channels[-1]
        self.conv5 = ConvBNActivation(input_channels, output_channels, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(output_channels, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3]) # global pooling
        x = self.fc(x)
        return x

def shufflenet_v2_x1_0(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x0_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth
    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024],
                         num_classes=num_classes)

    return model