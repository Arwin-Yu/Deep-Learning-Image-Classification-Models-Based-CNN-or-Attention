from custom_layers.CustomMethod import make_divisible8
from torch.nn import functional as F
import torch
import torch.nn as nn

# All neural networks are implemented with nn.Module. 
# If the layers are sequentially used (self.layer3(self.layer2(self.layer1(x))), 
# you can leverage nn.Sequential to not have to define the forward function of the model.

class ConvActivation(nn.Sequential):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=None, groups=1, bias=False, activate_layer=None):
        if padding is None:
            padding = (kernel_size - 1) // 2
        if activate_layer is None:
            activate_layer = nn.ReLU
        super(ConvActivation, self).__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            activate_layer(True)
       )

class ConvBatchNormalization(nn.Sequential):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=None, groups=1, bias=False, norm_layer=None):
        if padding is None:
            padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBatchNormalization, self).__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            norm_layer(output_channels)
        )
 
class ConvBNActivation(nn.Sequential):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=None, groups=1, bias=False, norm_layer=None, activation_layer=None):
        if padding is None:
            padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU
        super(ConvBNActivation, self).__init__(
        nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=bias),
        norm_layer(output_channels),
        activation_layer(inplace=True))

    
class Inception(nn.Module):
    def __init__(self, input_channels, out_nc1x1, out_nc3x3_reduce, out_nc3x3, out_nc5x5_reduce, out_nc5x5, out_nc_pool):
        super().__init__()
        self.branch1 = ConvActivation(input_channels, out_nc1x1, kernel_size=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvActivation(input_channels, out_nc3x3_reduce, kernel_size=1, padding=0),
            ConvActivation(out_nc3x3_reduce, out_nc3x3, kernel_size=3, padding=1))
        self.branch3 = nn.Sequential(
            ConvActivation(input_channels, out_nc5x5_reduce, kernel_size=1, padding=0),
            ConvActivation(out_nc5x5_reduce, out_nc5x5, kernel_size=5, padding=2))
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvActivation(input_channels, out_nc_pool, kernel_size=1, padding=0))
    def forward(self, x):
        branch_1 = self.branch1(x)
        branch_2 = self.branch2(x)
        branch_3 = self.branch3(x)
        branch_4 = self.branch4(x)
        output = [branch_1, branch_2, branch_3, branch_4]
        return torch.cat(output, dim=1)

class Inception_s(nn.Module):
    def __init__(self, input_channels, out_nc1x1, out_nc3x3, out_nc5x5, out_nc_pool):
        super().__init__()
        self.branch1 = ConvActivation(input_channels, out_nc1x1, kernel_size=1, padding=0)
        self.branch2 = nn.Sequential( 
            ConvActivation(input_channels, out_nc3x3, kernel_size=3, padding=1))
        self.branch3 = nn.Sequential( 
            ConvActivation(input_channels, out_nc5x5, kernel_size=5, padding=2))
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvActivation(input_channels, out_nc_pool, kernel_size=1, padding=0))
    def forward(self, x):
        branch_1 = self.branch1(x)
        branch_2 = self.branch2(x)
        branch_3 = self.branch3(x)
        branch_4 = self.branch4(x)
        output = [branch_1, branch_2, branch_3, branch_4]
        return torch.cat(output, dim=1)

class SqueezeExcitation(nn.Module):
    def __init__(self, pervious_layer_channels, scale_channels=None, scale_ratio=4):
        super().__init__()
        if scale_channels is None:
            scale_channels = pervious_layer_channels
        # assert input_channels > 16, 'input channels too small, Squeeze-Excitation is not necessary'
        squeeze_channels = make_divisible8(scale_channels//scale_ratio, 8)
        self.fc1 = nn.Conv2d(pervious_layer_channels, squeeze_channels, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(squeeze_channels, pervious_layer_channels, kernel_size=1, padding=0)
    def forward(self, x):
        weight = F.adaptive_avg_pool2d(x, output_size=(1,1))
        weight = self.fc1(weight)
        weight = F.relu(weight, True)
        weight = self.fc2(weight)
        weight = F.hardsigmoid(weight, True)
        return weight * x