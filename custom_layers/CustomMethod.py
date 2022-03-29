from torch import Tensor
import torch
import torch.nn as nn

def make_divisible8(channels, divisor=8, min_channel=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_channel is None:
        min_channel = divisor
    new_channels = max(min_channel, int(channels + divisor/2) // divisor*divisor)    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return new_channels

def channel_shuffle(x: Tensor, groups:int):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # shuffle
    x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)
    return x

def drop_path(x:Tensor, drop_probability:float=0., training:bool=False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    if drop_probability == 0. or not training:
        return x
    survival_probability = 1 - drop_probability
    shape = (x.shape[0],) + (1,)*(x.ndim -1) # work with diff dim tensors, not just 2D ConvNets. Output size = [batchsize, 1,1,1....]
    random_tensor = survival_probability + torch.rand(shape, dtype=x.dtype, device=x.device)
    # binarize(surival or dead)
    random_tensor.floor()
    # 假设一个神经元的输出激活值为a，在不使用dropout的情况下，其输出期望值为a，如果使用了dropout，神经元就可能有保留和关闭两种状态，
    # 把它看作一个离散型随机变量，它就符合概率论中的0-1分布，其输出激活值的期望变为(1-p)*a+p*0= (1-p)a，此时若要保持期望和不使用dropout时一致，就要除以 (1-p)
    output = x.div(survival_probability) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)