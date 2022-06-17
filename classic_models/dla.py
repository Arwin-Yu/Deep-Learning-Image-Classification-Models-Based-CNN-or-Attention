import math 
from torchsummary import summary
import torch
from torch import nn  

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1 ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,  stride=stride, padding=1,  bias=False )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x 
        out = self.conv1(x) 
        out = self.bn1(out)
        out = self.relu(out) 
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out
 
class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d( in_channels, out_channels, kernel_size, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,  root_residual=False):
        super(Tree, self).__init__()

        self.level_root = level_root
        self.levels = levels
        self.root_dim = root_dim
        self.downsample = None
        self.project = None


        if root_dim == 0:
            root_dim = 2 * out_channels

        if level_root:
            root_dim += in_channels

        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride)
            self.tree2 = block(out_channels, out_channels, stride=1)
            self.root = Root(root_dim, out_channels, root_kernel_size,  root_residual)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, 
                              root_dim=0, 
                              root_kernel_size=root_kernel_size, 
                              root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size, 
                              root_residual=root_residual)
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)

        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom

        if self.level_root:
            children.append(bottom)

        x1 = self.tree1(x, residual)

        if self.levels == 1:
            x2 = self.tree2(x1)
            out = self.root(x2, x1, *children)
        else:
            children.append(x1)
            out = self.tree2(x1, children=children)
        return out


class DLA(nn.Module):
    def __init__(self, layers, channels, num_classes=1000, block=BasicBlock, residual_root=False, pool_size=7 ):
        super().__init__()
        self.channels = channels 
        self.num_classes = num_classes

        self.patchfy_stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True))

        self.stage_0 = self._make_conv_level(channels[0], channels[0], layers[0])

        self.stage_1 = self._make_conv_level(channels[0], channels[1], layers[1], stride=2)

        self.stage_2 = Tree(layers[2], block, channels[1], channels[2], stride=2,
                           level_root=False, root_residual=residual_root)
        self.stage_3 = Tree(layers[3], block, channels[2], channels[3], stride=2,
                           level_root=True, root_residual=residual_root)
        self.stage_4 = Tree(layers[4], block, channels[3], channels[4], stride=2,
                           level_root=True, root_residual=residual_root)
        self.stage_5 = Tree(layers[5], block, channels[4], channels[5], stride=2,
                           level_root=True, root_residual=residual_root)

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
 

    def _make_conv_level(self, inplanes, planes, num_layers, stride=1 ):
        modules = []
        for i in range(num_layers):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=1, bias=False ),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        stages_features_list = []
        x = self.patchfy_stem(x)
        for i in range(6):
            x = getattr(self, 'stage_{}'.format(i))(x)
            stages_features_list.append(x) 
            
        x = self.avgpool(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        
        return x
 

def dla34(num_classes, **kwargs):  # DLA-34
    model = DLA(layers=[1, 1, 1, 2, 2, 1],
                channels=[16, 32, 64, 128, 256, 512],
                block=BasicBlock, num_classes=num_classes, **kwargs)
 
    return model

# net = dla34(5)
# summary(net.to('cuda'), (3, 224,224))
