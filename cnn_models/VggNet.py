from turtle import forward
import torch
import torch.nn as nn
from custom_layers.CustomLayers import ConvActivation
# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
} 
# note: if use pretrain parameters, minus the mean of ImageNet(123.68, 116.78, 103.94) to normalize the dataset

cfgs_feature = {
    'vgg11': [64, 'Pooling', 128, 'Pooling', 256, 256, 'Pooling', 512, 512, 'Pooling', 512, 512, 'Pooling'],
    'vgg13': [64, 64, 'Pooling', 128, 128, 'Pooling', 256, 256, 'Pooling', 512, 512, 'Pooling', 512, 512, 'Pooling'],
    'vgg16': [64, 64, 'Pooling', 128, 128, 'Pooling', 256, 256, 256, 'Pooling', 512, 512, 512, 'Pooling', 512, 512, 512, 'Pooling'],
    'vgg19': [64, 64, 'Pooling', 128, 128, 'Pooling', 256, 256, 256, 256, 'Pooling', 512, 512, 512, 512, 'Pooling', 512, 512, 512, 512, 'Pooling'],
}

def create_feature_layers(cfgs:list, input_channels=3):
    feature_layers=[] 
    for layer in cfgs:
        if layer == 'Pooling':
            feature_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else: 
            feature_layers += [ConvActivation(input_channels, layer, kernel_size=3, stride=1, padding=1)]
            input_channels = layer
    return nn.Sequential(*feature_layers)

class VggNet(nn.Module):
    def __init__(self, num_classes, feature_layers_type='vgg16', init_weights=True):
        super().__init__()
        assert feature_layers_type in cfgs_feature, "Warning: feature_layers_type not in cfgs dict!"

        self.feature_layers = create_feature_layers(cfgs=cfgs_feature[feature_layers_type]) 
        self.classifier_layers = nn.Sequential(*[
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),

            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(p=0.4),

            nn.Linear(4096, num_classes)
        ])
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                   nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.feature_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier_layers(x)
        return x
    
 