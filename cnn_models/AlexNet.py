from turtle import forward
import torch
import torch.nn as nn

from custom_layers.CustomLayers import ConvActivation

# 由于分类层结构是全连接，因此限制模型输入的图像大小为224*224
# 卷积后特征图尺寸不变的常用参数组合：（K=7,s=1,p=3）  （K=5,s=1,p=2） （K=3,s=1,p=1）

class AlexNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, init_weights = True):
        super().__init__()
        self.features = nn.Sequential(
            # in[3,224,224] ==> out[48,55,55] ==> out[48,27,27]
            ConvActivation(input_channels=input_channels, output_channels=48, kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # in[128,27,27] ==> out[128,27,27] ==> out[128,13,13]
            ConvActivation(input_channels=48, output_channels=128, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # in[128,13,13] ==> out[192,13,13]
            ConvActivation(input_channels=128, output_channels=192, kernel_size=3, padding=1),

            # in[192,13,13] ==> out[192,13,13] 
            ConvActivation(input_channels=192, output_channels=192, kernel_size=3, padding=1),
 
            # in[192,13,13] ==> out[128,13,13] ==> out[128,6,6]
            ConvActivation(input_channels=192, output_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6, 2048),
            nn.ReLU(True),

            nn.Dropout(p=0.4),
            nn.Linear(2048, 2048),
            nn.ReLU(True),

            nn.Linear(2048, num_classes)
        )

        if init_weights:
            self._initialize_weights()           
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

    def forward(self, x):
        assert x.shape[2] == 224 and x.shape[3] == 224, " input images size should be 224*224 "
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    