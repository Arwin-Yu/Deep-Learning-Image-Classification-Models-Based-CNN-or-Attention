import torch.nn as nn
import torch
from torchsummary import summary

# 与AlexNet有两处不同： 1. 第一次的卷积核变小，步幅减小。 2. 第3，4，5层的卷积核数量增加了。
class ZFNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=2),  # input[3, 224, 224]  output[96, 111, 111]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[96, 55, 55]

            nn.Conv2d(96, 256, kernel_size=5, padding=2),           # output[256, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[256, 27, 27]

            nn.Conv2d(256, 512, kernel_size=3, padding=1),          # output[512, 27, 27]
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),          # output[1024, 27, 27]
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 512, kernel_size=3, padding=1),          # output[512, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[512, 13, 13]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 13 * 13, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def zfnet(num_classes): 
    model = ZFNet(num_classes=num_classes)
    return model

# net = ZFNet(num_classes=1000)
# summary(net.to('cuda'), (3,224,224))
#########################################################################################################################################
# Total params: 386,548,840
# Trainable params: 386,548,840
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 57.77
# Params size (MB): 1474.57
# Estimated Total Size (MB): 1532.91
# ----------------------------------------------------------------
# conv_parameters:  11,247,744 相比于AelxNet的cnn层参数  3,747,200   增加 3 倍
# fnn_parameters:  375,301,096 相比于AelxNet的fnn层参数 58,631,144   增加 6.4 倍
# 卷积参数占全模型参数的 2% ；全连接层占 98%

 