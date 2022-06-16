import torch
import torch.nn as nn

MODULE_SUCCESSION = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class VGG(nn.Module):
    def __init__(self, in_channels=3, class_nb=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(in_channels)
        self.classifier = nn.Linear(512, class_nb)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        softmax_output = nn.Softmax()
        out = softmax_output(out)
        return out

    def _make_layers(self, in_channels_nb=3):
        layers = []
        in_channels = in_channels_nb
        reduce_pooling = False
        for x in MODULE_SUCCESSION:
            if x == 'M':
                if reduce_pooling:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
                else: 
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if in_channels == 512 and in_channels_nb == 1 :
                    reduce_pooling = True
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)       