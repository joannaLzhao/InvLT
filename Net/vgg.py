import torch
import torch.nn as nn
import torch.nn.functional as F


cfgs = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
              512, 512, 512, 512, "M",
              512, 512, 512, 512, "M"],
}


def make_layers(cfg, in_channels=3, batch_norm=True):
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, cfg, num_classes=10, temp=1.0, batch_norm=True):
        super(VGG, self).__init__()

        self.features = make_layers(cfg, batch_norm=batch_norm)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        self.temp = temp

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        out = out / self.temp
        return out


def vgg11(temp=1.0, **kwargs):
    return VGG(cfgs["VGG11"], temp=temp, **kwargs)


def vgg13(temp=1.0, **kwargs):
    return VGG(cfgs["VGG13"], temp=temp, **kwargs)


def vgg16(temp=1.0, **kwargs):
    return VGG(cfgs["VGG16"], temp=temp, **kwargs)


def vgg19(temp=1.0, **kwargs):
    return VGG(cfgs["VGG19"], temp=temp, **kwargs)
