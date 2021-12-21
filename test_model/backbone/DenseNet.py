# -*- coding = utf-8 -*-
# @Author : lyp2333
# @Time : 2021/10/31 21:22
# @File : DenseNet.py.py
# @Software : PyCharm
import torch
import os

import torchvision.models
from torch import nn
from torchsummary import summary
from torchstat import stat
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from torch.utils import data


def _DenseLayer(input_channels, channels):
    sequ = nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(inplace=True),
        nn.Conv2d(input_channels, out_channels=channels * 4, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(channels * 4), nn.ReLU(inplace=True),
        nn.Conv2d(channels * 4, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
    )
    return sequ


def _Transition(input_channel):
    sequ = nn.Sequential(
        nn.BatchNorm2d(input_channel), nn.ReLU(inplace=True),
        nn.Conv2d(input_channel, out_channels=input_channel // 2, kernel_size=1, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    return sequ


class _Denseblock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(_Denseblock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(_DenseLayer(input_channels + i * num_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, input):
        x = input
        for block in self.net:
            y = block(x)
            x = torch.cat((x, y), dim=1)
        return x


class DenseNet(nn.Module):
    def __init__(self, num_Denseblock, numbercls=1000):
        super(DenseNet, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                 nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.denseblock1 = _Denseblock(num_Denseblock[0], 64, 32)
        self.num1 = num_Denseblock[0] * 32 + 64
        self.transition1 = _Transition(self.num1)
        self.denseblock2 = _Denseblock(num_Denseblock[1], self.num1 // 2, 32)
        self.num2 = num_Denseblock[1] * 32 + self.num1 // 2
        self.transition2 = _Transition(self.num2)
        self.denseblock3 = _Denseblock(num_Denseblock[2], self.num2 // 2, 32)
        self.num3 = num_Denseblock[2] * 32 + self.num2 // 2
        self.transition3 = _Transition(self.num3)
        self.denseblock4 = _Denseblock(num_Denseblock[3], self.num3 // 2, 32)
        self.num4 = self.num3 // 2 + num_Denseblock[3] * 32
        self.net1 = nn.Sequential(
            nn.BatchNorm2d(self.num4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.num4, numbercls, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        x = self.net(input)
        y = self.denseblock4(self.transition3(self.denseblock3(self.transition2(
            self.denseblock2(self.transition1(self.denseblock1(x)))))))
        output = self.net1(y)
        return output


def densenet121(numcls):
    return DenseNet([6, 12, 24, 16], numcls)


def densenet169(numcls):
    return DenseNet([6, 12, 32, 32], numcls)


def densenet201(numcls):
    return DenseNet([6, 12, 48, 32], numcls)


def densenet264(numcls):
    return DenseNet([6, 12, 64, 48], numcls)


if __name__ == "__main__":
    # test_code
    # input_image = torch.randn(50, 3, 224, 224)
    densenet = torchvision.models.densenet121(pretrained=True)
    densenet121 = densenet121(1000)
    densenet169 = densenet169(1000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # input_image.to(device)
    # densenet121.to(device)
    # output = densenet169(input_image)
    # print(output.shape)
    # print(type(stat(densenet121, input_size=(3, 224, 224))))
    print(stat(densenet, input_size=(3, 224, 224)))
    # print(summary(densenet.cuda(), input_size=(3, 224, 224), batch_size=-1))
