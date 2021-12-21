# -*- coding = utf-8 -*-
# @Author : lyp2333
# @Time : 2021/10/26 16:10
# @File : ResNext.py
# @Software : PyCharm
import math

import torch
from torch import nn
from torch import cuda
from model.conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
from tqdm import tqdm
from time import sleep
import numpy as np


class BottleNeck_Resnext(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, groups=32, down_sample=None):
        super().__init__()
        self.channel = math.ceil(out_channel / 2)
        self.conv1 = nn.Conv2d(in_channel, self.channel, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channel)

        self.conv2 = nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, bias=False, stride=1,
                               groups=groups)
        self.bn2 = nn.BatchNorm2d(self.channel)

        self.conv3 = nn.Conv2d(self.channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(False)

        self.downsample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))  # bs,c,h,w
        out = self.relu(self.bn2(self.conv2(out)))  # bs,c,h,w
        out = self.relu(self.bn3(self.conv3(out)))  # bs,4c,h,w

        if (self.downsample != None):
            residual = self.downsample(residual)

        out += residual
        return self.relu(out)


class ResNeXt(nn.Module):
    stemlayer_channeloutput = 64

    def __init__(self, block, layers_number, number_class=1000):
        super(ResNeXt, self).__init__()
        # 公共layer
        self.conv1 = nn.Conv2d(3, self.stemlayer_channeloutput, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.stemlayer_channeloutput)
        self.relu1 = nn.ReLU(False)
        self.maxpooling1 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        # 自定义部分layer
        self.layer1 = self.stage_make(BottleNeck_Resnext, self.stemlayer_channeloutput,
                                      self.stemlayer_channeloutput * 4,
                                      layers_number[0], stride=1)
        self.layer2 = self.stage_make(BottleNeck_Resnext, self.stemlayer_channeloutput * 4,
                                      self.stemlayer_channeloutput * 4 * 2, layers_number[1], stride=2)
        self.layer3 = self.stage_make(BottleNeck_Resnext, self.stemlayer_channeloutput * 4 * 2,
                                      self.stemlayer_channeloutput * 4 * 2 * 2, layers_number[2], stride=2)
        self.layer4 = self.stage_make(BottleNeck_Resnext, self.stemlayer_channeloutput * 4 * 2 * 2,
                                      self.stemlayer_channeloutput * 4 * 2 * 2 * 2, layers_number[3], stride=2)

        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512 * 4, number_class)
        self.softmax = nn.Softmax(-1)##注意，为了更改类别，不能写确定的值

    def forward(self, input):
        # 公共部分连接
        out = self.maxpooling1(self.relu1(self.bn1(self.conv1(input))))
        # 自定义部分连接
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 分类器连接
        out = self.avgpool(out)  # bs,1,1,512*4
        out = out.reshape(out.shape[0], -1)  # bs,512*4
        out = self.classifier(out)  # bs,1000
        out = self.softmax(out)
        return out

    def stage_make(self, block_type, input_channel, output_channel, block_number, stride=1):
        layer = []
        # 存在尺度不匹配的问题，无法进行残差操作，需进行下采样,每一块的第一层均需进行升维
        downsample_ornot = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride, bias=False)
        block1 = block_type(input_channel, output_channel, down_sample=downsample_ornot, stride=stride)
        layer.append(block1)
        for remain in range(1, block_number):
            block_remain = block_type(output_channel, output_channel)
            layer.append(block_remain)
        return nn.Sequential(*layer)


def ResNeXt_50(Number_classes):
    return ResNeXt(BottleNeck_Resnext, [3, 4, 6, 3], number_class=Number_classes)


if __name__ == '__main__':
    # test
    # test_image = torch.randn(50, 3, 224, 224)
    # resnext_50 = ResNeXt_50(1000)
    # print(resnext_50(test_image).shape)
    # test = torch.ones(3,4)
    # print(test)
    # test1 = torch.ones_like(test,requires_grad=False)
    # print(test1)
    # mode_resnext = ResNeXt_50(1000)
    # if cuda.is_available():
    #     mode_resnext.to('cuda')
    # mode_resnext.to('cpu')
    list = [[1.,2,3,4],[1,2,3,4]]
    print(type(list))
    test = np.array(list)
    print(type(test))
    lyp = torch.tensor(list)
    print(type(lyp),'\n',lyp)
    lyp2 = lyp
    lyp3 = lyp ** lyp2
    lyp4 = lyp3.numpy()
    print(type(lyp4))
    list = [3,4]
    a = np.arange(0,10,1)
    a.reshape(2,5)

    test = np.array([1,2])
    b = torch.tensor([[1],[2]])
    c = torch.tensor(test).reshape(2,1)
    print("b:",b.size(),"c:",c.size())
    wo = (lyp,c,c)
    test2 = torch.cat(wo,dim=1)
    print(test2)


