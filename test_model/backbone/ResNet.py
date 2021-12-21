# -*- coding = utf-8 -*-
# @Author : lyp2333
# @Time : 2021/10/24 1:36
# @File : ResNet.py
# @Software : PyCharm

import math
from tensorboardX import SummaryWriter
import torch
import torchvision.models
from torchsummary import summary
from torch import nn
from test_model.utils.freeze_unfreeze import *
from test_model.utils.eval_acc import *

class BottleNeck_large(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, padding=1, down_sample=None, stride=1):
        super().__init__()
        self.channel = math.ceil(out_channel / 4)
        self.conv1 = nn.Conv2d(in_channel, self.channel, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channel)
        self.relu1 = nn.ReLU(False)
        self.conv2 = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.channel)
        self.relu2 = nn.ReLU(False)
        self.conv3 = nn.Conv2d(self.channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu3 = nn.ReLU(False)

        self.up_sample1 = down_sample
        self.bn4 = nn.BatchNorm2d(out_channel)
        self.stride1 = stride

        self.relu_final = nn.ReLU(False)

    def forward(self, input):
        input2 = input
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)))

        if (self.up_sample1):
            input1 = self.up_sample1(input)
            input2 = self.bn4(input1)
        out += input2

        return self.relu_final(out)


class ResNet(nn.Module):
    stemlayer_channeloutput = 64

    def __init__(self, block, layers_number, number_class=1000):
        super(ResNet, self).__init__()
        # 公共layer
        self.conv1 = nn.Conv2d(3, self.stemlayer_channeloutput, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.stemlayer_channeloutput)
        self.relu1 = nn.ReLU(False)
        self.maxpooling1 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        # 自定义部分layer
        self.layer1 = self.stage_make(BottleNeck_large, self.stemlayer_channeloutput, self.stemlayer_channeloutput * 4,
                                      layers_number[0], stride=1)
        self.layer2 = self.stage_make(BottleNeck_large, self.stemlayer_channeloutput * 4,
                                      self.stemlayer_channeloutput * 4 * 2, layers_number[1], stride=2)
        self.layer3 = self.stage_make(BottleNeck_large, self.stemlayer_channeloutput * 4 * 2,
                                      self.stemlayer_channeloutput * 4 * 2 * 2, layers_number[2], stride=2)
        self.layer4 = self.stage_make(BottleNeck_large, self.stemlayer_channeloutput * 4 * 2 * 2,
                                      self.stemlayer_channeloutput * 4 * 2 * 2 * 2, layers_number[3], stride=2)

        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512 * block.expansion, number_class)
        self.softmax = nn.Softmax(-1)

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


def ResNet_50(number_class1=1000):
    return ResNet(BottleNeck_large, [3, 4, 6, 3], number_class=number_class1)


def ResNet_101(number_class1=1000):
    return ResNet(BottleNeck_large, [3, 4, 23, 3], number_class=number_class1)


def ResNet_152(number_class1=1000):
    return ResNet(BottleNeck_large, [3, 8, 36, 3], number_class=number_class1)

# lr, num_epochs, batch_size = 0.05, 10, 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
# train(ResNet_50(10), train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

if __name__ == '__main__':
    ##测试代码
    input = torch.randn(50, 3, 224, 224)
    # resnet50 = ResNet_50(1000)
    # out = resnet50(input)
    # print(out.shape)
    # resnet152 = ResNet_152(1000)
    # out = resnet101(input)
    # print(out.shape)

    # summary(model.cuda(),input_size=(3,224,224),batch_size=-1)
    # stat(ResNet_101(1000),(3,224,224))
    # summary(model.to('cuda'),input_size=(3,224,224),batch_size=-1)
    # summary_ = summary(resnet50.to('cuda'), input_size=(3, 224, 224), batch_size=-1)
    # resnet50.to("cuda")
    # optimizer = optim.Adam(filter(lambda i: i.requires_grad, resnet50.parameters()), lr=0.1, weight_decay=0.01)

    # -------------------------------------------------------------------------------------

    #
    writer = SummaryWriter('runs/embedding_example1')
    mnist = torchvision.datasets.MNIST('mnist', download=True)
    writer.add_embedding(
        mnist.data.reshape((-1, 28 * 28))[:100, :],
        metadata=mnist.targets[:100],
        label_img=mnist.data[:100, :, :].reshape((-1, 1, 28, 28)).float() / 255,
        global_step=0
    )

    # writer = SummaryWriter('runs/scalar_example')
    # for i in range(10):
    #     writer.add_scalar('quadratic', i ** 2, global_step=i)
    #     writer.add_scalar('exponential', 2 ** i, global_step=i)
    # freeze_list = [i for i in range(10)]
    # freeze_by_idxs(resnet50, freeze_list)



