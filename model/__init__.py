import torch

from test_model.backbone.ResNet import *


if __name__ == '__main__':
    image_test = torch.randn(50,3,224,224)
    resnet_test = ResNet_50(1000)
    test_out  = resnet_test(image_test)
    print(test_out.shape)