# -*- coding: utf-8 -*-
import numpy as np
import mynet
import matplotlib.pyplot as plt
from tqdm import tqdm
from mynet import MyNet

if __name__ == '__main__':
    data1 = mynet.data
    data_raw = np.array(data1, dtype=float)
    data_ = data_raw[:,0:2]
    data_ = (data_-4)/4
    # print(data_)
    data = data_.tolist()
    label = data_raw[:, 2].tolist()
    epoch = 5000
    net_relu = MyNet('relu', 2, 10, 10, 10, 10,10, 10,10, 10,10, 10,1)
    net_sigmoid = MyNet('sigmoid', 2,10,10,10,1)
    #训练以及数据验证
    for i in tqdm(range(epoch)):
        for x, Y in zip(data, label):
            pre_y = net_sigmoid.forward(x)
            delta_weights, delta_bias = net_sigmoid.backpropagation(pre_y, Y)
            y = pre_y.item()
            net_sigmoid.updata_w_b(delta_weights, delta_bias, 0.1)
        print("----loss:{}---".format(net_sigmoid.loss_function(pre_y, Y)))
    print(net_sigmoid.weights)
    # # 数据验证
    # x = np.arange(0, 8, 0.1).tolist()
    # y = np.arange(0, 8, 0.1).tolist()
    #
    # data_val = []
    # result = []
    # for m in x:
    #     for n in y:
    #         data_test = [(m-4)/4, (n-4)/4]
    #         data_val.append(data_test)
    #
    # for i in range(len(data_val)):
    #     result.append(net_sigmoid.forward(data_val[i]).item())
    # result1 = np.array(result).reshape(len(x), len(y))
    #
    # # 图像展示
    # plt.figure()
    # ax3 = plt.axes(projection='3d')
    # xx, yy = np.meshgrid(x, y)
    # ax3.plot_surface(xx, yy, result1, cmap='rainbow')
    # plt.show()
