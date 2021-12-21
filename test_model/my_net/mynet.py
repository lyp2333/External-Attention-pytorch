# -*- coding: utf-8 -*-
import numpy as np

# 获取数据
data = [[8., 7, 1],
        [8, 8, 1],
        [0, 0, 1],
        [0, 1, 1],
        [8, 3, 0],
        [8, 2, 0],
        [1, 8, 0],
        [3, 8, 0],
        [2, 8, 0],
        [1, 1, 1],
        [1, 2, 1],
        [8, 1, 0],
        [0, 2, 1],
        [2, 0, 1],
        [2, 2, 1],
        [0, 6, 0],
        [6, 0, 0],
        [1, 1, 1],
        [2, 2, 1],
        [3, 3, 1],
        [4, 4, 1],
        [5, 5, 1],
        [6, 6, 1],
        [7, 7, 1]]


class MyNet:
    def __init__(self, activation, *args):

        # 初始化权重和偏差
        self.weights = [np.mat(np.zeros((y, x))) for x, y in zip(args[:-1], args[1:])]
        self.bias = [np.mat(np.zeros((y, 1))) for y in args[1:]]
        self.a = [np.mat(np.zeros((y, 1))) for y in args]  # 每个神经元的输出值，输入层直接是x，其他层则为a = sigmoid(z),a的长度比其他的要大1
        self.z = [np.mat(np.zeros_like(b)) for b in self.bias]  # 每个神经元的输入值，z = wx+b
        self.delta = [np.mat(np.zeros_like(b)) for b in self.bias]
        self.activation = activation

    def forward(self, input: list):
        # 前向运算,保存中间参数a,z，方便反向计算梯度
        self.a[0] = np.mat(input).reshape(len(input), 1)
        for i, w_b in enumerate(zip(self.weights, self.bias)):
            w, b = w_b
            self.z[i] = w.dot(self.a[i]) + b
            self.a[i + 1] = self.Sigmoid(self.z[i])
            put = self.a[i + 1]
        return self.a[-1]

    def backpropagation(self, y_pre, y_real):
        # 算出所有delta所处的位置
        i = len(self.delta) - 1
        # 先算出最后一层的delta
        self.delta[i] = np.multiply(np.mat(y_pre - y_real), self.Sigmoid_derivative(self.z[-1]))
        # 寻找到倒数第二层
        i -= 1
        while i >= 0:
            # 用下一层的权值与delta相×，然后在与当前层的倒数做点乘
            self.delta[i] = np.multiply(np.dot(self.weights[i + 1].T, self.delta[i + 1]),
                                        self.Sigmoid_derivative(self.z[i]))
            i -= 1

        # 利用delta算出所有参数的梯度，注意a的长度比其他的都要多一个，要去掉最后一个
        delta_weights = [D.dot(A.T) for D, A in zip(self.delta, self.a[:-1])]
        delta_bias = self.delta  # 无x的部分

        return delta_weights, delta_bias

    def updata_w_b(self, delta_weights, delta_bias, lr):
        for i in range(len(delta_weights)):
            self.weights[i] = self.weights[i] - lr * delta_weights[i]
            self.bias[i] = self.bias[i] - lr * delta_bias[i]

    def loss_function(self, y_pre, y_real):
        return 0.5 * pow(y_pre - y_real, 2).sum()

    def Sigmoid(self, x):
        # sigmoid函数
        s = 1 / (1 + np.exp(-x))
        return s

    def Sigmoid_derivative(self, x):
        # sigmoid函数对x的求导
        return np.multiply(self.Sigmoid(x), 1 - self.Sigmoid(x))

    def Relu(self, x):
        # relu
        return np.maximum(x, 0)

    def Relu_derivative(self, x):
        test = np.zeros_like(x)
        if x == None:
            return None
        else:
            test[x >= 0] = 1
            return test

    def activation_fn(self, x):
        if self.activation == 'relu':
            return self.Relu(x)
        elif self.activation == 'sigmoid':
            return self.Sigmoid(x)
        else:
            return False

    def derivative(self, x):
        if self.activation == 'relu':
            return self.Relu_derivative(x)
        elif self.activation == 'sigmoid':
            return self.Sigmoid_derivative(x)
        else:
            return False
