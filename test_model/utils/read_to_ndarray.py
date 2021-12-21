# -*- coding = utf-8 -*-
# @Author : lyp2333
# @Time : 2021/11/6 18:55
# @File : read_to_ndarray.py
# @Software : PyCharm
import numpy as np
import csv

def read_csv(path,h,w):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        label = []
        data = []
        for i, data1 in enumerate(reader):
            if i > 0:
                label.append(int(data1[0]))
                for j in data1[1:]:
                    data.append(int(j))

    label = np.array(label).reshape(1, -1)
    data = np.array(data).reshape((-1, h, w))
    return label, data