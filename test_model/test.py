# -*- coding = utf-8 -*-
# @Author : lyp2333
# @Time : 2021/10/29 13:33
# @File : test.py
# @Software : PyCharm
import multiprocessing
import os
import sys
import time
from typing import List
from multiprocessing import pool,Queue
from threading import Thread,Lock

import cv2
import csv
import copy
import numpy
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch.hub
import torchvision
import torchsummary
from torchvision import models
from sklearn import preprocessing
from PIL import Image
from utils.read_to_ndarray import *
from test_model.utils import xyxy2cxcywh
import argparse
from loguru import logger
from pycocotools.coco import COCO
# os.makedirs(os.path.join('..',"one"),exist_ok=True)
# wenjian = open(os.path.join('..','test','test.csv'))

# os.makedirs(os.path.join('..','data'),exist_ok=True)
# data_file = open(os.path.join('..','data','DATA.csv'))
# print(os.path.join('..','test'))
# f = open('123.txt','rw')
# f = open(os.path.join('..','data','123.txt'))
# str = os.path.abspath('.')
# f = open(str+'/123.txt','r+')
# f.close()
# os.makedirs(os.path.join('..','data'),exist_ok=True)
# data_file = os.path.join('..','data','house_tiny.csv')

# with open(data_file,'w+') as f:
#     f.write('NumRooms,alley,Price\n')
#     f.write('NA,Pave,127500\n')
#     f.write('2,NA,106000\n')
#     f.write('4.NA,178000\n')
#     f.write('NA,NA,140000\n')

# data_file = os.path.join('..','data','house_tiny.csv')
# data = pd.read_csv(data_file)
# input,output = data.iloc[:,0:2],data.iloc[:,2]
# # print(input,'\n',output)
# input = input.fillna(input.mean())#只有是数值域的nan才被填充
# input = pd.get_dummies(input,dummy_na=True)
#
# x,y = torch.tensor(input.values),torch.tensor(output.values)
# a = torch.arange(20*2).reshape((2,5,4))
# print(a)

# a = torch.randn([1,2,3,4],dtype=float,requires_grad=True)
# print(a)
# a = torch.ones((4,4))
# a = np.arange(12)
# a.reshape((3,4))
# b = torch.from_numpy(a)
# b = torch.sum(a,dim=0,keepdim=True)
# c = a / b
# print(c)
# model = models.densenet121(pretrained=True)
# label, data = read_csv('../data/train.csv', 28, 28)
# image = Image.fromarray(data[0, :, :])
# x = range(10)
# y = range(10)
# plt.figure('test')
# figure, axes = plt.subplot(2,2)
# axes.flatten()
# axes[0].plot(x,y,label = '1')
# axes[0].show()

# print(torch.hub.list('pytorch/vision:v0.4.2'))
# plt.subplot(221)
# plt.plot(x,y,label = '123')
# plt.xlabel('lyp')
# plt.ylabel('zjy')
# plt.legend()
#
# plt.subplot(222)
# plt.plot(x,y)
# figsize = (15,3)
# figure,axes = plt.subplot(10,2,figsize = figsize)
# axes.flatten()
# axes[0].set_xlabel('lyp')
# axes[0].show()
## transform

# image_all = pd.read_csv('../data/train.csv')
# label = image_all['label']
# label = np.array(label)
# image = image_all.drop('label', axis=1)
# feature_name = list(image.columns)
# image = np.array(image, dtype=int)
# image1 = image.reshape(-1, 28, 28)
# # draw
# test_image = []
# for i in range(image):
#     test_image.append(Image.fromarray(image[i, :, :]))
#
# plt.figure()
# plt.imshow(test_image[2])
# plt.show()
# 提取
# arr = np.ones((2,2))
# arr[0,0] = 0
# arr1 = np.array(arr,copy=True)
# arr1[arr>0] = 2 #将arr大于0的位置变为2
# 验证升维手段
# arr1 = np.ones((128,3,224,224))
# arr1 = torch.from_numpy(arr1)
#
# arr2 = torch.sum(arr1,(0,2,3),keepdim=True)
# arr3 = torch.sum(arr1,(0,2,3))[None,:,None,None]
# print(f'arr2: {arr2.shape}\narr3: {arr3.shape}')
#
# arr4 = torch.zeros((2,2))[:,:,None]#用于升维，
# print(arr4.shape)

# n = 10
# c = 8
# b = 15
# x = torch.rand((n, c, b))
#
# m = nn.BatchNorm2d(2, affine=True)  # affine参数设为True表示weight和bias将被使用
# input = torch.randn(1, 2, 3, 4)
# output = m(input)
# 测试任意位置的修改
# one = torch.ones((2,2,3))
# test = torch.ones((2,2,3))
# test[1,:,2] = 0
# print(one)
# one[test>0] = 3
# print(one)
# squeeze和unsqueeze
# one = torch.ones((2,2,3))
# b = torch.unsqueeze(one,2)
# print(b.shape)
# description = 'you need to add parameters'
# parser = argparse.ArgumentParser(description=description)
# parser.add_argument('--model', type=str, default='resnet_50')
# parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--task_id', type=int, default=0)
# parser.add_argument('--epoch_num', type=int, default=0)
#
# args = parser.parse_args()
#
# print(args.epoch_num)
#
# path = '/home/lyp/Data/people_dataset/train2017/FudanPed00006.png'
# img = cv2.imread(path)
# _,w,h = img.shape[::-1]
# print(_,w,h)
# path = '/home/lyp/code/LabelToolForDetection'
#
# for root, dirs, files in os.walk(path,topdown=True):
#     for file in files:
#         print(os.path.join(root, file))
#
#     for dir in dirs:
#         print(os.path.join(
#         path,dir))
#
# bboxes = [[1,2,3,4,1],[111,121,131,141,2],[221,231,241,242,3],[321,331,341,351,4]]
# bboxes = np.array(bboxes)
# bbox_trans = xyxy2cxcywh(bboxes)
# print(bboxes) pnt(ox_trans)

# dataset_dir = '/home/lyp/Data/dataset_anno'
# path = '/home/lyp/Data/dataset_anno/annotations'
# dir = 'instances_train2017.json'
# coco = COCO(os.path.join(path,dir))
# bear_ids = coco.getCatIds(catNms=['type_1'])
# train_2017 = os.path.join(dataset_dir,'train2017')
# img = []
# for img_content in coco.loadImgs(coco.getImgIds(catIds=bear_ids)):
#
#     img_path = os.path.join(train_2017,img_content['file_name'])
#     img.append(img_path)
# img_bear1 = cv2.imread(img[0])
# cv2.imshow('bear1',img_bear1)
# cv2.waitKey(0)

# def test(*args):
#     a = copy.copy(args[0])
#     a=4
#     return a
#
# b = 1
# print(test(b))
# print(b)



