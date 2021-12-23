import copy

import cv2
import numpy as np
import random
import math
import matplotlib.pyplot as plt

path = '/home/lyp/Data/dataset_anno/train2017/Img-845.jpg'
img = cv2.imread(path)
# img_convert = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# cv2.imshow('test',img_convert)
# cv2.waitKey(0)
# plt.figure('test')
# plt.subplot(221)
# plt.plot([1,2,3,4],[1,2,3,4])
# plt.title('test')
# plt.subplot(222)
# plt.imshow(img)# nly read,no display
# plt.axis('off')
# plt.show()

# img_resized = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
# h,w,c = img.shape
# M = cv2.getRotationMatrix2D((h//2,w//2),45,1)
# img_rotate = cv2.warpAffine(img,M,(600,600))
# cv2.imshow('test',img_rotate)
# cv2.waitKey(0)
# def get_aug_params(value, center=0):
#     if isinstance(value, float):
#         return random.uniform(center - value, center + value)
#     elif len(value) == 2:
#         return random.uniform(value[0], value[1])
#     else:
#         raise ValueError(
#             "Affine params should be either a sequence containing two values\
#                           or single float values. Got {}".format(
#                 value
#             )
#         )
# def get_affine_matrix(
#         target_size,
#         degrees=10,
#         translate=0.1,
#         scales=0.1,
#         shear=10,
# ):
#     twidth, theight = target_size
#
#     # Rotation and Scale
#     angle = get_aug_params(degrees)  # age rotate (-10,10)
#     scale = get_aug_params(scales, center=1.0)  # scale (0.9,1.1)
#
#     if scale <= 0.0:
#         raise ValueError("Argument scale should be positive")
#
#     R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)
#
#     M = np.ones([2, 3])
#     # Shear
#     shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
#     shear_y = math.tan(get_aug_params(shear) * math.pi / 180)
#
#     M[0] = R[0] + shear_y * R[1]
#     M[1] = R[1] + shear_x * R[0]
#
#     # Translation 平移
#     translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
#     translation_y = get_aug_params(translate) * theight  # y translation (pixels)
#
#     M[0, 2] = translation_x
#     M[1, 2] = translation_y
#
#     return M, scale
# def random_affine(
#         img,
#         targets=(),
#         target_size=(640, 640),
#         degrees=float(0),
#         translate=0.,
#         scales=0.,
#         shear=float(10),
# ):
#     M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)
#
#     img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))
#
#     # Transform label coordinates
#     return img
# img_test = random_affine(img)
# cv2.imshow('test',img_test)
# cv2.waitKey(0)

a = [[1, 2, 3], [3, 4, 4], [5, 6, 5]]
a = np.array(a)
b = a[:, :2]
b[0, 0] = 2
print(a)
