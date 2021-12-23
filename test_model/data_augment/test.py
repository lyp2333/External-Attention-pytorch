import cv2
import numpy as np
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
h,w,c = img.shape
M = cv2.getRotationMatrix2D((h//2,w//2),45,1)
img_rotate = cv2.warpAffine(img,M,(600,600))
cv2.imshow('test',img_rotate)
cv2.waitKey(0)

