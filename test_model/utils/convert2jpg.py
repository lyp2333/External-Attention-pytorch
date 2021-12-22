# encoding: utf-8
import os
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 文件夹路径
path = r'/to/your/imgfile'

def pilConvertJPG(path):
    for a, _, c in os.walk(path):
        for n in c:
            if '.jpg' in n or '.png' in n or '.jpeg' in n:
                img = Image.open(os.path.join(a, n))
                rgb_im = img.convert('RGB')
                error_img_path = os.path.join(a, n)
                os.remove(error_img_path)
                n = ''.join(filter(lambda n: ord(n) < 256, n))
                jpg_img_path = os.path.splitext(os.path.join(a, n).replace('\\', '/'))[0]
                jpg_img_path += '.jpg'
                print(jpg_img_path)
                rgb_im.save(jpg_img_path)

