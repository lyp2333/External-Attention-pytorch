import os
path = 'C:/Users/冷易鹏/Downloads/human_dataset/Images'
dir = os.listdir(path)
for name in dir:
    newname = name.split('.')[0]+'.jpg'
    os.rename(os.path.join(path,name),os.path.join(path,newname))