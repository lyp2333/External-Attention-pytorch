import os
import glob

dir_path = "C:/Users/冷易鹏/Downloads/human_dataset/Annotation"
Txt = "C:/Users/冷易鹏/Downloads/human_dataset/label.txt"


def get_boxnum_fudan(path):
    file_anno = glob.glob(dir_path + '/Fudan*')
    boxnum_all = []
    for file_txt in file_anno:
        with open(file_txt, 'r+') as f:
            content_txt = f.readlines()
            boxnum = eval(content_txt[4].split(':')[1].split('{')[0])
            boxnum_all.append(boxnum)
    return boxnum_all


def get_boxnum_penn(path):
    file_anno = glob.glob(dir_path + '/Penn*')
    boxnum_all = []
    for file_txt in file_anno:
        with open(file_txt, 'r+') as f:
            content_txt = f.readlines()
            boxnum = eval(content_txt[4].split(':')[1].split('{')[0])
            boxnum_all.append(boxnum)
    return boxnum_all


def coor_fudan2txt():
    label = open(Txt, 'a+')
    file_anno = glob.glob(dir_path + '/Fudan*')
    filenum = len(file_anno)
    filename = []
    for i in range(filenum):
        if i < 9:
            filename.append(f'FudanPed0000{i + 1}.jpg')
        else:
            filename.append(f'FudanPed000{i + 1}.jpg')
    for num, i in enumerate(file_anno):
        coor_lines = []

        with open(i, 'r+') as f:
            content = f.readlines()
            for j in range(get_boxnum_fudan(dir_path)[num]):
                coor_lines.append(content[(j + 2) * 5])
        for num1, coor in enumerate(coor_lines):
            position = []
            coor_tuple = coor.split(':')[1].split('-')
            for m in coor_tuple:
                position.append(m.strip('\n').strip(' ').strip('(').strip(')').split(',')[0])
                position.append(m.strip('\n').strip(' ').strip('(').strip(')').split(',')[1].strip(' '))
            cls = '1.0000'
            if num1 == 0:
                label.write(filename[num] + ' ')
                label.write('{:.4f}'.format(eval(position[0])) + ' ' + '{:.4f}'.format(
                    eval(position[1])) + ' ' + '{:.4f}'.format(eval(position[2])) + ' ' + '{:.4f}'.format(
                    eval(position[3])) + ' ' + cls + ' ')
            else:
                label.write('{:.4f}'.format(eval(position[0])) + ' ' + '{:.4f}'.format(
                    eval(position[1])) + ' ' + '{:.4f}'.format(eval(position[2])) + ' ' + '{:.4f}'.format(
                    eval(position[3])) + ' ' + cls + ' ')
        label.write('\n')

    print('Fudan导入成功')
    label.close()


def coor_penn2txt():
    label = open(Txt, 'a+')
    file_anno = glob.glob(dir_path + '/Penn*')
    filenum = len(file_anno)
    filename = []
    for i in range(filenum):
        if i < 9:
            filename.append(f'PennPed0000{i + 1}.jpg')
        else:
            filename.append(f'PennPed000{i + 1}.jpg')
    for num, i in enumerate(file_anno):
        coor_lines = []

        with open(i, 'r+') as f:
            content = f.readlines()
            for j in range(get_boxnum_penn(dir_path)[num]):
                coor_lines.append(content[(j + 2) * 5])
        for num1, coor in enumerate(coor_lines):
            coor_tuple = coor.split(':')[1].split('-')
            position = []
            for m in coor_tuple:
                position.append(m.strip('\n').strip(' ').strip('(').strip(')').split(',')[0])
                position.append(m.strip('\n').strip(' ').strip('(').strip(')').split(',')[1].strip(' '))
            cls = '1.0000'
            if num1 == 0:
                label.write(filename[num] + ' ')
                label.write('{:.4f}'.format(eval(position[0])) + ' ' + '{:.4f}'.format(
                    eval(position[1])) + ' ' + '{:.4f}'.format(eval(position[2])) + ' ' + '{:.4f}'.format(
                    eval(position[3])) + ' ' + cls + ' ')
            else:
                label.write('{:.4f}'.format(eval(position[0])) + ' ' + '{:.4f}'.format(
                    eval(position[1])) + ' ' + '{:.4f}'.format(eval(position[2])) + ' ' + '{:.4f}'.format(
                    eval(position[3])) + ' ' + cls + ' ')
        label.write('\n')
    print("Penn导入成功")
    label.close()


coor_fudan2txt()
coor_penn2txt()
