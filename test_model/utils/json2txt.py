import os
import json

path = 'C:/Users/冷易鹏/Desktop/animals_dataset/bndbox_anno/rhino.json'
save_txt = 'C:/Users/冷易鹏/Desktop/animals_dataset/bndbox_image/rhino/label.txt'
txt = open(save_txt, 'w+')
with open(path, 'r+') as f:
    dataset_anno = json.load(f)
    for i in dataset_anno:
        image_name = i
        txt.write(i + ' ')
        for num,j in enumerate(dataset_anno[i]):
            anno_h = j['bndbox']['ymax'] - j['bndbox']['ymin']
            anno_w = j['bndbox']['xmax'] - j['bndbox']['xmin']
            x1, y1 = j['bndbox']['xmin'], j['bndbox']['ymin']
            x2, y2 = j['bndbox']['xmax'], j['bndbox']['ymax']
            txt.write(
                str(format(x1, '.4f')) + ' ' + str(format(y1, '.4f')) + ' ' + str(format(x2, '.4f')) + ' ' + str(format(y2, '.4f')) + ' ' + str(format(4, '.4f')) + ' ')
        txt.write('\n')
txt.close()
