import json
import os
srt_path = '/home/lyp/桌面/MAE_论文逐段精读【论文精读】.457423264.zh-CN.srt'
json_path = '/home/lyp/桌面/caption.json'
txt_path = '/home/lyp/桌面'
def srt2txt(path):
    out_path= os.path.join(txt_path,path.split('.')[0]+'.txt')
    with open(path,'r+') as f:
        with open(out_path, 'w+') as out:
            for index,lines in enumerate(f.readlines()):
                if(index%5 == 2):
                    out.write(lines.split('>')[1].split('<')[0]+'\n')

def json2txt(path):
    out_path = out_path= os.path.join(txt_path,path.split('.')[0]+'.txt')
    with open(out_path,'w+') as out:
        with open(json_path,'r+') as f:
            caption_dict = json.load(f)
            # print(len(caption_dict['body']))
            for content_dict in caption_dict['body']:
                out.write(content_dict['content']+'\n')

if __name__ == '__main__':
    srt2txt(srt_path)
    json2txt(json_path)