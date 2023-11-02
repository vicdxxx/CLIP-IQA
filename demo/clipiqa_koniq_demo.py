# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch
import json
from mmedit.apis import init_model, restoration_inference, init_coop_model
from mmedit.core import tensor2img, srocc, plcc

import pandas as pd
from tqdm import tqdm
import numpy as np
from os.path import join


def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--config', default='configs/clipiqa/clipiqa_attribute_test.py', help='test config file path')
    parser.add_argument('--checkpoint', default=r'D:\BoyangDeng\WeedLambsquarter\CLIP-IQA\work_dirs\clipiqa_coop_koniq\latest.pth', help='checkpoint file')
    parser.add_argument('--file_path', default=r'D:\Dataset\koniq10k/512x384/', help='path to input image file')
    parser.add_argument('--csv_path', default='D:/Dataset/koniq10k/koniq10k_distributions_sets.csv', help='path to input image file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
def list_dir(path, list_name, extension):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_dir(file_path, list_name, extension)
        else:
            if file_path.endswith(extension):
                list_name.append(file_path)
    try:
        list_name = sorted(list_name, key=lambda k: int(os.path.split(k)[1].split(extension)[0].split('_')[-1]))
    except Exception as e:
        print(e)
    return list_name
im_dir = r'D:\Dataset\WeedData\DetectionLambsquarters\weed_all_object_in_box\Lambsquarters'
im_paths = list_dir(im_dir, [], '.jpg')
len(im_paths)


class Arg:
  def __init__(self):
    self.config='configs/clipiqa/clipiqa_attribute_test.py'
    # self.checkpoint=r'D:\BoyangDeng\WeedLambsquarter\CLIP-IQA\work_dirs\official\iter_80000.pth'
    # self.checkpoint=r'C:\Users\AFSALab\OneDriveBD\Dataset\WeedDataSample\DetectionLambsquarters\work_dirs\latest.pth'
    self.checkpoint=r'D:\BoyangDeng\WeedLambsquarter\CLIP-IQA\work_dirs\clipiqa_coop_koniq_6_attributes\latest.pth'
    self.device=0
    # self.csv_path='D:/Dataset/koniq10k/koniq10k_distributions_sets.csv'
    self.csv_path=r'C:\Users\AFSALab\OneDriveBD\Dataset\WeedDataSample\DetectionLambsquarters/Lambsquarters_distributions_sets.csv'
    self.file_path = im_dir

def main():
    args = Arg()
    
    result_dict = {}
    exception_dict = {}
    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    csv_list = pd.read_csv(args.csv_path, on_bad_lines='skip')
    img_test = csv_list[csv_list.set=='test'].reset_index()
    # img_test = im_paths

    txt_path = './koniq_resize.txt'
    y_true = csv_list[csv_list.set=='test'].MOS.values
    
    # save_dir = r'D:\Dataset\WeedData\DetectionLambsquarters\weed_all_object_in_box_IQA_finetuning_on_KonIQ'
    # result_name=  'result_dict.json'
    # exception_name=  'exception_dict.json'
    # result_path = join(save_dir, result_name)
    # exception_path = join(save_dir, exception_name)
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    cnt = 0
    attribute_list = ['Quality', 'Brightness', 'Sharpness', 'Noisiness', 'Colorfulness', 'Contrast']
    # attribute_list = ['Aesthetic', 'Happy', 'Natural', 'New', 'Scary', 'Complex']
    attribute_list = [*attribute_list, attribute_list[0]]

    angles = np.linspace(0, 2*np.pi, len(attribute_list), endpoint=False)
    
    pred_score = []
    for i in tqdm(range(len(img_test))):
        # im_path = img_test[i]
        # if cnt%50==0:
        #     with open(result_path, 'w') as f:
        #         json.dump(result_dict, f, indent=4)
        cnt+=1
        # im_name = os.path.basename(im_path)

        output, attributes = restoration_inference(model, os.path.join(args.file_path, img_test['image_name'][i]), return_attributes=True)
        try:
            # output, attributes = restoration_inference(model, im_path, return_attributes=True)
            output = output.float().detach().cpu().numpy()
            pred_score.append(output[0][0])
        
            # output = output.float().detach().cpu().numpy()
            attributes = attributes.float().detach().cpu().numpy()[0]
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print(attributes)

            attributes = [*attributes, attributes[0]]

            # result_dict[im_name] = {}
            # for i_att in range(len(attribute_list)):
            #     att_name = attribute_list[i_att]
            #     att_value = attributes[i_att]
            #     result_dict[im_name][att_name] = float(att_value)

            # fig = go.Figure(
            #     data=[
            #         go.Scatterpolar(r=attributes, theta=attribute_list, fill='toself'),
            #     ],
            #     layout=go.Layout(
            #         title=go.layout.Title(text='Attributes'),
            #         polar={'radialaxis': {'visible': True}},
            #         showlegend=False,
            #     )
            # )

            # fig.update_xaxes(tickfont_family="Arial Black")

            # fig.write_image(os.path.join(save_dir, im_name+'.svg'), engine="kaleido")
        except Exception as e:
            print(e)
            # print(im_name, e)
            # exception_dict[im_name] = str(e)
            # with open(exception_path, 'w') as f:
            #     json.dump(exception_dict, f, indent=4)
            #     continue
    pred_score = np.squeeze(np.array(pred_score))*100


    p_srocc = srocc(pred_score, y_true)
    p_plcc = plcc(pred_score, y_true)

    print(args.checkpoint)
    print('SRCC: {} | PLCC: {}'.\
          format(p_srocc, p_plcc))


if __name__ == '__main__':
    main()
