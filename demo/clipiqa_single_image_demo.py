import platform

print(platform.system())
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
if platform.system().lower().startswith('lin'):
    im_dir = '/content/onedrive/Dataset/WeedDataSample/DetectionLambsquarters/weed_all_object_in_box/Lambsquarters'

im_paths = list_dir(im_dir, [], '.jpg')
len(im_paths)
"""
save all results as json files

"""
# Copyright (c) OpenMMLab. All rights reserved
import argparse
import os

import mmcv
import torch

from mmedit.apis import init_model, restoration_inference, init_coop_model
from mmedit.core import tensor2img, srocc, plcc

import pandas as pd
from tqdm import tqdm
import numpy as np

import plotly.graph_objects as go
import plotly.offline as pyo
import json
from os.path import join
def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    # parser.add_argument('--config', default='configs/clipiqa/clipiqa_attribute_test.py', help='test config file path')
    # parser.add_argument('--checkpoint', default=r'D:\BoyangDeng\WeedLambsquarter\CLIP-IQA\work_dirs\clipiqa_coop_koniq\latest.pth', help='checkpoint file')
    # parser.add_argument('--file_path', default=im_paths[0], help='path to input image file')
    # parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

class Arg:
  def __init__(self):
    self.config='configs/clipiqa/clipiqa_attribute_test.py'
    # self.checkpoint=r'D:\BoyangDeng\WeedLambsquarter\CLIP-IQA\work_dirs\official\iter_80000.pth'
    self.checkpoint=r'D:\BoyangDeng\WeedLambsquarter\CLIP-IQA\work_dirs\clipiqa_coop_koniq\latest.pth'
    if platform.system().lower().startswith('lin'):
        self.checkpoint='/content/gdrive/MyDrive/WeedLambsquarter/work_dirs/latest.pth'
    self.device=0

def main():
    args = Arg()

    result_dict = {}
    exception_dict = {}
    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))


    attribute_list = ['Quality', 'Brightness', 'Sharpness', 'Noisiness', 'Colorfulness', 'Contrast']
    # attribute_list = ['Quality', ]
    # attribute_list = ['Aesthetic', 'Happy', 'Natural', 'New', 'Scary', 'Complex']
    attribute_list = [*attribute_list, attribute_list[0]]

    angles = np.linspace(0, 2*np.pi, len(attribute_list), endpoint=False)
    save_dir = r'D:\Dataset\WeedData\DetectionLambsquarters\weed_all_object_in_box_IQA_finetuning_on_Weed'
    if platform.system().lower().startswith('lin'):
        save_dir = '/content/onedrive/Dataset/WeedDataSample/DetectionLambsquarters/weed_all_object_in_box_IQA_finetuning_on_Weed'
    result_name=  'result_dict.json'
    exception_name=  'exception_dict.json'
    result_path = join(save_dir, result_name)
    exception_path = join(save_dir, exception_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result_dict = json.load(f)
    if os.path.exists(exception_path):
            with open(exception_path, 'r') as f:
                exception_dict = json.load(f)
    cnt = 0
    for im_path in tqdm(im_paths):
        if cnt % 50 == 0:
            with open(result_path, 'w') as f:
                json.dump(result_dict, f, indent=4)
        cnt += 1
        im_name = os.path.basename(im_path)
        #ipdb.set_trace(context=20)
        save_path = os.path.join(save_dir, im_name+'.svg')
        if os.path.exists(save_path):
            continue
        try:
            output, attributes = restoration_inference(model, os.path.join(im_path), return_attributes=True)
            
            output = output.float().detach().cpu().numpy()
            attributes = attributes.float().detach().cpu().numpy()[0]
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print(attributes)
            if len(attributes) < len(attribute_list):
                add_num = len(attribute_list) - len(attributes) -1
                attributes = np.array(attributes.tolist()+[1.0]*add_num)
            attributes = [*attributes, attributes[0]]

            result_dict[im_name] = {}
            for i_att in range(len(attribute_list)):
                att_name = attribute_list[i_att]
                att_value = attributes[i_att]
                result_dict[im_name][att_name] = float(att_value)

            fig = go.Figure(
                data=[
                    go.Scatterpolar(r=attributes, theta=attribute_list, fill='toself'),
                ],
                layout=go.Layout(
                    title=go.layout.Title(text='Attributes'),
                    polar={'radialaxis': {'visible': True}},
                    showlegend=False,
                )
            )

            fig.update_xaxes(tickfont_family="Arial Black")

            fig.write_image(, engine="kaleido")
        except Exception as e:
            print(im_name, e)
            exception_dict[im_name] = str(e)
            with open(exception_path, 'w') as f:
                json.dump(exception_dict, f, indent=4)
                continue

    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=4)

    with open(exception_path, 'w') as f:
        json.dump(exception_dict, f, indent=4)
main()