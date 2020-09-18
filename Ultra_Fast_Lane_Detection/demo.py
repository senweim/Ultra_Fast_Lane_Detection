#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
from os import makedirs
from os.path import join, exists
import cv2
import torch
import torch.optim as optim
import numpy as np
import argparse

from data.dataloader import get_dataloader
from model.model import LaneDetectNet
from data.transforms import DeNormalize
from config import TuSimpleConfig as cfg


# In[58]:


def visualization(model, dataloader, save_dir, row_anchor, num_grids):
    model.eval()
    
    T = DeNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    lane_color = [(60, 76, 231), (18, 156, 243), (113, 204, 46), (219, 152, 52)]
    num_anchors = len(row_anchor)
    scale = int(800.0 / num_grids)
    anchor_width = row_anchor[1] - row_anchor[0]
    
    
    
    for step, batch in enumerate(dataloader):
        inputs = batch['input_tensor'].to(device)
        image_names = batch['image_name']
        
        pred_cls = model(inputs)
        pred_cls = np.array(torch.argmax(pred_cls, dim=-1).cpu())
        
        for input_tensor, name, pred in zip(inputs.cpu(), image_names, pred_cls):
            img = np.array((T(input_tensor)*255).permute(1,2,0).byte())
            
            name = '_'.join(name.split('/'))
            img_dir = os.path.join(save_dir, name)
            
            #mask = np.zeros((288, 800, 3), dtype=np.uint8)
            for idx, lane in enumerate(pred):
                pos_valid = (lane != num_grids)
                lane_valid = lane[pos_valid] * scale
                anchor_valid = np.array(row_anchor)[pos_valid]
                scale_map = np.tile(np.arange(0, scale), anchor_width * len(np.where(pos_valid)[0]))          
                lane_valid = lane_valid.repeat(scale * anchor_width) + scale_map
                anchor_valid = anchor_valid.repeat(anchor_width) + np.tile(np.arange(0, anchor_width), len(anchor_valid))
                anchor_valid = anchor_valid.repeat(scale)
                img[anchor_valid, lane_valid] = lane_color[idx]
            
            cv2.imwrite(img_dir, img)
            cv2.imshow('result', img)
            cv2.waitKey(1)
        


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/senwei/data/TuSimple/test_set/',help='path to TuSimple Benchmark dataset')
    flags = parser.parse_args()
    data_dir = flags.data_dir
    meta_dir = os.path.join(data_dir, 'test.txt')
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = LaneDetectNet(num_lanes=cfg.num_lanes, 
                          num_anchors=len(cfg.row_anchor), 
                          num_grids=cfg.num_grids, 
                          using_auxilary=False, 
                          pretrained=True).to(device)


    logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
    chosen_folder = logs[-1]
    weight_path = os.path.join(chosen_folder, 'LaneDetectNet.pth')

    save_dir = os.path.join('demo/')
    makedirs(save_dir) if not exists(save_dir) else None

    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict, strict=False)
    
    test_loader = get_dataloader(data_dir, meta_dir, phase='test', batch_size=1, shuffle=False, num_workers=1)

    visualization(model, test_loader, save_dir, cfg.row_anchor, cfg.num_grids)


# In[ ]:




