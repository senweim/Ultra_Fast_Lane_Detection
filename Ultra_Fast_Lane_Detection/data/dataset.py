#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# In[2]:


class LaneDataset(Dataset):
    def __init__(self, 
                 data_root, 
                 meta_root, 
                 num_lanes=4, 
                 num_grids=50,
                 row_anchor=None,
                 using_auxilary=True,
                 pre_transform=None, 
                 img_transform=None, 
                 seg_transform=None):
        super(LaneDataset, self).__init__()
        
        img_path = []
        label_path = []      
        with open(meta_root, 'r') as f:
            for line in f:
                data_info = line.split()
                img_name, label_name = data_info[0].lstrip('/'), data_info[1].lstrip('/')
                img_path.append(os.path.join(data_root, img_name))
                label_path.append(os.path.join(data_root, label_name))
        
        
        self.img_path = img_path
        self.label_path = label_path
        self.num_lanes = num_lanes
        self.num_grids = num_grids
        self.row_anchor = row_anchor
        self.using_auxilary = using_auxilary
        self.pre_transform = pre_transform
        self.img_transform = img_transform
        self.seg_transform = seg_transform
                
    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx]).convert('RGB')
        label = Image.open(self.label_path[idx])
        
        # for data augmentation
        if self.pre_transform is not None:
            img, label = self.pre_transform(img, label)
        
        # get cls_label from label
        self.label = np.array(label)
        cls_label = self._get_cls_label()
        
        if self.img_transform is not None:
            img = self.img_transform(img)
        
        data = {}
        data['input_tensor'] = img
        data['cls_label'] = cls_label
        
        # using auxilary segmentation network to help training
        if self.using_auxilary:
            data['seg_label'] = self.seg_transform(Image.fromarray(self.label))
        
        return data
        
    def _get_cls_label(self):
        h, w = self.label.shape
        
        # cause we resize the image to (288 x 800)
        # the predefined anchor is chosen from this scale
        if h != 288:
            row_anchor = (np.array(self.row_anchor) / 288.0 * h).round().astype(int)
        
        lane_pts = np.zeros((self.num_lanes, len(row_anchor), 2))
        grid_width = (w * 1.0) / self.num_grids
        
        # we sample row for each row anchor
        for lane in range(1, self.num_lanes + 1):
            for row_idx, row in enumerate(row_anchor):
                pos = np.where(self.label[row] == lane)[0]
                if len(pos) == 0:
                    lane_pts[lane - 1, row_idx] = (row, -1)
                else:
                    lane_pts[lane - 1, row_idx] = (row, np.mean(pos))
            
            # data augmentation when the bottom pixel of lane sensor
            if lane_pts[lane - 1, -1, 1] == -1:
                pos_valid = np.where(lane_pts[lane - 1, :, 1] != -1)[0]
                pts_valid = lane_pts[lane - 1, pos_valid, :]
                
                # if the lane is too short, we omit it
                if len(pts_valid) >= 6:
                    # we use bottom half of the lane points to predict the unknown
                    pts_valid_half = pts_valid[len(pts_valid) // 2:, :]
                    param_line_fit = np.polyfit(pts_valid_half[:, 0], pts_valid_half[:, 1], 1)

                    # find the first unkown position
                    predict_from = pos_valid[-1] + 1 
                    bottom_predict = np.polyval(param_line_fit, lane_pts[lane - 1, predict_from:, 0])

                    # remove the points predict outside the image
                    for idx, predict in enumerate(bottom_predict):
                        if predict < 0 or predict >= w:
                            bottom_predict[idx] = -1

                    lane_pts[lane - 1, predict_from:, 1] = bottom_predict

                    # we also augment the label
                    pts_to_draw = lane_pts[lane - 1, pos_valid[-1]:, :]
                    for idx in range(1, len(pts_to_draw)):
                        if pts_to_draw[idx][1] == -1:
                            break
                        # using cls_label to draw line in original label
                        start_pt = (int(pts_to_draw[idx - 1][1]), int(pts_to_draw[idx - 1][0]))
                        end_pt = (int(pts_to_draw[idx][1]), int(pts_to_draw[idx][0]))
                        cv2.line(self.label, 
                                 start_pt, 
                                 end_pt, 
                                 (lane,),
                                 thickness = 16)

            # we predict the label in a course resolusion
            pos_invalid_bool = (lane_pts[lane - 1, :, 1] == -1)
            lane_pts[lane - 1, :, 1] //= grid_width
            # here we change value of invalid pos from -1 to num_grids, we use 100 in this code
            # because we will predict position for each anchor row, which means we will add 1 in training phace
            # to represent certain lane does not exist
            lane_pts[lane - 1, pos_invalid_bool, 1] = self.num_grids
                
        return lane_pts[:, :, 1].astype(np.int)
    

    def __len__(self):
        return len(self.img_path)


# In[ ]:


class LaneTestSet(Dataset):
    def __init__(self, 
                 data_root, 
                 meta_root, 
                 img_transform=None):
        super(LaneTestSet, self).__init__()
        
        img_name = []   
        with open(meta_root, 'r') as f:
            for line in f:
                name = line.split()[0].lstrip('/')
                img_name.append(name)
        
        self.img_name = img_name
        self.data_root = data_root
        self.img_transform = img_transform
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.img_name[idx])
        img = Image.open(img_path).convert('RGB')
        
        if self.img_transform is not None:
            img = self.img_transform(img)
        
        data = {}
        data['input_tensor'] = img
        data['image_name'] = self.img_name[idx]
        
        return data
    
    def __len__(self):
        return len(self.img_name)


# In[3]:


if __name__ == '__main__':
    pass


# In[ ]:




