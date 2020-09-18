#!/usr/bin/env python
# coding: utf-8

# In[27]:


import sys
import random
from torch.utils.data import DataLoader
from PIL import Image

import transforms as transforms
import dataset as dataset


# In[ ]:


def get_dataloader(data_root, 
                   meta_root,
                   phase='train',
                   num_lanes=None,
                   num_grids=None,
                   row_anchor=None,
                   using_auxilary=False,
                   batch_size=1,
                   shuffle=True,
                   num_workers=0):
    # basic transform apply for all data
    img_transform = transforms.Compose([
        transforms.Resize((288, 800), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if using_auxilary:
        seg_transform = transforms.Compose([
            transforms.Resize((36, 100), interpolation=0),
            transforms.MaskToTensor()
        ])
    
    if phase == 'train':
        # transform for img and label together to augment data
        pre_transform = transforms.Compose([
            transforms.RandomRotate(6),
            transforms.RandomVerticalShift(100),
            transforms.RandomHorizontalShift(200),
        ])

        train_set = dataset.LaneDataset(data_root=data_root, 
                                        meta_root=meta_root, 
                                        num_lanes=num_lanes, 
                                        num_grids=num_grids, 
                                        row_anchor=row_anchor,
                                        using_auxilary=using_auxilary,
                                        pre_transform=pre_transform,
                                        img_transform=img_transform,
                                        seg_transform=seg_transform)
        dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    elif phase == 'val':
        val_set = dataset.LaneDataset(data_root=data_root, 
                                      meta_root=meta_root,
                                      num_lanes=num_lanes,
                                      num_grids=num_grids,
                                      row_anchor=row_anchor,
                                      using_auxilary=using_auxilary,
                                      pre_transform=None,
                                      img_transform=img_transform,
                                      seg_transform=seg_transform)
        dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    elif phase == 'test':
        test_set = dataset.LaneTestSet(data_root=data_root,
                                       meta_root=meta_root,
                                       img_transform=img_transform)
        dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
    else:
        raise NotImplementedError
    
    return dataloader


# In[ ]:


if __name__ == '__main__':
    sys.path.append('..')
    from config import TuSimpleConfig as cfg
    import numpy as np
    import cv2
    
    data_loader = get_dataloader(phase='train', 
                                 data_root='/home/senwei/data/TuSimple/train_set/', 
                                 meta_root='/home/senwei/data/TuSimple/train_set/train_gt.txt', 
                                 num_lanes=cfg.num_lanes,
                                 num_grids=cfg.num_grids,
                                 row_anchor=cfg.row_anchor,
                                 using_auxilary=True)
    for batch in data_loader:
        # img check
        img = batch['input_tensor'][-1]
        T = transforms.DeNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        img = np.array((T(img)*255).permute(1,2,0).byte())
        #cv2.imwrite('test_img.png',x)
      
        seg_label = np.array(batch['seg_label'][-1]) * 60
        print(seg_label.shape)
        cv2.imwrite('test_aug_label.png',seg_label)

        lane_color = [(60, 76, 231), (18, 156, 243), (113, 204, 46), (219, 152, 52)]
        cls_label = np.zeros((56, 100, 3), dtype=np.int)
        label = np.array(batch['cls_label'][-1])

        h, w, c = img.shape
        scale = w * 1.0 / cfg.num_grids
        for idx, lane in enumerate(label):
            start = False
            start_pt = None
            for y, x in enumerate(lane[1:], 1):
                if x == 100:
                    continue
                if not start:
                    start_pt = (int(x * scale), cfg.row_anchor[y])
                    start = True
                    continue
                end_pt = (int(x * scale), cfg.row_anchor[y])
                cv2.line(img, start_pt, end_pt, lane_color[idx], thickness = 5)
                start_pt = end_pt
        cv2.imwrite('test_img.png', img)              
        break

