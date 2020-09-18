#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class TuSimpleConfig():
    batch_size = 16
    num_workers = 4
    
    num_lanes = 4
    num_grids = 100
    row_anchor = [ 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
              116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
              168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
              220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
              272, 276, 280, 284]
    using_auxilary = True
    
    cls_num_per_lane = 56
    
    num_epoch = 50
    lr = 0.0000001
    step_info_interval = 50

