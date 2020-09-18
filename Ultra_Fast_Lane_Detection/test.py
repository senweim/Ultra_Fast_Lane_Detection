#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from os import makedirs
from os.path import join, exists
import time
import torch
import numpy as np

from utils import log_out


def test_model(model, save_dir, device):
    model.eval()
    model.to(device)
    
    log_file = open(os.path.join(save_dir, 'log_test_summary.txt'), 'a')
    test_info = '-' * 30 + '\n' +\
             'running test on ' + device + '\n' +\
             '#test image: {}\n' +\
             'average time: {:6.4f}ms\n'+\
             'average FPS: {}\n' +\
             'fastest time: {:6.4f}ms\n'+\
             'fastest FPS: {}\n' +\
             'slowest time: {:6.4f}ms\n'+\
             'slowest FPS: {}\n' +\
             '-' * 30

    img_count = 100
    time_consuming = []
    x = torch.ones(1, 3, 288, 800).to(device)
    
    # warm up for cuda initilization
    for _ in range(10):
        logits = model(x)
    
    for _ in range(img_count):
        since = time.time()
        logits = model(x)
        time_elapsed = time.time()
        time_consuming.append(time_elapsed - since)
    
    max_time = np.max(time_consuming)
    min_time = np.min(time_consuming)
    avg_time = np.mean(time_consuming)
    

    log_out(test_info.format(img_count, avg_time * 1000, 1. / avg_time, min_time * 1000, 1. / min_time, max_time * 1000, 1. / max_time), log_file)
