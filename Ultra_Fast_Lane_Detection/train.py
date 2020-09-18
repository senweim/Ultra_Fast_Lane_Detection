#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
from os import makedirs
from os.path import join, exists
import time
import copy
import torch
import numpy as np

from utils import log_out
from model.loss import get_loss


def train_model(model, 
                train_loader, 
                val_loader, 
                optimizer, 
                num_epoch, 
                num_lanes, 
                num_grids,
                step_info_interval):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
    makedirs(saving_path) if not exists(saving_path) else None
    log_file = open(os.path.join(saving_path, 'log_train_summary.txt'), 'a')
    
    step_info = "step {:05d}, loss = {:6.4f}, loss_cls = {:6.4f}, loss_seg = {:6.4f}, " +\
                "accuracy = {:6.4f}, IoU = {:6.4f} --- {:5.2f} ms/batch"
    epoch_train_info = '-' * 30 + '\n' +\
                       "train summary: " + '\n' +\
                       "Total loss = {:6.4f}, loss_cls = {:6.4f}, loss_seg = {:6.4f}, " +\
                       "accuracy = {:6.4f}, IoU = {:6.4f}" + '\n' +\
                       '-' * 30 + '\n'
    epoch_val_info = '-' * 30 + '\n' +\
                     "validation summary: " + '\n' +\
                     "accuracy = {:6.4f}, IoU = {:6.4f}" + '\n' +\
                     '-' * 30 + '\n' 
    
    best_acc = 0.0
    best_weight = copy.deepcopy(model.state_dict())
    torch.save(best_weight, os.path.join(saving_path, 'LaneDetectNet.pth'))
    
    data_loader = {'train': train_loader, 'val': val_loader}
    
    for epoch in range(1, num_epoch + 1):
        log_out('****epoch {}/{}****'.format(epoch, num_epoch), log_file)
        
        sum_cls_T, sum_cls_all = 0, 0
        sum_seg_bin_iou = np.zeros(2)
        running_loss, running_loss_cls, running_loss_seg = 0, 0, 0
        since = time.time()
        for phase in ['train', 'val']:
            for step, batch in enumerate(data_loader[phase]):
                # prepare data
                inputs = batch['input_tensor'].to(device)
                labels_cls = batch['cls_label'].to(device)
                labels_seg = batch['seg_label'].to(device)
                
                labels = {}
                labels['cls'] = labels_cls
                labels['seg'] = labels_seg
             
                if phase == 'train':
                    model.train()
                    # clean gradient
                    optimizer.zero_grad()
                else:
                    model.eval()

                # forward
                preds_cls, preds_seg = model(inputs)
                preds_cls = preds_cls.permute(0, 3, 1, 2)
                
                logits = {}
                logits['cls'] = preds_cls
                logits['seg'] = preds_seg
                
                if phase == 'train':
                    # compute loss
                    loss_cls, loss_seg = get_loss(logits, labels)
                    loss = loss_cls + loss_seg

                    running_loss += loss.item()
                    running_loss_cls += loss_cls.item()
                    running_loss_seg += loss_seg.item()
                    
                    # backward
                    loss.backward()
                    optimizer.step()
      
                # flatten pred and label and compute metrics
                preds_cls_1d = torch.argmax(preds_cls.detach(), dim=1).view(-1)
                preds_seg_1d = torch.argmax(preds_seg.detach(), dim=1).view(-1)      
                labels_cls_1d = labels_cls.detach().view(-1)
                labels_seg_1d = labels_seg.detach().view(-1)
                
                sum_cls_T += torch.sum(preds_cls_1d == labels_cls_1d).item()
                sum_cls_all += len(preds_cls_1d)
                
                sum_seg_bin_iou[0] += torch.sum((preds_seg_1d != 0) & (labels_seg_1d != 0)).item()
                sum_seg_bin_iou[1] += torch.sum(preds_seg_1d != 0).item() + torch.sum(labels_seg_1d != 0).item()
                
                if phase == 'train' and step % step_info_interval == (step_info_interval - 1):
                    count_input = (step + 1) * inputs.size(0)
                    acc = (sum_cls_T * 1.) / sum_cls_all
                    iou = (sum_seg_bin_iou[0] * 1.) / (sum_seg_bin_iou[1] - sum_seg_bin_iou[0])
                    time_elapsed = time.time()
                    str_out = step_info.format(step + 1, 
                                               running_loss / count_input, 
                                               running_loss_cls / count_input, 
                                               running_loss_seg / count_input,
                                               acc,
                                               iou,
                                               (time_elapsed - since) * 1000.0 / count_input)                                                                  
                    log_out(str_out, log_file)
                    
            # summary every epoch
            acc = (sum_cls_T * 1.) / sum_cls_all
            iou = (sum_seg_bin_iou[0] * 1.) / (sum_seg_bin_iou[1] - sum_seg_bin_iou[0])
            if phase == 'train':
                str_out = epoch_train_info.format(running_loss,
                                                  running_loss_cls,
                                                  running_loss_seg,
                                                  acc,
                                                  iou)
            else:
                str_out = epoch_val_info.format(acc, iou)
                if acc > best_acc:
                    best_acc = acc
                    best_weight = copy.deepcopy(model.state_dict())
                    torch.save(best_weight, os.path.join(saving_path, 'LaneDetectNet.pth'))
            log_out(str_out, log_file)




