#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# In[29]:


def get_loss(logits, labels):
    loss_cls = FocalLoss(gamma=2)(logits['cls'], labels['cls'])
    loss_seg = nn.CrossEntropyLoss()(logits['seg'], labels['seg'])

    return loss_cls, loss_seg


# In[30]:


class FocalLoss(nn.Module):
    def __init__(self, gamma, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(weight=weight)
    
    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        
        return loss


# In[51]:


if __name__ == '__main__':
    cls_out = torch.randn(2, 4, 56, 101)
    cls_out = cls_out.permute(0, 3, 1, 2)
    
    seg_out = torch.randn(2, 5, 36, 100)
    
    label_cls = torch.argmax(torch.randn(2, 101, 4, 56), dim=1)
    label_seg = torch.argmax(torch.randn(2, 5, 36, 100), dim=1)
    
    logits = {}
    labels = {}
    logits['cls'] = cls_out
    logits['seg'] = seg_out
    labels['cls'] = label_cls
    labels['seg'] = label_seg
    
    loss_cls, loss_seg = get_loss(logits, labels)
    print('loss_cls = {}, loss_seg = {}'.format(loss_cls, loss_seg))


# In[ ]:




