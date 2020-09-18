#!/usr/bin/env python
# coding: utf-8

# In[14]:


import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import random
from PIL import Image


# In[2]:


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, target_1, target_2=None):
        if target_2 is None:
            for T in self.transforms:
                target_1 = T(target_1)
            return target_1
        
        for T in self.transforms:
            target_1, target_2 = T(target_1, target_2)
        return target_1, target_2


# In[3]:


class ToTensor():
    def __call__(self, target_1, target_2=None):
        if target_2 is None:
            return F.to_tensor(target_1)
        
        return F.to_tensor(target_1), F.to_tensor(target_2)


# In[ ]:


class MaskToTensor():
    def __call__(self, mask):
        return torch.tensor(np.array(mask)).long()


# In[4]:


class Resize():
    def __init__(self, size, interpolation='Image.BILINEAR'):
        self.size = size
        self.interpolation = interpolation
        
    def __call__(self, target_1, target_2=None):
        if target_2 is None:
            return F.resize(target_1, self.size, self.interpolation)
        
        return F.resize(target_1, self.size, self.interpolation), F.resize(target_2, self.size, self.interpolation)


# In[5]:


class Normalize():
    """
    only support one object
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return F.normalize(tensor, self.mean, self.std)


# In[6]:


class DeNormalize():
    """
    only support one object
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        for t, m, d in zip(tensor, self.mean, self.std):
            t.mul_(d).add_(m)
        return tensor


# In[7]:


class RandomRotate():
    def __init__(self, angle):
        self.angle = angle
    
    def __call__(self, img, label):
        angle = random.randint(-self.angle, self.angle)
        img = F.rotate(img, angle, resample=2)
        label = F.rotate(label, angle, resample=0)
        return img, label


# In[8]:


class RandomHorizontalShift():
    def __init__(self, offset):
        self.offset = offset
    
    def __call__(self, img, label):
        offset = random.randint(-self.offset, self.offset)
        w = img.width
        
        img = np.array(img)
        label = np.array(label)
    
        img_o = np.zeros_like(img)
        label_o = np.zeros_like(label)
        
        begin = max(0, offset)
        end = min(w, w+offset)
        
        img_o[:, begin:end, :] = img[:, begin-offset:end-offset, :]
        label_o[:, begin:end] = label[:, begin-offset:end-offset]
        
        return Image.fromarray(img_o), Image.fromarray(label_o)


# In[9]:


class RandomVerticalShift():
    def __init__(self, offset):
        self.offset = offset
        
    def __call__(self, img, label):
        offset = random.randint(-self.offset, self.offset)
        h = img.height
        
        img = np.array(img)
        label = np.array(label)
        
        img_o = np.zeros_like(img)
        label_o = np.zeros_like(label)
        
        begin = max(0, offset)
        end = min(h, h+offset)
        
        img_o[begin:end, :, :] = img[begin-offset:end-offset, :, :]
        label_o[begin:end, :] = label[begin-offset:end-offset, :]
        
        return Image.fromarray(img_o), Image.fromarray(label_o)


# In[ ]:


if __name__ == '__main__':
    pass


# In[ ]:




