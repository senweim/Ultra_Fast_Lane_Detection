import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet34

class Conv_BN_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv_BN_Relu, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        out = self.block(x)
        
        return out


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1 = resnet34().conv1
        self.bn1 = resnet34().bn1
        self.relu = resnet34().relu
        self.maxpool = resnet34().maxpool
        self.layer1 = resnet34().layer1
        self.layer2 = resnet34().layer2
        self.layer3 = resnet34().layer3
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        c2 = out
        out = self.layer3(out)
        c3 = out
        
        return c2, c3


# In[4]:


class ASPP(nn.Module):
    def __init__(self, in_channels=256, out_channels=128):
        super(ASPP, self).__init__()

        self.conv1 = Conv_BN_Relu(in_channels, out_channels, kernel_size=1)
        self.conv2 = Conv_BN_Relu(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv3 = Conv_BN_Relu(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.conv4 = Conv_BN_Relu(in_channels, out_channels, kernel_size=3, padding=5, dilation=5)
        self.conv5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Conv_BN_Relu(in_channels, out_channels, kernel_size=1)
        )
        
        
        self.conv6 = Conv_BN_Relu(out_channels * 5, out_channels, kernel_size=1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)        
        x5 = self.conv1(x)
        x5 = F.interpolate(x5, size = x.shape[-2:], mode='bilinear', align_corners=True)
          
        out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.conv6(out)
        
        return out


class LaneDetectNet(nn.Module):
    def __init__(self, num_lanes=4, num_anchors=56, num_grids=100, using_auxilary=True, pretrained=True):
        super(LaneDetectNet, self).__init__()
        self.feature = Backbone()
        self.aspp = ASPP()
                
        # detection head
        # shape of input: (n, 128, 18, 50)
        # shape of output: (n, num_lanes, 18, 50) 
        # after bilinear interpolation -> (n, num_lanes, num_anchors, num_grids + 1)
        self.pool = Conv_BN_Relu(128, num_lanes, kernel_size=1)
        self.cls_head = nn.Sequential(
            Conv_BN_Relu(num_lanes, num_lanes, kernel_size=3, padding=1, dilation=1),
            Conv_BN_Relu(num_lanes, num_lanes, kernel_size=3, padding=1, dilation=1),
        )
        
        # segmentation head
        # shape of input: (n, 256, 36, 100)
        # shape of output: (n, num_lanes + 1, 36, 100)
        if using_auxilary:
            self.seg_head = nn.Sequential(
                Conv_BN_Relu(256, 128, kernel_size=1),
                Conv_BN_Relu(128, 128, kernel_size=3, padding=1, dilation=1),
                nn.Conv2d(128, num_lanes + 1, kernel_size=1)
            )
            # initialize weights for segmentation branch
            init_weights(self.seg_head)
        
        # initialize weights for encoder and detection branch
        if not pretrained:
            init_weights(self.feature, self.aspp, self.cls_head)
        else:
            state_dict = torch.load('/home/senwei/data/ultra_fast_lane_detection/model/resnet34.pth')
            self.feature.load_state_dict(state_dict, strict=False)
            init_weights(self.aspp, self.cls_head)
        
        self.num_anchors = num_anchors
        self.num_grids = num_grids
        self.using_auxilary = using_auxilary
        
    def forward(self, x):
        # encoder
        # shape of c2: (n, 128, 36, 100)
        # shape of c3: (n, 256, 18, 50) 
        c2, c3 = self.feature(x)
        out = self.aspp(c3)
        
        # decoder branch of detection
        out_cls = self.pool(out)
        out_cls = F.interpolate(out_cls, (self.num_anchors, self.num_grids + 1), mode='bilinear', align_corners=True)
        out_cls = self.cls_head(out_cls)

        # decoder branch of segmentation, only active in train phase
        if self.using_auxilary:
            out_seg = F.interpolate(out, c2.shape[-2:], mode='bilinear', align_corners=True)
            out_seg = torch.cat((out_seg, c2), dim=1)
            out_seg = self.seg_head(out_seg)
            
            return out_cls, out_seg
        
        return out_cls     


def init_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):    
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    x = torch.randn(5,3,288,800)
    net = LaneDetectNet()
    out_cls, out_seg = net(x)
    print("shape of output from detection branch:", out_cls.shape)
    print("shape of output from segmentation branch:", out_seg.shape)





