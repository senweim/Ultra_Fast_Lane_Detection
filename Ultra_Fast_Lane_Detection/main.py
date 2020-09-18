import os
import torch
import torch.optim as optim
import numpy as np
import argparse

from data.dataloader import get_dataloader
from model.model import LaneDetectNet
from config import TuSimpleConfig as cfg
from train import train_model
from test import test_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='option: trian, test')
    parser.add_argument('--data_dir', type=str, default='/home/senwei/data/TuSimple/train_set/',help='path to TuSimple Benchmark dataset')
    parser.add_argument('--restart', type=bool, default=False, help='restart from last train weight')
    flags = parser.parse_args()
    mode = flags.mode
    data_dir = flags.data_dir
    restart = flags.restart
    
    if torch.cuda.is_available():
        device = 'cuda:0'
        print('using gpu!')
    else:
        device = 'cpu'
        print('cpu only')
    model = LaneDetectNet(num_lanes=cfg.num_lanes, 
                          num_anchors=len(cfg.row_anchor), 
                          num_grids=cfg.num_grids, 
                          using_auxilary=cfg.using_auxilary, 
                          pretrained=True).to(device)

    if restart:
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
        assert len(logs) > 0
        chosen_folder = logs[-1]
        weight_path = os.path.join(chosen_folder, 'LaneDetectNet.pth')
        state_dict = torch.load(weight_path)
        model.load_state_dict(state_dict, strict=False)
        print('restart from latest trained weights!')
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    if mode == 'train':
        train_meta_dir = os.path.join(data_dir, 'train_gt.txt')
        val_meta_dir = os.path.join(data_dir, 'val_gt.txt')
        train_loader = get_dataloader(data_dir, 
                                      train_meta_dir,
                                      phase='train',
                                      num_lanes=cfg.num_lanes,
                                      num_grids=cfg.num_grids,
                                      row_anchor=cfg.row_anchor,
                                      using_auxilary=cfg.using_auxilary,
                                      batch_size=cfg.batch_size,
                                      shuffle=True,
                                      num_workers=cfg.num_workers)
        val_loader = get_dataloader(data_dir,
                                    val_meta_dir,
                                    phase='val',
                                    num_lanes=cfg.num_lanes,
                                    num_grids=cfg.num_grids,
                                    row_anchor=cfg.row_anchor,
                                    using_auxilary=cfg.using_auxilary,
                                    batch_size=cfg.batch_size,
                                    shuffle=False,
                                    num_workers=cfg.num_workers)
        train_model(model, train_loader, val_loader, optimizer, cfg.num_epoch, cfg.num_lanes, cfg.num_grids, cfg.step_info_interval)
        
        
    
    elif mode == 'test':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = LaneDetectNet(num_lanes=cfg.num_lanes, 
                       num_anchors=len(cfg.row_anchor), 
                       num_grids=cfg.num_grids, 
                       using_auxilary=False, 
                       pretrained=False).to(device)
        
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
        assert len(logs) > 1
        chosen_folder = logs[-1]
        
        weight_path = os.path.join(chosen_folder, 'LaneDetectNet.pth')
        state_dict = torch.load(weight_path)
        model.load_state_dict(state_dict, strict=False)
  
        test_model(model, save_dir=chosen_folder, device=device)
    
    else:
        raise NotImplementedError




