# -*- coding: utf-8 -*-
# @File : train_amp.py
# @Project: BP-Net
# @Author : jie
# @Time : 10/27/21 3:58 PM

import torch
from tqdm import tqdm
import hydra
from PIL import Image
import os
from omegaconf import OmegaConf
from utils import *
import vis_utils

def test(run, mode='test', save=True):
    dataloader = run.testloader
    net = run.net_ema.module
    net.eval()
    tops = [AverageMeter() for i in range(len(run.metric.metric_name))]
    if save:
        dir_path = f'{run.cfg.data.testset.path}'
        os.makedirs(dir_path, exist_ok=True)
    with torch.no_grad():
        for datas in tqdm(dataloader, desc="test ", dynamic_ncols=True, leave=False, disable=run.rank):
            # breakpoint()            
            rgb, lidar, K_cam, depth, sample_idx = datas

            # move tensors to GPU
            rgb, lidar, K_cam, depth = run.init_cuda(rgb, lidar, K_cam, depth)

            # forward pass (net expects rgb, lidar, K_cam)
            output = net(rgb, lidar, K_cam)
            if isinstance(output, (list, tuple)):
                output = output[-1]
            precs = run.metric(output,depth)
            for prec, top in zip(precs, tops):
                top.update(prec.mean().detach().cpu().item())
            if save:
                for depth_i, true_idx in zip(output, sample_idx):
                    vis_utils.save_depth_as_points(depth_i, int(true_idx), dir_path)
    logs = ""
    for name, top in zip(run.metric.metric_name, tops):
        logs += f" {name}:{top.avg:.7f} "
    run.ddp_log(logs, always=True)


@hydra.main(config_path='configs', config_name='config', version_base='1.2')
def main(cfg):
    with Trainer(cfg) as run:
        test(run, mode=cfg.data.testset.mode, save=OmegaConf.select(cfg, 'save', default=False))



if __name__ == '__main__':
    main()
