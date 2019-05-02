#!/usr/bin/env python
'''
train foldingnet

author  : Ruoyu Wang; Yuqiong Li
created : 10/25/18 1:29 PM
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from foldingnet import FoldingNetVanilla
from foldingnet import FoldingNetShapes
from foldingnet import ChamfersDistance
from datasets import pcdDataset
from torch.utils.data import DataLoader
import numpy as np
from utils import check_exist_or_remove
import math


def train(dataset, model, batch_size, lr, epoches, log_interval, save_along_training):
    """train implicit version of foldingnet
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    chamfer_distance_loss = ChamfersDistance()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    model = model.train().cuda()   # set attributes
    check_exist_or_remove("../log/train_loss_log.txt")
    loss_log = open('../log/train_loss_log.txt', 'w')
    for ep in range(0, epoches):
        running_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            opt.zero_grad()
            data = batch.cuda()
            # print(data.shape)
            points_pred = model(data)
            loss = chamfer_distance_loss(data, points_pred)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            if batch_idx % log_interval == log_interval - 1:
                print('[%d, %5d] loss: %.6f' %
                    (ep + 1, batch_idx + 1, running_loss / log_interval))
                print('[%d, %5d] loss: %.6f' %
                    (ep + 1, batch_idx + 1, running_loss / log_interval), file=loss_log)
                running_loss = 0.0
        if save_along_training:
            torch.save(model.state_dict(), os.path.join('../model', 'ep_%d.pth' % ep))
    if save_along_training:   # the last one
        torch.save(model.state_dict(), os.path.join('../model', 'ep_%d.pth' % ep))
    loss_log.close()
    return


if __name__ == '__main__':
    ROOT = "../data/nyc/"    # root path
    TRIAN_PATH = "../data/catelog/train.txt"
    MLP_DIMS = (3,64,64,64,128,1024)
    FC_DIMS = (1024, 512, 512)
    FOLDING1_DIMS = (521, 512, 512, 3)   # change the input feature of the first fc because now has 9 dims instead of 2
    FOLDING2_DIMS = (515, 512, 512, 3)
    MLP_DOLASTRELU = False
    if not os.path.exists('../model'):
        os.makedirs('../model')
    kwargs = {
        'lr': 0.0001,
        'epoches': 330,
        'batch_size': 16,
        'log_interval': 10,
        'save_along_training': True
    }

    with open(TRIAN_PATH) as fp:
        catelog = fp.readlines()
    catelog = [x.strip() for x in catelog]

    dataset = pcdDataset(ROOT, catelog)
    model = FoldingNetShapes(MLP_DIMS, FC_DIMS, FOLDING1_DIMS, FOLDING2_DIMS)
    train(dataset, model, **kwargs)
    print("End training!!!")


