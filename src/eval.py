#!/usr/bin/env python
'''
evaluate point cloud generative results

author  : Yuqiong Li, Ruoyu Wang
created : 05/02/19  10:30AM
'''
import sys
import torch
from foldingnet import FoldingNetVanilla, FoldingNetShapes
from torch.utils.data import DataLoader
import torch
from foldingnet import ChamfersDistance
from datasets import pcdDataset
import numpy as np
from utils import check_exist_or_remove
import scipy

def eval(dataset, model, batch_size, log_interval):
    """test implicit version of foldingnet
    TODO: return indices of a specific sample for comparison with meshes and voxels too
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    chamfer_distance_loss = ChamfersDistance()
    model = model.eval().cuda()
    check_exist_or_remove("../log/eval_loss_log.txt")
    loss_log = open('../log/eval_loss_log.txt', 'w')
    running_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        data = batch.cuda()
        points_pred = model(data)
        np.save("../res/o{}.npy".format(batch_idx), data.cpu().detach().numpy())   # original
        np.save("../res/p{}.npy".format(batch_idx), points_pred.cpu().detach().numpy())    # predicted
        loss = chamfer_distance_loss(data, points_pred)
        running_loss += loss.item()
        if batch_idx % log_interval == log_interval - 1:
            print('[%d, %5d] loss: %.6f' %
            (1, batch_idx + 1, running_loss / log_interval))
            print('[%d, %5d] loss: %.6f' %
            (1, batch_idx + 1, running_loss / log_interval), file=loss_log)
            running_loss = 0.0
    loss_log.close()
    return


def jsd(o, p, base=np.e):
    """
    compute Jensen-Shannon Divergence between original and predicted point clouds
    https://gist.github.com/zhiyzuo/f80e2b1cfb493a5711330d271a228a3d
    :param o: original pcd, a 3D numpy array of N x 4096 x 3
    :param p: predicted pcd, same size as o
    :return:
    """
    ## normalize p, q to probabilities
    o, p = o / o.sum(), p / p.sum()
    m = 1. / 2 * (o + p)
    return scipy.stats.entropy(p, m, base=base) / 2. + scipy.stats.entropy(o, m, base=base) / 2.




def main(modelpath):
    ROOT = "../data/nyc/"    # root path
    TEST_PATH = "../data/catelog/test.txt"
    MLP_DIMS = (3,64,64,64,128,1024)
    FC_DIMS = (1024, 512, 512)
    FOLDING1_DIMS = (521, 512, 512, 3)   # change the input feature of the first fc because now has 9 dims instead of 2
    FOLDING2_DIMS = (515, 512, 512, 3)

    with open(TEST_PATH) as fp:
        catelog = fp.readlines()
    catelog = [x.strip() for x in catelog]

    testset = pcdDataset(ROOT, catelog)
    model = FoldingNetShapes(MLP_DIMS, FC_DIMS, FOLDING1_DIMS, FOLDING2_DIMS)
    model.load_state_dict(torch.load(modelpath))
    batch_size = 16
    log_interval = 10
    eval(testset, model, batch_size, log_interval)
    print("End evaluation!!!")


if __name__ == "__main__":
    main(*sys.argv[1:])