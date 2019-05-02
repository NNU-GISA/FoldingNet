#!/usr/bin/env python
'''
visualize point cloud from numpy array
TODO: change visualize code and save pcd files
author  : Yuqiong Li, Ruoyu Wang
created : 05/02/19  11:26 AM
'''
import numpy as np
from matplotlib import pyplot as plt
import utils
import pptk

def main():
    o = np.load("../val/o0.npy")   # original data
    p = np.load("../val/p0.npy")   # predicted data
    # print(o[0].shape)
    # print(o[0])
    pptk.viewer(o[0])
    pptk.viewer(p[0])
    # utils.vis_pts(o[0], 'b', 'tab20')
    # plt.show()
    # utils.vis_pts(p[0], 'b', 'tab20')
    # plt.show()
    return


if __name__ == "__main__":
    main()
