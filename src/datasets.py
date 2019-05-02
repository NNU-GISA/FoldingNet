'''
PointCloudDataset in deepgeom
ShapeNetDataset in carvingnet

author 1: cfeng
created : 1/26/18 11:21 PM

author 2: Ruoyu Wang
created : 10/11/18 7:14 PM
'''

import os
import sys
import argparse
import glog as logger

from torch.utils.data import Dataset
import numpy as np
import torch

class PointCloudDataset(Dataset):

    def __init__(self, pkl_path, shuffle_point_order='no'):
        self.shuffle_point_order = shuffle_point_order

        logger.info('loading: '+pkl_path)
        # with open(pkl_path) as f:

        raw_data = np.load(pkl_path, encoding='bytes').item()
        self.all_data = raw_data[b'data'] #[BxNx3]
        if shuffle_point_order=='preprocess':
            for i in xrange(self.all_data.shape[0]):
                np.random.shuffle(self.all_data[i])
        self.all_label = np.asarray(raw_data[b'label'], dtype=np.int64)

        logger.info('pkl loaded: data '+str(self.all_data.shape)+', label '+str(self.all_label.shape))

        logger.check_eq(len(self.all_data.shape), 3,
                        'data field should of size BxNx3!')
        logger.check_eq(self.all_data.shape[-1], 3,
                        'data field the last dimension size should be 3!')
        logger.check_eq(len(self.all_label.shape), 1,
                        'label field should be one dimensional!')
        logger.check_eq(self.all_data.shape[0], self.all_label.shape[0],
                        'data field and label field should have the same size along the first dimension!')


    def __len__(self):
        return self.all_data.shape[0]


    def __getitem__(self, idx):
        if self.shuffle_point_order=='online':
            np.random.shuffle(self.all_data[idx])
        return {'data':self.all_data[idx], 'label':self.all_label[idx]}


class ShapeNetDataset:
    """shapenet dataset

    load shape net point cloud: BxNx3

    Attributes:
        shuffle_point_order -- define the mode of shuffling point
        all_data -- MxNx3 all point cloud data
    """

    def __init__(self, data_path, shuffle_point_order='no', rand_rot=True):
        self.shuffle_point_order = shuffle_point_order
        logger.info('loading' + data_path)
        self.all_data = np.load(data_path)['data']
        self.rand_rot = rand_rot
        if shuffle_point_order == 'preprocess':
            [np.random.shuffle(pc) for pc in self.all_data]
        logger.info('loaded: data ' + str(self.all_data.shape))
        logger.check_eq(len(self.all_data.shape), 3,
                        'data field should of size BxNx3!')
        logger.check_eq(self.all_data.shape[-1], 3,
                        'data field the last dimension size should be 3!')

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        if self.shuffle_point_order == 'online':
            np.random.shuffle(self.all_data[idx])
        if self.rand_rot:
            r = rand_ortho_rotation_matrix()
            return {'data': np.matmul(self.all_data[idx], r)}
        else:
            return {'data': self.all_data[idx]}


class pcdDataset(Dataset):
    """
    point cloud dataset
    load individual .npy files in a folder
    """
    def __init__(self, root, roster):
        """
        :param root: root folder for all .npy files
        :param roster: a list of file names for this dataset. subset of os.listdir(root). could be trian, test, val file names
        """
        self.root = root
        self.roster = roster
        return


    def __len__(self):
        return len(self.roster)


    def __getitem__(self, idx):
        pcd_name = os.path.join(self.root, self.roster[idx])
        self.x = np.load(pcd_name)
        self.x = torch.FloatTensor(self.x)
        return self.x

