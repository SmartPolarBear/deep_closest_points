import os
import json
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset

from scipy.spatial.transform import Rotation as R

BASEDIR = Path('/home/xkz/forest')


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def random_sample(pcd, nums):
    N, C = pcd.shape
    idxs = np.random.randint(0, N, size=nums)
    return pcd[idxs]


class ForestDataset(Dataset):
    def __init__(self, split, gt, mode='train', num_points=1024, gaussian_noise=False, unseen=False):
        super().__init__()
        self.split_path = BASEDIR / Path(split)
        self.gt_T_path = BASEDIR / Path(gt)
        self.uls_path = BASEDIR / Path('uls')
        self.als_path = BASEDIR / Path('als')

        self.num_points = num_points
        self.gaussian_noise = gaussian_noise

        with open(self.split_path, encoding='utf8') as f:
            pcds_split = json.load(f)
            self.pcds_idx = pcds_split[mode]
            if not unseen and mode == 'train':
                self.pcds_idx = list(
                    self.pcds_idx)+list(pcds_split['test'])+list(pcds_split['valid'])

        with open(self.gt_T_path, encoding='utf8') as f:
            gt_Ts = json.load(f)
            self.gt_Ts = gt_Ts

    def __len__(self):
        return len(self.pcds_idx)

    def __getitem__(self, idx):
        pcd_idx = self.pcds_idx[idx]

        uls_filename = Path('uls_point_{}.npy'.format(pcd_idx))
        als_filename = Path('als_point_{}.npy'.format(pcd_idx))

        p0, p1 = np.load(str(self.als_path / als_filename)
                         ), np.load(str(self.uls_path / uls_filename))

        if self.gaussian_noise:
            jitter_pointcloud(p0)
            jitter_pointcloud(p1)

        p0 = random_sample(p0, self.num_points)
        p1 = random_sample(p1, self.num_points)

        gt = np.array(self.gt_Ts[str(pcd_idx)]).astype(np.float32)

        rot_ab = gt[:3, :3]
        translation_ab = gt[:3, 3]

        rot_ba = rot_ab.T
        translation_ba = -rot_ba @ translation_ab

        euler_ab = R.from_matrix(rot_ab).as_euler('zyx')
        euler_ba = R.from_matrix(rot_ba).as_euler('zyx')

        # ignore intensity (dim #4)
        p0 = p0[:, :3].T
        p1 = p1[:, :3].T

        return p0.astype('float32'), p1.astype('float32'), \
            rot_ab.astype('float32'), translation_ab.astype('float32'), \
            rot_ba.astype('float32'), translation_ba.astype('float32'),\
            euler_ab.astype('float32'), euler_ba.astype('float32')


def get_forest_dataset(num_points, partition='train', gaussian_noise=False, unseen=False, factor=4):
    # factor is unused
    ret = ForestDataset(
        '../forest/split_train_valid_test2.json', 'gt.txt', mode=partition, num_points=num_points,
        gaussian_noise=gaussian_noise, unseen=unseen)
    return ret


if __name__ == '__main__':
    dataset = ForestDataset(
        '../forest/split_train_valid_test2.json', 'gt.txt', mode='train', gaussian_noise=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, pin_memory=True, drop_last=True, num_workers=6,
                                              shuffle=True)
    index = 0
    for i, (p0, p1, r0, t0, r1, t1, e0, e1) in enumerate(data_loader):
        print(i, ": ")
        print(p0.shape, p1.shape, r0.shape, t0.shape,
              r1.shape, t1.shape, e0.shape, e1.shape)

        rab = R.from_matrix(r0)
        rba = R.from_matrix(r1)

        print(rab.apply(p0[0, :, :3]) + t0.detach().cpu().numpy())
        print(rba.apply(p0[0, :, :3]) + t1.detach().cpu().numpy())

        print(p1[0, :, :3])

        break
