import os
import json
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset

BASEDIR = Path('/home/xkz/forest')


class ForestDataset(Dataset):
    def __init__(self, split_path, gt, mode='train', transform=None):
        super().__init__()
        self.split_path = split_path
        self.gt_T_path = BASEDIR / Path(gt)
        self.uls_path = BASEDIR / Path('uls-50000')
        self.als_path = BASEDIR / Path('als-50000')

        # 取序号
        with open(self.split_path, encoding='utf8') as f:
            pcds_split = json.load(f)
            self.pcds_idx = pcds_split[mode]

        with open(self.gt_T_path, encoding='utf8') as f:
            gt_Ts = json.load(f)
            self.gt_Ts = gt_Ts

        self.transform = transform

    def __len__(self):
        return len(self.pcds_idx)

    def __getitem__(self, idx):
        pcd_idx = self.pcds_idx[idx]

        # found file name
        uls_filename = Path('uls_point_{}.npy'.format(pcd_idx))
        als_filename = Path('als_point_{}.npy'.format(pcd_idx))

        # Get points in each vehicles' coordinate system (no transforms)
        p0, p1 = np.load(str(self.als_path / als_filename)), np.load(str(self.uls_path / uls_filename))
        pts = np.concatenate((p0[None], p1[None]), axis=0).astype(np.float32)
        trans = np.array(self.gt_Ts[str(pcd_idx)]).astype(np.float32)

        if self.transform:
            pts, trans = self.transform(pts, trans)
        return pts, trans


if __name__ == '__main__':
    dataset = ForestDataset('')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=6, pin_memory=True, drop_last=True, num_workers=6,
                                              shuffle=True)
    index = 0
    for i, (p, t) in enumerate(data_loader):
        print(p.shape)
        print(p[0][1].shape)
        # print(p[0][1])
        print(t.shape)
        print(t[0])
        index += 1
        if index == 10:
            break
