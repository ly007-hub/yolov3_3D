import torch
import numpy as np
from copy import deepcopy
import os.path as osp
import os
import sys
sys.path.append("..")
from data.config import anchor_size_3D_try
import tools

try:
    # from mayavi import mlab
    # from qweqwe import mlab
    print ('mayavi already imported')
    mayavi_exist_flag = True
except:
    print ('no mayavi')
    mayavi_exist_flag = 0

# 数据存放在这个类里
class datanih():
    def __init__(self, data):
        self.data = data
        self.ids = list()
        self.root = "/data/liyi219/pnens_3D_data/v1_data/NIH"
        rootpath = osp.join(self.root, 'data')
        for root, dirs, files in os.walk(rootpath, topdown=False):
            self.ids.append((rootpath, files))

    def __len__(self):
        return len(self.data[0]['train_set'])

    def __getitem__(self, index):
        """
        return:
        im [1, 128, 128, 64] torch
        gt [y1, y2, x1, x2, z1, z2, 0] 相对
        """
        im, gt, mt, h, w, d = self.pull_item(index)
        im = torch.from_numpy(im)
        im = im[np.newaxis, :]
        # 绝对2相对
        gt = gt / [w, w, h, h, d, d]
        gt = gt.tolist()
        gt.append(0)
        return im, gt

    def pull_item(self, index):
        """
        return:
        冠状面、矢状面、水平面[y, x, z]
        img [128, 128, 64]
        bbox[y1, y2, x1, x2, z1, z2] 绝对
        mask[128, 128, 64]
        """
        dataset = self.data[0]
        dataset = dataset['train_set']
        img = dataset[index]['img']
        mask = dataset[index]['mask']
        height, width, depth = img.shape
        bbox = dataset[index]['bbox']


        return img, bbox, mask, height, width, depth

    def get_bbox_juedui(self, index):
        """
        return:
        bbox[y1, y2, x1, x2, z1, z2] 绝对
        """
        _, bbox, _, _, _, _ = self.pull_item(index)
        return bbox

    def get_bbox_whd(self, index):
        bbox = self.get_bbox_juedui(index)  # bbox[y1, y2, x1, x2, z1, z2]
        h = bbox[1] - bbox[0]
        w = bbox[3] - bbox[2]
        d = bbox[5] - bbox[4]
        return w, h, d

    def adjustWindow(self, scan, WW, WL):
        """
        调整窗宽窗位的函数
        :param img:
        :param WW: 窗宽
        :param WL: 窗位
        :return:
        """
        img = deepcopy(scan)
        img[img > WL + WW * 0.5] = WL + WW * 0.5
        img[img < WL - WW * 0.5] = WL - WW * 0.5
        return img





