# -*- coding: utf-8 -*-
import os
import numpy as np
from mypath import Path
from data import transforms as tr
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import torch

class Segmentationc(Dataset):
    NUM_CLASSES = 1

    def __init__(self, args, split='train'):
        super(Segmentationc, self).__init__()
        self._base_dir = Path.db_root_dir(args.dataset)
        self._image_dir = os.path.join(self._base_dir, 'train', 'images')
        self._cat_dir = os.path.join(self._base_dir, 'train', 'gt')
        self._edge_dir = os.path.join(self._base_dir, 'train','edges')
        self.args = args

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        _splits_dir = os.path.join(self._base_dir)
        self.batchsize = torch.cuda.device_count() * args.batch_size
        self.im_ids = []
        self.images = []
        self.categories = []
        self.edges = []
        # 将train_crops.txt文件中逐个分别加入到self.im_ids、self.images、self.categories列表中
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir,'train', splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + '_sat.jpg')
                _cat = os.path.join(self._cat_dir, line + '_mask.png')
                _ed = os.path.join(self._edge_dir, line + '.jpg')
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.edges.append(_ed)
        assert (len(self.images) == len(self.categories))

        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        if self.split[0] == 'test':
            return len(self.images)
        else:
            return len(self.images) // self.batchsize * self.batchsize

    # 逐个进行数据增强
    def __getitem__(self, index):
        _img, _target, _edge = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'edge': _edge}
        # print(sample.get('image', '111'))
        # print(sample.get('edge', '222'))
        for split in self.split:
            if split == 'train':
                sample = self.transform_tr(sample)
                # print(sample.get('image', '111'))
                # print(sample.get('edge', '222'))
                return sample
            # elif split == 'val':
            #     return self.transform_val(sample), self.im_ids[index]
            elif split == 'test':
                return self.transform_test(sample), self.im_ids[index]

    # 将images转为RGB图像
    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])
        _edge = Image.open(self.edges[index]).convert('RGB')
        return _img, _target, _edge

    # 数据增强
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomRotate(180),
            tr.RandomHorizontalFlip(),  # 以0.5的概率水平翻转图像
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # 将图像随机裁剪为不同的大小和宽高比，然后缩放裁剪得到的图像为指定大小。
            tr.RandomGaussianBlur(),  # 高斯模糊
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 用均值和标准差归一化张量图像[-1，1]
            tr.ToTensor()])  # ToTensor把灰度范围从0-255变换到0-1之间

        return composed_transforms(sample)

    # def transform_val(self, sample):
    #     composed_transforms = transforms.Compose([
    #         tr.FixedResize(size=self.args.crop_size),
    #         tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         tr.ToTensor()])
    #
    #     return composed_transforms(sample)

    def transform_test(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)
