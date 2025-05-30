import os
import numpy as np
from mypath import Path
from data import transforms as tr
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class Segmentation_testc(Dataset):

    def __init__(self, args, split='test'):

        super(Segmentation_testc, self).__init__()
        self._base_dir = Path.db_root_dir(args.dataset)
        self._image_dir = os.path.join(self._base_dir, 'test', 'images')
        self._cat_dir = os.path.join(self._base_dir,'test', 'gt')
        self._edge_dir = os.path.join(self._base_dir, 'test', 'edges')
        self.args = args

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        _splits_dir = os.path.join(self._base_dir)

        self.im_ids = []
        self.images = []
        self.categories = []
        self.edges = []
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, 'test',splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + '_sat.jpg')
                _cat = os.path.join(self._cat_dir, line + '_mask.png')
                _edge = os.path.join(self._edge_dir, line + '.jpg')
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.edges.append(_edge)
        assert (len(self.images) == len(self.categories))

        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        if self.split[0] == 'test':
            return len(self.images)
        else:
            return len(self.images) // self.args.batch_size * self.args.batch_size

    def __getitem__(self, index):
        _img, _target, _edge = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'edge': _edge}
        samplpr = self.transform_test(sample)
        return samplpr, self.im_ids[index]

    def _make_img_gt_point_pair(self, index):
        # _img = Image.open(self.categories[index]).convert('RGB')
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])  # 转为灰色图像
        _edge = Image.open(self.edges[index]).convert('RGB')
        return _img, _target, _edge

    def transform_test(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize_test(size=self.args.crop_size),
            tr.Normalize_test(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor_test()])

        return composed_transforms(sample)


