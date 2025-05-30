from data import deepglobe, deepglobe_crop
from torch.utils.data import DataLoader
import os
import numpy as np
from mypath import Path
from data import transforms as tr
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset


class Segmentation(Dataset):
    NUM_CLASSES = 1

    def __init__(self, args, base_dir=Path.db_root_dir('deepglobe'), split='train'):
        super(Segmentation, self).__init__()
        self._base_dir = Path.db_root_dir(args.dataset)
        self._image_dir = os.path.join(self._base_dir, 'train', 'crops', 'images')
        self._cat_dir = os.path.join(self._base_dir, 'train', 'crops', 'gt')
        self._con_dir = os.path.join(self._base_dir, 'train', 'crops', 'connect_8_d1')
        self._con_d1_dir = os.path.join(self._base_dir, 'train', 'crops', 'connect_8_d3')
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
        self.connect_label = []
        self.connect_d1_label = []
        # 将train_crops.txt文件中逐个分别加入到self.im_ids、self.images、self.categories列表中
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '_crops' + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line)
                _cat = os.path.join(self._cat_dir, line)
                _con = os.path.join(self._con_dir, line)
                _con_d1 = os.path.join(self._con_d1_dir, line)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.connect_label.append(_con)
                self.connect_d1_label.append(_con_d1)

        assert (len(self.images) == len(self.categories))

        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        if self.split[0] == 'test':
            return len(self.images)
        else:
            return len(self.images) // self.args.batch_size * self.args.batch_size

    # 逐个进行数据增强
    def __getitem__(self, index):
        _img, _target, _connect0, _connect1, _connect2, _connect_d1_0, _connect_d1_1, _connect_d1_2 = self._make_img_gt_point_pair(
            index)
        sample = {'image': _img, 'label': _target, 'connect0': _connect0, 'connect1': _connect1, 'connect2': _connect2,
                  'connect_d1_0': _connect_d1_0, 'connect_d1_1': _connect_d1_1, 'connect_d1_2': _connect_d1_2}

        for split in self.split:
            if split == 'train':
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample), self.im_ids[index]
            elif split == 'test':
                return self.transform_test(sample), self.im_ids[index]

    # 将images转为RGB图像
    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        _connect0 = Image.open(self.connect_label[index].split('.png')[0] + '_0.png').convert("RGB")
        _connect1 = Image.open(self.connect_label[index].split('.png')[0] + '_1.png').convert("RGB")
        _connect2 = Image.open(self.connect_label[index].split('.png')[0] + '_2.png').convert("RGB")
        _connect_d1_0 = Image.open(self.connect_d1_label[index].split('.png')[0] + '_0.png').convert("RGB")
        _connect_d1_1 = Image.open(self.connect_d1_label[index].split('.png')[0] + '_1.png').convert("RGB")
        _connect_d1_2 = Image.open(self.connect_d1_label[index].split('.png')[0] + '_2.png').convert("RGB")

        return _img, _target, _connect0, _connect1, _connect2, _connect_d1_0, _connect_d1_1, _connect_d1_2

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

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_test(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)




def make_data_loader(args, **kwargs):

    if args.dataset == 'deepglobe':
        train_set = deepglobe_crop.Segmentation(args, split='train')
        val_set = deepglobe_crop.Segmentation(args, split='val')
        test_set = deepglobe.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        # train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        # val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        # test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        dataset = ImageFolder(root_path=Constants.ROOT, datasets='DRIVE')
        train_loader = torch.utils.data.DataLoader(dataset,batch_size=batchsize,shuffle=True,drop_last=True,num_workers=0)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False, drop_last=True,num_workers=0)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False, drop_last=True,num_workers=0)

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError