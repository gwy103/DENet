# -*- coding: utf-8 -*-
import torch
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter

class Normalize(object):

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']  # 从序列image中随机抽取n个元素，并将n个元素以list形式返回
        mask = sample['label']
        edge = sample['edge']

        img = np.array(img).astype(np.float32)
        edge = np.array(edge).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        img /= 255.0
        img -= self.mean
        img /= self.std
        edge /= 255.0
        edge -= self.mean
        edge /= self.std
        mask /= 255.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0

        return {'image': img, 'label': mask, 'edge': edge}

class ToTensor(object):
    "将数组转变为tensor"
    "np image: H W C"
    "torch image: C H W"

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        edge = sample['edge']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))  # transpose()：转置
        edge = np.array(edge).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()  # 将数组变成tensor
        edge = torch.from_numpy(edge).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img, 'label': mask, 'edge': edge}

class RandomHorizontalFlip(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        edge = sample['edge']

        if random.random() < 0.5:  # random.random()用于生成一个0到1的随机符点数: 0 <= n < 1.0
            img = img.transpose(Image.FLIP_LEFT_RIGHT)  # Image.FLIP_LEFT_RIGHT左右翻转
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img, 'label': mask, 'edge':edge}

class RandomRotate(object):

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        edge = sample['edge']

        rotate_degree = random.uniform(-1 * self.degree, self.degree)  # random.uniform随机生成一个实数，在[x,y]范围内
        img = img.rotate(rotate_degree, Image.BILINEAR)  # 双线性插值[旋转]
        edge = edge.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)  # 最近邻插值[旋转]

        return {'image': img, 'label': mask, 'edge':edge}

class RandomGaussianBlur(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        edge = sample['edge']

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))  # filter:调用滤波函数对图像进行滤波
            # 图象滤波：高斯模糊[本质是数据光滑技术]

        return {'image': img, 'label': mask, 'edge': edge}

class RandomScaleCrop(object):

    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        edge = sample['edge']

        # 随机范围
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))  # 用于生成一个指定范围内的整数，a<=n<=b
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        img = img.resize((ow, oh), Image.BILINEAR)  # 图片缩放，双线性
        edge = edge.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)  # 低质量

        # pad 裁剪
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0

            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)  # ImageOps.expand：左上右下padding像素尺寸，fill为填充的颜色
            edge = ImageOps.expand(edge, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)

        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))  # 裁剪图片
        edge = edge.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img, 'label': mask, 'edge':edge}

class FixScaleCrop(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        edge = sample['edge']

        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)

        img = img.resize((ow, oh), Image.BILINEAR)
        edge = edge.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # 中心裁剪
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        edge = edge.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img, 'label': mask,'edge':edge}

class FixedResize(object):

    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        edge = sample['edge']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        edge = edge.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img, 'label': mask, 'edge': edge}

class FixedResize_test(object):

    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        edge = sample['edge']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        edge = edge.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img, 'label': mask, 'edge':edge}

class Normalize_test(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        edge = sample['edge']
        img = np.array(img).astype(np.float32)
        edge = np.array(edge).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        img /= 255.0
        img -= self.mean
        img /= self.std
        edge /= 255.0
        edge -= self.mean
        edge /= self.std
        mask /= 255.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0

        return {'image': img,'label': mask, 'edge':edge}

class ToTensor_test(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        edge = sample['edge']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        edge = np.array(edge).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        edge = torch.from_numpy(edge).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img, 'label': mask, 'edge':edge}
