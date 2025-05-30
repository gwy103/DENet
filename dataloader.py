# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
import torch
from data.data_aug_c import Segmentationc
from data.test_data_aug_c import Segmentation_testc
from data.data_aug_d import Segmentationd
from data.test_data_aug_d import Segmentation_testd

def make_data_loader(args, **kwargs):

    if args.dataset == 'CHN':
        train_set = Segmentationc(args, split='train')
        #val_set = Segmentation(args, split='val')
        test_set = Segmentation_testc(args, split='test')
        num_class = 1
        # 处理完毕的数据集
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size*torch.cuda.device_count(),shuffle=True,drop_last=True,num_workers=0,**kwargs)
        #val_loader = torch.utils.data.DataLoader(val_set,batch_size=args.batch_size*torch.cuda.device_count(), shuffle=False, drop_last=True,num_workers=0,**kwargs)
        test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size*torch.cuda.device_count(), shuffle=False, drop_last=True,num_workers=0,**kwargs)
        return train_loader, test_loader, num_class

    elif args.dataset == 'deepglobe':
        train_set = Segmentationd(args, split='train')
        # val_set = Segmentation(args, split='val')
        test_set = Segmentation_testd(args, split='test')
        num_class = 1
        # 处理完毕的数据集
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size * torch.cuda.device_count(), shuffle=True, drop_last=True, num_workers=0, **kwargs)
        # val_loader = torch.utils.data.DataLoader(val_set,batch_size=args.batch_size*torch.cuda.device_count(), shuffle=False, drop_last=True,num_workers=0,**kwargs)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size * torch.cuda.device_count(), shuffle=False, drop_last=True, num_workers=0, **kwargs)
        return train_loader, test_loader, num_class

    else:
        raise NotImplementedError