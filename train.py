# -*- coding: utf-8 -*-
import argparse
import os, time
import numpy as np
import torch
from tqdm import tqdm
from time import time
import torch.optim as optim
from models.replicate import patch_replication_callback
from utils.saver import Saver
from utils.metrics import Evaluator
from utils.lr_scheduler import LR_Scheduler
from utils.loss import dice_bce_loss
from mypath import Path
from data.dataloader import make_data_loader
from models.res34_edge import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7'

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # define Tensorboard Summary
        #self.summary = TensorboardSummary(self.saver.experiment_dir)
        #self.writer = self.summary.create_summary()

        self.batchsize = torch.cuda.device_count() * args.batch_size
        print(self.batchsize)
        # define Dataloader
        #kwargs = {'num_workers': args.workers, 'pin_memory': True}
        #self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        self.train_loader, self.test_loader, self.nclass = make_data_loader(args)

        # define network
        model = Model()
        # model.half()
        # train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
        #                 {'params': model.get_2x_lr_params(), 'lr': args.lr * 2}]

        # define Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.weight_decay)
            print('Optimizer: Adam')
        elif args.optimizer == 'SGD':
            #optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,weight_decay=args.weight_decay)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,nesterov=args.nesterov)
            print('Optimizer: SGD')

        # optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)


        # define criterion
        # whether to use class balanced weights
        # if args.use_balanced_weights:
        #     classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
        #     if os.path.isfile(classes_weights_path):
        #         weight = np.load(classes_weights_path)
        #     else:
        #         weight = calculate_weights_labels(args.dataset, self.train_loader, self.nclass)
        #     weight = torch.from_numpy(weight.astype(np.float32))
        # else:
        #     weight = None
        self.criterion = dice_bce_loss()# 返回a, b
        #self.criterion_edge = Focalloss()
        self.model, self.optimizer = model, optimizer

        # define Evaluator
        self.evaluator = Evaluator(2)
        # define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))
        # if args.lr_update:
        #     self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.gamma)
        # else:
        #     self.scheduler = None

        # using cuda
        if args.cuda:
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model,device_ids=range(torch.cuda.device_count()))
            patch_replication_callback(self.model)

        # resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])  # 加载训练好的参数
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        # clear start epoch if finetuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        self.evaluator.reset()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        tic = time()
        for i, sample in enumerate(tbar):
            image, target, edge = sample['image'], sample['label'], sample['edge']
            if self.args.cuda:
                image, target, edge = image.cuda(), target.cuda(), edge.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            output = self.model(image, edge)
            target = torch.unsqueeze(target, 1)  # 升维
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            #self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            pred = output.data.cpu().numpy()
            target_n = target.cpu().numpy()
            # add batch sample into evaluator
            pred[pred > 0.1] = 1
            pred[pred < 0.1] = 0
            self.evaluator.add_batch(target_n, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        IoU = self.evaluator.Intersection_over_Union()
        Precision = self.evaluator.Pixel_Precision()
        Recall = self.evaluator.Pixel_Recall()
        F1 = self.evaluator.Pixel_F1()
        mylogt = open('/home/students/master/2022/gaoy/experiments/e4/logs/' + 'train_log4.txt', 'a+')
        print('********', file=mylogt)
        print('Train:', file=mylogt)
        print('epoch:', epoch, '   numImages:', i * self.batchsize + image.data.shape[0], '    time:', int(time() - tic), file=mylogt)
        print('Acc:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{} '.format(Acc, mIoU, IoU, Precision, Recall, F1), file=mylogt)
        print('Loss: %.3f' % (train_loss), file=mylogt)
        mylogt.close()

        new_pred = IoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred, }, is_best)

    # def validation(self, epoch):
    #     self.model.eval()
    #     self.evaluator.reset()
    #     tbar = tqdm(self.val_loader, desc='\r')
    #     val_loss = 0.0
    #     num_img_tr = len(self.val_loader)
    #     tic = time()
    #     for i, sample in enumerate(tbar):
    #         image, target, edge = sample[0]['image'], sample[0]['label'], sample[0]['edge']
    #         image, target, edge = image.cuda(), target.cuda(), edge.cuda()
    #         with torch.no_grad():
    #             output = self.model(image, edge)
    #         target = torch.unsqueeze(target, 1)
    #         loss = self.criterion(output, target)
    #         val_loss += loss.item()
    #         tbar.set_description('val loss: %.3f' % (val_loss / (i + 1)))
    #         pred = output.data.cpu().numpy()
    #         target_n = target.cpu().numpy()
    #         # add batch sample into evaluator
    #         pred[pred >= 0.5] = 1
    #         pred[pred < 0.5] = 0
    #         self.evaluator.add_batch(target_n, pred)
    #         # if i % (num_img_tr // 1) == 0:
    #         #     self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, i, split='val')
    #
    #     # fast test during the training
    #     Acc = self.evaluator.Pixel_Accuracy()
    #     mIoU = self.evaluator.Mean_Intersection_over_Union()
    #     IoU = self.evaluator.Intersection_over_Union()
    #     Precision = self.evaluator.Pixel_Precision()
    #     Recall = self.evaluator.Pixel_Recall()
    #     F1 = self.evaluator.Pixel_F1()
    #     mylogv = open('/home/students/master/2022/gaoy/experiments/e4/logs/' + 'val_log9.txt', 'a+')
    #     print('********', file=mylogv)
    #     print('val:', file=mylogv)
    #     print('epoch:', epoch, '   numImages:', i * self.batchsize + image.data.shape[0], '    time:', int(time() - tic), file=mylogv)
    #     print('Acc:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{} '.format(Acc, mIoU, IoU, Precision, Recall, F1), file=mylogv)
    #     print('Loss: %.3f' % (val_loss), file=mylogv)
    #     mylogv.close()
    #
    #     new_pred = IoU
    #     if new_pred > self.best_pred:
    #         is_best = True
    #         self.best_pred = new_pred
    #         self.saver.save_checkpoint({
    #             'epoch': epoch + 1,
    #             'state_dict': self.model.module.state_dict(),
    #             'optimizer': self.optimizer.state_dict(),
    #             'best_pred': self.best_pred, }, is_best)

def main():
    parser = argparse.ArgumentParser(description="Pytorch new Training")
    # parser.add_argument('--backbone', type=str, default='resnet',
    #                     help='backbone name (default: resnet)')
    # parser.add_argument('--out_stride', type=int, default=8,
    #                     help='network output stride (default:8)')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['adam', 'SGD'],
                        help='optimizer type')
    parser.add_argument('--lr_update', type=bool, default=True,
                        help='lr update type')
    parser.add_argument('--dataset', type=str, default='CHN',
                        choices=['deepglobe', 'Mas', 'CHN'],
                        help='dataset name (default:deepglobe)')
    parser.add_argument('--workers', type=int, default=16,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='crop image size')
    # parser.add_argument('--sync_bn', type=bool, default=False,
    #                     help='whether to use sync bn')
    # parser.add_argument('--freeze_bn', type=bool, default=False,
    #                     help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss_type', type=str, default='focal',
                        choices=['ce', 'con_ce', 'focal'],
                        help='loss func type')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default:16)')
    # parser.add_argument('--use_balanced_weights', action='store_true', default=False,
    #                     help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default:0.01)')
    # parser.add_argument('--lr_step', type=int, default=10000,
    #                     metavar='N', help='')
    # parser.add_argument('--gamma', type=float, default=0.95, metavar='M',
    #                     help='')
    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default:poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default:5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default:False)')
    # cuda, seed and logging
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='5',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='res34_edge',
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no_val', action='store_true', default=True,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.checkname is None:
        args.checkname = 'res34_edge'
    print(args)
    torch.manual_seed(args.seed)

    # learning_rate = [0.01,0.007]
    # for i in learning_rate:
    #     args.lr = i
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
        #     trainer.validation(epoch)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

