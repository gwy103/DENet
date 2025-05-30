# -*- coding: utf-8 -*-
import os
import shutil
import torch
from collections import OrderedDict
import glob
from datetime import datetime

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))  # 遍历文件夹
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]  # 初始化1个
                for run in self.runs:  # 创建对比list
                    run_id = run.split('t_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred >= max_miou:
                    shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best.pth.tar'))

    # 网络参数
    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['dataset'] = self.args.dataset
        # p['backbone'] = self.args.backbone
        # p['out_stride'] = self.args.out_stride
        p['optimizer'] = self.args.optimizer
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        #p['lr_step'] = self.args.lr_step
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()

