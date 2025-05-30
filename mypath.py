# -*- coding: utf-8 -*-
class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'deepglobe':
            return '/home/students/master/2022/gaoy/dataset/deepglobe'
        elif dataset == 'Mas':
            return '/home/students/master/2022/gaoy/dataset/Massachsetts'
        elif dataset == 'CHN':
            return '/home/students/master/2022/gaoy/dataset/CHN'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError