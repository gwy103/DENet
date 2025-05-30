# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
from PIL import Image
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt

def edge_conv2d(im):

    conv_op1 = nn.Conv2d(3, 3, kernel_size=3, padding=1,bias=False)
    sobel_kernel1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_kernel1 = sobel_kernel1.reshape((1, 1, 3, 3))
    conv_op1.weight.data = torch.from_numpy(sobel_kernel1)
    edge_detect1 = torch.abs(conv_op1(Variable(im)))

    conv_op2 = nn.Conv2d(3, 3, kernel_size=3, padding=1,bias=False)
    sobel_kernel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel2 = sobel_kernel2.reshape((1, 1, 3, 3))
    conv_op2.weight.data = Variable(torch.from_numpy(sobel_kernel2))
    edge_detect2 = torch.abs(conv_op2(Variable(im)))
    
    conv_op3 = nn.Conv2d(3, 3, kernel_size=3,padding=1, bias=False)
    sobel_kernel3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel3 = sobel_kernel3.reshape((1, 1, 3, 3))
    conv_op3.weight.data = Variable(torch.from_numpy(sobel_kernel3))
    edge_detect3 = torch.abs(conv_op3(Variable(im)))

    conv_op4 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel4 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel4 = sobel_kernel4.reshape((1, 1, 3, 3))
    conv_op4.weight.data = Variable(torch.from_numpy(sobel_kernel4))
    edge_detect4 = torch.abs(conv_op4(Variable(im)))

    sobel_out = edge_detect1 + edge_detect2 + edge_detect3 + edge_detect4
    sobel_out = sobel_out.squeeze().detach().numpy()
    return sobel_out

def main():
    edge_dir = '/home/students/master/2022/gaoy/dataset/CHN/train'
    with open(os.path.join(os.path.join(edge_dir, 'train' + '.txt')), "r") as f:
        lines = f.read().splitlines()
    for ii, line in enumerate(lines):
        _edge = os.path.join(edge_dir, 'images', line + '_sat.jpg')
        im = Image.open(_edge).convert('L')
        im = np.array(im, dtype='float32')
        im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))  # 1,1,512,512
        edge_detect = edge_conv2d(im)
        im = Image.fromarray(edge_detect)
        if im.mode == 'F':
            im = im.convert('RGB')
        im.save(os.path.join('/home/students/master/2022/gaoy/dataset/CHN/train/edges_2sobel',line + '.jpg'), quality=95)
    f.close()
    print('end')

    # im = Image.open('/home/students/master/2022/gaoy/dataset/CHN/train/images/am100009_sat.jpg').convert('L')
    # im = np.array(im, dtype='float32')
    # im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1]))) #1,1,512,512
    # edge_detect = edge_conv2d(im)
    # im = Image.fromarray(edge_detect)
    # if im.mode == 'F':
    #     im = im.convert('RGB')
    # im.save('/home/students/master/2022/gaoy/experiments/e3/edges/11.jpg',quality=95)

if __name__ == "__main__" :
	 main()
