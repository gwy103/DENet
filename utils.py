# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_seg_map_sequence(label_masks):  #[]
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, plot=False):
    n_classes = 2
    # label_colours = get_labels()

    # r = label_mask.copy()
    # g = label_mask.copy()
    # b = label_mask.copy()
    # for ll in range(0, n_classes):
    #     r[label_mask == ll] = label_colours[ll, 0]
    #     g[label_mask == ll] = label_colours[ll, 0]
    #     b[label_mask == ll] = label_colours[ll, 0]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:,:,1] = label_mask
    rgb[np.where(rgb==1)]=255
    # rgb[:, :, 0] = r / 255.0
    # rgb[:, :, 1] = g / 255.0
    # rgb[:, :, 2] = b / 255.0

    if plot:
        plt.imshow(rgb)  # 展示热力图，即将二维数组中的元素用颜色表示
        plt.show()
    else:
        return rgb

def get_labels():
    return np.asarray([[0, 0, 0], [0, 255, 0]])

def show_distance(label_masks,preds):
    rgb_masks = []
    for i in range(len(label_masks)):
        rgb_mask = show_dis(label_masks[i],preds[i])
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def show_dis(label_mask,pred,plot=False):

    distance = label_mask-pred  #0:正确 1:没预测到 blue  -1：预测错 red

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 2][np.where((label_mask == 1) & (pred==0))] = 255 #blue
    rgb[:, :, 0][np.where((label_mask == 0) & (pred==1))] = 255 #red
    rgb[:, :, 1][np.where((label_mask == 1) & (pred==1))] = 255 #green
    if plot:
        plt.imshow(rgb)  # 展示热力图，即将二维数组中的元素用颜色表示
        plt.show()
    else:
        return rgb


