import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from models.dsconv import DSConv_pro
from matplotlib import pyplot as plt
import matplotlib as mpl
import torchvision.transforms.functional as TF
from PIL import Image
import os


# cmap = mpl.cm.rainbow
#cmap = mpl.colors.ListedColormap(['r','g','b'])

class Edge_block(nn.Module):
    def __init__(self, in_c, out_c):
        super(Edge_block, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(out_c),
                                  nn.ReLU(inplace=True))
        self.conv0_x = DSConv_pro(out_c, out_c//8, kernel_size=5, morph=0, if_offset=True)
        self.conv0_y = DSConv_pro(out_c, out_c//8, kernel_size=5, morph=1, if_offset=True)
        self.conv2 = nn.Sequential(nn.Conv2d(out_c//4, out_c, 1, bias=False),
                                   nn.BatchNorm2d(out_c),
                                   nn.ReLU(inplace=True))
        
    def forward(self, x):
        x = self.conv1(x)
        x_0 = self.conv0_x(x)
        x_1 = self.conv0_y(x)
        x_out = torch.cat([x_0, x_1], dim=1)
        out = self.conv2(x_out)
        return out

class Decoder(nn.Module):
    def __init__(self, in_c, out_c, inp=False):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_c),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_c),
                                   nn.ReLU(inplace=True))
        self.inp = inp
    def forward(self, x, y): # x:decoder  y:encoder
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
        x = self.conv1(x)
        x = torch.cat((x, y), dim=1)
        out = self.conv2(x)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class Res34(nn.Module):
    def __init__(self, block=BasicBlock, pretrained=True):
        super(Res34, self).__init__()
        self.inplanes = 64
        layers = [3, 4, 6, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):  # 3,512,512
        x = self.conv1(x)  # 64,512,512
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 64,256,256
        x1 = self.layer1(x)  # 64,256,256
        x2 = self.layer2(x1)  # 128,128,128
        x3 = self.layer3(x2)  # 256,64,64
        x4 = self.layer4(x3)  # 512,64,64
        return x1, x2, x3, x4

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class Direction(nn.Module):
    def __init__(self, in_c):
        super(Direction, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, in_c // 4, 1),
                                   nn.BatchNorm2d(in_c // 4),
                                   nn.ReLU())
        self.conv_v = nn.Conv2d(in_c // 4, in_c // 8, (1, 9), padding=(0, 4))
        self.conv_h = nn.Conv2d(in_c // 4, in_c // 8, (9, 1), padding=(4, 0))
        self.conv_l = nn.Conv2d(in_c // 4, in_c // 8, (9, 1), padding=(4, 0))
        self.conv_r = nn.Conv2d(in_c // 4, in_c // 8, (1, 9), padding=(0, 4))
        self.conv2 = nn.Sequential(nn.BatchNorm2d(in_c // 4 + in_c // 4),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_c // 4 + in_c // 4, in_c, 1),
                                   nn.BatchNorm2d(in_c),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv_v(x)
        x2 = self.conv_h(x)
        x3 = self.inv_h_transform(self.conv_l(self.h_transform(x)))
        x4 = self.inv_v_transform(self.conv_r(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.conv2(x)
        return x

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class ACM(nn.Module):
    def __init__(self, channels, r=4):
        super(ACM, self).__init__()
        in_c = channels // r
        self.topdown = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(channels, in_c, 1),
                                     nn.BatchNorm2d(in_c),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_c, channels, 1),
                                     nn.BatchNorm2d(channels),
                                     nn.Sigmoid())
        self.bottomup = nn.Sequential(nn.Conv2d(channels, in_c, 1),
                                      nn.BatchNorm2d(in_c),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_c, channels, 1),
                                      nn.BatchNorm2d(channels),
                                      nn.Sigmoid())
        self.post = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(channels),
                                  nn.ReLU(inplace=True))

    def forward(self, x, y):
        bottomup_wei = self.bottomup(x)  # 低级
        topdown_wei = self.topdown(y)  # 高级
        xs = 2 * torch.mul(x, topdown_wei) + 2 * torch.mul(y, bottomup_wei)
        xs = self.post(xs)
        return xs

class Channel(nn.Module):
    def __init__(self, channels):
        super(Channel, self).__init__()
        self.dilation1 = nn.Sequential(nn.Conv2d(channels // 4, channels // 4, kernel_size=3, dilation=1, padding=1),
                                       nn.BatchNorm2d(channels // 4),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(nn.Conv2d(channels // 4, channels // 4, kernel_size=3, dilation=2, padding=2),
                                       nn.BatchNorm2d(channels // 4),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(nn.Conv2d(channels // 4, channels // 4, kernel_size=3, dilation=4, padding=4),
                                       nn.BatchNorm2d(channels // 4),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(nn.Conv2d(channels // 4, channels // 4, kernel_size=3, dilation=8, padding=8),
                                       nn.BatchNorm2d(channels // 4),
                                       nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(channels, channels, 1),
                                   nn.BatchNorm2d(channels),
                                   nn.ReLU(inplace=True))
        self.acm = ACM(channels // 4)

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = x[:, 0:c // 4, :, :]
        x2 = x[:, c // 4:c // 2, :, :]
        x3 = x[:, c // 2:-c // 4, :, :]
        x4 = x[:, -c // 4:, :, :]
        x1 = self.dilation1(x1)

        x2_0 = self.acm(x2, x1)
        x2 = self.dilation2(x2_0)

        x3_0 = self.acm(x3, x2)
        x3 = self.dilation3(x3_0)

        x4_0 = self.acm(x4, x3)
        x4 = self.dilation4(x4_0)

        x_out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.conv2(x_out)
        out = out + x
        return out

class Spacial(nn.Module):
    def __init__(self, channels):
        super(Spacial, self).__init__()
        self.conv1 = nn.Conv2d(channels, 1, kernel_size=(7, 1), padding=(3, 0), bias=False)
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=(1, 7), padding=(0, 3), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, y, x):
        x_y = x + y
        y1 = self.conv1(x_y)
        y1 = self.sigmoid(y1)
        y2 = self.conv2(x_y)
        y2 = self.sigmoid(y2)
        out = y1 * x + y2 * x + x
        return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        filter = [64, 128, 256, 512]
        self.resnet = Res34()

        self.edge1 = Direction(filter[0])
        self.edge2 = Direction(filter[1])

        self.conv_edge1 = Edge_block(3, filter[0])
        self.conv_edge2 = Edge_block(filter[0], filter[1])
        self.conv_edge3 = Edge_block(filter[1], filter[2])

        self.decoder3 = Decoder(filter[3], filter[2])
        self.decoder2 = Decoder(filter[2], filter[1], inp=True)
        self.decoder1 = Decoder(filter[1], filter[0], inp=True)

        # self.fconv = nn.Conv2d(filter[0], 1, 1)
        self.fconv = nn.Sequential(nn.ConvTranspose2d(filter[0], filter[0], kernel_size=2, stride=2, bias=False),
                                   nn.BatchNorm2d(filter[0]),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(filter[0], 1, 1),
                                   nn.Sigmoid())
                                   
        self.conv1 = nn.Sequential(nn.Conv2d(filter[2], filter[3], 1, bias=False),
                                   nn.BatchNorm2d(filter[3]),
                                   nn.ReLU(inplace=True))
        self.spacial1 = Spacial(filter[0])
        self.spacial2 = Spacial(filter[1])                        
        self.channel = Channel(filter[2])
        
    def forward(self, x, y):  # 3,512,512
        x1, x2, x3, x4 = self.resnet(x)

        o1 = F.interpolate(x1, scale_factor=2, mode='bilinear')
        o2 = F.interpolate(x2, scale_factor=4, mode='bilinear')
        o3 = F.interpolate(x3, scale_factor=8, mode='bilinear')
        o4 = F.interpolate(x4, scale_factor=8, mode='bilinear')
        # n1 = x1.detach().squeeze()
        # plt.figure()
        # plt.imshow(n1[0, 0, :, :].cpu().numpy())
        # plt.show()
        #
        # n2 = x2.detach().squeeze()
        # plt.figure()
        # plt.imshow(n2[0, 0, :, :].cpu().numpy())
        # plt.show()
        #
        # n3 = x3.detach().squeeze()
        # plt.figure()
        # plt.imshow(n3[0, 0, :, :].cpu().numpy())
        # plt.show()
        #
        # n4 = x4.detach().squeeze()
        # plt.figure()
        # plt.imshow(n4[0, 0, :, :].cpu().numpy())
        # plt.show()

        e1 = self.edge1(x1)
        e2 = self.edge2(x2)

        y1 = e1 + self.conv_edge1(y)
        y2 = e2 + self.conv_edge2(y1)
        y3 = x3 + self.conv_edge3(y2)


        a1 = F.interpolate(y1, scale_factor=2, mode='bilinear')
        # a1 = a1.detach().squeeze()
        # plt.figure()
        # plt.imshow(a1[0, 0, :, :].cpu().numpy(), cmap=cmap)
        # plt.show()
        #draw_heatmap(a1.cpu().numpy())

        a2 = F.interpolate(y2, scale_factor=4, mode='bilinear')
        # a2 = a2.detach().squeeze()
        # plt.figure()
        # plt.imshow(a2[0, 0, :, :].cpu().numpy(), cmap=cmap)
        # plt.show()

        a3 = F.interpolate(y3, scale_factor=8, mode='bilinear')
        # a3 = a3.detach().squeeze()
        # plt.figure()
        # plt.imshow(a3[0, 0, :, :].cpu().numpy(), cmap=cmap)
        # plt.show()

        m1 = self.spacial1(y1, x1)

        b1 = F.interpolate(m1, scale_factor=2, mode='bilinear')
        # b1 = b1.detach().squeeze()
        # plt.figure()
        # plt.imshow(b1[0, 0, :, :].cpu().numpy(), cmap=cmap)
        # plt.show()

        m2 = self.spacial2(y2, x2)

        b2 = F.interpolate(m2, scale_factor=4, mode='bilinear')
        # b2 = b2.detach().squeeze()
        # plt.figure()
        # plt.imshow(b2[0, 0, :, :].cpu().numpy(), cmap=cmap)
        # plt.show()

        y3 = self.channel(y3)

        b3 = F.interpolate(y3, scale_factor=8, mode='bilinear')
        # b3 = b3.detach().squeeze()
        # plt.figure()
        # plt.imshow(b3[0, 0, :, :].cpu().numpy(), cmap=cmap)
        # plt.show()


        x4 = self.conv1(x3) + x4
        b4 = F.interpolate(x4, scale_factor=8, mode='bilinear')

        d3 = self.decoder3(x4, y3)
        l1 = F.interpolate(d3, scale_factor=8, mode='bilinear')

        d2 = self.decoder2(d3, m2)
        l2 = F.interpolate(d2, scale_factor=4, mode='bilinear')

        d1 = self.decoder1(d2, m1)
        l3 = F.interpolate(d1, scale_factor=2, mode='bilinear')

        out = self.fconv(d1)  # 1,512,512
        l4 = F.interpolate(out, scale_factor=1, mode='bilinear')
        return out, o1, o2, o3, o4, a1, a2, a3, b1, b2, b3, b4, l1, l2, l3, l4

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# def generate_data(batch_size=256):
#     return np.random.rand(batch_size, 10, 10, 10), np.random.rand(batch_size, 10, 10, 10)


# 绘制热图




if __name__ == "__main__":
    x1 = torch.tensor(np.random.rand(16, 3, 512, 512).astype(np.float32))
    x2 = torch.tensor(np.random.rand(16, 3, 512, 512).astype(np.float32))
    model = Model()
    y1 = model(x1, x2)
    print(y1.size())


