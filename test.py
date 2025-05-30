import argparse
import torch
import os
import numpy as np
import time
from data import transforms as tr
from PIL import Image
from torchvision import transforms
from data.utils import  *
from torchvision.utils import make_grid #, save_image
from data.dataloader import make_data_loader
from utils.metrics import Evaluator
from utils.loss import dice_bce_loss
from tqdm import tqdm
from models.res34_edge import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):

    #from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize([512, 512])
    im.save(filename)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--out-path', type=str, default='/home/students/master/2022/gaoy/experiments/e4/run/CHN/res34_edge/experiment_20231129_154444/',
                        help='mask image to save')
    # parser.add_argument('--backbone', type=str, default='resnet',
    #                     help='backbone name (default: resnet)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for test ')
    parser.add_argument('--ckpt', type=str, default='/home/students/master/2022/gaoy/experiments/e4/run/CHN/res34_edge/experiment_20231129_154444/checkpoint.pth.tar', #####
                        help='saved model')
    # parser.add_argument('--out-stride', type=int, default=8,
    #                     help='network output stride (default: 8)')
    parser.add_argument('--loss-type', type=str, default='con_ce',
                        choices=['ce', 'con_ce', 'focal'],
                        help='loss func type')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # parser.add_argument('--gpu-ids', type=str, default='0',
    #                     help='use which gpu to train, must be a \
    #                     comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='CHN',
                        choices=['Deepglobe', 'Mas', 'CHN'],
                        help='dataset name (default:deepglobe)')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size. spacenet:1280, DeepGlobe:1024.')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size. spacenet:1280, DeepGlobe:1024.')
    # parser.add_argument('--sync-bn', type=bool, default=False,
    #                     help='whether to use sync bn')
    # parser.add_argument('--freeze-bn', type=bool, default=False,
    #                     help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # if args.cuda:
    #     try:
    #         args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    #     except ValueError:
    #         raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    #kwargs = {'num_workers': args.workers, 'pin_memory': False}
    train_loader, test_loader, nclass = make_data_loader(args)

    model = Model()

    model = model.cuda()
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['state_dict'])

    out_path = os.path.join(args.out_path, 'other111/')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    evaluator = Evaluator(2)
    model.eval()
    evaluator.reset()
    tbar = tqdm(test_loader, desc='\r')
    for i, sample in enumerate(tbar):
        image, target, edge = sample[0]['image'], sample[0]['label'], sample[0]['edge']
        image = image.cpu().numpy()
        image1 = image[:, :, ::-1, :]
        image2 = image[:, :, :, ::-1]
        image3 = image[:, :, ::-1, ::-1]
        image = np.concatenate((image,image1,image2,image3), axis=0)
        image = torch.from_numpy(image).float()

        edge = edge.cpu().numpy()
        edge1 = edge[:, :, ::-1, :]
        edge2 = edge[:, :, :, ::-1]
        edge3 = edge[:, :, ::-1, ::-1]
        edge = np.concatenate((edge, edge1, edge2, edge3), axis=0)
        edge = torch.from_numpy(edge).float()

        img_name = sample[1][0].split('.')[0]
        if args.cuda:
            image, target, edge = image.cuda(), target.cuda(), edge.cuda()
        with torch.no_grad():
            output = model(image, edge)
        pred_full = []
        pred = output.data.cpu().numpy()
        target_n = target.cpu().numpy()
        pred_full.append(pred[0, ...])
        pred_full.append(pred[1, :, ::-1, :])
        pred_full.append(pred[2, :, :, ::-1])
        pred_full.append(pred[3, :, ::-1, ::-1])
        pred_full = np.asarray(pred_full).mean(axis=0)

        pred_full[pred_full > 0.1] = 1
        pred_full[pred_full < 0.1] = 0

        #evaluator.add_batch(target_n, su.astype(int))#
        evaluator.add_batch(target_n, pred_full)

        # save imgs
        out_image = make_grid(image[0,:].clone().cpu().data, 3, normalize=True)
        out_GT = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
        out_pred_label_sum = make_grid(decode_seg_map_sequence(pred_full), 3, normalize=False, range=(0, 255))
        out_pred_label_dis = make_grid(show_distance(torch.squeeze(target[:3], 1).detach().cpu().numpy(),pred_full), 3, normalize=False, range=(0, 255))

        save_image(out_image, out_path + img_name + '_sat.png')
        save_image(out_GT, out_path + img_name + '_GT' + '.png')
        save_image(out_pred_label_sum, out_path + img_name + '_pred' + '.png')
        save_image(out_pred_label_dis, out_path + img_name + '_dis' + '.png')

    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    mIoU = evaluator.Mean_Intersection_over_Union()
    IoU = evaluator.Intersection_over_Union()
    Precision = evaluator.Pixel_Precision()
    Recall = evaluator.Pixel_Recall()
    F1 = evaluator.Pixel_F1()
    print('Validation:')
    print('[numImages: %5d]' % (i * args.batch_size + image.data.shape[0]))
    print("Acc:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
          .format(Acc, mIoU, IoU, Precision, Recall, F1))

if __name__ == "__main__":
   main()