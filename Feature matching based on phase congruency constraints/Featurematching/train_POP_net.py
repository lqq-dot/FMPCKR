import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
import warnings
import glob
import time
import os
from typing import List
from torch.utils.data import DataLoader
from POP_net_class import POPNet
from reconstruct_net import ReconstructNet
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_dilation
from dataset_class import PointDataset,PairPointDataset
from point_select import point_select

warnings.filterwarnings("ignore")

# train POP network

def train_POP(args):
    device = args.device
    device_recon = args.device
    save_path = args.save_path

    save_epoch_interval = args.save_epoch_interval
    print_interval = args.print_iter_interval
    save_interval = args.save_iter_interval

    # point_dataset = PointDataset(args.train_image_path, args.random_aug_num,
    #                              args.train_height, args.train_width)
    point_dataset = PairPointDataset(args.train_image_path, args.random_aug_num,
                                 args.train_height, args.train_width)

    dataloader = DataLoader(point_dataset, batch_size=1, shuffle=True,
                            num_workers=args.data_num_workers)

    net_recon = ReconstructNet(feature_len=64, in_f_len=64)
    need_info_mark = False
    if args.recon_net_path is not None:
        need_info_mark = True
        checkpoint_recon = torch.load(args.recon_net_path, map_location=device_recon)
        net_recon.load_state_dict(checkpoint_recon['model_state_dict_recon'], strict=True)
    net_recon.eval()
    net_recon.to(device_recon)

    net = POPNet()
    epoch_init = 0
    checkpoint = None
    if args.POP_checkpoint_path is not None:
        checkpoint = torch.load(args.POP_checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)
        net.train()
        epoch_init = checkpoint['epoch']
    net.to(device)

    optimizer = optim.Adam(net.parameters())
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss_MSE = nn.MSELoss()

    image_row, image_col = point_dataset.image_row, point_dataset.image_col
    resp_thre = 0.5
    # the local radius of the suppression of local non-maximum
    local_rad = 4
    back1_weight = 1
    # the region near to the border of image will be ignored in the computation of loss
    out_dist = 7
    # the pair of points whose reprojection error is less than soft_dist will be
    # considered as the correspondence
    soft_dist = 1

    cum_iter_num = 0
    max_epoch = epoch_init + args.max_epoch
    iter_num = len(dataloader)
    def convert_list2tensor(lst:List[List[torch.Tensor]])->torch.Tensor:
        # 将每个子列表的 Tensor 堆叠为一个 Tensor
        stacked_sublist_tensors = [torch.stack(sublist) for sublist in lst]
        # 将所有堆叠的子列表 Tensor 再次堆叠为一个 Tensor
        res = torch.stack(stacked_sublist_tensors)
        return res

    for epoch in range(epoch_init, max_epoch):
        loss_score_cum = 0
        disc_value_cum = 0
        disc_ratio_cum = 0
        info_value_cum = 0
        fore_num_cum = 0
        fore_loss_cum = 0
        back_num_cum = 0
        back_loss_cum = 0
        repeat_mean_cum = 0
        image_base_num = 0
        fore_num_cum_runtime = 0
        image_real_num = 0

        time_last = time.time()

        out_rad = 5
        inner_rad = 3

        for idx, batch in enumerate(dataloader):
            cum_iter_num += 1
            optimizer.zero_grad()
            # 提取所有 "image" 键对应的 Tensor
            image_tensors = []
            H = []
            base_gray = []
            image_ori = []
            for d in batch:
                image_tensors.append([i["image"] for i in d])
                H .append([i["H"] for i in d])
                base_gray.append( [i['base_gray'] for i in d])
                image_ori.append([i["image_ori"] for i in d])

            # 合并所有提取出的 Tensor
            input = torch.squeeze(convert_list2tensor(image_tensors),dim=2)
            H = torch.squeeze(convert_list2tensor(H),dim=2)
            base_gray =torch.squeeze(convert_list2tensor(base_gray),dim=2)
            image_ori = torch.squeeze(convert_list2tensor(image_ori),dim=2)

            # input, H, image_ori, base_gray = \
            #     batch['image'], batch['H'], batch['image_ori'], \
            #     batch['base_gray']

            batch_size = input.size(0)
            input_each_num = input.size(2)
            input = input.view(tuple(np.r_[input.shape[0] * input.shape[1] * input.shape[2],
                                           input.shape[3:]]))
            H = H.view(tuple(np.r_[H.shape[0] * H.shape[1]* H.shape[2], H.shape[3:]]))
            image_ori = image_ori.view(tuple(np.r_[image_ori.shape[0] * image_ori.shape[1]* image_ori.shape[2],
                                                   image_ori.shape[3:]]))
            base_gray = base_gray.view(tuple(np.r_[base_gray.shape[0] * base_gray.shape[1]* base_gray.shape[2],
                                                   base_gray.shape[3:]]))

            input = input.to(device)
            score_output, desc = net(input)
            score_detach = torch.sigmoid(score_output).detach()
            gt = score_output.detach().clone()

            loss_disc = 0
            loss_disc_exist = False
            for batch_id in range(batch_size):
                range_now = range(input_each_num * batch_id, input_each_num * (batch_id + 1))
                loss_disc_now, loss_disc_exist_now, loss_info_now, print_dict = \
                    point_select(score_output, score_detach, desc, gt, H, range_now,
                                 local_rad, out_rad, inner_rad, soft_dist,
                                 back1_weight, out_dist, image_ori,
                                 input, net_recon, need_info_mark)
                loss_disc += loss_disc_now

                loss_disc_exist |= loss_disc_exist_now
                if loss_disc_exist_now:
                    fore_num_cum += print_dict['fore_num_cum']
                    fore_loss_cum += print_dict['fore_loss_cum']
                    back_num_cum += print_dict['back_num_cum']
                    back_loss_cum += print_dict['back_loss_cum']
                    repeat_mean_cum += print_dict['repeat_mean_cum']
                    image_base_num += print_dict['image_base_num']
                    fore_num_cum_runtime += print_dict['fore_num_cum_runtime']
                    image_real_num += print_dict['image_real_num']
                    disc_value_cum += print_dict['disc_value']
                    info_value_cum += print_dict['info_value']
                    disc_ratio_cum += print_dict['disc_ratio']

            gt = gt.view(batch_size,2,input_each_num,gt.shape[1],gt.shape[2],gt.shape[3])
            score_output:torch.Tensor = score_output.view(batch_size,2,input_each_num,score_output.shape[1],score_output.shape[2],score_output.shape[3])
            splited = score_output.split(1,dim=1)
            ori_output = torch.squeeze(splited[0],dim=1)
            X_output =  torch.squeeze(splited[1],dim=1)

            additional_loss = loss_MSE(ori_output,X_output)
            loss_score = loss_MSE(score_output, gt)
            loss_score_value = 0

            if loss_disc_exist:
                loss_score_value = loss_score.item()
                loss_disc =  loss_score + loss_disc
                # loss = loss_score + loss_disc
            loss = loss_score + additional_loss + loss_disc

            loss.backward()
            optimizer.step()

            loss_score_cum += loss_score_value

            if cum_iter_num % print_interval == print_interval - 1:
                print('## ep:%d' % epoch, 'iter:%d/%d ##' % (idx, iter_num - 1),
                      ' loss_detection:%.3f' % (loss_score_cum * 1000 / (image_base_num + 0.1)),
                      ' loss_description:%.4f' % (disc_value_cum / (image_base_num + 0.1)),
                      ' loss_informativeness:%.4f \n' % (info_value_cum / (image_base_num + 0.1)),
                      ' loss_det_foreground:%.4f' % (fore_loss_cum / (fore_num_cum + 0.1)),
                      ' loss_det_background:%.4f' % (back_loss_cum / (back_num_cum + 0.1)),
                      ' repeatability:%.4f' % (repeat_mean_cum / (image_base_num + 0.1)),
                      ' point_number:%.2f \n' % (fore_num_cum_runtime / (image_real_num + 0.1)))
                loss_score_cum = disc_value_cum = info_value_cum = disc_ratio_cum = 0
                fore_loss_cum = fore_num_cum = back_loss_cum = back_num_cum = 0
                repeat_mean_cum = image_base_num = 0
                fore_num_cum_runtime = image_real_num = 0

            if cum_iter_num % save_interval == save_interval - 1:
                torch.save({
                    # 'model_state_dict': net.module.state_dict(),
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                }, os.path.join(save_path, 'POP_net_temp.pth'))
        time_end = time.time()
        time_batch = time_end-time_last
        print("耗时：",time_batch)

        if epoch % save_epoch_interval == save_epoch_interval - 1:
            torch.save({
                # 'model_state_dict': net.module.state_dict(),
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }, os.path.join(save_path, 'POP_net_epoch_end%d.pth' % (epoch)))


def main():
    parser = argparse.ArgumentParser(description="POP interest point Training")

    # 'demo_images'
    # '/home/ubuntu/yanpei/data/data_yanpei/train2014'
    parser.add_argument('--train-image-path', type=str,
                        default='demo_input_images',
                        help='the path of training images, which should be ' \
                             'a directory containing only images')
    parser.add_argument('--POP-checkpoint-path', type=str,
                        default=None,
                        help='s he path of checkpoint if needed, ' \
                             'and use the default when train the model from scratch')
    parser.add_argument('--device', type=str, default='cpu',
                        help='the device used to train the model, ' \
                             'which is represented with the pytorch format')
    parser.add_argument('--max-epoch', type=int, default=2,
                        help='the maximal number of epochs')
    parser.add_argument('--data-num-workers', type=int, default=2,
                        help='the num_workers of DataLoader')
    parser.add_argument('--train-width', type=int, default=320,
                        help='every training image will be resized with train_width')
    parser.add_argument('--train-height', type=int, default=240,
                        help='every training image will be resized with train_height')
    parser.add_argument('--save-epoch-interval', type=int, default=1,
                        help='the model will be saved every ' \
                             'save_epoch_interval iterations')
    parser.add_argument('--recon-net-path', type=str,
                        default=None,
                        help='set the path of the reconstructor model. ' \
                             'If this path is not None, the informativeness properties ' \
                             'will take effect in the model optimization')
    parser.add_argument('--save-path', type=str,
                        default='save_POP_model',
                        help='the path used to save the trained model')
    parser.add_argument('--print-iter-interval', type=int, default=4,
                        help='the training status will be printed every ' \
                             'print_iter_interval iterations')
    parser.add_argument('--save-iter-interval', type=int, default=1000,
                        help='the temporary model will be saved every ' \
                             'save_iter_interval iteration, and the filename is POP_net_temp.pth')
    parser.add_argument('--random-aug-num', type=int, default=10,
                        help='every training image will be transformed to ' \
                             'random_aug_num new images')

    args = parser.parse_args()
    # create the storage directory if needed
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    train_POP(args)


if __name__ == '__main__':
    main()
