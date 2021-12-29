import argparse
import math

import torch
import os


def Parser():
    # init
    args = argparse.ArgumentParser(description='universa torch parser')

    # device
    args.add_argument('-d', '--device', default=torch.device('cuda0' if torch.cuda.is_available() else 'cpu'))

    # data
    args.add_argument('--data_name', default='ImageNet', type=str)
    args.add_argument('--data_dir', default='/home/lyp/Data/dateset', type=str, required=True)
    args.add_argument('--image_size', default=256, type=int)
    args.add_argument('--num_workers', default=8, type=int)

    # model
    # - use ViT-Base whose parameters are referred from "Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers
    # - for Image Recognition at Scale. ICLR 2021. https://openreview.net/forum?id=YicbFdNTTy".
    args.add_argument('--patch_size', default=32, type=int)
    args.add_argument('--vit_dim', default=768, type=int)
    args.add_argument('--vit_depth', default=12, type=int)
    args.add_argument('--vit_heads', default=12, type=int)
    args.add_argument('--vit_mlp_dim', default=3072, type=int)
    args.add_argument('--masking_ratio', default=0.80, type=float)
    args.add_argument('--decoder_dim', default=512, type=int)
    args.add_argument('--decoder_depth', default=1, type=int)

    # train
    args.add_argument('--batch_size', default=4096, type=int)
    args.add_argument('--epochs', default=800, type=int)
    args.add_argument('base_lr', default=1.5e-4, type=float)
    args.add_argument('--weight_decay', default=5e-2, type=float)
    args.add_argument('--momentum', default=(0.9, 0.95), type=tuple)
    args.add_argument('epoch_warmup', default=40, type=int)
    args.add_argument('--warmup_from', default=1e-4, type=float)
    args.add_argument('--lr_decay_rate', default=1e-2, type=float)
    eta_min = args.lr * (args.lr_decay_rate ** 3)
    args.add_argument('--warmup_to', default=eta_min + (args.lr - eta_min) *
                                             (1 + torch.cos(math.pi * args.epochs_warmup / args.epochs)) / 2)

    # print and save
    args.add_argument('--print_freq', default=5, type=int)
    args.add_argument('--save_freq', default=100, type=int)

    # tensorboard
    trail = 0
    args.add_argument('tb_folder', default=os.path.join(os.getcwd(), 'log', '{}_{}'.format(args.data_name, trail)))
    if not os.path.exists(args.tb_folder):
        os.makedirs(args.tb_folder)

    # ckpt
    args.ckpt_folder = os.path.join(os.getcwd(), 'ckpt', '{}_{}'.format(args.data_name, trail))
    if not os.path.isdir(args.ckpt_folder):
        os.makedirs(args.ckpt_folder)
