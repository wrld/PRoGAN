import time
import kornia
from tensorboardX import SummaryWriter
from datetime import datetime
import argparse
from models.PRoGAN import PRoGAN
import os
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import importlib
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from util.util import *
from util.visualizer import Visualizer


def options():
    """
    Options for training and validating
    """
    desc = "Pytorch implementation of PRoGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--display_id',
                        type=int,
                        default=10,
                        help='window id of the web display')
    parser.add_argument('--display_server',
                        type=str,
                        default="http://localhost",
                        help='visdom server of the web display')
    parser.add_argument(
        '--display_env',
        type=str,
        default='main',
        help='visdom display environment name (default is "main")')
    parser.add_argument('--display_port',
                        type=int,
                        default=8097,
                        help='visdom port of the web display')
    parser.add_argument(
        '--update_html_freq',
        type=int,
        default=500,
        help='frequency of saving training results to html')
    parser.add_argument(
        '--print_freq',
        type=int,
        default=20,
        help='frequency of showing training results on console')
    parser.add_argument(
        '--no_html',
        action='store_true',
        help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/'
    )
    parser.add_argument('--save_latest_freq',
                        type=int,
                        default=500,
                        help='frequency of saving the latest results')
    parser.add_argument(
        '--save_epoch_freq',
        type=int,
        default=1,
        help='frequency of saving checkpoints at the end of epochs')

    parser.add_argument(
        '--name',
        type=str,
        default='experiment_name',
        help='name of the experiment'
    )
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='0',
                        help='The ids of gpu to use, gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir',
                        type=str,
                        default='./checkpoints',
                        help='The place to save models')

    parser.add_argument(
        '--input_nc',
        type=int,
        default=3,
        help='The channel of input image')
    parser.add_argument(
        '--output_nc',
        type=int,
        default=3,
        help='The channel of output image')
    parser.add_argument('--ngf',
                        type=int,
                        default=64,
                        help='# of gen filters in the last conv layer')
    parser.add_argument(
        '--ndf',
        type=int,
        default=64,
        help='# of discrim filters in the first conv layer')
    parser.add_argument(
        '--netG',
        type=str,
        default='resnet_9blocks',
        help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]'
    )
    parser.add_argument('--n_layers_D',
                        type=int,
                        default=3,
                        help='only used if netD==n_layers')

    parser.add_argument('--no_dropout',
                        action='store_true',
                        help='no dropout for the generator')
    parser.add_argument('--num_threads',
                        default=4,
                        type=int,
                        help='# threads for loading data')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='The batch size of input')
    parser.add_argument('--load_size',
                        type=int,
                        default=256,
                        help='The size of input image for network (128 / 256)')
    parser.add_argument(
        '--display_winsize',
        type=int,
        default=256,
        help='display window size for both visdom and HTML')
    parser.add_argument(
        '--epoch',
        type=str,
        default='latest',
        help='The start epoch for continuing to train'
    )

    parser.add_argument('--save_by_iter',
                        action='store_true',
                        help='whether saves model by iteration')
    parser.add_argument('--continue_train',
                        action='store_true',
                        help='Continue to train')
    parser.add_argument(
        '--epoch_count',
        type=int,
        default=1,
        help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...'
    )
    parser.add_argument('--phase',
                        type=str,
                        default='train',
                        help='Choose to train or validate (train / val)')
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=100,
        help='number of epochs with the initial learning rate')
    parser.add_argument(
        '--n_epochs_decay',
        type=int,
        default=100,
        help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--beta1',
                        type=float,
                        default=0.5,
                        help='momentum term of adam')
    parser.add_argument('--lr',
                        type=float,
                        default=0.00015,
                        help='initial learning rate for adam')
    parser.add_argument(
        '--pool_size',
        type=int,
        default=0,
        help='the size of image buffer that stores previously generated images')
    parser.add_argument(
        '--lr_decay_iters',
        type=int,
        default=50,
        help='multiply by a gamma every lr_decay_iters iterations')

    parser.add_argument('--train_writer_path',
                        type=str,
                        default="./checkpoints/log/",
                        help='Where to write the Log of training')
    parser.add_argument('--val_writer_path',
                        type=str,
                        default="./outputs/",
                        help='Where to save the images of validating')
    parser.add_argument('--dataset',
                        type=str,
                        default="night",
                        help='dataset')
    parser.add_argument('--lambda_L1',
                        type=float,
                        default=100.0,
                        help='weight for L1 loss')

    parser.add_argument(
        '--display_freq',
        type=int,
        default=20,
        help='frequency of showing training results on screen')
    parser.add_argument('--use_scale',
                        action='store_true',
                        help='Use scale for transformation')
    parser.add_argument('--aeroground',
                        action='store_true',
                        help='Use dataset: AeroGround')
    parser.add_argument('--carla',
                        action='store_true',
                        help='Use dataset: carla')

    return parser


def config(opt):
    """The configuration of options"""
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
    if opt.aeroground:
        opt.dataset = 'stereo'
        opt.netG = 'resnet_6blocks'
        opt.load_size = 128
        opt.input_nc = 1
        opt.output_nc = 1
        opt.name = 'aeroground_train'
    if opt.carla:
        opt.dataset = 'night'
        opt.netG = 'unet_256'
        opt.load_size = 256
        opt.input_nc = 3
        opt.output_nc = 3
        opt.name = 'carla_train'
    if opt.continue_train:
        opt.epoch_count = int(opt.epoch) + 1
    return opt


def print_options(opt, parser):
    """Print the options for training and validating"""
    message = ''
    message += '----------------- PRoGAN Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def print_current_losses(args, epoch, iters, losses, t_comp, t_data):
    log_name = os.path.join(args.checkpoints_dir, args.name,
                            'loss_log.txt')
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (
        epoch, iters, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)  # print the message
    print('\n')
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message
