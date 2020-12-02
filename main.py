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
from data.PRoGAN_dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from util.utils import *
from util.util import *
from util.visualizer import Visualizer
from options import *


def main():
    # get options
    parser = options()
    args = config(parser.parse_args())
    print_options(args, parser)
    if args is None:
        exit()
    if args.phase == 'train':
        args.isTrain = 1
    # load dataset 
    dataset = dataloader(args)
    # build PRoGAN model
    gan = PRoGAN(args)
    # tensorboard summary writer
    writer = SummaryWriter(
        log_dir=args.train_writer_path + str(args.name) + "/")
    total_iters = 0
    # training 
    if args.phase == 'train':
        # start visdom to visualize training process
        visualizer = Visualizer(args)
        for epoch in range(args.epoch_count, args.n_epochs + args.n_epochs_decay + 1):
            epoch_iter = 0
            # update learning rate
            gan.update_learning_rate()
            iter_data_time = time.time()
            visualizer.reset()
            # read images from dataset
            for origin, target, path in dataset:
                iter_start_time = time.time()
                if total_iters % args.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_iters += args.batch_size
                epoch_iter += args.batch_size
                # input the origin and target images to PRoGAN network
                gan.set_input(
                    origin, target, args.phase
                )
                # start training
                gan.train()
                # display the training process on visdom and tensorboard
                if total_iters % args.display_freq == 0:  # display images on visdom and save images to a HTML file
                    save_result = total_iters % args.update_html_freq == 0
                    visualizer.display_current_results(gan.get_current_visuals(),
                                                       epoch, save_result)
                    for label, image in gan.get_current_visuals().items():
                        writer.add_image(
                            label, image[0, :, :].cpu(), total_iters)
                # print training losses and save logging information to the disk
                if total_iters % args.print_freq == 0:  
                    losses = gan.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / args.batch_size
                    print_current_losses(args, epoch, epoch_iter, losses,
                                         t_comp, t_data)
                    for k, v in losses.items():
                        writer.add_scalar(k, v, total_iters)
                    if args.display_id > 0:
                        visualizer.plot_current_losses(
                            epoch,
                            float(epoch_iter) / len(dataset), losses)
                # save models
                if total_iters % args.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' %
                          (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if args.save_by_iter else 'latest'
                    gan.save_networks(save_suffix)
            # save models for epoch 
            if epoch % args.save_epoch_freq == 0:  
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_iters))
                gan.save_networks('latest')
                gan.save_networks(epoch)

            print(" [*] Training finished!")
    # validating
    elif args.phase == 'val':
        i = 0
        for origin, target, origin_path in dataset:
            i += 1
            # input the origin and target images to PRoGAN network
            gan.set_input(origin, target, args.phase)
            # test the model
            gan.test()
            # set the image save path
            save_path = args.val_writer_path + args.name + "_" + str(args.epoch) + "/"
            mkdirs(save_path)
            path = save_path + str(origin_path[0].split("/")[-1])
            save_image(tensor2im(gan.fake_B), path)
            if i % 100 == 0:
                print('Saving image %d / %d' % (i, len(dataset)))

        print(" [*] Test finished!")


if __name__ == '__main__':
    main()
