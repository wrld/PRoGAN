from data.data_utils import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import random

sys.path.append(os.path.abspath(".."))


class PRoGAN_dataset(Dataset):
    """ PRoGAN dataloader
    The implementation of PRoGAN dataloader
    """
    def __init__(self,
                 origin_train_list,
                 target_train_list,
                 opt,
                 loader=default_loader):
        self.origin_path_list = origin_train_list
        self.target_path_list = target_train_list
        self.opt = opt
        self.loader = loader

    def __len__(self):
        return min(len(self.origin_path_list), len(self.target_path_list))

    def load_data(self):
        return self

    def __getitem__(self, index):
        this_origin_path = self.origin_path_list[index]
        this_target_path = self.target_path_list[index]
        this_origin = self.loader(
            this_origin_path,
            resize_shape=self.opt.load_size,
            channel=self.opt.output_nc)
        this_target = self.loader(
            this_target_path,
            resize_shape=self.opt.load_size,
            channel=self.opt.output_nc)

        return [
            this_origin, this_target, this_origin_path
        ]


def dataloader(opt):
    phase = opt.phase
    path = './datasets/' + opt.dataset
    # load trainA and trainB for training process
    if phase == 'train':
        origin_name = os.listdir(path + "/trainA/")
        origin_name.sort()
        origin_train_list = [
            os.path.join(path + "/trainA/", file_path)
            for file_path in origin_name
        ]
        target_name = os.listdir(path + "/trainB/")
        target_name.sort()
        target_train_list = [
            os.path.join(path + "/trainB/", file_path)
            for file_path in target_name
        ]
        data_set = PRoGAN_dataset(
            origin_train_list, target_train_list, opt)
    # load testA and testB for validating process
    elif phase == 'val':
        origin_name = os.listdir(path + "/testA/")
        origin_name.sort()
        origin_val_list = [
            os.path.join(path + "/testA/", file_path)
            for file_path in origin_name
        ]
        target_name = os.listdir(path + "/testB/")
        target_name.sort()
        target_val_list = [
            os.path.join(path + "/testB/", file_path)
            for file_path in target_name
        ]
        data_set = PRoGAN_dataset(
            origin_val_list, target_val_list, opt)
    dataloaders = DataLoader(data_set,
                             batch_size=opt.batch_size,
                             shuffle=True,
                             num_workers=int(opt.num_threads))
    return dataloaders
