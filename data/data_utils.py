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
sys.path.append(os.path.abspath(".."))


def default_loader(path=None,
                   resize_shape=None,
                   channel=3):
    transform_list = []
    if channel == 1:
        transform_list.append(transforms.Grayscale(1))
    target_size = [resize_shape, resize_shape]
    transform_list.append(transforms.Resize(target_size, Image.BICUBIC))

    transform_list += [transforms.ToTensor()]
    if channel == 1:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    image = Image.open(path).convert('RGB')
    trans = transforms.Compose(transform_list)
    image = trans(image)
    return image


def get_gt_tensor(this_gt, size):
    this_gt = this_gt + 180
    gt_tensor_self = torch.zeros(size, size)
    angle_convert = this_gt * size / 360
    angle_index = angle_convert // 1 + (angle_convert % 1 + 0.5) // 1
    if angle_index.long() == size:
        angle_index = size - 1
        gt_tensor_self[angle_index, 0] = 1
    else:
        gt_tensor_self[angle_index.long(), 0] = 1

    return gt_tensor_self
