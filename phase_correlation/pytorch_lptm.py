from torchvision import models
from util.utils import *
from log_polar.log_polar import polar_transformer
from phase_correlation.phase_corr import phase_corr
import numpy as np
import torch.nn as nn
import torch
import sys
import os
sys.path.append(os.path.abspath(".."))
print("sys path", sys.path)


class LogPolar(nn.Module):
    def __init__(self, out_size, device):
        super(LogPolar, self).__init__()
        self.out_size = out_size
        self.device = device

    def forward(self, input):
        return polar_transformer(input, self.out_size, self.device)


class PhaseCorr(nn.Module):
    def __init__(self, device, logbase, trans=False):
        super(PhaseCorr, self).__init__()
        self.device = device
        self.logbase = logbase
        self.trans = trans

    def forward(self, template, source):
        return phase_corr(template,
                          source,
                          self.device,
                          self.logbase,
                          trans=self.trans)


class FFT2(nn.Module):
    def __init__(self, device):
        super(FFT2, self).__init__()
        self.device = device

    def forward(self, input):
        # print("fft input", np.shape(input))
        median_output = torch.rfft(input, 2, onesided=False)
        # print("median_output", np.shape(median_output))
        median_output_r = median_output[:, :, :, 0]  # real
        median_output_i = median_output[:, :, :, 1]  # virtual
        # print("median_output r", median_output_r)
        # print("median_output i", median_output_i)
        output = torch.sqrt(median_output_r**2 + median_output_i**2 + 1e-15)
        # print("output before shift", np.shape(output))
        # output = median_outputW_r
        output = fftshift2d(output)
        # print("output after shift", np.shape(output))
        # h = logpolar_filter((output.shape[1],output.shape[2]), self.device)
        # output = output.squeeze(0) * h
        # output = output.unsqueeze(-1)
        output = output.unsqueeze(-1)
        return output


def double_conv(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1),
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))



class Corr2Softmax(nn.Module):
    def __init__(self, weight, bias):

        super(Corr2Softmax, self).__init__()
        softmax_w = torch.tensor((weight), requires_grad=True)
        softmax_b = torch.tensor((bias), requires_grad=True)
        self.softmax_w = torch.nn.Parameter(softmax_w)
        self.softmax_b = torch.nn.Parameter(softmax_b)
        self.register_parameter("softmax_w", self.softmax_w)
        self.register_parameter("softmax_b", self.softmax_b)

    def forward(self, x):
        x1 = self.softmax_w * x + self.softmax_b
        # print("w = ",self.softmax_w, "b = ",self.softmax_b)
        # x1 = 1000. * x
        return x1
