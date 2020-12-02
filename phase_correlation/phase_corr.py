import sys
import os
sys.path.append(os.path.abspath(".."))

import cv2
import math
import torch
import kornia
import numpy as np
import torch.nn as nn
from numpy.fft import fft2, ifft2, fftshift
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from util.utils import *
from log_polar.log_polar import *


def phase_corr(a, b, device, logbase, trans=False):
    # a: template; b: source
    # imshow(a.squeeze(0).float())
    G_a = torch.rfft(a, 2, onesided=False)
    G_b = torch.rfft(b, 2, onesided=False)
    eps = 1e-15

    real_a = G_a[:, :, :, 0]
    real_b = G_b[:, :, :, 0]
    imag_a = G_a[:, :, :, 1]
    imag_b = G_b[:, :, :, 1]

    # compute a * b.conjugate; shape=[B,H,W,C]
    R = torch.FloatTensor(G_a.shape[0], G_a.shape[1], G_a.shape[2],
                          2).to(device)
    R[:, :, :, 0] = real_a * real_b + imag_a * imag_b
    R[:, :, :, 1] = real_a * imag_b - real_b * imag_a

    r0 = torch.sqrt(real_a**2 + imag_a**2 + eps) * torch.sqrt(real_b**2 +
                                                              imag_b**2 + eps)
    R[:, :, :, 0] = R[:, :, :, 0].clone() / (r0 + eps).to(device)
    R[:, :, :, 1] = R[:, :, :, 1].clone() / (r0 + eps).to(device)

    r = torch.ifft(R, 2)
    r_real = r[:, :, :, 0]
    r_imag = r[:, :, :, 1]
    r = torch.sqrt(r_real**2 + r_imag**2 + eps)
    r = fftshift2d(r)
    if trans:
        r[:, 0:60, :] = 0.
        r[:, G_a.shape[1] - 60:G_a.shape[1], :] = 0.
        r[:, :, 0:60] = 0.
        r[:, :, G_a.shape[1] - 60:G_a.shape[1]] = 0.
    # imshow(r[0,:,:])
    # plt.show()

    angle_resize_out_tensor = torch.sum(r.clone(), 2, keepdim=False)
    scale_reszie_out_tensor = torch.sum(r.clone(), 1, keepdim=False)
    # get the argmax of the angle and the scale
    angle_out_tensor = torch.argmax(angle_resize_out_tensor.clone().detach(),
                                    dim=-1)
    scale_out_tensor = torch.argmax(scale_reszie_out_tensor.clone().detach(),
                                    dim=-1)
    if not trans:
        angle_out_tensor = angle_out_tensor * 180.00 / r.shape[1]
        for batch_num in range(angle_out_tensor.shape[0]):
            if angle_out_tensor[batch_num].item() > 90:
                angle_out_tensor[batch_num] -= 90.00
            else:
                angle_out_tensor[batch_num] += 90.00

        logbase = logbase.to(device)

        # sca_f = scale_out_tensor.clone() * 256 / r.shape[2] - 256 // 2
        scale_out_tensor = 1 / torch.pow(
            logbase, scale_out_tensor.clone())  #logbase ** sca_f

    return scale_out_tensor, angle_out_tensor, r, logbase


def highpass(shape):
    """Return highpass filter to be multiplied with fourier transform."""
    i1 = torch.cos(torch.linspace(-np.pi / 2.0, np.pi / 2.0, shape[0]))
    i2 = torch.cos(torch.linspace(-np.pi / 2.0, np.pi / 2.0, shape[1]))
    x = torch.einsum('i,j->ij', i1, i2)
    return (1.0 - x) * (1.0 - x)


def logpolar_filter(shape):
    """
    Make a radial cosine filter for the logpolar transform.
    This filter suppresses low frequencies and completely removes
    the zero freq.
    """
    yy = np.linspace(-np.pi / 2., np.pi / 2., shape[0])[:, np.newaxis]
    xx = np.linspace(-np.pi / 2., np.pi / 2., shape[1])[np.newaxis, :]
    # Supressing low spatial frequencies is a must when using log-polar
    # transform. The scale stuff is poorly reflected with low freqs.
    rads = np.sqrt(yy**2 + xx**2)
    filt = 1.0 - np.cos(rads)**2
    # vvv This doesn't really matter, very high freqs are not too usable anyway
    filt[np.abs(rads) > np.pi / 2] = 1
    filt = torch.from_numpy(filt)
    return filt


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


